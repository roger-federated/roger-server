"""aggregate.py — secure-aggregation round lifecycle + FedAvg accumulation (server core).

Lock-held methods (register, seal, stage bookkeeping, claim_finalize) run with no await, so they are
atomic under the single event-loop thread; the I/O-heavy `run_finalize` runs OFF it in a threadpool,
touching only a round already pulled from `collecting` + storage, so it can't race the lock-held state.

What is aggregated is ONE LoRA factor per epoch (`delta.py`): in a B-epoch every member trains B
against the same frozen A, uploads ΔB, and the server folds G_B += η·mean(ΔB); an A-epoch mirrors it.
That is what makes secure aggregation exact here — the server only ever sees Σ Δ, and Σ(ΔB_i)·A =
Σ(ΔB_i·A), while summing whole factor PAIRS would leave the cross terms B_i A_j behind. The epoch
advances after `epoch_folds` successful folds and is stamped on every upload, so an update trained in
a past epoch (against a factor that has since moved) is voided rather than folded.

Lifecycle (per model_id, one registering round at a time; on seal it gets a round_id and moves aside,
so cohorts collect concurrently, self-routed by round_id):
  REGISTERING  /round/register collects X25519 pubkeys, issuing each registrant a secret upload token.
  -> sealed    at K_target, or at the W deadline if >= K_min (else FAILED: sub-K_min leaks a lone Δ).
  COLLECTING   /contribute streams each masked upload to its OWN temp object (never a RAM running-sum),
               gated on the registrant's token so only cohort members upload, each at most once.
  -> finalized all uploaded, or U deadline: sum staged slices per module (masks cancel), fold into the
               global, voiding if ‖ΣΔ‖ exceeds k·clip. Masks cancel only if EVERY member uploaded.

Staging (not a RAM running-sum) because a masked vector spans the epoch's whole factor basis, and a
cohort's worth of those resident for the whole window × concurrent cohorts overruns serverless tiers;
per-module keeps RAM ~one factor. Bootstrap (sparse `mode()`): one unmasked DP Δ to /contribute_dp,
`dp_fold` (k=1) — the same per-module streamed fold, just no cohort.
"""
import json, secrets, uuid

import torch

from roger_server import delta
from roger_server import secure_agg
from roger_server import store


class Round:
    """Mutable state for one aggregation round. A class (not a dict) because the lifecycle flags and the
    staged-upload bookkeeping are genuinely stateful and mutated in place across many calls."""
    def __init__(self, model_id: str, created: float):
        self.model_id = model_id
        self.created = created                  # time.monotonic() at first registration (W deadline)
        # pubkey hex -> secret upload token (insertion order = arrival). The token is this registrant's
        # proof that it's the same party /contribute must accept for that slot (not the source IP, which
        # is spoofable/NAT-shared).
        self.registrants: dict[str, str] = {}
        self.spent_tokens: set = set()          # tokens already claimed (reserved or completed) at /contribute
        self.sealed = False
        self.failed = False                     # cohort too small, or a mismatched contribution
        self.finalized = False
        self.finalize_scheduled = False         # app spawns the U-timeout task once
        self.round_id: str | None = None         # assigned on seal; clients echo it on /contribute
        self.sealed_peers: list[str] | None = None
        # set by the first valid contribution; the cohort must agree on layout for masks to cancel
        self.compat: str | None = None
        self.base: dict | None = None           # {module: (out, in)} of the base this cohort targets
        self.epoch: int | None = None           # the epoch every member of this cohort trained in
        self.spec: list | None = None           # [(factor_key, shape_tuple), ...]
        self.length: int | None = None
        self.staged: set = set()                # slot ids of fully-uploaded temp objects
        self.inflight = 0                       # reservations whose upload is still streaming
        self.received = 0

    def token_pubkey(self, token: str) -> str | None:
        for pubkey, tok in self.registrants.items():
            if tok == token:
                return pubkey
        return None


def _parse_spec(spec_json: str) -> tuple:
    """JSON [[key,[shape...]],...] -> ([(key, shape_tuple)], total_flat_length)."""
    spec, length = [], 0
    for key, shape in json.loads(spec_json):
        shape = tuple(shape)
        spec.append((key, shape))
        n = 1
        for d in shape:
            n *= d
        length += n
    return spec, length


class Aggregator:
    def __init__(self, datadir: str, *, k_min: int = 3, k_target: int = 5, eta: float = 1.0,
                 eta_boot: float | None = None, clip_norm: float = 1.0,
                 allowlist: set | None = None, busy_threshold: int | None = None,
                 busy_window: float = 180.0, rank: int = 16, epoch_folds: int = 20):
        self.datadir = datadir
        self.k_min = k_min
        self.k_target = k_target
        self.eta = eta                          # server learning rate: G <- G + η·mean(Δ)
        self.eta_boot = eta if eta_boot is None else eta_boot  # rate for a single async DP upload
        self.clip_norm = clip_norm              # aggregate L2 cap (honest cohort sums to ≤ k·clip_norm)
        self.allowlist = allowlist              # None = accept any model_id
        self.busy_threshold = k_target if busy_threshold is None else busy_threshold
        self.busy_window = busy_window
        self.rank = rank                        # rank CAP a NEW model's global is created at (delta.rank_for)
        self.epoch_folds = epoch_folds          # folds per epoch before the trained factor swaps
        self.recent: dict[str, dict[str, float]] = {}  # model_id -> {ip: last-seen monotonic time}
        self.registering: dict[str, Round] = {} # model_id -> the round open for registration
        self.collecting: dict[str, Round] = {}  # round_id -> a sealed round awaiting uploads (many at once)
        self._state: dict[str, dict] = {}       # model_id -> cached factor state (see `state`)

    # --- mode / density ---------------------------------------------------------------------

    def mode(self, model_id: str, now: float) -> str:
        """"unsupported" when an allowlist is set and excludes this model — surfaced here (not only as the
        register/contribute 403) so a client can warn the user up front + skip, instead of wasting a
        session's training and upload on a gradient no federation will take. Else "busy" (secure-agg
        cohorts) once >= busy_threshold distinct contributors fall inside the busy_window, else
        "bootstrap" (async DP). Prunes the window in passing."""
        if self.allowlist is not None and model_id not in self.allowlist:
            return "unsupported"                # permanent + actionable, unlike a transient seal failure
        seen = self.recent.get(model_id)
        if not seen:
            return "bootstrap"
        cutoff = now - self.busy_window
        live = {ip: t for ip, t in seen.items() if t >= cutoff}
        self.recent[model_id] = live
        return "busy" if len(live) >= self.busy_threshold else "bootstrap"

    def _note_contributor(self, model_id: str, ip: str, now: float) -> None:
        self.recent.setdefault(model_id, {})[ip] = now

    # --- factor state (which factor to train, at which rank, in which epoch) -----------------

    def state(self, model_id: str) -> dict:
        """The federation's factor state for a model: {epoch, phase, rank, folds, compat}. It is durable
        in the global's metadata; cached here because the server is single-instance by construction, so
        nothing else can move it (every fold updates the cache in the same breath as the blob). Does
        storage I/O on a cache miss, so callers hold no lock while warming it."""
        st = self._state.get(model_id)
        if st is None:
            meta = store.load_meta(self.datadir, model_id)
            epoch = int(meta.get("epoch", 0))
            st = {"epoch": epoch, "phase": delta.phase_of(epoch), "folds": int(meta.get("folds", 0)),
                  "rank": int(meta.get("rank", self.rank)),   # the CAP; a stored global keeps its own
                  "compat": meta.get("compat")}
            self._state[model_id] = st
        return st

    def round_model(self, round_id: str) -> str | None:
        """The model a collecting round belongs to (so the wire layer can warm `state` before locking)."""
        rnd = self.collecting.get(round_id)
        return rnd.model_id if rnd is not None else None

    def check_upload(self, model_id: str, compat: str, base_json: str, epoch: int, spec: list) -> tuple:
        """Shared validation for both upload paths. Returns ({module: (out, in)}, "ok") or (None, error).
        An upload must target this model's current epoch (so its frozen factor is the one still in the
        global), carry only that epoch's factor at the federation's rank, and agree with the stored
        global's base shapes — otherwise it is not addable to the global at all."""
        st = self.state(model_id)
        if epoch != st["epoch"]:
            return None, "stale epoch"          # trained against a factor the federation has moved past
        base = delta.base_from_json(base_json)
        if not base or delta.compat_from_shapes(base) != compat:
            return None, "bad base shapes"      # `base` must be exactly what `compat` digests
        if st["compat"] is not None and compat != st["compat"]:
            return None, "compat mismatch"      # a different architecture claiming this model_id
        if not spec:
            return None, "empty delta"
        suffix, cap = delta.SUFFIX[st["phase"]], st["rank"]
        for key, shape in spec:
            module = delta.module_of(key)
            if module is None or not key.endswith(suffix):
                return None, f"expected {st['phase']}-factor updates this epoch"
            if module not in base or len(shape) != 2:
                return None, "unknown module"
            out_dim, in_dim = base[module]
            rank = delta.rank_for(out_dim, in_dim, cap)   # the map is a pure function of shape + cap
            want = (rank, in_dim) if suffix == delta.LORA_A else (out_dim, rank)
            if tuple(shape) != want:
                got_rank = shape[0] if suffix == delta.LORA_A else shape[1]
                return None, (f"rank mismatch ({module} trains at rank {rank} here)" if got_rank != rank
                              else f"{key} does not match its base module {base[module]}")
        return base, "ok"

    # --- registration / sealing -------------------------------------------------------------

    def add_registrant(self, model_id: str, pubkey: str, ip: str, now: float) -> tuple[Round, str]:
        """Join this model_id's open registering round, opening a fresh one if none is registering.
        Returns (round, token): a fresh secret this registrant must present at /contribute to prove
        it's the one that received this cohort's peer set (not just anyone sharing its IP)."""
        self._note_contributor(model_id, ip, now)
        rnd = self.registering.get(model_id)
        if rnd is None:
            rnd = Round(model_id, created=now)
            self.registering[model_id] = rnd
        token = secrets.token_urlsafe(32)
        rnd.registrants[pubkey] = token
        return rnd, token

    def try_seal(self, rnd: Round, *, final: bool = False) -> None:
        """Seal at k_target (immediate) or, at the W deadline (`final`), at k_min. A `final` round below
        k_min FAILS — sealing it would hand the server a near-unmasked single contribution."""
        if rnd.sealed or rnd.failed:
            return
        n = len(rnd.registrants)
        if n >= self.k_target or (final and n >= self.k_min):
            rnd.sealed = True
            rnd.sealed_peers = list(rnd.registrants.keys())
            rnd.round_id = uuid.uuid4().hex
            self.collecting[rnd.round_id] = rnd
            self._clear_registering(rnd)
        elif final:
            rnd.failed = True
            self._clear_registering(rnd)

    # --- contribution staging (lock-held bookkeeping; the I/O is in app.py) ------------------

    def begin_stage(self, round_id: str, compat: str, spec_json: str, base_json: str, epoch: int,
                    token: str, *, masked_i64: bool = True, masked_len: int | None = None) -> tuple:
        """Validate an incoming upload and reserve a slot. Returns (round, slot_id, "ok") or
        (None, None, error). The caller streams the body to `store.stage_writer(.., round_id, slot)`,
        then calls mark_received (success) or unreserve (failure). `token` must be the secret this
        round's /round/register handed to a specific registrant, proving the uploader is that same
        party (not just anyone sharing its IP) and letting each registrant upload at most once —
        the cohort is capped at k by construction (there are only k valid tokens), with no separate
        count check needed. masked_i64/masked_len come from the upload's safetensors header (we never
        decode the whole tensor here)."""
        rnd = self.collecting.get(round_id)
        if rnd is None or rnd.finalized:
            return None, None, "no active round"
        if rnd.token_pubkey(token) is None:
            return None, None, "invalid token"
        if token in rnd.spent_tokens:
            return None, None, "token already used"
        if not masked_i64:
            return None, None, "bad tensor"          # masked must be a 1-D int64 residue vector
        spec, length = _parse_spec(spec_json)
        if masked_len is not None and masked_len != length:
            return None, None, "length mismatch"
        base, res = self.check_upload(rnd.model_id, compat, base_json, epoch, spec)
        if res != "ok":
            return None, None, res
        if rnd.compat is None:                        # first contribution fixes the cohort's layout
            rnd.compat, rnd.base, rnd.epoch, rnd.spec, rnd.length = compat, base, epoch, spec, length
        elif compat != rnd.compat or length != rnd.length or epoch != rnd.epoch:
            rnd.failed = True                         # mixed layouts can't cancel coordinate-wise -> void
            return None, None, "compat mismatch"
        rnd.spent_tokens.add(token)                   # reserve now; freed by unreserve() on stream failure
        rnd.inflight += 1
        return rnd, uuid.uuid4().hex, "ok"

    def mark_received(self, rnd: Round, slot: str) -> None:
        rnd.inflight -= 1
        rnd.staged.add(slot)
        rnd.received = len(rnd.staged)

    def unreserve(self, rnd: Round, token: str) -> None:
        rnd.inflight -= 1
        rnd.spent_tokens.discard(token)

    def complete(self, rnd: Round) -> bool:
        return rnd.received == len(rnd.sealed_peers)

    # --- bootstrap DP fold (per-module streamed, k=1) ---------------------------------------

    def dp_precheck(self, model_id: str, ip: str, now: float) -> str:
        """Cheap + lock-held: allowlist + count the uploader toward the busy threshold."""
        if self.allowlist is not None and model_id not in self.allowlist:
            return "model_id not accepted"
        self._note_contributor(model_id, ip, now)
        return "ok"

    def dp_fold(self, model_id: str, base: dict, epoch: int, round_id: str, slot: str) -> str:
        """I/O-bound (run off-thread): per-module fold of ONE staged unmasked factor Δ (k=1). Reads the
        upload module-by-module from its staged temp object, so nothing is whole in RAM."""
        up = store.open_staged_reader(self.datadir, round_id, slot)
        if up is None or not up.keys:
            return "empty delta"
        shapes = {key: shape for key, (_, shape) in up.keys.items()}
        return self._fold_streamed(model_id, base, shapes, lambda key, shape: up.read(key),
                                   k=1, eta=self.eta_boot, epoch=epoch)

    def submit_dp(self, model_id: str, factors: dict, base: dict, ip: str, now: float) -> str:
        """Sync bootstrap submit (tests / direct use): stage the factor Δ then fold it. The app instead
        streams the upload straight into the temp object. "ok" or an error/void reason."""
        pre = self.dp_precheck(model_id, ip, now)
        if pre != "ok":
            return pre
        spec = [(key, tuple(t.shape)) for key, t in factors.items()]
        checked, res = self.check_upload(model_id, delta.compat_from_shapes(base),
                                         delta.base_to_json(base), self.state(model_id)["epoch"], spec)
        if res != "ok":
            return res
        round_id, slot = "dp-" + uuid.uuid4().hex, "u"
        w = store.stage_writer(self.datadir, round_id, slot)
        w.write(delta.to_bytes(factors, model_id, {"base": delta.base_to_json(base)}))
        w.commit()
        try:
            return self.dp_fold(model_id, checked, self.state(model_id)["epoch"], round_id, slot)
        finally:
            store.stage_cleanup(self.datadir, round_id)

    # --- cohort finalization (claim under lock, aggregate off-thread) ------------------------

    def claim_finalize(self, rnd: Round) -> str:
        """Atomically claim a round for finalization (idempotent). Returns "aggregate" (this caller owns
        a complete cohort), "void" (this caller owns an incomplete/failed round — cleanup only), or
        "done" (already claimed). The heavy work then runs in run_finalize off the event loop."""
        if rnd.finalized:
            return "done"
        rnd.finalized = True
        if rnd.round_id:
            self.collecting.pop(rnd.round_id, None)
        if (rnd.failed or not rnd.sealed or rnd.sealed_peers is None
                or rnd.received != len(rnd.sealed_peers) or rnd.spec is None):
            return "void"
        return "aggregate"

    def run_finalize(self, rnd: Round, status: str) -> bool:
        """I/O-bound; run in a threadpool (no event-loop lock). Aggregate a complete cohort into the
        global, then always delete the round's temp objects. Returns whether the global was updated."""
        updated = False
        try:
            if status == "aggregate":
                updated = self._aggregate_streamed(rnd, len(rnd.sealed_peers))
        except Exception:
            updated = False                      # writer aborted inside; global untouched -> void
        finally:
            if rnd.round_id:
                store.stage_cleanup(self.datadir, rnd.round_id)
        return updated

    def _module_sum(self, rnd: Round, slots: list, data_starts: dict, key: str, shape: tuple,
                    moff: int, n: int) -> torch.Tensor:
        """ΣΔ for one factor: sum each slot's int64 slice mod R (masks cancel per coord), dequantize."""
        acc = None
        for slot in slots:
            start = data_starts[slot] + moff * 8         # staged "masked" is int64 (8 B/elem)
            s = torch.frombuffer(bytearray(store.stage_read_slice(self.datadir, rnd.round_id, slot, start, n * 8)),
                                 dtype=torch.int64)
            acc = s.clone() if acc is None else (acc + s) % secure_agg.R
        return secure_agg.dequantize(acc % secure_agg.R, [(key, tuple(shape))])[key]

    def _aggregate_streamed(self, rnd: Round, k: int) -> bool:
        """Cohort finalize: reconstruct ΣΔ per factor from the staged slices, then stream-fold it."""
        slots = list(rnd.staged)
        data_starts = {}
        for slot in slots:
            ds, length = store.stage_header(self.datadir, rnd.round_id, slot)
            if length != rnd.length:
                return False                             # malformed staged object -> void
            data_starts[slot] = ds

        offsets, off = {}, 0                             # key -> (flat-offset, flat-len), spec order
        for key, shape in rnd.spec:
            n = 1
            for d in shape:
                n *= d
            offsets[key] = (off, n)
            off += n
        shapes = {key: shape for key, shape in rnd.spec}
        module = lambda key, shape: self._module_sum(rnd, slots, data_starts, key, shape, *offsets[key])
        return self._fold_streamed(rnd.model_id, rnd.base, shapes, module,
                                   k=k, eta=self.eta, epoch=rnd.epoch) == "ok"

    def _fold_streamed(self, model_id: str, base: dict, upd_shape: dict, factor_delta, *,
                       k: int, eta: float, epoch: int | None = None) -> str:
        """Fold η·mean(Δ) into the epoch's trainable factor, one module at a time (peak RAM ~one factor),
        in a SINGLE pass: stream the untouched frozen factor + the updated trainable one while measuring
        ‖ΣΔ‖, then commit iff within k·clip. An honest clipped cohort always is (triangle ineq); an
        over-norm aggregate only comes from a client that skipped its clip, so we VOID it (consistent
        with dropout-void). `factor_delta(key, shape)` yields ΣΔ for one factor (cohort: summed staged
        slices; bootstrap: the uploaded tensor). `epoch` re-checks, now that same-model folds are
        serialized, that a cohort sealed in one epoch isn't landing in the next."""
        st = self.state(model_id)
        if epoch is not None and epoch != st["epoch"]:
            return "stale epoch"                         # another fold advanced the epoch while we waited
        if st["compat"] is not None and delta.compat_from_shapes(base) != st["compat"]:
            return "compat mismatch"
        cap, suffix = st["rank"], delta.SUFFIX[st["phase"]]
        reader = store.open_global_reader(self.datadir, model_id)
        old = dict(reader.keys) if reader is not None else {}

        order = []                                       # every module carries BOTH factors, A first
        for module in sorted(base):
            out_dim, in_dim = base[module]
            rank = delta.rank_for(out_dim, in_dim, cap)
            order.append((module + delta.LORA_A, (rank, in_dim)))
            order.append((module + delta.LORA_B, (out_dim, rank)))
        folds = st["folds"] + 1
        advance = folds >= self.epoch_folds              # the trained factor swaps at the epoch boundary
        new_epoch = st["epoch"] + 1 if advance else st["epoch"]
        meta = {"rank": str(cap), "epoch": str(new_epoch), "phase": delta.phase_of(new_epoch),
                "folds": "0" if advance else str(folds),
                "scaling": "1"}                          # the scale is already inside the factors
        writer = store.global_writer(self.datadir, model_id, [(key, "F32", shape) for key, shape in order],
                                     meta)
        sq, coef = 0.0, eta / k
        try:
            for key, shape in order:
                if key in old:
                    val = reader.read(key).float()
                elif key.endswith(delta.LORA_A):
                    # A fresh federation's frozen A is derived, not contributed, so a cold client can
                    # train its ΔB against the very A that lands here (delta.init_A).
                    val = delta.init_A(model_id, delta.module_of(key), shape[0], shape[1])
                else:
                    val = torch.zeros(shape)             # LoRA starts at B = 0
                if key.endswith(suffix) and key in upd_shape:
                    d = factor_delta(key, shape).float()
                    if not torch.isfinite(d).all():
                        writer.abort()
                        return "non-finite delta"        # void: global untouched
                    sq += float((d ** 2).sum())
                    val = val + coef * d
                writer.write(val.contiguous().to(torch.float32).numpy().tobytes())
        except Exception:
            writer.abort()
            raise
        if sq ** 0.5 > self.clip_norm * k:               # over the honest bound -> void, don't fold abuse
            writer.abort()
            return "norm exceeded"
        writer.commit(store.load_version(self.datadir, model_id) + 1)
        self._state[model_id] = {"epoch": new_epoch, "phase": delta.phase_of(new_epoch),
                                 "folds": 0 if advance else folds, "rank": cap,
                                 "compat": delta.compat_from_shapes(base)}
        return "ok"

    # --- broadcast --------------------------------------------------------------------------

    def serve_global(self, model_id: str, since: str):
        """(chunk_iter, version) for the global LoRA adapter, or None when nothing newer than `since`
        (cursor == version) or no global yet. The stream is exactly what the client attaches."""
        version = store.load_version(self.datadir, model_id)
        if version == 0 or since == str(version):
            return None
        res = store.open_global_stream(self.datadir, model_id)
        if res is None:
            return None
        return res[0], version

    def _clear_registering(self, rnd: Round) -> None:
        if self.registering.get(rnd.model_id) is rnd:
            del self.registering[rnd.model_id]
