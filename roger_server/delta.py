"""delta.py — the wire serialization + LoRA-factor contract for the federated update.

The server half of the client's `roger.federated.delta` (main repo). Both sides exchange **LoRA
factors only**, never a dense ΔW: the global IS the adapter the runtime attaches, stored and
broadcast as the PEFT pair `<module>.lora_A.weight` [r, in] / `<module>.lora_B.weight` [out, r]
(F32, `scaling` = "1" in the metadata since the scale is already folded into the factors). Each module
has its own rank, fixed by `rank_for` from its shape and the federation's cap; see below.

Why one factor at a time (the alternating contract, RoLoRA; Chen et al. 2024, arXiv:2410.07739).
Secure aggregation only ever hands the server the SUM of the cohort's uploads, and factors do not
sum: (ΣB)(ΣA) = Σ B_i A_i + Σ_{i≠j} B_i A_j, the cross terms being pure error. So a federation
alternates epochs: in a B-epoch A is frozen and identical for everyone, clients train B only and
upload ΔB, and Σ(ΔB_i)·A = Σ(ΔB_i·A) exactly; an A-epoch does the reverse. Epoch 0 trains B, since
LoRA starts at B=0 and A's gradient would be zero. Which factor to train is advertised at /status
and stamped on every upload, so work from a past epoch (not representable against the new frozen
factor) is rejected rather than silently corrupting the global.

The frozen A a fresh federation starts from is DERIVED, not transmitted (`init_A`): a cold client
must train against the same A the server will fold into, before any global exists to pull.

Compatibility between federation members is just "same base model" = identical per-module (out, in)
weight shapes, captured by `compat_hash` / `compat_from_shapes`. Every upload carries that shape map
verbatim (`base`), both so the server can check it against the digest and so it learns the `in`
dimensions it needs to seed A and to derive the rank map.

Rank is per-module but STATIC: `rank_for` derives it from the module's shape and the federation's cap
(`/status`'s `rank`), so both sides agree on the map before anyone trains, and it never moves for a
given model. That keeps the one property the protocol needs (everyone in a cohort lays out the same
vector) while not spending a 4096² module's rank budget on a 1024-row one.
"""
import hashlib, json, struct

import torch
from safetensors.torch import load as st_load, save as st_save

LORA_A = ".lora_A.weight"
LORA_B = ".lora_B.weight"
SUFFIX = {"A": LORA_A, "B": LORA_B}          # phase -> the factor key that phase trains

# Rank rule (contract): a module's rank is ~1/128 of its full rank, floored so even a small matrix gets
# a usable subspace, and capped by the federation (ROGER_AGG_RANK). Both sides compute it from the base
# shapes, so a cold client knows the map before any global exists. Changing either constant is a
# breaking change on both sides, exactly like changing SCALE.
RANK_FLOOR = 8
RANK_DIVISOR = 128


def module_of(key: str) -> str | None:
    """The base module a factor key belongs to, or None if it is not a factor key at all."""
    for suffix in (LORA_A, LORA_B):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def phase_of(epoch: int) -> str:
    """Even epochs train B (against the frozen A), odd ones train A. Derived rather than stored, so
    the durable state is a single integer and the two sides cannot disagree about the order."""
    return "B" if epoch % 2 == 0 else "A"


def rank_for(out_dim: int, in_dim: int, cap: int) -> int:
    """This federation's rank for ONE module. Wider matrices earn more rank than narrow ones (a 4096²
    q_proj gets the cap; a GQA-shrunk v_proj [1024, 4096] gets the floor), which is the whole point of a
    per-module map: rank is what an upload costs, so spending it where the matrix is small is waste.
    The map is STATIC per (model, federation) — the cap is pinned in the global's metadata when the
    model is created and the base shapes never move — so it is agreed before anyone trains and no
    upload can ever be stale because of it."""
    full = min(out_dim, in_dim)
    return min(max(full // RANK_DIVISOR, RANK_FLOOR), cap, full)


def rank_map(base: dict, cap: int) -> dict:
    """{module: rank} for a whole base shape map; what a client builds its adapter from."""
    return {module: rank_for(out_dim, in_dim, cap) for module, (out_dim, in_dim) in base.items()}


def init_A(model_id: str, module: str, rank: int, in_dim: int) -> torch.Tensor:
    """The frozen A a federation starts from, derived deterministically from (model_id, module, shape).
    A cold client has no global to pull yet but must train its ΔB against the very A the server will
    fold it into, so A is seeded from a SHAKE-256 XOF (same primitive as the secure-agg PRG) instead of
    being transmitted or chosen by whoever contributes first. Values are uniform in ±1/√in, matching
    PEFT's kaiming-uniform init of lora_A; the exact expansion is part of the client contract."""
    raw = hashlib.shake_256(f"{model_id}|{module}|{rank}x{in_dim}".encode()).digest(rank * in_dim * 4)
    u = torch.frombuffer(bytearray(raw), dtype=torch.int32).to(torch.float64)
    u = (u + 2.0 ** 31) / 2.0 ** 32           # int32 residues -> [0, 1)
    bound = 1.0 / in_dim ** 0.5
    return ((2.0 * u - 1.0) * bound).to(torch.float32).reshape(rank, in_dim)


def compat_from_shapes(shapes: dict) -> str:
    """The compat digest from an already-extracted {module: (out, in)} map. The server recomputes it
    while rebuilding the global per-module (it has shapes, not whole tensors)."""
    blob = ";".join(f"{m}:{s[0]}x{s[1]}" for m, s in sorted(shapes.items()))
    return hashlib.sha1(blob.encode()).hexdigest()


def compat_hash(tensors: dict) -> str:
    """Stable digest of the base architecture this update targets: sorted module → (out, in), read off
    the factors (out from lora_B's rows, in from lora_A's columns). A one-factor upload only pins one
    of the two dimensions, which is why `base` travels alongside; this stays the canonical digest of a
    complete factor pair (a stored global, a broadcast)."""
    shapes = {}
    for key, t in tensors.items():
        if key.endswith(LORA_A):
            shapes.setdefault(key[: -len(LORA_A)], [None, None])[1] = t.shape[1]    # in
        elif key.endswith(LORA_B):
            shapes.setdefault(key[: -len(LORA_B)], [None, None])[0] = t.shape[0]    # out
    return compat_from_shapes(shapes)


def base_to_json(shapes: dict) -> str:
    return json.dumps([[m, list(s)] for m, s in sorted(shapes.items())])


def base_from_json(text: str) -> dict:
    """{module: (out, in)} from an upload's `base` metadata; {} when it is missing or malformed (the
    caller rejects the upload, since the digest can never match an empty map)."""
    try:
        return {m: (int(s[0]), int(s[1])) for m, s in json.loads(text)}
    except Exception:
        return {}


def _read_metadata(buf: bytes) -> dict:
    # safetensors layout: u64 LE header length, then the JSON header (whose "__metadata__" holds our
    # str→str fields). load() drops it, so parse the header directly rather than round-tripping a file.
    n = struct.unpack("<Q", buf[:8])[0]
    return json.loads(buf[8 : 8 + n]).get("__metadata__", {})


def to_bytes(tensors: dict, model_id: str, meta: dict | None = None) -> bytes:
    """Serialize a factor dict (one factor's Δ for an upload, both factors for a broadcast) with
    model_id + the base-compat digest; `meta` carries the rest of the stamp (base, epoch, rank, ...).
    A one-factor payload pins only one dimension per module, so the digest comes from `base` whenever
    that is present and from the tensors themselves otherwise."""
    meta = dict(meta or {})
    compat = (compat_from_shapes(base_from_json(meta["base"])) if "base" in meta
              else compat_hash(tensors))
    return st_save(tensors, metadata={"model_id": model_id, "compat": compat, **meta})


def from_bytes(buf: bytes) -> tuple[dict, dict]:
    return st_load(buf), _read_metadata(buf)
