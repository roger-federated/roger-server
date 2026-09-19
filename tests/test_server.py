"""Tests for the federated aggregation server (roger_server).

CPU-only, download-free. The core tests drive the synchronous Aggregator directly (deterministic, no
event loop); the HTTP test exercises the FastAPI wire layer + the register/seal barrier via concurrent
TestClient calls. Synthetic clients reuse the real client crypto (secure_agg.quantize/mask), so the
mask-cancellation the server relies on is genuinely tested end-to-end.

Every upload carries ONE LoRA factor's update (delta.py): a B-epoch cohort uploads ΔB against the
frozen A, an A-epoch cohort the reverse. The cohort path stages each masked upload to its own object in
`store` and aggregates one factor at a time at finalize (so server RAM is ~one factor, not the whole
model). `_stage` mirrors the wire path: validate+reserve via begin_stage, write the blob through
store.stage_writer, mark_received.
"""
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from safetensors.torch import save as st_save

from roger_server import delta, secure_agg
from roger_server import store
from roger_server.aggregate import Aggregator

KEY = "base_model.model.model.layers.0.self_attn.q_proj"
OUT, IN, RANK = 6, 4, 2                       # a tiny module + the rank these tests' federations run at
BASE = {KEY: (OUT, IN)}                       # {module: (out, in)} — what `base`/`compat` describe
B_KEY, A_KEY = KEY + delta.LORA_B, KEY + delta.LORA_A
COMPAT = delta.compat_from_shapes(BASE)
BASE_JSON = delta.base_to_json(BASE)


def _agg(tmp_path, **kw) -> Aggregator:
    kw.setdefault("rank", RANK)
    return Aggregator(str(tmp_path), **kw)


def _dB(scale=0.05):
    return {B_KEY: torch.randn(OUT, RANK) * scale}


def _dA(scale=0.05):
    return {A_KEY: torch.randn(RANK, IN) * scale}


def _mask_cohort(payloads, compat=COMPAT):
    """Mirror the client: quantize each factor Δ, pairwise-mask against the whole cohort's pubkeys.
    Returns (uploads, pubs) where uploads = [(masked int64, compat, spec_json)]."""
    pairs = [secure_agg.gen_keypair() for _ in payloads]
    pubs = [pub for _, pub in pairs]
    uploads = []
    for (priv, _), p in zip(pairs, payloads):
        q, spec = secure_agg.quantize(p)
        masked = secure_agg.mask(q, priv, pubs)
        uploads.append((masked, compat, json.dumps([[k, list(s)] for k, s in spec])))
    return uploads, pubs


def _pack(masked, compat, spec_json, round_id, model_id="m", token="", epoch=0, base_json=BASE_JSON):
    return st_save({"masked": masked}, metadata={"model_id": model_id, "compat": compat,
                                                  "spec": spec_json, "base": base_json,
                                                  "epoch": str(epoch), "round_id": round_id,
                                                  "token": token})


def _seal_with(agg, pubs, ip="1.2.3.4", model="m"):
    """Returns (round, tokens): tokens[i] is the secret /contribute must present for pubs[i]."""
    rnd, tokens = None, []
    for pub in pubs:
        rnd, token = agg.add_registrant(model, pub.hex(), ip, now=0.0)
        tokens.append(token)
    agg.try_seal(rnd)
    return rnd, tokens


def _stage(agg, rnd, masked, compat, spec_json, token, model="m", epoch=0, base_json=BASE_JSON):
    """Validate + stage one upload exactly as the wire layer does; returns the begin_stage status."""
    r, slot, res = agg.begin_stage(rnd.round_id, compat, spec_json, base_json, epoch, token,
                                   masked_i64=(masked.dtype == torch.int64 and masked.dim() == 1),
                                   masked_len=masked.numel())
    if res != "ok":
        return res
    w = store.stage_writer(agg.datadir, rnd.round_id, slot)
    w.write(_pack(masked, compat, spec_json, rnd.round_id, model, token, epoch, base_json))
    w.commit()
    agg.mark_received(r, slot)
    return res


def _round(agg, payloads, epoch=0):
    """One whole cohort: seal, stage every member's masked factor Δ, finalize."""
    uploads, pubs = _mask_cohort(payloads)
    rnd, tokens = _seal_with(agg, pubs)
    for token, (masked, compat, spec_json) in zip(tokens, uploads):
        _stage(agg, rnd, masked, compat, spec_json, token, epoch=epoch)
    _finalize(agg, rnd)
    return rnd


def _finalize(agg, rnd):
    status = agg.claim_finalize(rnd)
    if status != "done":
        agg.run_finalize(rnd, status)


def _pull(agg, model="m", since=""):
    res = agg.serve_global(model, since)
    if res is None:
        return None
    chunks, version = res
    return b"".join(chunks), version


def _global(agg, model="m"):
    """(tensors, metadata) of the stored global, or None when nothing has been folded yet."""
    res = _pull(agg, model)
    return None if res is None else delta.from_bytes(res[0])


def _dp(model_id="m", factors=None, epoch=0, base=BASE):
    return delta.to_bytes(factors, model_id, {"base": delta.base_to_json(base), "epoch": str(epoch)})


# --- core: recovery + FedAvg accumulation ----------------------------------------------------

def test_aggregator_recovers_mean(tmp_path):
    torch.manual_seed(0)
    N = 4
    payloads = [_dB() for _ in range(N)]
    uploads, pubs = _mask_cohort(payloads)
    agg = _agg(tmp_path, k_min=2, k_target=N)
    rnd, tokens = _seal_with(agg, pubs)
    assert rnd.sealed and set(rnd.sealed_peers) == {p.hex() for p in pubs}
    for token, (masked, compat, spec_json) in zip(tokens, uploads):
        assert _stage(agg, rnd, masked, compat, spec_json, token) == "ok"
    _finalize(agg, rnd)
    tensors, meta = _global(agg)
    expected = sum(p[B_KEY] for p in payloads) / N          # η=1, no clip ⇒ plain mean of ΔB
    assert torch.allclose(tensors[B_KEY], expected, atol=1e-4)
    # A was never uploaded: the global carries the derived frozen A the cohort trained against.
    assert torch.equal(tensors[A_KEY], delta.init_A("m", KEY, RANK, IN))
    assert meta["scaling"] == "1" and meta["rank"] == str(RANK) and meta["phase"] == "B"
    assert _pull(agg)[1] == 1
    assert not (tmp_path / "tmp").exists() or not list((tmp_path / "tmp").iterdir())   # temp cleaned
    print("PASS test_aggregator_recovers_mean")


def test_two_rounds_accumulate(tmp_path):
    torch.manual_seed(1)
    agg = _agg(tmp_path, k_min=2, k_target=2)
    totals = torch.zeros(OUT, RANK)
    for _ in range(2):
        payloads = [_dB(0.02) for _ in range(2)]
        _round(agg, payloads)
        totals += sum(p[B_KEY] for p in payloads) / 2        # cumulative Σ mean(ΔB)
    blob, version = _pull(agg)
    tensors, _ = delta.from_bytes(blob)
    assert version == 2 and torch.allclose(tensors[B_KEY], totals, atol=1e-4)
    print("PASS test_two_rounds_accumulate")


def test_epoch_swaps_the_trained_factor(tmp_path):
    # After `epoch_folds` folds the federation flips to training A: the same-shaped ΔA folds into the
    # frozen-A baseline while B is carried over untouched, and a straggler still stamped with the old
    # epoch is refused (its ΔB was trained against an A that has since moved).
    torch.manual_seed(7)
    agg = _agg(tmp_path, k_min=2, k_target=2, epoch_folds=1)
    dB = [_dB() for _ in range(2)]
    _round(agg, dB)
    assert agg.state("m")["epoch"] == 1 and agg.state("m")["phase"] == "A"

    stale, pubs = _mask_cohort([_dB()] * 2)
    rnd, tokens = _seal_with(agg, pubs)
    assert _stage(agg, rnd, stale[0][0], COMPAT, stale[0][2], tokens[0], epoch=0) == "stale epoch"
    wrong_factor, _ = _mask_cohort([_dB()])
    assert _stage(agg, rnd, wrong_factor[0][0], COMPAT, wrong_factor[0][2], tokens[0],
                  epoch=1).startswith("expected A-factor")

    dA = [_dA() for _ in range(2)]
    _round(agg, dA, epoch=1)
    tensors, meta = _global(agg)
    assert torch.allclose(tensors[A_KEY],
                          delta.init_A("m", KEY, RANK, IN) + sum(p[A_KEY] for p in dA) / 2, atol=1e-4)
    assert torch.allclose(tensors[B_KEY], sum(p[B_KEY] for p in dB) / 2, atol=1e-4)   # B carried over
    assert meta["epoch"] == "2" and meta["phase"] == "B"    # back to training B against the new A
    print("PASS test_epoch_swaps_the_trained_factor")


def test_per_module_rank_map(tmp_path):
    # Rank is per module: a narrow matrix earns less of it than a wide one, and the map is a pure
    # function of (shape, cap), so every member derives the same one before training.
    assert delta.rank_for(4096, 4096, 16) == 16 and delta.rank_for(1024, 4096, 16) == 8   # q vs GQA v
    assert delta.rank_for(4096, 4096, 64) == 32                                           # cap not binding
    assert delta.rank_for(2, 2, 16) == 2                                                  # never exceed the matrix

    torch.manual_seed(9)
    narrow = KEY.replace("q_proj", "v_proj")
    base = {KEY: (OUT, 4), narrow: (OUT, 2)}                  # ⇒ ranks 4 and 2 under cap 8
    compat, base_json = delta.compat_from_shapes(base), delta.base_to_json(base)
    agg = _agg(tmp_path, k_min=2, k_target=2, rank=8)
    payloads = [{KEY + delta.LORA_B: torch.randn(OUT, 4) * 0.02,
                 narrow + delta.LORA_B: torch.randn(OUT, 2) * 0.02} for _ in range(2)]
    uploads, pubs = _mask_cohort(payloads, compat)
    rnd, tokens = _seal_with(agg, pubs)
    for token, (masked, _c, spec_json) in zip(tokens, uploads):
        assert _stage(agg, rnd, masked, compat, spec_json, token, base_json=base_json) == "ok"
    _finalize(agg, rnd)
    tensors, _ = _global(agg)
    assert tensors[KEY + delta.LORA_A].shape == (4, 4)
    assert tensors[narrow + delta.LORA_A].shape == (2, 2)
    assert torch.allclose(tensors[narrow + delta.LORA_B],
                          sum(p[narrow + delta.LORA_B] for p in payloads) / 2, atol=1e-4)

    # Uploading the wide module's rank for the narrow one is refused, naming the module's own rank.
    wide = [{narrow + delta.LORA_B: torch.randn(OUT, 4)} for _ in range(2)]
    up, pubs2 = _mask_cohort(wide, compat)
    rnd2, tok2 = _seal_with(agg, pubs2)
    assert _stage(agg, rnd2, up[0][0], compat, up[0][2], tok2[0],
                  base_json=base_json) == f"rank mismatch ({narrow} trains at rank 2 here)"
    print("PASS test_per_module_rank_map")


def test_dropout_voids_round(tmp_path):
    payloads = [_dB() for _ in range(3)]
    uploads, pubs = _mask_cohort(payloads)
    agg = _agg(tmp_path, k_min=2, k_target=3)
    rnd, tokens = _seal_with(agg, pubs)
    for token, (masked, compat, spec_json) in zip(tokens, uploads[:-1]):  # one sealed member never uploads
        _stage(agg, rnd, masked, compat, spec_json, token)
    _finalize(agg, rnd)
    assert _pull(agg) is None                               # global untouched (masks wouldn't cancel)
    assert not (tmp_path / "tmp" / rnd.round_id).exists()   # voided round's staged objects cleaned up
    print("PASS test_dropout_voids_round")


def test_subquorum_register_fails(tmp_path):
    # Only 1 registrant at the deadline with k_min=2 ⇒ the round FAILS rather than leaking a lone Δ.
    agg = _agg(tmp_path, k_min=2, k_target=8)
    rnd, _token = agg.add_registrant("m", "aa", "1.2.3.4", now=0.0)
    agg.try_seal(rnd, final=True)
    assert rnd.failed and not rnd.sealed
    print("PASS test_subquorum_register_fails")


def test_norm_bound_voids_aggregate(tmp_path):
    torch.manual_seed(2)
    N = 3
    payloads = [_dB(3.0) for _ in range(N)]      # ΣΔB ≫ k·clip ⇒ over the honest bound
    agg = _agg(tmp_path, k_min=2, k_target=N, clip_norm=1.0, eta=1.0)
    _round(agg, payloads)
    assert _pull(agg) is None                 # only a non-clipping client exceeds k·clip ⇒ round voided
    print("PASS test_norm_bound_voids_aggregate")


def test_rejects_bad_uploads(tmp_path):
    payloads = [_dB() for _ in range(2)]
    uploads, pubs = _mask_cohort(payloads)
    agg = _agg(tmp_path, k_min=2, k_target=2)
    rnd, tokens = _seal_with(agg, pubs)
    masked, compat, spec_json = uploads[0]
    assert _stage(agg, rnd, masked.float(), compat, spec_json, tokens[0]) == "bad tensor"  # wrong dtype
    # `compat` must be exactly the digest of the `base` shape map travelling with it, so a client can't
    # hand the server a digest that hides what it is really uploading.
    assert _stage(agg, rnd, masked, "deadbeef", spec_json, tokens[0]) == "bad base shapes"
    # A factor at the wrong rank can't be added to this federation's global at all.
    wide, _ = _mask_cohort([{B_KEY: torch.randn(OUT, RANK + 1)}])
    assert _stage(agg, rnd, wide[0][0], compat, wide[0][2], tokens[0]).startswith("rank mismatch")
    assert _stage(agg, rnd, masked, compat, spec_json, "not-a-real-token") == "invalid token"
    assert _stage(agg, rnd, masked, compat, spec_json, tokens[0]) == "ok"
    # tokens[0] already completed an upload above: reusing it is exactly the same-registrant
    # double-upload this token scheme is meant to block (it would otherwise corrupt mask cancellation).
    assert _stage(agg, rnd, masked, compat, spec_json, tokens[0]) == "token already used"
    # A second member disagreeing about the base layout is refused: its ΔB is not even the shape its
    # own declared base implies, so nothing could cancel coordinate-wise.
    other = {KEY: (OUT + 2, IN)}
    assert _stage(agg, rnd, uploads[1][0], delta.compat_from_shapes(other), spec_json, tokens[1],
                  base_json=delta.base_to_json(other)).endswith("does not match its base module (8, 4)")
    print("PASS test_rejects_bad_uploads")


def test_serve_global_cursor(tmp_path):
    agg = _agg(tmp_path, k_min=2, k_target=2)
    _round(agg, [_dB() for _ in range(2)])
    assert agg.serve_global("m", "1") is None            # since == current version ⇒ nothing new
    assert agg.serve_global("m", "") is not None
    assert agg.serve_global("other", "") is None         # unknown model
    print("PASS test_serve_global_cursor")


def test_concurrent_cohorts_per_model(tmp_path):
    # While cohort A is sealed and collecting, new registrants must form a SEPARATE cohort B (not be
    # rejected), and the two collect + finalize independently, each routed by its own round_id.
    torch.manual_seed(4)
    agg = _agg(tmp_path, k_min=2, k_target=2)
    payloads_a = [_dB() for _ in range(2)]
    payloads_b = [_dB() for _ in range(2)]
    up_a, pubs_a = _mask_cohort(payloads_a)
    up_b, pubs_b = _mask_cohort(payloads_b)

    rnd_a, tokens_a = _seal_with(agg, pubs_a)                # cohort A seals, now COLLECTING
    rnd_b, tokens_b = _seal_with(agg, pubs_b)                # B forms + seals while A still collects
    assert rnd_a.round_id != rnd_b.round_id
    assert rnd_a.round_id in agg.collecting and rnd_b.round_id in agg.collecting  # both live at once

    # Interleave uploads; each routes by its round_id (and stages to its own temp prefix).
    _stage(agg, rnd_b, up_b[0][0], up_b[0][1], up_b[0][2], tokens_b[0])
    _stage(agg, rnd_a, up_a[0][0], up_a[0][1], up_a[0][2], tokens_a[0])
    _stage(agg, rnd_b, up_b[1][0], up_b[1][1], up_b[1][2], tokens_b[1])
    _stage(agg, rnd_a, up_a[1][0], up_a[1][1], up_a[1][2], tokens_a[1])
    _finalize(agg, rnd_a)
    _finalize(agg, rnd_b)

    blob, version = _pull(agg)
    expected = (sum(p[B_KEY] for p in payloads_a) + sum(p[B_KEY] for p in payloads_b)) / 2
    assert version == 2 and torch.allclose(delta.from_bytes(blob)[0][B_KEY], expected, atol=1e-4)
    print("PASS test_concurrent_cohorts_per_model")


def test_late_cohort_voids_across_an_epoch_boundary(tmp_path):
    # Two cohorts sealed in the same epoch, but the first fold advances it: the second's ΔB was trained
    # against the A that just stopped being frozen, so its fold is refused instead of corrupting G.
    torch.manual_seed(8)
    agg = _agg(tmp_path, k_min=2, k_target=2, epoch_folds=1)
    up_a, pubs_a = _mask_cohort([_dB() for _ in range(2)])
    up_b, pubs_b = _mask_cohort([_dB() for _ in range(2)])
    rnd_a, tok_a = _seal_with(agg, pubs_a)
    rnd_b, tok_b = _seal_with(agg, pubs_b)
    for rnd, tokens, ups in ((rnd_a, tok_a, up_a), (rnd_b, tok_b, up_b)):
        for token, (masked, compat, spec_json) in zip(tokens, ups):
            _stage(agg, rnd, masked, compat, spec_json, token)
    _finalize(agg, rnd_a)
    _finalize(agg, rnd_b)
    assert _pull(agg)[1] == 1                    # only cohort A landed; B voided on the stale epoch
    print("PASS test_late_cohort_voids_across_an_epoch_boundary")


def test_global_persists_across_restart(tmp_path):
    agg = _agg(tmp_path, k_min=2, k_target=2, epoch_folds=1)
    _round(agg, [_dB() for _ in range(2)])
    reloaded = _agg(tmp_path)                             # fresh instance serves from storage, no eager load
    blob, version = _pull(reloaded)
    assert version == 1 and B_KEY in delta.from_bytes(blob)[0]
    # The factor state (which epoch/phase, at which rank) rides in the global's metadata, so a restarted
    # server keeps telling clients to train the same factor.
    assert reloaded.state("m") == {"epoch": 1, "phase": "A", "folds": 0, "rank": RANK, "compat": COMPAT}
    print("PASS test_global_persists_across_restart")


def test_s3_backend_round_trip(tmp_path, monkeypatch):
    # The scale-to-zero deploy keeps the global in S3-compatible object storage, not on the (ephemeral)
    # container disk, and STAGES uploads there too (multipart) then aggregates per-module via range-GET.
    # Prove a fresh Aggregator — the cold-start-after-scale-to-zero analog — rehydrates the persisted
    # global purely from the bucket, writing nothing to local disk.
    pytest.importorskip("boto3")
    pytest.importorskip("moto")
    import boto3
    from moto import mock_aws

    # An AWS-style endpoint so moto's request interception matches (it ignores non-amazonaws hosts);
    # the real deploy points ROGER_S3_ENDPOINT at the Scaleway/Koyeb S3 endpoint instead.
    endpoint = "https://s3.us-east-1.amazonaws.com"
    monkeypatch.setenv("ROGER_SERVER_STORAGE", "s3")
    monkeypatch.setenv("ROGER_S3_ENDPOINT", endpoint)
    monkeypatch.setenv("ROGER_S3_REGION", "us-east-1")     # avoids moto's LocationConstraint requirement
    monkeypatch.setenv("ROGER_S3_BUCKET", "roger-test")
    monkeypatch.setenv("ROGER_S3_KEY", "k")
    monkeypatch.setenv("ROGER_S3_SECRET", "s")

    with mock_aws():
        boto3.client("s3", endpoint_url=endpoint, region_name="us-east-1",
                     aws_access_key_id="k", aws_secret_access_key="s").create_bucket(Bucket="roger-test")

        payloads = [_dB() for _ in range(2)]
        agg = _agg(tmp_path, k_min=2, k_target=2)          # datadir is unused by the s3 backend
        _round(agg, payloads)

        reloaded = _agg(tmp_path)                          # cold start: rehydrate from object storage only
        blob, version = _pull(reloaded)
        expected = sum(p[B_KEY] for p in payloads) / 2
        assert version == 1 and torch.allclose(delta.from_bytes(blob)[0][B_KEY], expected, atol=1e-4)
        assert not list(tmp_path.iterdir())                # s3 mode never touches local disk
    print("PASS test_s3_backend_round_trip")


# --- bootstrap (async DP) mode ---------------------------------------------------------------

def test_dp_bootstrap_accumulates(tmp_path):
    # A single unmasked factor Δ folds straight into the global (k=1), no cohort. Two async uploads
    # accumulate as Σ η_boot·ΔB.
    torch.manual_seed(5)
    agg = _agg(tmp_path, eta=1.0, clip_norm=10.0)   # high clip ⇒ no scaling, exact sum
    d1, d2 = _dB(), _dB()
    assert agg.submit_dp("m", d1, BASE, "1.1.1.1", now=0.0) == "ok"
    assert agg.submit_dp("m", d2, BASE, "2.2.2.2", now=1.0) == "ok"
    tensors, _ = _global(agg)
    assert _pull(agg)[1] == 2
    assert torch.allclose(tensors[B_KEY], d1[B_KEY] + d2[B_KEY], atol=1e-4)
    print("PASS test_dp_bootstrap_accumulates")


def test_dp_bootstrap_norm_bound_and_rejects(tmp_path):
    agg = _agg(tmp_path, clip_norm=1.0, eta=1.0)
    assert agg.submit_dp("m", _dB(5.0), BASE, "1.1.1.1", now=0.0) == "norm exceeded"   # ‖ΔB‖ ≫ 1 ⇒ void
    assert _pull(agg) is None                                 # nothing folded
    assert agg.submit_dp("m", _dB(), BASE, "1.1.1.1", now=1.0) == "ok"   # within bound ⇒ folds, sets G
    # a base whose shapes disagree with the established global is refused (would corrupt the sum)
    other = {KEY: (OUT + 2, IN)}
    assert agg.submit_dp("m", {B_KEY: torch.randn(OUT + 2, RANK)}, other, "1.1.1.1", now=2.0) == "compat mismatch"
    assert agg.submit_dp("m", {B_KEY: torch.full((OUT, RANK), float("nan"))}, BASE,
                         "1.1.1.1", now=3.0) == "non-finite delta"
    print("PASS test_dp_bootstrap_norm_bound_and_rejects")


def test_dp_bootstrap_bf16(tmp_path):
    # A client may upload bf16 factors (training dtype); the staged per-module reader must read them
    # back. f32 tests don't exercise that path.
    agg = _agg(tmp_path, eta=1.0, clip_norm=10.0)
    dB = {B_KEY: (torch.randn(OUT, RANK) * 0.05).to(torch.bfloat16)}
    assert agg.submit_dp("m", dB, BASE, "1.1.1.1", now=0.0) == "ok"
    tensors, _ = _global(agg)
    assert torch.allclose(tensors[B_KEY], dB[B_KEY].float(), atol=1e-2)   # bf16 round-trip precision
    print("PASS test_dp_bootstrap_bf16")


def test_mode_flips_at_density_threshold(tmp_path):
    agg = _agg(tmp_path, busy_threshold=3, busy_window=100.0)
    assert agg.mode("m", now=0.0) == "bootstrap"               # nothing seen yet
    for i, ip in enumerate(["a", "b"]):
        agg.submit_dp("m", _dB(0.01), BASE, ip, now=float(i))
    assert agg.mode("m", now=2.0) == "bootstrap"               # only 2 distinct contributors < 3
    agg.submit_dp("m", _dB(0.01), BASE, "c", now=3.0)
    assert agg.mode("m", now=3.0) == "busy"                    # 3rd distinct contributor ⇒ busy
    assert agg.mode("m", now=3.0 + 200.0) == "bootstrap"       # all aged out of the window ⇒ sparse again
    print("PASS test_mode_flips_at_density_threshold")


def test_defaults(tmp_path):
    agg = Aggregator(str(tmp_path))
    assert agg.k_min == 3 and agg.k_target == 5
    assert agg.rank == 16 and agg.epoch_folds == 20
    # A model nobody has contributed to yet still has a well-defined factor state to train against.
    assert agg.state("fresh") == {"epoch": 0, "phase": "B", "folds": 0, "rank": 16, "compat": None}
    print("PASS test_defaults")


def test_allowlist_reports_unsupported(tmp_path):
    # An allowlisted model keeps the density logic; an excluded one reports "unsupported" at /status so
    # the client can warn + skip rather than waste a training round the 403/400 would reject anyway.
    from fastapi.testclient import TestClient
    from roger_server.app import create_app
    agg = _agg(tmp_path, allowlist={"ok"})
    assert agg.mode("ok", now=0.0) == "bootstrap"
    assert agg.mode("nope", now=0.0) == "unsupported"
    with TestClient(create_app(agg)) as client:
        body = client.get("/status", params={"model_id": "nope"}).json()
        assert body["mode"] == "unsupported" and body["models"] == ["ok"]   # the list is what's actionable
        assert client.get("/status", params={"model_id": "ok"}).json()["mode"] == "bootstrap"
    # No allowlist ⇒ every model is supported (never "unsupported"), advertised as models: null.
    assert _agg(tmp_path, allowlist=None).mode("anything", now=0.0) == "bootstrap"
    with TestClient(create_app(_agg(tmp_path, allowlist=None))) as client:
        assert client.get("/status", params={"model_id": "anything"}).json()["models"] is None
    print("PASS test_allowlist_reports_unsupported")


def test_status_advertises_client_version(tmp_path, monkeypatch):
    # /status echoes the deployment's client-version policy (ROGER_MIN_CLIENT / ROGER_LATEST_CLIENT) so
    # an out-of-date client self-skips + nudges an update. Absent env ⇒ 0 (no opinion), never blocks.
    from fastapi.testclient import TestClient
    from roger_server.app import create_app
    with TestClient(create_app(_agg(tmp_path))) as client:
        body = client.get("/status", params={"model_id": "m"}).json()
        assert body["min_client"] == 0 and body["latest_client"] == 0
    monkeypatch.setenv("ROGER_MIN_CLIENT", "3")
    monkeypatch.setenv("ROGER_LATEST_CLIENT", "5")
    with TestClient(create_app(_agg(tmp_path))) as client:
        body = client.get("/status", params={"model_id": "m"}).json()
        assert body["min_client"] == 3 and body["latest_client"] == 5
    print("PASS test_status_advertises_client_version")


def test_healthz_reports_store(tmp_path):
    # /healthz exercises the write+read+delete path; fs backend is always healthy here.
    from fastapi.testclient import TestClient
    from roger_server.app import create_app
    assert store.health_check(str(tmp_path)) == "ok"
    with TestClient(create_app(_agg(tmp_path))) as client:
        r = client.get("/healthz")
        assert r.status_code == 200 and r.json()["storage"] == "ok"
    print("PASS test_healthz_reports_store")


def test_register_rejects_malformed_json(tmp_path):
    # A malformed body must be a clean 400, not a 500 from the unguarded req.json().
    from fastapi.testclient import TestClient
    from roger_server.app import create_app
    with TestClient(create_app(_agg(tmp_path))) as client:
        r = client.post("/round/register", content=b"{not json",
                        headers={"Content-Type": "application/json"})
        assert r.status_code == 400
    print("PASS test_register_rejects_malformed_json")


def test_absent_global(tmp_path):
    assert store.open_global_reader(str(tmp_path), "nope") is None   # never-folded model ⇒ None, not a crash
    assert store.open_global_stream(str(tmp_path), "nope") is None
    assert store.load_version(str(tmp_path), "nope") == 0
    assert store.load_meta(str(tmp_path), "nope") == {}
    print("PASS test_absent_global")


def test_seeded_A_is_stable_and_shaped(tmp_path):
    # The frozen A a cold federation starts from is DERIVED on both sides, so it must be reproducible
    # bit-for-bit and distinct per (model, module).
    a1 = delta.init_A("m", KEY, RANK, IN)
    assert a1.shape == (RANK, IN) and a1.dtype == torch.float32
    assert torch.equal(a1, delta.init_A("m", KEY, RANK, IN))
    assert not torch.equal(a1, delta.init_A("other", KEY, RANK, IN))
    assert a1.abs().max() <= 1.0 / IN ** 0.5                  # PEFT's kaiming-uniform bound
    print("PASS test_seeded_A_is_stable_and_shaped")


# --- HTTP wire layer + seal barrier ----------------------------------------------------------

def test_http_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from roger_server.app import create_app

    monkeypatch.setenv("ROGER_AGG_W", "10")              # bound any hang if requests serialized
    torch.manual_seed(3)
    N = 3
    payloads = [_dB() for _ in range(N)]
    uploads, pubs = _mask_cohort(payloads)
    agg = _agg(tmp_path, k_min=2, k_target=N)

    with TestClient(create_app(agg)) as client:
        # What the client reads right before training: which factor, at which rank, in which epoch.
        st = client.get("/status", params={"model_id": "m"}).json()
        assert (st["phase"], st["rank"], st["epoch"]) == ("B", RANK, 0)

        # Concurrent registration: the cohort seals once all N are in-flight, returning the same peers.
        with ThreadPoolExecutor(max_workers=N) as ex:
            regs = list(ex.map(
                lambda pub: client.post("/round/register", json={"model_id": "m", "pubkey": pub.hex()}),
                pubs))
        assert all(r.status_code == 200 for r in regs)
        assert set(regs[0].json()["peers"]) == {p.hex() for p in pubs}
        round_id = regs[0].json()["round_id"]                 # all members of one cohort share it
        assert all(r.json()["round_id"] == round_id for r in regs)
        tokens = [r.json()["token"] for r in regs]            # ex.map preserves pubs' order in regs

        for token, (masked, compat, spec_json) in zip(tokens, uploads):
            r = client.post("/contribute",
                            content=_pack(masked, compat, spec_json, round_id, token=token,
                                          epoch=st["epoch"]),
                            headers={"Content-Type": "application/octet-stream"})
            assert r.status_code == 200

        r = client.get("/global", params={"model_id": "m", "since": ""})
        assert r.status_code == 200
        tensors, meta = delta.from_bytes(r.content)
        # The broadcast is the adapter itself: both factors, scale already folded in.
        assert torch.allclose(tensors[B_KEY], sum(p[B_KEY] for p in payloads) / N, atol=1e-4)
        assert torch.equal(tensors[A_KEY], delta.init_A("m", KEY, RANK, IN))
        assert meta["scaling"] == "1"
        cursor = r.headers["X-Cursor"]
        assert client.get("/global", params={"model_id": "m", "since": cursor}).status_code == 204
    print("PASS test_http_end_to_end")


def test_http_bootstrap_path(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from roger_server.app import create_app

    torch.manual_seed(6)
    agg = _agg(tmp_path, busy_threshold=3, clip_norm=10.0)
    with TestClient(create_app(agg)) as client:
        # A fresh model is sparse ⇒ /status says bootstrap; the client uploads one unmasked factor Δ.
        assert client.get("/status", params={"model_id": "m"}).json()["mode"] == "bootstrap"
        dB = _dB()
        r = client.post("/contribute_dp", content=_dp(factors=dB),
                        headers={"Content-Type": "application/octet-stream"})
        assert r.status_code == 200
        # An upload stamped with the wrong epoch never reaches the global.
        r = client.post("/contribute_dp", content=_dp(factors=_dB(), epoch=9),
                        headers={"Content-Type": "application/octet-stream"})
        assert r.status_code == 400 and "stale epoch" in r.text
        g = client.get("/global", params={"model_id": "m", "since": ""})
        assert g.status_code == 200
        assert torch.allclose(delta.from_bytes(g.content)[0][B_KEY], dB[B_KEY], atol=1e-4)
    print("PASS test_http_bootstrap_path")


if __name__ == "__main__":
    import tempfile, pathlib
    d = pathlib.Path(tempfile.mkdtemp())
    test_aggregator_recovers_mean(d / "a"); test_two_rounds_accumulate(d / "b")
    test_dropout_voids_round(d / "c"); test_subquorum_register_fails(d / "d")
    test_norm_bound_voids_aggregate(d / "e"); test_rejects_bad_uploads(d / "f")
    test_serve_global_cursor(d / "g"); test_global_persists_across_restart(d / "h")
    test_dp_bootstrap_accumulates(d / "i"); test_dp_bootstrap_norm_bound_and_rejects(d / "j")
    test_mode_flips_at_density_threshold(d / "k"); test_defaults(d / "l")
    test_absent_global(d / "m"); test_dp_bootstrap_bf16(d / "n")
    test_allowlist_reports_unsupported(d / "o"); test_epoch_swaps_the_trained_factor(d / "p")
    test_per_module_rank_map(d / "s")
    test_late_cohort_voids_across_an_epoch_boundary(d / "q"); test_seeded_A_is_stable_and_shaped(d / "r")
