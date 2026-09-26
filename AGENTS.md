# roger-server — federated aggregation server

## What this is
The server half of **Roger Federated** (the client lives in the separate `roger-federated` repo:
`github.com/roger-federated/roger-federated`). A federation is just this server's URL. It seals
secure-aggregation cohorts, sums the masked client uploads (individual gradients stay hidden), folds
η·mean(Δ) into a per-model cumulative **global LoRA adapter**, and broadcasts that adapter for members
to attach to the model their runtime serves. It does **no** model inference — only int64 tensor sums +
safetensors I/O.

**The factor contract (the load-bearing design decision).** Everything on the wire is LoRA factors,
never a dense ΔW, because the client is now a wrapper around llama-server/vllm and can only apply an
*adapter*. Factors do not sum ((ΣB)(ΣA) leaves the cross terms BᵢAⱼ behind), and secure aggregation
only ever gives the server the sum — so a federation trains **one factor per epoch** (RoLoRA, Chen et
al. 2024, arXiv:2410.07739): in a B-epoch A is frozen and identical for everyone, clients upload ΔB,
and Σ(ΔBᵢ)·A = Σ(ΔBᵢ·A) exactly; an A-epoch mirrors it. Epoch 0 trains B (LoRA starts at B=0, so A's
gradient would be zero). The epoch advances after `epoch_folds` folds, is advertised at `/status`
(`rank`/`epoch`/`phase`) and stamped on every upload; an upload from a past epoch is voided, since it
was trained against a factor the federation has moved past. Rank is **per module but static**
(`delta.rank_for` = ~1/128 of the matrix's full rank, floored at 8, capped by `ROGER_AGG_RANK`, which is
pinned per model once its global exists): a pure function of the base shapes, so every member derives
the same map before training and it never moves. The first frozen A is *derived* too (`delta.init_A`,
SHAKE-256 over model_id|module|shape), never transmitted, so a client with nothing to pull yet trains
against the exact A the server will fold into.

**Client-side policy this assumes** (decided 2026-09-17, to be implemented in the client repo):
contribute to **exactly one** federation, training against `W₀ + that federation's factors only` at
scale 1, so the gradient is measured at the base that federation's aggregate assumes. A client may
*pull* from several: the wrapper attaches them by concatenating along the rank axis (`B_cat @ A_cat =
ΣB_f A_f`, exact), scaled by 1/N as a merge coefficient, since independently cumulated globals overlap
and summing them at full scale overshoots. Adaptive/renegotiated rank was considered and dropped in
favour of the static map above; if it ever comes back, the cheap route is a factor-space SVD at an
epoch boundary (QR both factors, SVD the r×r core) rather than materializing a dense ΔW.

This repo was **extracted from the `roger-federated` monorepo** (server code was `src/roger/federated/
server/`). Its git history is the server's slice of that history, with paths renamed to `roger_server/`.

## Wire contract with the client (keep in lockstep BY HAND)
The server never imports the client package; coupling is purely over HTTP + two mirrored modules:
- `roger_server/secure_agg.py` — the canonical secure-aggregation protocol (Bonawitz 2017). `SCALE`,
  `R`, the **sorted-key flatten layout** (every client lays the epoch's factor keys out in
  `sorted(keys)` order so the server can locate each one's slice in the flat upload), the SHAKE-256
  PRG, and the X25519 pairwise masks must match the client's `roger.federated.secure_agg` exactly or
  masks stop cancelling. `SCALE` is 2^22 (not 2^16): a factor update is far smaller in magnitude than
  the dense ΔW this used to carry. The server only *calls* `dequantize`+`R`; `quantize`/`mask` are kept
  here because they are the contract `dequantize` inverts and the tests drive them to prove
  mask-cancellation end-to-end. (The client's copy omits `dequantize` — it is server-only.)
- `roger_server/delta.py` — the factor key layout (`LORA_A`/`LORA_B`, `phase_of`), the seeded frozen A
  (`init_A`), the safetensors wire (de)serialization and the base-compat hash, mirroring the client's
  `roger.federated.delta`. `compat_hash` = sorted module → (out, in) digest = "same base model"; every
  upload carries that shape map verbatim as `base` (a one-factor payload pins only one dimension, and
  the server needs `in` to seed A).

If you change the quantization, the flatten order, `init_A`, the epoch/phase rule, the safetensors
metadata keys (`model_id`, `compat`, `spec`, `base`, `epoch`, `round_id`, `token`; `scaling`/`rank`/
`phase` on the broadcast), or any endpoint shape, make the matching change in the `roger-federated`
client.

## Package layout
- `roger_server/aggregate.py` — round lifecycle + per-factor streamed FedAvg (`begin_stage`/
  `mark_received`/`claim_finalize`/`run_finalize`); the epoch/rank state machine (`state`,
  `check_upload`); bootstrap `submit_dp`; density `mode`.
- `roger_server/app.py` — FastAPI streaming endpoints (`/round/register` seal barrier, `/contribute`,
  `/contribute_dp`, `/global`, `/status`); finalize off the event loop (threadpool, per-model lock).
- `roger_server/store.py` — durable global (its safetensors metadata carries the durable factor state:
  `rank`/`epoch`/`folds`) + per-round upload staging; per-module reader / multipart `GlobalWriter` /
  streamed broadcast; `fs` + `s3` backends (`ROGER_SERVER_STORAGE`).
- `roger_server/secure_agg.py`, `roger_server/delta.py` — the mirrored client contract (above).
- `roger_server/aliases.py` — the curated table of repos accepted as the same weights as a canonical id
  (quants, reuploads; QAT is its own canonical), served at `/status` as `aliases`. Hand-verified, never
  name-inferred: the hub is full of same-named forks. Uploads/allowlist stay canonical-only.
- `roger_server/__main__.py` — `python -m roger_server` (uvicorn).
- `Dockerfile`, `README.md` — scale-to-zero container deploy (the README is also the deploy guide).
- `tests/test_server.py` — CPU-only, download-free; simulates clients with the real `secure_agg` crypto
  (B-epoch and A-epoch cohorts, epoch flips, stale uploads, rank mismatches).

## Architecture notes (read before assuming)
- **Intrinsically single-instance.** Secure aggregation needs every cohort member in ONE process
  (one in-memory masked sum, one barrier); concurrency = rounds *inside* the process, never more
  processes. Default deploy = a scale-to-zero container with `max-instances=1` and the durable global
  (+ per-round upload staging under `tmp/<round_id>/`, GC'd by an S3 lifecycle rule) in S3.
- **Per-module streaming.** Each masked upload streams to its own object; finalize sums the cohort one
  factor at a time (range-GET; masks cancel per coordinate) and folds η·mean(Δ) into the epoch's
  trainable factor in one pass, carrying the frozen one over untouched. Peak RAM ~one factor at any
  model size.
- **All-or-nothing rounds.** Void on any dropout, on a stale epoch (including a cohort that straddles a
  boundary, re-checked at fold time), or if `‖ΣΔ‖ > k·clip` in factor space (only a non-clipping client
  can exceed it; reject rather than damp).
- **Cold-start = DP-noised async bootstrap.** While a model has < `busy_threshold` recent contributors,
  `/status` serves `bootstrap`: clients skip the cohort and upload ONE faux-DP-noised unmasked factor Δ
  to `/contribute_dp` (k=1 fold). Once busy, secure-agg only; quorum k_min=3 / k_target=5. (Noise on a
  factor update is cleaner than the old dense scheme: the frozen factor is public and the map is
  linear, so the weight-space noise Δ·A is exactly Gaussian.)
- NOT yet built: Shamir/double-mask dropout recovery, central ground-truth anti-poison gate. See the
  readme TODO in the client repo. (Membership auth is built: a secret token issued at
  `/round/register` and required back at `/contribute` — see `Round.token_pubkey`/`spent_tokens` in
  `aggregate.py`.)

## Dev environment
- Python: conda env **`roger`** (Python 3.13, CUDA torch) — `~/.conda/envs/roger/python.exe`. Bare
  `python`/`python3` hit the Windows Store stub. On the macOS checkout there is no conda env and the
  system python is 3.9 (too old for the `X | None` annotations): `uv venv --python 3.12` + `uv pip
  install` the deps, then run pytest from that venv.
- Install/test: `pip install -e ".[test]"` then `python -m pytest tests/` (CPU-only, no model download).
- Run locally: `python -m roger_server` (uvicorn on `0.0.0.0:8000`; env knobs in `app.create_app`).

## Conventions
- Functional-first Python; a class only when isolated mutable state genuinely requires it.
- Minimum necessary changes; no speculative abstraction. Dense *why*-not-*what* comments; no docstrings
  that merely restate the signature. Avoid em-dashes in code/comments/text; prefer ; , ().
- Deploy/setup docs stay generic + provider-agnostic (an expert, or a novice with a chatbot, on any
  S3-compatible provider), not a one-provider walkthrough; keep portable caveats like
  `ROGER_AGG_TRUSTED_PROXIES`.
- When something here is wrong/stale and a future session would benefit, update this file as part of the work.
