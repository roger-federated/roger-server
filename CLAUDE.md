# roger-server — federated aggregation server

## What this is
The server half of **Roger Federated** (the client lives in the separate `roger-federated` repo:
`github.com/roger-federated/roger-federated`). A federation is just this server's URL. It seals
secure-aggregation cohorts, sums the masked client uploads (individual gradients stay hidden), folds
η·mean(ΔW) into a per-model cumulative dense global, and broadcasts that global for members to fold
into their base model. It does **no** model inference — only int64 tensor sums + safetensors I/O.

This repo was **extracted from the `roger-federated` monorepo** (server code was `src/roger/federated/
server/`). Its git history is the server's slice of that history, with paths renamed to `roger_server/`.

## Wire contract with the client (keep in lockstep BY HAND)
The server never imports the client package; coupling is purely over HTTP + two mirrored modules:
- `roger_server/secure_agg.py` — the canonical secure-aggregation protocol (Bonawitz 2017). `SCALE`,
  `R`, the **sorted-key flatten layout** (every client lays ΔW modules out in `sorted(keys)` order so
  the server can locate each module's slice in the flat upload), the SHAKE-256 PRG, and the X25519
  pairwise masks must match the client's `roger.federated.secure_agg` exactly or masks stop cancelling.
  The server only *calls* `dequantize`+`R`; `quantize`/`mask` are kept here because they are the
  contract `dequantize` inverts and the tests drive them to prove mask-cancellation end-to-end. (The
  client's copy omits `dequantize` — it is server-only.)
- `roger_server/delta.py` — the safetensors wire (de)serialization + base-compat hash, mirroring the
  client's `roger.federated.delta`. The server keeps only the serialization/compat half (no client-only
  densify / fold_into / DP-noise). `compat_hash` = sorted module → (out, in) digest = "same base model".

If you change the quantization, the flatten order, the safetensors metadata keys (`model_id`, `compat`,
`spec`, `round_id`), or any endpoint shape, make the matching change in the `roger-federated` client.

## Package layout
- `roger_server/aggregate.py` — round lifecycle + per-module streamed FedAvg (`begin_stage`/
  `mark_received`/`claim_finalize`/`run_finalize`); bootstrap `submit_dp`; density `mode`.
- `roger_server/app.py` — FastAPI streaming endpoints (`/round/register` seal barrier, `/contribute`,
  `/contribute_dp`, `/global`, `/status`); finalize off the event loop (threadpool, per-model lock).
- `roger_server/store.py` — durable global + per-round upload staging; per-module reader / multipart
  `GlobalWriter` / streamed broadcast; `fs` + `s3` backends (`ROGER_SERVER_STORAGE`).
- `roger_server/secure_agg.py`, `roger_server/delta.py` — the mirrored client contract (above).
- `roger_server/__main__.py` — `python -m roger_server` (uvicorn).
- `Dockerfile`, `DEPLOY.md` — scale-to-zero container deploy (see DEPLOY.md).
- `tests/test_server.py` — CPU-only, download-free; simulates clients with the real `secure_agg` crypto.

## Architecture notes (read before assuming)
- **Intrinsically single-instance.** Secure aggregation needs every cohort member in ONE process
  (one in-memory masked sum, one barrier); concurrency = rounds *inside* the process, never more
  processes. Default deploy = a scale-to-zero container with `max-instances=1` and the durable global
  (+ per-round upload staging under `tmp/<round_id>/`, GC'd by an S3 lifecycle rule) in S3.
- **Per-module streaming.** Each masked upload streams to its own object; finalize sums the cohort one
  module at a time (range-GET; masks cancel per coordinate) and folds η·mean(ΔW) into the global in one
  pass. Peak RAM ~one weight matrix at any model size; the binding constraint is per-round S3 I/O.
- **All-or-nothing rounds.** Void on any dropout or if `‖ΣΔW‖ > k·clip` (only a non-clipping client
  can exceed it; reject rather than damp).
- **Cold-start = DP-noised async bootstrap.** While a model has < `busy_threshold` recent contributors,
  `/status` serves `bootstrap`: clients skip the cohort and upload ONE faux-DP-noised unmasked dense ΔW
  to `/contribute_dp` (k=1 fold). Once busy, secure-agg only; quorum k_min=3 / k_target=5.
- NOT yet built: Shamir/double-mask dropout recovery, central ground-truth anti-poison gate, membership
  auth (round-token/signature). See the readme TODO in the client repo.

## Dev environment
- Python: conda env **`roger`** (Python 3.13, CUDA torch) — `~/.conda/envs/roger/python.exe`. Bare
  `python`/`python3` hit the Windows Store stub.
- Install/test: `pip install -e ".[test]"` then `python -m pytest tests/` (CPU-only, no model download).
- Run locally: `python -m roger_server` (uvicorn on `0.0.0.0:8000`; env knobs in `app.create_app`).

## Conventions
- Functional-first Python; a class only when isolated mutable state genuinely requires it.
- Minimum necessary changes; no speculative abstraction. Dense *why*-not-*what* comments; no docstrings
  that merely restate the signature. Avoid em-dashes in code/comments/text; prefer ; , ().
- Deploy/setup docs stay generic + provider-agnostic (an expert, or a novice with a chatbot, on any
  S3-compatible provider), not a one-provider walkthrough; keep portable caveats like `ROGER_AGG_IPBIND`.
- When something here is wrong/stale and a future session would benefit, update this file as part of the work.
