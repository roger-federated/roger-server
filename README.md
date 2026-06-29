# roger-server

The aggregation server for [Roger Federated](https://github.com/roger-federated/roger-federated). A
federation is just this server's URL: it seals secure-aggregation cohorts, sums the masked client
gradient uploads (individual gradients stay hidden), folds `η·mean(ΔW)` into a per-model cumulative
global, and broadcasts that global for members to fold into their base model. It does no model
inference — only int64 tensor sums + safetensors I/O — so it runs lean and CPU-only.

It is intrinsically single-instance (secure aggregation needs every cohort member in one process), so
the default deploy is a **scale-to-zero container** (`max-instances=1`) with the durable global in
S3-compatible object storage — ~zero idle cost.

## Quick start
```bash
pip install -e ".[test]"      # add the package + test deps
python -m pytest tests/        # CPU-only, no model download
python -m roger_server         # serve (uvicorn on 0.0.0.0:8000)
```

## Deploy
See **[DEPLOY.md](DEPLOY.md)** for the scale-to-zero container deploy (S3 storage) and the legacy
always-on VM path.

## Relationship to the client
The gradient-sharing client lives in the separate `roger-federated` repo. This server never imports
the client; they interoperate purely over HTTP. The secure-aggregation + wire-serialization contract
is mirrored by hand in `roger_server/secure_agg.py` and `roger_server/delta.py` — see `CLAUDE.md` for
what must stay in lockstep across the two repos.
