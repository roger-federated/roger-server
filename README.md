# roger-server

The aggregation server for [Roger Federated](https://github.com/roger-federated/roger-federated). It is
what makes a *federation*: it seals secure-aggregation cohorts, sums the masked client uploads
(individual gradients stay hidden), and serves the cumulative **global LoRA adapter** that members
attach to the model their runtime serves. Anyone can run one; a federation is just its URL.

## How a federation trains (read this first)
Everything on the wire is **LoRA factors**, never a dense weight delta: the global *is* the adapter
(`<module>.lora_A.weight` [r, in] and `.lora_B.weight` [out, r]), so a member can attach it to
llama.cpp or vllm directly.

That forces one rule on the protocol. Secure aggregation only ever hands the server the *sum* of a
cohort's uploads, and factors do not sum: `(ΣB)(ΣA) = Σ BᵢAᵢ + Σ_{i≠j} BᵢAⱼ`, and those cross terms are
pure error. So a federation trains **one factor at a time**:

- In a **B-epoch**, `A` is frozen and identical for every member. Clients train `B` only and upload
  `ΔB`, and `Σ(ΔBᵢ)·A = Σ(ΔBᵢ·A)` is exact. An **A-epoch** mirrors it. Epoch 0 trains `B`, since LoRA
  starts at `B = 0` and `A`'s gradient would be zero.
- The epoch advances after `ROGER_AGG_EPOCH_FOLDS` successful folds. `GET /status` advertises
  `{rank, epoch, phase}`, and a client reads it **immediately before training** so it trains the right
  factor; every upload is stamped with its epoch, and an upload from a past epoch is **voided** (it was
  trained against a factor the federation has since moved).
- **Rank is per module but static.** `ROGER_AGG_RANK` is a *cap*; each module's actual rank is
  `min(max(min(out, in) / 128, 8), cap, min(out, in))`, so a 4096² `q_proj` gets the cap while a
  GQA-shrunk `v_proj` gets the floor and costs half the upload. The map is a pure function of the base
  shapes and the cap, so every member derives the same one before training, it never moves for a given
  model, and uploads whose factor shapes disagree are rejected.
- The initial frozen `A` is **derived, not transmitted** (seeded from the model id + module + shape via
  SHAKE-256), so a client with no global to pull yet still trains against the exact `A` the server will
  fold its `ΔB` into.

A side effect is that uploads are now factor-sized (`r·(out + in)` per module) rather than the full
dense basis, which is roughly two orders of magnitude less per-round traffic.

The server is **intrinsically single-instance**: secure aggregation only works when every cohort member
reaches the *same* process (one in-memory masked sum, one barrier), and concurrency is handled by
spawning rounds *inside* that one process — never by adding processes. So you want exactly **0 or 1**
instance, never more. That makes it a perfect fit for a **scale-to-zero container**: it runs the one
process while there's traffic and costs ~nothing while idle. The only durable state — the cumulative
global LoRA adapter — lives in **S3-compatible object storage**, so it survives the container scaling to zero.

Any managed scale-to-zero container platform works (Scaleway Serverless Containers, Koyeb, ...). The
instructions below are deliberately platform-agnostic; map them onto your provider's console or CLI.

> **Read the [memory sizing](#memory-sizing) section.** The server stages every upload to object storage
> and aggregates **one module at a time**, so peak RAM is ~a single factor at *any* model size, and
> factor-only uploads keep per-round I/O small as well. Add the **`tmp/` lifecycle rule** (below) so a
> crashed round can't leave staged objects behind.

## Quick start
```bash
pip install -e ".[test]"      # package + test deps (moto, httpx)
python -m pytest tests/        # CPU-only, no model download
python -m roger_server         # serve (uvicorn on 0.0.0.0:8000)
```

## Relationship to the client
The gradient-sharing client lives in the separate `roger-federated` repo. This server never imports the
client; they interoperate purely over HTTP. The secure-aggregation + LoRA-factor wire format is mirrored
by hand in `roger_server/secure_agg.py` and `roger_server/delta.py`; see `AGENTS.md` for what must stay
in lockstep across the two repos.

## What you need
- An **S3-compatible object-storage bucket** plus an access key + secret. The container holds these
  creds; **clients never touch storage** (no client keys, no user accounts). Some scale-to-zero
  platforms have no managed object storage of their own — that's fine, point the server at a bucket
  from any S3-compatible provider; only the endpoint/creds differ.
- A **container registry** the platform can pull from (the platform's own is usually simplest).
- The platform provisions HTTPS on a generated URL; that URL *is* the federation.

## Steps
1. **Build and push the image.** Build context is the repo root, then tag and push to your registry
   (example values are a real Scaleway registry namespace; substitute your own):
   ```bash
   docker build -t roger-agg .
   docker tag  roger-agg rg.nl-ams.scw.cloud/roger-containers/roger-agg:latest
   docker push rg.nl-ams.scw.cloud/roger-containers/roger-agg:latest
   ```
   The image bundles CPU-only PyTorch + boto3 and runs `python -m roger_server` (uvicorn on
   `0.0.0.0:8000`).
2. **Create the bucket** and an access key/secret for it. Keep it **private** (clients reach the global
   only through the server, never the bucket) and leave **versioning off** (the server overwrites one
   blob per model each round; versioning just retains dead copies forever). Add a **lifecycle rule that
   expires objects under the `tmp/` prefix** after a few hours: the server streams each round's uploads
   to `tmp/<round_id>/` and deletes them at finalize, but a container killed mid-round (scale-to-zero,
   crash) would otherwise orphan them. The rule is the backstop GC.
3. **Deploy the container** from the image with the [settings](#container-settings) and
   [environment](#environment-variables) below.
4. **Smoke-test:** `curl -i "https://<your-url>/global?model_id=x"` should return **204** before any
   upload, and `curl "https://<your-url>/status?model_id=x"` should report the `rank`/`epoch`/`phase`
   clients will train against. Members then add the URL to `~/.roger/config.json`: `"federations": ["https://<your-url>"]`.

## Container settings
| Setting | Value | Why |
|---|---|---|
| **max instances** | **1** | **Load-bearing.** A 2nd instance would split a cohort across processes and the secure-agg masks would never cancel. Never scale out. |
| min instances | 0 | Scale to zero when idle = ~zero cost. Bump to 1 only while a federation is *actively* busy, to skip cold-starts during live cohorts. |
| concurrency | high (e.g. 80) | All cohort members must share the one instance; serve them concurrently rather than spilling to a new instance. Keep it above your largest cohort. |
| request timeout | **≥ `ROGER_AGG_W` + margin** (e.g. 60 s) | `/round/register` long-polls until the cohort seals (up to `W`, default 20 s). A shorter platform timeout would kill the barrier mid-seal. Keep `W` well under the client's 30 s call timeout. |
| CPU | modest (≈1 vCPU) | The server does **no inference** — only per-module tensor sums + safetensors I/O. CPU is not the constraint. |
| memory | see [memory sizing](#memory-sizing) | Per-module aggregation keeps peak RAM to ~one weight matrix at any model size; a small tier (e.g. 2–4 GB) suffices. |
| port | 8000 | What uvicorn binds. |

**Cold start:** after idle, the first request pays a cold-start (pull image + start uvicorn); the global
is read from object storage lazily, only for the model actually requested (nothing is pre-loaded).
Subsequent cohort members arrive warm. A live cohort holds the instance warm via the open
`/round/register` long-polls, so an in-flight round is not dropped. Losing in-memory density/round state
on scale-to-zero is harmless — idle means "bootstrap" is the correct mode anyway. (Uploads already
staged to `tmp/` survive a scale-to-zero, but their round's cohort barrier does not, so the round voids
and the `tmp/` lifecycle rule reclaims them.)

## Client IP behind a managed platform
Cohort membership is proven by a per-registration secret token (issued at seal, required back at
`/contribute`), not by client IP — so IP no longer needs to be trustworthy for security. It's only
used to estimate "how many distinct contributors recently," which drives the bootstrap→busy mode
switch. A managed platform terminates TLS in front of the container, so the real client IP arrives
via `X-Forwarded-For`; the server only trusts that header from proxies listed in
`ROGER_AGG_TRUSTED_PROXIES` (these platforms rarely publish a *stable* proxy CIDR to whitelist, but
getting this wrong now just skews when a model flips to busy mode, not who can upload). On a VM
where you control the proxy (next section), the default trusted set covers localhost.

## Memory sizing
Memory used to be the load-bearing constraint; three changes removed it.

- **Narrow basis.** The client trains the federated LoRA on `q_proj`/`v_proj` only (`lora_utils.FED_TARGETS`),
  the fixed basis every member shares. That is ~a few percent of an `all-linear` target set.
- **Factors, not dense deltas.** An upload is one LoRA factor's update, `r·out` or `r·in` numbers per
  module instead of `out·in`, with `r` per module (see the rank rule above). At cap 16 on the q/v basis
  of a 7B model that is tens of MB per member per round, against several GB when dense deltas were on
  the wire.
- **Stage + aggregate per module.** A secure-agg round must sum every member's masked vector over the
  epoch's whole factor basis; summed in RAM, × concurrent cohorts, that is what blew past serverless
  tiers. Instead the server **streams each upload to its own object** under `tmp/<round_id>/` and, at
  finalize, reads back **one factor at a time** (range-GET), sums it (masks cancel per coordinate),
  folds it into the global, and streams the new global out via multipart. Peak RAM is ~one factor plus
  small buffers, at **any** model size, and concurrent cohorts no longer multiply it (their partial
  state lives in the bucket, not RAM). The server also no longer pre-loads every model's global; it
  touches only the model in the request.

Consequences:
- A **small memory tier suffices** (2–4 GB is ample). **Ephemeral/scratch storage is irrelevant** — on
  Scaleway Serverless Containers it is RAM-backed tmpfs (writing N bytes to `/tmp` raises memory use by N
  and counts against the limit), so it was never a real spill target; we stage to the S3 bucket instead.
- **Per-round S3 I/O** is what scales with model size now, and factor-sized uploads keep it modest even
  for large bases. The one full-size object per fold is the **global itself**, rebuilt streamed each
  time (`r·(out + in)` per module, so also factor-sized). `ROGER_AGG_MODELS` still lets you allowlist
  which base models a given deployment serves.

## Environment variables
A filled-in `s3` example (real Scaleway region/bucket; supply your own key/secret as platform secrets):
```
ROGER_SERVER_STORAGE=s3
ROGER_S3_ENDPOINT=https://s3.nl-ams.scw.cloud   # region endpoint, no bucket prefix
ROGER_S3_REGION=nl-ams
ROGER_S3_BUCKET=roger-agg
ROGER_S3_KEY=<access key>                        # store as a secret
ROGER_S3_SECRET=<secret key>                     # store as a secret
```

**Storage**
| Var | Default | Meaning |
|---|---|---|
| `ROGER_SERVER_STORAGE` | `fs` | `s3` for the scale-to-zero deploy (durable global in object storage); `fs` for a local-disk/always-on VM. |
| `ROGER_S3_ENDPOINT` / `_REGION` / `_BUCKET` / `_KEY` / `_SECRET` | — | S3-compatible object store (required when `=s3`). `ENDPOINT` is the *region* endpoint with no bucket prefix — boto3 adds the bucket from `_BUCKET`. Store `_KEY`/`_SECRET` as platform secrets, not plain env vars. |
| `ROGER_S3_PREFIX` | `` | Optional key prefix so several deployments can share one bucket. |
| `ROGER_SERVER_DATA` | `./server-data` | Local data dir (only used when `ROGER_SERVER_STORAGE=fs`). |
| `ROGER_SERVER_HOST` / `ROGER_SERVER_PORT` | `0.0.0.0` / `8000` | uvicorn bind. |

**Aggregation knobs**
| Var | Default | Meaning |
|---|---|---|
| `ROGER_AGG_KMIN` | `3` | Min cohort size to seal (a cohort of 1 is unmasked; at 2 each peer can subtract its own Δ to recover the other's). Also the collusion margin: unmasking one member needs `KMIN−1` colluding peers. |
| `ROGER_AGG_KTARGET` | `5` | Seal immediately at this many registrants. Also the dropout blast radius (a no-show voids its whole cohort), so keep it modest. Defaults to the busy-mode threshold below. |
| `ROGER_AGG_W` | `20` | Registration window, seconds — must stay below the client's 30 s timeout *and* below the platform's request timeout. |
| `ROGER_AGG_U` | `20` | Seconds to wait for every sealed member to upload before voiding the round. |
| `ROGER_AGG_ETA` | `1.0` | Server learning rate: `G ← G + η·mean(Δ)` on the epoch's trainable factor. Lower to damp noisy rounds. |
| `ROGER_AGG_ETA_BOOT` | *(=`ETA`)* | Learning rate for a single async bootstrap upload (`G ← G + η_boot·Δ`, k=1). Lower it to damp the noisier per-upload bootstrap gradients. |
| `ROGER_AGG_CLIP` | `1.0` | Per-client L2 budget, in **factor space** (the norm of the uploaded `ΔB`/`ΔA`). The server **voids** a round whose aggregate `‖ΣΔ‖` exceeds `cohort_size · CLIP` (a bootstrap upload exceeding `CLIP`). Honest clients clip below this, so only a non-clipping client trips it. |
| `ROGER_AGG_RANK` | `16` | The federation's rank **cap**, advertised at `/status`. Per-module rank is `min(max(min(out, in) / 128, 8), cap, min(out, in))`; raise the cap to give wide modules more capacity (and bigger uploads), lower it to bound every module. It applies to globals created from here on; a model whose global already exists keeps the cap it was created with. |
| `ROGER_AGG_EPOCH_FOLDS` | `20` | Successful folds before the epoch advances and the trained factor swaps (`B` ⇄ `A`). Short epochs spread training over both factors sooner; long ones waste less work at boundaries, since an upload trained in a past epoch is voided. |
| `ROGER_AGG_BUSY_THRESHOLD` | *(=`KTARGET`)* | Distinct recent contributors needed to switch a model from bootstrap (async DP) to busy (secure-agg cohorts). |
| `ROGER_AGG_BUSY_WINDOW` | `180` | Rolling window, seconds, over which those distinct contributors are counted. |
| `ROGER_AGG_MODELS` | *(any)* | Comma-separated `model_id` allowlist; empty accepts any base model. Scope which models a deployment serves (e.g. to bound per-round S3 I/O for very large models; see memory sizing). Advertised verbatim as `models` at `/status` (`null` when empty) so clients can tell users which models to run. List **canonical** ids only: the quants/reuploads accepted as the same weights (MLX, GGUF, AWQ, FP8 builds) come from the curated table in `roger_server/aliases.py`, advertised as `aliases` for the ids listed here. |
| `ROGER_AGG_TRUSTED_PROXIES` | `127.0.0.1/32,::1/128,10.0.0.0/8` | Proxies whose `X-Forwarded-For` is trusted for the real client IP (only used for the busy-mode contributor count; cohort membership is proven by the `/round/register` token, not IP). |
| `ROGER_MIN_CLIENT` | `0` | Minimum client protocol version (`CLIENT_VERSION`) this deployment accepts. Clients below it self-skip contributing and tell the user to update; `0` = no floor. Raise it after a breaking protocol change so stale clients stop poisoning the global. |
| `ROGER_LATEST_CLIENT` | `0` | Newest client version to advertise. A client below it (but at/above `ROGER_MIN_CLIENT`) prints an advisory "update available" notice without being blocked; `0` = no notice. |

## Notes & limits
- **Cold-start is handled by bootstrap mode.** While a model has fewer than `BUSY_THRESHOLD` recent
  contributors, `/status` reports `bootstrap` and clients upload a single DP-noised, *unmasked* factor Δ
  to `/contribute_dp`, folded asynchronously — no cohort, no arrival coincidence, no 503. Privacy then
  rests on the client's (faux-)DP factor noise, not secure aggregation, so bootstrap is a temporary
  obfuscation regime: expect noisy, modest per-upload gradients until the federation fills up. Once
  `BUSY_THRESHOLD` distinct contributors appear within `BUSY_WINDOW`, the model flips to busy mode.
- In **busy** mode a round only aggregates when ≥`KMIN` members register and upload within the same
  ~`W`-second window; a sub-`KMIN` cohort gets a 503 and retries (safe behaviour — it would otherwise
  expose an individual Δ).
- No dropout recovery yet: one sealed member that never uploads voids only its own cohort.
- **Work can go stale at an epoch boundary.** A client trains against the frozen factor of the epoch it
  read at `/status`; if the epoch advances before its upload folds, the upload is voided (at
  `/round/register` or `/contribute`, or at finalize for a cohort that straddles the boundary). Raise
  `ROGER_AGG_EPOCH_FOLDS` if you see that often.
- Federations are open; `/contribute` requires the secret token issued to that registrant at
  `/round/register`, proving the uploader sealed into this cohort (not IP matching, which is
  spoofable/NAT-shared). Secure aggregation hides individual Δ but can't filter a *well-formed*
  poisoned upload from a genuine cohort member — only the aggregate norm bound (over-norm rounds are
  voided), small cohorts, and small `ETA` bound the damage. Strong per-client bounds need ZK range
  proofs (future work).
- **State durability:** only the cumulative global is durable (one blob + version per model). A round's
  in-flight uploads are staged under `tmp/<round_id>/` and deleted at finalize; open cohorts and the
  bootstrap↔busy density window are in-memory and intentionally ephemeral. The bucket `tmp/` lifecycle
  rule is the only cleanup for uploads orphaned by a mid-round crash.

## Legacy: always-on VM (self-hosted, `fs` storage)
If you'd rather run a plain VM (no object storage, fixed monthly cost), use local-disk storage and put
a TLS-terminating reverse proxy in front. The `fs` backend stages uploads under `<data>/tmp/` and cleans
them per round, just like `s3`; give the volume room for a round's in-flight uploads. Cost: paying 24/7.
```bash
docker run -d --name roger-agg --restart unless-stopped \
    -p 127.0.0.1:8000:8000 -v roger-agg-data:/data \
    -e ROGER_SERVER_STORAGE=fs -e ROGER_SERVER_DATA=/data roger-agg
```
Without Docker: `pip install -e .` then `python -m roger_server`. Front it with a
reverse proxy that provisions TLS — Caddy is simplest. Point an `A`/`AAAA` record at the VM, open
inbound 80 (cert challenge) and 443, and use a minimal `/etc/caddy/Caddyfile`:
```
fed.example.org {
    reverse_proxy 127.0.0.1:8000
}
```
`--restart unless-stopped` plus the data volume keeps the cumulative global across reboots.
