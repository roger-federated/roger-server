"""delta.py — the wire serialization + base-compat contract for the federated ΔW blob.

The server half of the client's `roger.federated.delta` (main repo). The client additionally densifies
LoRA factors into dense ΔW and folds a broadcast global back into base weights; the server never does
either, so only the safetensors (de)serialization + compat-hash helpers live here. These MUST stay
byte-compatible with the client copy — they define the on-the-wire format the two exchange.

Compatibility between federation members is just "same base model" = identical per-module (out, in)
weight shapes, captured by `compat_hash` / `compat_from_shapes`.
"""
import hashlib, json, struct

from safetensors.torch import load as st_load, save as st_save


def compat_from_shapes(shapes: dict) -> str:
    """The compat digest from an already-extracted {module: (out, in)} map. The server recomputes it
    while rebuilding the global per-module (it has shapes, not whole tensors)."""
    blob = ";".join(f"{m}:{s[0]}x{s[1]}" for m, s in sorted(shapes.items()))
    return hashlib.sha1(blob.encode()).hexdigest()


def compat_hash(tensors: dict) -> str:
    """Stable digest of the base architecture this delta targets: sorted module → (out, in). Dense
    ΔW carries (out, in) directly; LoRA factors give out from lora_B[:,0], in from lora_A[0,:], so a
    dense upload and the re-factored broadcast of the same base hash identically."""
    shapes = {}
    for key, t in tensors.items():
        if key.endswith(".lora_A.weight"):
            shapes.setdefault(key[: -len(".lora_A.weight")], [None, None])[1] = t.shape[1]   # in
        elif key.endswith(".lora_B.weight"):
            shapes.setdefault(key[: -len(".lora_B.weight")], [None, None])[0] = t.shape[0]   # out
        else:                                              # dense ΔW [out, in]
            shapes[key] = [t.shape[0], t.shape[1]]
    return compat_from_shapes(shapes)


def _read_metadata(buf: bytes) -> dict:
    # safetensors layout: u64 LE header length, then the JSON header (whose "__metadata__" holds our
    # str→str fields). load() drops it, so parse the header directly rather than round-tripping a file.
    n = struct.unpack("<Q", buf[:8])[0]
    return json.loads(buf[8 : 8 + n]).get("__metadata__", {})


def to_bytes(tensors: dict, model_id: str) -> bytes:
    """Serialize a tensor dict (dense ΔW for the DP-fold, or factors for a broadcast) with model_id +
    base-compat hash in the metadata."""
    return st_save(tensors, metadata={"model_id": model_id, "compat": compat_hash(tensors)})


def from_bytes(buf: bytes) -> tuple[dict, dict]:
    return st_load(buf), _read_metadata(buf)
