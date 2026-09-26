"""aliases.py: the repos a federation accepts as *the same weights* as a canonical model id.

Members run whatever build of a model suits their hardware: an MLX quant on Apple silicon, a GGUF for
llama-server, an AWQ/FP8 checkpoint for vllm. As far as a LoRA global is concerned each of those IS the
canonical base (a quantization is a lossy encoding of the same weights, and training against a quantized
copy is ordinary QLoRA). What must never be conflated is a *different* model that merely shares the name:
an abliterated/"uncensored" fork, an agentic or coding finetune, a distill. Their deltas would pull the
global toward another model, and on the Hugging Face hub they outnumber the faithful quants
(`gemma-4-12B-it-*` alone has dozens). So membership is curated here, by hand, never inferred from a name.

Admission rule for an entry, in order of strength:
  - byte-identical reupload: same safetensors LFS sha256 as the canonical repo;
  - a quantization with no provenance metadata: dequantize one projection and check it sits within
    quantization error of the canonical weights and far from any sibling release (done for the
    mlx-community repos: 0.7-12% relative error to their parent vs ~85% to the other release);
  - otherwise the hub's `base_model:quantized:<canonical>` tag, and only from an established quantizer.
QAT is NOT an alias of its release: Google's QAT weights differ from the release by ~46% relative norm on
q_proj (it rescales weights to quantize well), so a global trained on one does not transfer to the other.
It is its own canonical id, with its own quants.

Served at `/status` as `aliases` ({canonical: [alias, ...]}, for the canonical ids this deployment
accepts). The client resolves what its runtime serves against canonical ∪ aliases (hub ids are
case-insensitive, so it compares casefolded) and uploads under the CANONICAL id only: the allowlist and
every upload check stay on canonical ids, because `delta.init_A` is seeded from the model id, so an upload
stamped with an alias would have trained against a different frozen A than the one folded here.
"""

ALIASES: dict[str, tuple[str, ...]] = {
    "google/gemma-4-12B-it": (
        # byte-identical reuploads (same model.safetensors sha256)
        "unsloth/gemma-4-12b-it",
        "RedHatAI/gemma-4-12B-it",
        # MLX (Apple silicon), verified numerically; OptiQ is mixed 4/8-bit affine
        "mlx-community/gemma-4-12B-it-4bit",
        "mlx-community/gemma-4-12B-it-5bit",
        "mlx-community/gemma-4-12B-it-6bit",
        "mlx-community/gemma-4-12B-it-8bit",
        "mlx-community/gemma-4-12B-it-bf16",
        "mlx-community/gemma-4-12B-it-mxfp4",
        "mlx-community/gemma-4-12B-it-nvfp4",
        "mlx-community/gemma-4-12B-it-mxfp8",
        "mlx-community/gemma-4-12B-it-OptiQ-4bit",
        "lmstudio-community/gemma-4-12B-it-MLX-4bit",
        "lmstudio-community/gemma-4-12B-it-MLX-5bit",
        "lmstudio-community/gemma-4-12B-it-MLX-6bit",
        "lmstudio-community/gemma-4-12B-it-MLX-8bit",
        # GGUF (llama-server)
        "unsloth/gemma-4-12b-it-GGUF",
        "ggml-org/gemma-4-12B-it-GGUF",
        "bartowski/gemma-4-12B-it-GGUF",
        "lmstudio-community/gemma-4-12B-it-GGUF",
        "mradermacher/gemma-4-12B-it-GGUF",
        "mradermacher/gemma-4-12B-it-i1-GGUF",
        # vllm checkpoints (CUDA)
        "unsloth/gemma-4-12b-it-NVFP4",
        "RedHatAI/gemma-4-12B-it-FP8-Dynamic",
        "RedHatAI/gemma-4-12B-it-NVFP4",
        "cyankiwi/gemma-4-12B-it-AWQ-INT4",
    ),
    "google/gemma-4-12B-it-qat-q4_0-unquantized": (
        "unsloth/gemma-4-12B-it-qat-q4_0-unquantized",   # byte-identical reupload
        "google/gemma-4-12B-it-qat-q4_0-gguf",
        "google/gemma-4-12B-it-qat-w4a16-ct",
        "unsloth/gemma-4-12B-it-qat-GGUF",
        "unsloth/gemma-4-12B-it-qat-w4a16",
        "lmstudio-community/gemma-4-12B-it-QAT-GGUF",
        "mradermacher/gemma-4-12B-it-qat-q4_0-unquantized-GGUF",
        "cyankiwi/gemma-4-12B-it-qat-AWQ-INT4",
        "mlx-community/gemma-4-12B-it-qat-4bit",
        "mlx-community/gemma-4-12B-it-qat-5bit",
        "mlx-community/gemma-4-12B-it-qat-6bit",
        "mlx-community/gemma-4-12B-it-qat-8bit",
        "mlx-community/gemma-4-12B-it-qat-bf16",
        "mlx-community/gemma-4-12B-it-qat-mxfp4",
        "mlx-community/gemma-4-12B-it-qat-nvfp4",
        "mlx-community/gemma-4-12B-it-qat-mxfp8",
        "mlx-community/gemma-4-12B-it-qat-OptiQ-4bit",
    ),
}


def advertised(allowlist: set | None) -> dict[str, list[str]]:
    """The alias table for the canonical ids this deployment accepts (all of it when any model is
    accepted: a client still needs it then, or every quantization would fold into a global of its own)."""
    return {c: list(a) for c, a in ALIASES.items() if allowlist is None or c in allowlist}
