"""Stage 9 — Convert merged fp16 checkpoints to Q8_0 + Q5_K_M GGUFs.

Runs llama.cpp's HF-to-GGUF converter and quantizer on a Modal CPU
container that mounts the project volume. Operates on both
checkpoints in one call by default:

- ``/vol/checkpoints/merged-fp16/``       (DPO, primary)
- ``/vol/checkpoints/sft-merged-fp16/``   (SFT-only, secondary)

Outputs land at ``/vol/gguf/<model_name>/{Q8_0,Q5_K_M}.gguf`` so the
upload script can read them off the volume directly without a 30+
GB local round-trip.

Usage::

    modal run publish/export_gguf.py                   # both models
    modal run publish/export_gguf.py --model dpo       # DPO only
    modal run publish/export_gguf.py --model sft       # SFT only

Per Unsloth's Gemma 4 small-model guidance: Q8_0 is the Pareto
starting point (near-lossless, ~2× smaller than fp16); Q5_K_M
trades a small quality hit for another ~40% size reduction. We
ship both so operators can pick the cost/quality point that fits
their hardware.

The image clones llama.cpp at a pinned tag and builds the
``llama-quantize`` binary from source; ``convert_hf_to_gguf.py``
ships in the repo as a Python script. Build is cached after first
run.
"""

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import modal

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "dpo": {
        "source_dir": "/vol/checkpoints/merged-fp16",
        "display_name": "gemma4-social-bias-judge",
    },
    "sft": {
        "source_dir": "/vol/checkpoints/sft-merged-fp16",
        "display_name": "gemma4-social-bias-judge-sft",
    },
}
GGUF_DIR = "/vol/gguf"
LLAMA_CPP_DIR = "/opt/llama.cpp"
# Pinned to ``master`` because Gemma 4 conversion support landed
# across multiple PRs (#21309 model arch, #21343/#21534 tokenizer,
# #21390 final_logit_softcapping, #21418 chat parser). Specific
# tags around the Gemma 4 release (b6800-era) ship the C++ runtime
# fixes but the Python ``convert_hf_to_gguf.py`` rejected
# ``Gemma4ForConditionalGeneration`` until later. ``master`` has
# the full set; reproducibility cost is acceptable for a one-shot
# export step. Bump to a specific tag (e.g. ``b7500+``) once the
# upstream releases stabilize.
LLAMA_CPP_TAG = "master"
QUANT_TYPES = ("Q8_0", "Q5_K_M")
MIN_GGUF_BYTES = 2_000_000_000  # ~2 GB floor; Q5_K_M of 8B model ~4 GB

# ----------------------------------------------------------------------------
# Modal app
# ----------------------------------------------------------------------------
VOLUME_NAME = "judge-from-scratch"

app = modal.App("judge-export-gguf")

# CPU base; the conversion+quantization step is CPU-bound (no GPU
# math). debian_slim is fine here because we don't need flashinfer's
# JIT path.
gguf_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake", "libcurl4-openssl-dev")
    .run_commands(
        [
            f"git clone --depth 1 --branch {LLAMA_CPP_TAG} "
            f"https://github.com/ggerganov/llama.cpp {LLAMA_CPP_DIR}",
            f"cmake -S {LLAMA_CPP_DIR} -B {LLAMA_CPP_DIR}/build "
            "-DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF",
            f"cmake --build {LLAMA_CPP_DIR}/build --target llama-quantize -j 4",
            "pip install --no-cache-dir uv",
            (
                # llama.cpp b6800 is C++-clean for Gemma 4 (the
                # llama-quantize binary handles the model arch
                # correctly), but its top-level ``requirements.txt``
                # pulls a stale
                # ``transformers @ v4.56.0-Embedding-Gemma-preview``
                # branch that doesn't recognize the ``gemma4``
                # model_type — ``convert_hf_to_gguf.py`` then aborts
                # at ``AutoConfig.from_pretrained``. llama.cpp master's
                # ``requirements/requirements-convert_legacy_llama.txt``
                # has updated to ``transformers==5.5.1``; we install
                # that exact dep set here directly (numpy,
                # sentencepiece, transformers, gguf-via-bundled-py,
                # protobuf, safetensors, torch). The ``protobuf<5``
                # upper bound from master's manifest is dropped
                # (Modal's mirror only carries protobuf 6.x; the
                # 4→6 API delta doesn't affect anything
                # ``convert_hf_to_gguf.py`` uses).
                # ``--index-strategy unsafe-best-match`` lets uv pull
                # torch CPU wheels and sentencepiece from secondary
                # indexes — same fix the vLLM image uses.
                "uv pip install --system --no-cache "
                "--index-strategy unsafe-best-match "
                "numpy "
                "'sentencepiece>=0.1.98,<0.3.0' "
                "'transformers>=5.5.0' "
                "'protobuf>=4.21.0' "
                "safetensors "
                "torch "
                # b6800's convert_hf_to_gguf.py imports mistral_common
                # at module load (not behind a try/except guard like
                # master). The Mistral path is dead code for Gemma 4
                # but the import still has to resolve, so install the
                # package regardless.
                "mistral-common"
            ),
        ]
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# ----------------------------------------------------------------------------
# Pure helpers (importable for tests)
# ----------------------------------------------------------------------------


def validate_model_name(name: str) -> None:
    """Reject unknown names with the valid set in the message."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}; valid: {sorted(MODEL_REGISTRY)}")


def gguf_paths(model_name: str) -> dict[str, Path]:
    """Resolve the per-model GGUF output paths.

    Returns a dict keyed by ``f16`` plus each entry in ``QUANT_TYPES``.
    The ``f16`` intermediate is kept on disk between the convert and
    quantize steps so the conversion is run only once.
    """
    out_dir = Path(GGUF_DIR) / model_name
    return {
        "f16": out_dir / "f16.gguf",
        **{q: out_dir / f"{q}.gguf" for q in QUANT_TYPES},
    }


def _verify_quant(path: Path) -> dict[str, Any]:
    """Confirm a quantized GGUF exists with a non-trivial size."""
    if not path.exists():
        raise AssertionError(f"GGUF not produced: {path}")
    size = path.stat().st_size
    if size < MIN_GGUF_BYTES:
        raise AssertionError(
            f"GGUF suspiciously small: {path} = {size:,} bytes "
            f"(< {MIN_GGUF_BYTES:,})"
        )
    return {"path": str(path), "bytes": size}


# ----------------------------------------------------------------------------
# Modal-side worker
# ----------------------------------------------------------------------------


@app.function(
    image=gguf_image,
    cpu=8.0,
    memory=32768,  # 32 GB; quantization needs to mmap the f16 weights
    timeout=3600,
    volumes={"/vol": volume},
)
def export_one(model_name: str) -> dict[str, Any]:
    """Convert + quantize one model's merged fp16 checkpoint to GGUFs.

    Idempotent on the per-quant level: if both Q8_0 and Q5_K_M
    already exist with the right size floor, returns ``status="cached"``
    without rerunning the converter.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    validate_model_name(model_name)
    info = MODEL_REGISTRY[model_name]
    source_dir = Path(info["source_dir"])
    if not (source_dir / "config.json").exists():
        raise FileNotFoundError(
            f"No config.json at {source_dir}; the merged fp16 "
            "checkpoint is missing. Run merge_adapter.py "
            "(SFT) / Stage 7 merge (DPO) first."
        )

    paths = gguf_paths(model_name)
    out_dir = paths["f16"].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cached = all(paths[q].exists() for q in QUANT_TYPES) and all(
        paths[q].stat().st_size >= MIN_GGUF_BYTES for q in QUANT_TYPES
    )
    if cached:
        logger.info("All quants already present at %s — skipping convert.", out_dir)
        return {
            "status": "cached",
            "model_name": model_name,
            **{q: _verify_quant(paths[q]) for q in QUANT_TYPES},
        }

    # Convert HF -> f16 GGUF (single intermediate). The converter is a
    # Python script in the llama.cpp tree that reads HF-format
    # safetensors and emits a GGUF; the heavy lifting is dtype packing,
    # not arithmetic, so CPU is fine.
    convert_script = Path(LLAMA_CPP_DIR) / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found at {convert_script}. "
            f"llama.cpp checkout at {LLAMA_CPP_DIR} is malformed."
        )

    t0 = time.time()
    if not paths["f16"].exists():
        logger.info("Converting %s -> %s (f16 GGUF)", source_dir, paths["f16"])
        subprocess.run(
            [
                "python",
                str(convert_script),
                str(source_dir),
                "--outfile",
                str(paths["f16"]),
                "--outtype",
                "f16",
            ],
            check=True,
        )
        logger.info("Convert took %.1fs", time.time() - t0)

    # Quantize f16 -> Q8_0 and Q5_K_M.
    quantize_bin = Path(LLAMA_CPP_DIR) / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        raise FileNotFoundError(
            f"llama-quantize binary not at {quantize_bin}. "
            "Image build did not produce it."
        )

    results: dict[str, Any] = {"status": "produced", "model_name": model_name}
    for quant in QUANT_TYPES:
        if paths[quant].exists() and paths[quant].stat().st_size >= MIN_GGUF_BYTES:
            logger.info("%s already exists, skipping", paths[quant])
            results[quant] = _verify_quant(paths[quant])
            continue
        t1 = time.time()
        logger.info("Quantizing %s -> %s", paths["f16"], paths[quant])
        subprocess.run(
            [
                str(quantize_bin),
                str(paths["f16"]),
                str(paths[quant]),
                quant,
            ],
            check=True,
        )
        logger.info("Quantize %s took %.1fs", quant, time.time() - t1)
        results[quant] = _verify_quant(paths[quant])

    # Drop the f16 intermediate (~16 GB) once both quants are produced
    # — keeping it would balloon the volume for no reason. Re-running
    # the script will re-convert if needed.
    if all(paths[q].exists() for q in QUANT_TYPES):
        logger.info("Removing f16 intermediate %s", paths["f16"])
        paths["f16"].unlink(missing_ok=True)

    volume.commit()
    return results


@app.local_entrypoint()
def main(model: str = "all") -> None:
    """Operator entrypoint: export GGUFs for one or both models.

    ``model`` is one of ``"dpo"``, ``"sft"``, or ``"all"``.
    """
    targets = list(MODEL_REGISTRY) if model == "all" else [model]
    if model not in (*MODEL_REGISTRY, "all"):
        raise SystemExit(
            f"--model must be 'all' or one of {sorted(MODEL_REGISTRY)}; "
            f"got {model!r}"
        )
    for name in targets:
        print(f"[export_gguf] === {name} ===")
        result = export_one.remote(name)
        print(json.dumps(result, indent=2, default=str))
    print("[export_gguf] all targets done")


# Re-export for tests
__all__ = [
    "MODEL_REGISTRY",
    "QUANT_TYPES",
    "MIN_GGUF_BYTES",
    "validate_model_name",
    "gguf_paths",
    "_verify_quant",
]
