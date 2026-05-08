"""Merge the Stage 6 SFT LoRA adapter into a fresh bf16 checkpoint.

The Stage 8 vLLM eval pivot needs a *merged* SFT checkpoint at fp16
because vLLM does not load PEFT adapters directly the way Unsloth's
``FastLanguageModel`` does. Stage 7 already produced a merged DPO
artifact at ``/vol/checkpoints/merged-fp16/``; this script does the
same trick for the SFT adapter, writing to
``/vol/checkpoints/sft-merged-fp16/``.

Pattern lifted from ``train/modal/dpo.py:_merge_to_fp16`` (lines
975–1036). Hardcoded source/dest paths because this is a one-shot
utility, not a configurable trainer step.

Run from project root::

    modal run eval/modal/merge_adapter.py

A100, ~3–5 min wallclock, ~$0.20-0.25.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

import modal

logger = logging.getLogger(__name__)

# --- Constants ------------------------------------------------------------
SFT_ADAPTER_DIR = "/vol/checkpoints/sft-final"
SFT_MERGED_DIR = "/vol/checkpoints/sft-merged-fp16"
MAX_SEQ_LENGTH = 2048  # matches train/configs/sft.yaml
MIN_TOTAL_BYTES = 5_000_000_000
# Same required-tokenizer pair as download_baseline; tokenizer.json
# bundles special tokens for Gemma 4 so special_tokens_map.json
# is optional.
REQUIRED_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")

# --- Modal app ------------------------------------------------------------
MODAL_GPU = os.environ.get("MODAL_GPU", "A100")
VOLUME_NAME = "judge-from-scratch"

app = modal.App("judge-merge-adapter")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
    .add_local_file("uv.lock", "/root/uv.lock", copy=True)
    .uv_sync()
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]


def _verify_merged(merged_dir: Path) -> dict[str, Any]:
    """Run the post-merge integrity checks.

    Asserts the things vLLM will need at load time: ``config.json``,
    weight shards, the two required tokenizer files, and a non-tiny
    total size. Returns the summary dict on success; raises with a
    clear recovery hint on failure.
    """
    if not (merged_dir / "config.json").exists():
        raise AssertionError(
            f"Merged checkpoint at {merged_dir} missing config.json. "
            f"`rm -rf {merged_dir}` and re-run."
        )
    shards = list(merged_dir.glob("*.safetensors")) or list(
        merged_dir.glob("pytorch_model*.bin")
    )
    if not shards:
        raise AssertionError(
            f"Merged checkpoint at {merged_dir} has no model weights. "
            f"`rm -rf {merged_dir}` and re-run."
        )
    total_bytes = sum(s.stat().st_size for s in shards)
    if total_bytes < MIN_TOTAL_BYTES:
        raise AssertionError(
            f"Merged checkpoint suspiciously small: {total_bytes:,} bytes. "
            f"`rm -rf {merged_dir}` and re-run."
        )
    for fname in REQUIRED_TOKENIZER_FILES:
        if not (merged_dir / fname).exists():
            raise AssertionError(
                f"Merged checkpoint at {merged_dir} missing tokenizer "
                f"file {fname!r} (vLLM needs it). "
                f"`rm -rf {merged_dir}` and re-run."
            )
    return {
        "merged_dir": str(merged_dir),
        "shard_count": len(shards),
        "total_bytes": total_bytes,
    }


@app.function(
    image=image,
    gpu=MODAL_GPU,
    timeout=900,
    volumes={"/vol": volume},
    secrets=secrets,
)
def merge_sft_to_fp16_remote() -> dict[str, Any]:
    """Reload the SFT adapter at fp16, merge, save, and verify.

    Idempotent: ``shutil.rmtree(merged_dir, ignore_errors=True)``
    at the start so a stale shard layout from a killed prior run
    can't survive into the new merge.
    """
    import torch
    from unsloth import FastLanguageModel

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    adapter_dir = Path(SFT_ADAPTER_DIR)
    merged_dir = Path(SFT_MERGED_DIR)

    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"No adapter_config.json at {adapter_dir}. Stage 6 SFT "
            "must have completed before the merge can run."
        )

    shutil.rmtree(merged_dir, ignore_errors=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    merge_model, merge_tok = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,  # critical — fp16 merge requires fp16 base
        dtype=torch.bfloat16,
    )
    merge_model.save_pretrained_merged(
        str(merged_dir),
        merge_tok,
        save_method="merged_16bit",
    )
    volume.commit()

    summary = _verify_merged(merged_dir)
    summary["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
    logger.info(
        "Merged SFT adapter at %s: %d shards, %.2f GB, peak_vram=%.2fGB",
        merged_dir,
        summary["shard_count"],
        summary["total_bytes"] / 1e9,
        summary["peak_vram_gb"],
    )
    return summary


@app.local_entrypoint()
def main() -> None:
    """Operator entrypoint: merge the SFT adapter to fp16."""
    import json

    result = merge_sft_to_fp16_remote.remote()
    print(json.dumps(result, indent=2, default=str))
