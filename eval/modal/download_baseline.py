"""Download the bf16 baseline Gemma 4 E4B mirror to the Modal volume.

The Stage 8 vLLM eval pivot needs all three columns (baseline, SFT,
DPO) loaded at the same precision — bf16 — so any κ delta reflects
training, not the 4-bit/bf16 backend gap that disqualified the v1
Stage 8 runs. Stage 6 was trained from
``unsloth/gemma-4-E4B-it-unsloth-bnb-4bit`` (4-bit packaging); for
the eval column we instead pull ``unsloth/gemma-4-E4B-it`` (the
non-quant mirror) so vLLM has a clean bf16 checkpoint.

Run from project root::

    modal run eval/modal/download_baseline.py

CPU-only, ~5–10 min on a fresh volume; idempotent — skips if the
target directory already has a `config.json`, ≥1 safetensors shard,
the three tokenizer files, and total bytes ≥ 5 GB.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import modal

logger = logging.getLogger(__name__)

# --- Constants ------------------------------------------------------------
BASELINE_REPO_ID = "unsloth/gemma-4-E4B-it"
BASELINE_VOL_DIR = "/vol/checkpoints/baseline-bf16"
MIN_TOTAL_BYTES = 5_000_000_000  # ~8 GB expected; 5 GB is the floor
# Gemma 4's tokenizer.json bundles special tokens, so
# special_tokens_map.json is *optional*. tokenizer.json and
# tokenizer_config.json are the required pair.
REQUIRED_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
ALLOWED_DTYPES = {"bfloat16", "float16"}

# --- Modal app ------------------------------------------------------------
VOLUME_NAME = "judge-from-scratch"

app = modal.App("judge-baseline-download")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
    .add_local_file("uv.lock", "/root/uv.lock", copy=True)
    .uv_sync()
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]


def _verify_local(target_dir: Path) -> dict[str, Any] | None:
    """Return verifier summary if ``target_dir`` is already complete, else None.

    Used both as the idempotent-skip gate and the post-download check;
    keeping the predicate in one place stops the two paths from drifting.
    """
    if not target_dir.exists():
        return None
    config_path = target_dir / "config.json"
    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError:
        return None
    dtype = config.get("torch_dtype")
    if dtype not in ALLOWED_DTYPES:
        return None
    for fname in REQUIRED_TOKENIZER_FILES:
        if not (target_dir / fname).exists():
            return None
    shards = list(target_dir.glob("*.safetensors")) or list(
        target_dir.glob("pytorch_model*.bin")
    )
    if not shards:
        return None
    total_bytes = sum(s.stat().st_size for s in shards)
    if total_bytes < MIN_TOTAL_BYTES:
        return None
    return {
        "torch_dtype": dtype,
        "shard_count": len(shards),
        "total_bytes": total_bytes,
    }


@app.function(
    image=image,
    cpu=2.0,
    timeout=1800,
    volumes={"/vol": volume},
    secrets=secrets,
)
def download_baseline_remote() -> dict[str, Any]:
    """Probe the HF repo, download if missing, verify the local tree.

    Aborts with a clear message if the repo's ``config.json`` is
    missing (mirror moved / repo gated under different ID), if the
    declared ``torch_dtype`` is anything other than ``bfloat16`` /
    ``float16``, or if the post-download tree fails the integrity
    check (size, tokenizer files, safetensors shards).
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    target_dir = Path(BASELINE_VOL_DIR)

    cached = _verify_local(target_dir)
    if cached is not None:
        logger.info(
            "Baseline already present at %s (%d shards, %.2f GB, %s) — skipping.",
            target_dir,
            cached["shard_count"],
            cached["total_bytes"] / 1e9,
            cached["torch_dtype"],
        )
        return {"status": "already_present", **cached}

    # Pre-flight probe: try to fetch only config.json so we fail fast
    # with a clear error if the mirror is gone.
    target_dir.mkdir(parents=True, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    try:
        snapshot_download(
            repo_id=BASELINE_REPO_ID,
            local_dir=str(target_dir),
            allow_patterns=["config.json"],
            token=hf_token,
        )
    except RepositoryNotFoundError as exc:
        raise RuntimeError(
            f"HF repo {BASELINE_REPO_ID!r} not found. The non-quant "
            "mirror may have been renamed; try `google/gemma-4-e4b-it` "
            "or check the Unsloth org page."
        ) from exc

    config_path = target_dir / "config.json"
    config = json.loads(config_path.read_text())
    dtype = config.get("torch_dtype")
    if dtype not in ALLOWED_DTYPES:
        raise RuntimeError(
            f"Baseline mirror declares torch_dtype={dtype!r}; expected "
            f"one of {sorted(ALLOWED_DTYPES)}. This is the int8/nf4 "
            "mirror surprise the plan flagged — abort before downloading."
        )

    # Full download: model weights + tokenizer + small JSON files.
    snapshot_download(
        repo_id=BASELINE_REPO_ID,
        local_dir=str(target_dir),
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "tokenizer.model",
            "*.txt",
            "*.jinja",
        ],
        token=hf_token,
    )
    volume.commit()

    verified = _verify_local(target_dir)
    if verified is None:
        raise AssertionError(
            f"Post-download verifier failed for {target_dir}. "
            "`rm -rf` the directory and re-run."
        )
    logger.info(
        "Downloaded baseline to %s: %d shards, %.2f GB, %s",
        target_dir,
        verified["shard_count"],
        verified["total_bytes"] / 1e9,
        verified["torch_dtype"],
    )
    return {"status": "downloaded", **verified}


@app.local_entrypoint()
def main() -> None:
    """Operator entrypoint: download baseline (or confirm already present)."""
    result = download_baseline_remote.remote()
    print(json.dumps(result, indent=2, default=str))
