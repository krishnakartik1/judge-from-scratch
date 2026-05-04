"""Idempotent uploader for SFT training data and probe holdout.

Pushes ``data/formatted/sft.jsonl`` (the Stage 5 SFT artifact) and
``data/pairs/eval_set_unlabeled.jsonl`` (the religion-only OOD
holdout, used as the dry-run/full-run probe source) to the Modal
volume ``judge-from-scratch`` at ``/vol/data/``.

Run from project root:
    modal run train/modal/upload_data.py

Idempotent: re-running overwrites the same paths on the volume,
so you can re-upload after a Stage 5 re-run without volume cleanup.
The DPO upload in Stage 7 reuses this same pattern.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import modal

app = modal.App("judge-upload-data")

VOLUME_NAME = "judge-from-scratch"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

SFT_LOCAL = "data/formatted/sft.jsonl"
SFT_REMOTE = "/vol/data/sft.jsonl"
EVAL_LOCAL = "data/pairs/eval_set_unlabeled.jsonl"
EVAL_REMOTE = "/vol/data/eval_set_unlabeled.jsonl"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_file(SFT_LOCAL, "/root/sft.jsonl", copy=True)
    .add_local_file(EVAL_LOCAL, "/root/eval_set_unlabeled.jsonl", copy=True)
)


@app.function(image=image, volumes={"/vol": volume}, timeout=600)
def upload() -> dict[str, int]:
    Path("/vol/data").mkdir(parents=True, exist_ok=True)
    Path("/vol/checkpoints/sft").mkdir(parents=True, exist_ok=True)
    Path("/vol/checkpoints/sft-final").mkdir(parents=True, exist_ok=True)

    shutil.copyfile("/root/sft.jsonl", SFT_REMOTE)
    shutil.copyfile("/root/eval_set_unlabeled.jsonl", EVAL_REMOTE)

    volume.commit()

    sft_size = Path(SFT_REMOTE).stat().st_size
    eval_size = Path(EVAL_REMOTE).stat().st_size
    print(f"Uploaded {SFT_REMOTE} ({sft_size:,} bytes)")
    print(f"Uploaded {EVAL_REMOTE} ({eval_size:,} bytes)")
    print("\n/vol/data/ contents:")
    for p in sorted(Path("/vol/data").iterdir()):
        print(f"  {p.name}  {p.stat().st_size:>12,} bytes")

    return {"sft_bytes": sft_size, "eval_bytes": eval_size}


@app.local_entrypoint()
def main() -> None:
    """CPU-only volume upload. Records to the cost ledger; no prompt
    (the run is < $0.05 and a budget gate would be more friction
    than value)."""
    import time

    from train.modal._cost_ledger import record_cost, total_spend

    t0 = time.time()
    result = upload.remote()
    actual_wallclock_s = time.time() - t0

    print(f"\nUpload complete: {result}")
    print(f"\nVerify with: `modal volume ls {VOLUME_NAME} /data/`")

    row = record_cost(
        stage="stage6",
        function="upload_data",
        gpu="CPU",
        wallclock_s=actual_wallclock_s,
        notes=f"sft_bytes={result.get('sft_bytes')} "
        f"eval_bytes={result.get('eval_bytes')}",
    )
    print(
        f"[cost] Upload recorded: ${row['est_cost_usd']:.4f} "
        f"({actual_wallclock_s:.1f}s wallclock). "
        f"Cumulative spend: ${total_spend():.2f}"
    )
