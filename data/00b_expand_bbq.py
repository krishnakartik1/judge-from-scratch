"""Stage 0b — append additional BBQ rows to an existing sample.

Reads the locked ``data/raw/bbq_sample.jsonl`` (preserves it byte-for-byte)
and adds ``--n-additional N`` more rows drawn from BBQ records NOT already
in the sample, stratified by ``(category x context_condition x question_polarity)``.
The combined file is then written atomically.

Stage 1 (candidate generator) reads the expanded file and resumes; existing
``pair_key`` values are skipped, only the new rows trigger API calls.

Usage:
    uv run python data/00b_expand_bbq.py --n-additional 1500 [--seed 43]

Default seed for the expansion is 43 (one off from the original sample's 42)
so the per-cell rng stream is independent.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset

from scripts.common import jsonl_read

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
SAMPLE_PATH = RAW_DIR / "bbq_sample.jsonl"
SAMPLE_META_PATH = RAW_DIR / "bbq_sample.meta.json"

CATEGORIES: tuple[str, ...] = (
    "age",
    "disability_status",
    "gender_identity",
    "nationality",
    "physical_appearance",
    "race_ethnicity",
    "race_x_gender",
    "race_x_ses",
    "religion",
    "ses",
    "sexual_orientation",
)
CONDITIONS: tuple[str, ...] = ("ambig", "disambig")
POLARITIES: tuple[str, ...] = ("neg", "nonneg")
CELLS: tuple[tuple[str, str], ...] = tuple(
    (c, p) for c in CONDITIONS for p in POLARITIES
)
DATASET_REPO = "Elfsong/BBQ"


def make_question_id(category: str, record: dict[str, Any]) -> str:
    """Match Stage 0's id construction so existing rows compare cleanly."""
    return (
        f"{category}::{record['question_index']}::"
        f"{record['example_id']}::{record['question_polarity']}"
    )


def extract_stereotyped_groups(metadata: dict[str, Any]) -> list[str]:
    sg = metadata.get("stereotyped_groups")
    if isinstance(sg, list):
        return [str(s) for s in sg if s]
    ksg = metadata.get("known_stereotyped_groups")
    if isinstance(ksg, list):
        return [str(s) for s in ksg if s]
    if isinstance(ksg, str) and ksg and ksg.lower() != "nan":
        return [s.strip() for s in ksg.split(",") if s.strip()]
    return []


def existing_qids(path: Path) -> set[str]:
    """Read question_ids already in the sample."""
    return {row["question_id"] for row in jsonl_read(path)}


def per_cell_additional_target(n_additional: int) -> int:
    """How many extra rows per cell. 11 categories x 4 cells = 44 cells.

    For n_additional=1500, that's about 34 per cell, matching the original
    Stage 0 cell target. We round up so the total is at least n_additional;
    surplus from one cell (if pool exhausted) is absorbed by the next.
    """
    return -(-n_additional // (len(CATEGORIES) * len(CELLS)))


def sample_additions_for_category(
    category: str,
    existing: set[str],
    additional_per_cell: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Sample additional rows for one category, skipping existing question_ids."""
    logger.info("Loading split %s", category)
    split = load_dataset(DATASET_REPO, split=category)

    bucket: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    valid_cells = set(CELLS)
    for record in split:
        cell = (record["context_condition"], record["question_polarity"])
        if cell not in valid_cells:
            continue
        qid = make_question_id(category, record)
        if qid in existing:
            continue
        bucket[cell].append(record)

    out: list[dict[str, Any]] = []
    warnings: list[str] = []
    for cell in CELLS:
        condition, polarity = cell
        pool = bucket[cell]
        if len(pool) >= additional_per_cell:
            picked = rng.sample(pool, additional_per_cell)
        else:
            picked = list(pool)
            msg = (
                f"cell {category}/{condition}/{polarity}: "
                f"requested {additional_per_cell} additional, "
                f"only {len(pool)} disjoint records available, taking all."
            )
            logger.warning(msg)
            warnings.append(msg)
        for record in picked:
            out.append(
                {
                    "question_id": make_question_id(category, record),
                    "category": category,
                    "context_condition": record["context_condition"],
                    "question_polarity": record["question_polarity"],
                    "example_id": int(record["example_id"]),
                    "question_index": int(record["question_index"]),
                    "context": record["context"],
                    "question": record["question"],
                    "ans0": record["ans0"],
                    "ans1": record["ans1"],
                    "ans2": record["ans2"],
                    "answer_label": int(record["answer_label"]),
                    "target_label": int(record["target_label"]),
                    "answer_info": record["answer_info"],
                    "stereotyped_groups": extract_stereotyped_groups(
                        record["additional_metadata"]
                    ),
                }
            )
    return out, warnings


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write all rows to ``<path>.tmp`` then rename to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
    tmp.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Append additional BBQ rows to bbq_sample.jsonl. Existing rows "
            "are preserved; new rows are sampled from BBQ records not in "
            "the existing sample, stratified by category x condition x polarity."
        )
    )
    parser.add_argument(
        "--n-additional", type=int, required=True, metavar="N",
        help="How many extra rows to append.",
    )
    parser.add_argument(
        "--seed", type=int, default=43,
        help="RNG seed for the expansion (default 43; original was 42).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute additions and report counts; do not write files.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not SAMPLE_PATH.exists():
        logger.error("%s not found. Run data/00_sample_bbq.py first.", SAMPLE_PATH)
        return 2

    existing = existing_qids(SAMPLE_PATH)
    logger.info("Existing sample has %d rows.", len(existing))

    additional_per_cell = per_cell_additional_target(args.n_additional)
    logger.info(
        "Targeting +%d rows total = +%d per cell over %d cells.",
        args.n_additional,
        additional_per_cell,
        len(CATEGORIES) * len(CELLS),
    )

    rng = random.Random(args.seed)
    new_rows: list[dict[str, Any]] = []
    all_warnings: list[str] = []
    for category in CATEGORIES:
        cat_rows, cat_warnings = sample_additions_for_category(
            category, existing, additional_per_cell, rng
        )
        new_rows.extend(cat_rows)
        all_warnings.extend(cat_warnings)

    # Trim to exactly n_additional in case rounding-up overshoots, picking
    # deterministically based on the rng's current state.
    if len(new_rows) > args.n_additional:
        rng.shuffle(new_rows)
        new_rows = new_rows[: args.n_additional]
    logger.info("Sampled %d additional rows.", len(new_rows))

    if args.dry_run:
        logger.info("--dry-run: not writing files.")
        return 0

    # Read existing rows in their on-disk order so the prefix is preserved.
    existing_rows = list(jsonl_read(SAMPLE_PATH))
    combined = existing_rows + new_rows
    atomic_write_jsonl(SAMPLE_PATH, combined)
    logger.info("Wrote expanded sample (%d total rows) to %s", len(combined), SAMPLE_PATH)

    # Update the metadata sidecar.
    if SAMPLE_META_PATH.exists():
        with SAMPLE_META_PATH.open() as fh:
            meta = json.load(fh)
    else:
        meta = {}
    meta.setdefault("expansions", []).append(
        {
            "expansion_seed": args.seed,
            "n_additional": args.n_additional,
            "n_total_after_expansion": len(combined),
            "expanded_at_utc": datetime.now(tz=UTC).isoformat(),
            "warnings": all_warnings,
        }
    )
    meta["n_rows_actual"] = len(combined)
    atomic_write_json(SAMPLE_META_PATH, meta)
    logger.info("Updated meta sidecar at %s", SAMPLE_META_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
