"""Stage 1 sampler — produce a stratified BBQ sample for REVAL Judge.

Loads HuggingFace ``Elfsong/BBQ`` across all 11 splits and draws a 1,500-row
sample stratified by ``(category × context_condition × question_polarity)``.
Writes ``data/raw/bbq_sample.jsonl`` plus a metadata sidecar.

Run before ``data/01_generate_candidates.py``. Production output is locked
once written — re-sampling requires manually deleting ``bbq_sample.jsonl``.

Usage:
    uv run python data/00_sample_bbq.py [--dry-run-categories N] [--seed N]
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

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
SAMPLE_PATH = RAW_DIR / "bbq_sample.jsonl"
SAMPLE_META_PATH = RAW_DIR / "bbq_sample.meta.json"
DRYRUN_PATH = RAW_DIR / "bbq_sample.dryrun.jsonl"

# Sorted alphabetically; the last category absorbs the per-category remainder.
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
TOTAL_TARGET = 1500
DATASET_REPO = "Elfsong/BBQ"


def per_category_target(category: str) -> int:
    """Per-category target. Last alphabetical category absorbs the remainder."""
    return 140 if category == CATEGORIES[-1] else 136


def per_cell_target(category: str) -> int:
    """Per-(condition × polarity) target inside one category. 4 cells/category."""
    return per_category_target(category) // 4


def make_question_id(category: str, record: dict[str, Any]) -> str:
    """Composite ``{category}::{question_index}::{example_id}::{polarity}``.

    ``category`` is the lowercase split name we iterated over, not
    ``record['category']`` (which is capitalized in BBQ).
    """
    return (
        f"{category}::{record['question_index']}::"
        f"{record['example_id']}::{record['question_polarity']}"
    )


def extract_stereotyped_groups(metadata: dict[str, Any]) -> list[str]:
    """Flatten BBQ's stereotyped-group annotation to a list of strings.

    BBQ exposes two related fields under ``additional_metadata``:
    ``stereotyped_groups`` (a proper ``list[str]`` populated for every
    category including the intersectional ones) and
    ``known_stereotyped_groups`` (a comma-separated string that is
    literally ``"nan"`` for intersectional categories). Prefer the list
    field; fall back to splitting the string only if the list is absent.
    """
    sg = metadata.get("stereotyped_groups")
    if isinstance(sg, list):
        return [str(s) for s in sg if s]
    ksg = metadata.get("known_stereotyped_groups")
    if isinstance(ksg, list):
        return [str(s) for s in ksg if s]
    if isinstance(ksg, str) and ksg and ksg.lower() != "nan":
        return [s.strip() for s in ksg.split(",") if s.strip()]
    return []


def sample_category(
    category: str, rng: random.Random
) -> tuple[list[dict[str, Any]], list[str], dict[tuple[str, str], int]]:
    """Stratified sample for one BBQ category.

    Verifies the per-split assumption that
    ``(example_id, question_index)`` is unique, then buckets rows by
    ``(context_condition, question_polarity)`` and samples each cell to its
    target. Cells with too few rows take everything available and emit a
    warning rather than failing.

    Args:
        category: Lowercase BBQ split name (e.g. ``"age"``).
        rng: A seeded ``random.Random`` shared across categories so the full
            run is reproducible.

    Returns:
        ``(rows, warnings, per_cell_counts)``.
    """
    logger.info("Loading split %s", category)
    split = load_dataset(DATASET_REPO, split=category)

    seen_keys: set[tuple[int, int]] = set()
    bucket: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    valid_cells = set(CELLS)
    for record in split:
        key = (int(record["example_id"]), int(record["question_index"]))
        if key in seen_keys:
            raise AssertionError(
                f"BBQ split {category!r} has duplicate "
                f"(example_id, question_index)={key!r}; "
                "schema assumption violated."
            )
        seen_keys.add(key)
        cell = (record["context_condition"], record["question_polarity"])
        if cell not in valid_cells:
            continue
        bucket[cell].append(record)

    target = per_cell_target(category)
    out: list[dict[str, Any]] = []
    counts: dict[tuple[str, str], int] = {}
    warnings: list[str] = []

    for cell in CELLS:
        condition, polarity = cell
        pool = bucket[cell]
        if len(pool) >= target:
            picked = rng.sample(pool, target)
        else:
            picked = list(pool)
            msg = (
                f"cell {category}/{condition}/{polarity}: "
                f"target {target}, only {len(pool)} available, "
                f"taking {len(pool)}"
            )
            logger.warning(msg)
            warnings.append(msg)
        counts[cell] = len(picked)
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
    return out, warnings, counts


def get_dataset_revision() -> str | None:
    """Best-effort fetch of the HF revision SHA for the metadata sidecar."""
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(DATASET_REPO).sha
    except Exception as exc:  # noqa: BLE001 — forensic field, not load-bearing
        logger.warning("Could not fetch HF dataset revision: %s", exc)
        return None


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Buffer rows in memory, write to ``<path>.tmp``, then rename to ``path``.

    Tmp lives in the same directory as the destination so ``rename`` stays
    on one filesystem and is atomic.
    """
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
    """Tmp + rename for a single JSON document (the metadata sidecar)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    tmp.replace(path)


def cleanup_stale_tmp_files() -> None:
    """Remove leftover ``.tmp`` files from prior killed runs.

    Atomicity guarantees these can never represent committed state; lingering
    cruft is just noise.
    """
    for target in (SAMPLE_PATH, SAMPLE_META_PATH, DRYRUN_PATH):
        tmp = target.with_name(target.name + ".tmp")
        if tmp.exists():
            logger.info("Removing stale tmp file: %s", tmp)
            tmp.unlink()


def format_table(
    counts_by_category: dict[str, dict[tuple[str, str], int]],
    targets_by_category: dict[str, int],
) -> str:
    """Render the per-cell summary table.

    Cells under target are shown as ``actual(target)``. Header groups
    ``ambig`` over its two polarity columns and ``disambig`` over its two,
    with a small gap between the two groups.
    """
    cat_w = 22
    cell_w = 9
    gap = "  "
    pad = " " * cat_w
    group_w = 2 * cell_w
    top = pad + f"{'ambig':^{group_w}s}" + gap + f"{'disambig':^{group_w}s}"
    sub = (
        pad
        + f"{'neg':>{cell_w}s}{'nonneg':>{cell_w}s}"
        + gap
        + f"{'neg':>{cell_w}s}{'nonneg':>{cell_w}s}"
    )
    lines = [top, sub]
    for category in CATEGORIES:
        if category not in counts_by_category:
            continue
        target = targets_by_category[category]
        cells = counts_by_category[category]
        parts = [f"{category:<{cat_w}s}"]
        for i, cell in enumerate(CELLS):
            actual = cells.get(cell, 0)
            text = f"{actual}({target})" if actual < target else str(actual)
            parts.append(f"{text:>{cell_w}s}")
            if i == 1:
                parts.append(gap)
        lines.append("".join(parts))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sample BBQ for REVAL Judge Stage 1. Stratifies by "
            "(category × condition × polarity)."
        )
    )
    parser.add_argument(
        "--dry-run-categories",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Sample only the first N categories (alphabetical). Writes "
            "data/raw/bbq_sample.dryrun.jsonl; never the locked sample."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42). Recorded in the metadata sidecar.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    is_dry_run = args.dry_run_categories is not None
    output_path = DRYRUN_PATH if is_dry_run else SAMPLE_PATH

    cleanup_stale_tmp_files()

    if not is_dry_run and SAMPLE_PATH.exists():
        logger.error("%s already exists — delete to re-sample.", SAMPLE_PATH)
        return 1

    if is_dry_run:
        if args.dry_run_categories <= 0:
            logger.error("--dry-run-categories must be a positive integer.")
            return 2
        if args.dry_run_categories > len(CATEGORIES):
            logger.error(
                "--dry-run-categories=%d exceeds the %d available categories.",
                args.dry_run_categories,
                len(CATEGORIES),
            )
            return 2
        categories: list[str] = list(CATEGORIES[: args.dry_run_categories])
    else:
        categories = list(CATEGORIES)

    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    counts_by_category: dict[str, dict[tuple[str, str], int]] = {}
    targets_by_category: dict[str, int] = {}
    warnings: list[str] = []

    for category in categories:
        cat_rows, cat_warnings, cat_counts = sample_category(category, rng)
        rows.extend(cat_rows)
        counts_by_category[category] = cat_counts
        targets_by_category[category] = per_cell_target(category)
        warnings.extend(cat_warnings)

    target_total = (
        sum(per_category_target(c) for c in categories) if is_dry_run else TOTAL_TARGET
    )

    atomic_write_jsonl(output_path, rows)
    logger.info("Wrote %d rows to %s", len(rows), output_path)

    if not is_dry_run:
        per_cell_serializable = {
            cat: {f"{cond}/{pol}": n for (cond, pol), n in cells.items()}
            for cat, cells in counts_by_category.items()
        }
        meta = {
            "seed": args.seed,
            "n_rows_target": target_total,
            "n_rows_actual": len(rows),
            "per_cell_counts": per_cell_serializable,
            "warnings": warnings,
            "sampled_at_utc": datetime.now(UTC).isoformat(),
            "bbq_dataset_revision": get_dataset_revision(),
        }
        atomic_write_json(SAMPLE_META_PATH, meta)
        logger.info("Wrote metadata sidecar to %s", SAMPLE_META_PATH)

    pct = 100.0 * len(rows) / target_total if target_total else 0.0
    n_primary = sum(1 for c in categories if "_x_" not in c)
    n_inter = sum(1 for c in categories if "_x_" in c)
    cat_descr = f"{n_primary} primary"
    if n_inter:
        cat_descr += f" + {n_inter} intersectional"

    print()
    print(
        f"Sampling complete. Sample size: {len(rows)} / {target_total} "
        f"({pct:.1f}%)."
    )
    print(f"Categories: {len(categories)} ({cat_descr}).")
    print()
    print("Per-cell counts (target shown in parens if shortfall):")
    print(format_table(counts_by_category, targets_by_category))
    print()
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")
        if not is_dry_run:
            short = target_total - len(rows)
            print(f"Total sample: {len(rows)} / {target_total} ({short} short).")
    else:
        print("Warnings: (none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
