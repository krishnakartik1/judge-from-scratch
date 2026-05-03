"""Stage 3a — carve a 300-pair human-eval holdout out of pairs.jsonl.

Reads ``data/pairs/pairs.jsonl`` (Stage 2 output) and emits three
artifacts plus a sentinel sidecar:

    eval_set_unlabeled.jsonl    300 pairs awaiting human labels
                                  (IN_DIST_TOTAL in_dist + OOD_TOTAL ood_religion)
    pairs_to_label.jsonl        non-religion pairs for Stage 3 Claude
                                  labeling (zero religion — holdout protected)
    pairs_unused_religion.jsonl  religion pairs not picked for OOD;
                                  quarantined out of training in v1
    eval_holdout.meta.json     written LAST; presence is the
                                  "stage finished cleanly" sentinel

Exact counts depend on the input pairs.jsonl size and the religion
pool size; the script prints them at the end of every run. The
sidecar records achieved counts under ``n_eval_unlabeled``,
``n_pairs_to_label``, and ``n_unused_religion``.

Per primer §"Stage 5: Hold out a human-labeled eval set" and the
religion-only OOD design (primer line 451). v2b will add UnQover for a
"different dataset entirely" OOD claim.

Determinism: seed=42 by default. Sort the input by ``pair_id`` first,
then partition, then sub-bucket, then ``rng.sample``. Same seed + same
input SHA-256 ⇒ same output.

Resume model: this stage is **not per-record resumable** (sampling
depends on the full pool). The sidecar is the sentinel — if it exists
and matches current state, exit 0; otherwise exit 1 and require
``--force`` to redo. ``--force`` unlinks the eight enumerated paths
(four real outputs + four ``.tmp`` siblings) before rebuilding.

Usage:
    uv run python data/03a_holdout_eval.py [--seed N] [--force]
                                            [--dry-run] [--preview-eval]

Exit codes: 0 success, 1 assertion or resume-state mismatch, 2 missing
input file.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

from scripts.common import (
    atomic_write_json,
    atomic_write_jsonl,
    file_sha256,
    jsonl_read,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
PAIRS_DIR = REPO_ROOT / "data" / "pairs"

INPUT_PATH = PAIRS_DIR / "pairs.jsonl"
EVAL_PATH = PAIRS_DIR / "eval_set_unlabeled.jsonl"
TO_LABEL_PATH = PAIRS_DIR / "pairs_to_label.jsonl"
UNUSED_RELIGION_PATH = PAIRS_DIR / "pairs_unused_religion.jsonl"
META_PATH = PAIRS_DIR / "eval_holdout.meta.json"

# In-dist eval split per the prompt: stratified across pair_category to
# roughly match the overall training mix while upweighting clear-bias
# and tracked-bias-vs-alternate buckets (the high-signal ones).
IN_DIST_TARGETS: dict[str, int] = {
    "clear_bias_vs_clean": 110,
    "subtle_bias_vs_clean": 50,
    "tracked_bias_vs_alternate": 35,
    "both_clean_tie": 25,
    "adversarial": 20,
}
IN_DIST_TOTAL = sum(IN_DIST_TARGETS.values())  # 240

# OOD religion holdout size; bucket targets are derived from
# IN_DIST_TARGETS via allocate_ood_targets() so the OOD slice mirrors
# the in-dist mix proportionally.
OOD_TOTAL = 60

RELIGION_VALUE = "religion"

# All paths (real + .tmp) that --force unlinks. Enumerated literally so
# a future helper rename can't silently widen or narrow the cleanup.
_ALL_OUTPUT_PATHS: tuple[Path, ...] = (
    EVAL_PATH,
    EVAL_PATH.with_name(EVAL_PATH.name + ".tmp"),
    TO_LABEL_PATH,
    TO_LABEL_PATH.with_name(TO_LABEL_PATH.name + ".tmp"),
    UNUSED_RELIGION_PATH,
    UNUSED_RELIGION_PATH.with_name(UNUSED_RELIGION_PATH.name + ".tmp"),
    META_PATH,
    META_PATH.with_name(META_PATH.name + ".tmp"),
)


# -----------------------------------------------------------------------------
# Pure functions (unit-tested)
# -----------------------------------------------------------------------------


def allocate_ood_targets(in_dist_targets: dict[str, int], total: int) -> dict[str, int]:
    """Stratify ``total`` across buckets proportionally to in-dist targets.

    Largest-residual rounding with a fully deterministic three-key sort:
    ``(residual desc, in-dist-target desc, bucket-name asc)``. Buckets
    in the input dict whose share rounds to zero still receive zero
    unless they win a residual bump.

    Args:
        in_dist_targets: Per-bucket in-distribution targets (positive ints).
        total: The total OOD count to distribute.

    Returns:
        Per-bucket OOD targets summing to exactly ``total``.
    """
    in_dist_sum = sum(in_dist_targets.values())
    if in_dist_sum <= 0:
        raise ValueError("in_dist_targets must sum to > 0")
    if total < 0:
        raise ValueError("total must be >= 0")

    # Integer arithmetic — residual_num / in_dist_sum is the fractional
    # part of the raw share. Avoids float-precision quirks that would
    # silently reorder buckets with mathematically equal residuals.
    floors = {b: (t * total) // in_dist_sum for b, t in in_dist_targets.items()}
    residual_num = {b: (t * total) % in_dist_sum for b, t in in_dist_targets.items()}
    bumps_needed = total - sum(floors.values())

    # Sort: residual desc, in-dist desc, name asc — tertiary key locks
    # determinism when the first two tie.
    ranked = sorted(
        in_dist_targets.keys(),
        key=lambda b: (-residual_num[b], -in_dist_targets[b], b),
    )

    result = dict(floors)
    for b in ranked[:bumps_needed]:
        result[b] += 1
    return result


def partition_pairs(
    pairs: list[dict[str, Any]], religion_value: str = RELIGION_VALUE
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split pairs into (religion_pool, nonreligion_pool).

    Both lists preserve the input order. Caller is responsible for
    sorting the input first if reproducibility is required.

    Args:
        pairs: All Stage 2 pair records.
        religion_value: The ``bias_category`` value flagged as OOD.

    Returns:
        ``(religion_pool, nonreligion_pool)``.
    """
    religion = [p for p in pairs if p["bias_category"] == religion_value]
    nonreligion = [p for p in pairs if p["bias_category"] != religion_value]
    return religion, nonreligion


def annotate_eval_record(record: dict[str, Any], eval_slice: str) -> dict[str, Any]:
    """Return a deep copy with ground-truth stripped + label stubs added.

    Strips ``suspected_bias_level`` from both responses (Stage 2
    docstring excludes ground-truth fields from labeling input; same
    reasoning applies to the human labeler). Keeps the ``model`` field
    on each response — the human labeler grades under production
    conditions, where model identity is visible.

    Args:
        record: Stage 2 pair record.
        eval_slice: Either ``"in_dist"`` or ``"ood_religion"``.

    Returns:
        Annotated deep copy ready for human labeling.
    """
    out = copy.deepcopy(record)
    for side in ("response_a", "response_b"):
        out[side].pop("suspected_bias_level", None)
    out["human_verdict"] = None
    out["confidence"] = None
    out["notes"] = None
    out["eval_slice"] = eval_slice
    return out


def _bucket_by_category(
    pool: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group pairs by ``pair_category`` preserving input order."""
    buckets: dict[str, list[dict[str, Any]]] = {b: [] for b in IN_DIST_TARGETS}
    for p in pool:
        cat = p["pair_category"]
        if cat not in buckets:
            raise AssertionError(
                f"unknown pair_category {cat!r} in input pool — "
                f"expected one of {sorted(IN_DIST_TARGETS)}"
            )
        buckets[cat].append(p)
    return buckets


def sample_holdout(
    pairs: list[dict[str, Any]],
    in_dist_targets: dict[str, int],
    total_ood: int,
    seed: int,
    religion_value: str = RELIGION_VALUE,
) -> dict[str, list[dict[str, Any]]]:
    """Compute the four output sets in memory.

    Sorts input by ``pair_id``, partitions into religion / non-religion,
    sub-buckets each by ``pair_category``, and samples per-bucket via
    ``random.Random(seed).sample``. Asserts per-bucket sufficiency
    BEFORE any sampling so a supply failure raises clearly.

    Args:
        pairs: All Stage 2 pair records.
        in_dist_targets: Per-bucket in-dist targets (used both directly
            and to derive OOD targets via :func:`allocate_ood_targets`).
        total_ood: Total OOD picks (e.g. 60).
        seed: RNG seed.
        religion_value: ``bias_category`` value treated as OOD.

    Returns:
        Dict with keys ``eval_in_dist``, ``eval_ood``, ``to_label``,
        ``unused_religion``. Each value is a list of records (annotated
        with ``eval_slice`` for the eval lists; raw for the others).
    """
    sorted_pairs = sorted(pairs, key=lambda r: r["pair_id"])

    religion_pool, nonreligion_pool = partition_pairs(sorted_pairs, religion_value)
    religion_buckets = _bucket_by_category(religion_pool)
    nonreligion_buckets = _bucket_by_category(nonreligion_pool)

    ood_targets = allocate_ood_targets(in_dist_targets, total_ood)

    for b, target in ood_targets.items():
        supply = len(religion_buckets[b])
        assert (
            supply >= target
        ), f"OOD bucket {b!r}: religion supply {supply} < target {target}"
    for b, target in in_dist_targets.items():
        supply = len(nonreligion_buckets[b])
        assert (
            supply >= target
        ), f"in-dist bucket {b!r}: non-religion supply {supply} < target {target}"

    rng = random.Random(seed)
    eval_in_dist: list[dict[str, Any]] = []
    eval_ood: list[dict[str, Any]] = []
    in_dist_picked_ids: set[str] = set()
    ood_picked_ids: set[str] = set()

    for b in sorted(IN_DIST_TARGETS):
        picks = rng.sample(nonreligion_buckets[b], in_dist_targets[b])
        for p in picks:
            eval_in_dist.append(annotate_eval_record(p, "in_dist"))
            in_dist_picked_ids.add(p["pair_id"])

    for b in sorted(IN_DIST_TARGETS):
        picks = rng.sample(religion_buckets[b], ood_targets[b])
        for p in picks:
            eval_ood.append(annotate_eval_record(p, "ood_religion"))
            ood_picked_ids.add(p["pair_id"])

    to_label = [p for p in nonreligion_pool if p["pair_id"] not in in_dist_picked_ids]
    unused_religion = [p for p in religion_pool if p["pair_id"] not in ood_picked_ids]

    return {
        "eval_in_dist": eval_in_dist,
        "eval_ood": eval_ood,
        "to_label": to_label,
        "unused_religion": unused_religion,
    }


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def _bucket_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {b: 0 for b in IN_DIST_TARGETS}
    for r in records:
        counts[r["pair_category"]] += 1
    return counts


def _print_report(splits: dict[str, list[dict[str, Any]]]) -> None:
    in_dist_counts = _bucket_counts(splits["eval_in_dist"])
    ood_counts = _bucket_counts(splits["eval_ood"])
    total = (
        len(splits["eval_in_dist"])
        + len(splits["eval_ood"])
        + len(splits["to_label"])
        + len(splits["unused_religion"])
    )

    bar = "-" * 72
    print(bar)
    print("Stage 3a — eval-set holdout report")
    print(bar)
    print(
        f"  eval_set_unlabeled    : {len(splits['eval_in_dist']) + len(splits['eval_ood']):>5}"
        f"  ({len(splits['eval_in_dist'])} in_dist + {len(splits['eval_ood'])} ood_religion)"
    )
    print(f"  pairs_to_label        : {len(splits['to_label']):>5}")
    print(f"  pairs_unused_religion : {len(splits['unused_religion']):>5}")
    print(f"  TOTAL accounted for   : {total:>5}")
    print(bar)
    print("  in-dist per pair_category:")
    for b in IN_DIST_TARGETS:
        print(f"    {b:<28} {in_dist_counts[b]:>4} / target {IN_DIST_TARGETS[b]}")
    print("  ood_religion per pair_category:")
    ood_targets = allocate_ood_targets(IN_DIST_TARGETS, OOD_TOTAL)
    for b in IN_DIST_TARGETS:
        print(f"    {b:<28} {ood_counts[b]:>4} / target {ood_targets[b]}")
    print(bar)


def _print_preview(splits: dict[str, list[dict[str, Any]]]) -> None:
    """Print first 3 records of each candidate set, ≥1 in_dist + ≥1 ood."""
    bar = "=" * 72
    for name, records in (
        ("eval_in_dist", splits["eval_in_dist"]),
        ("eval_ood_religion", splits["eval_ood"]),
        ("to_label", splits["to_label"]),
        ("unused_religion", splits["unused_religion"]),
    ):
        print(bar)
        print(f"PREVIEW — {name} (first 3 of {len(records)})")
        print(bar)
        for r in records[:3]:
            print(json.dumps(r, ensure_ascii=False))


# -----------------------------------------------------------------------------
# Resume-state inspection
# -----------------------------------------------------------------------------


def _count_jsonl_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def _check_existing_state(
    expected_input_sha: str,
) -> tuple[str, str | None]:
    """Inspect on-disk outputs and return ``(state, diagnostic)``.

    State is one of:
      - ``"empty"`` — no real outputs and no sidecar; safe to run.
      - ``"clean"`` — sidecar + 3 jsonl files exist with matching counts
                       and matching input SHA. Already done.
      - ``"mismatch"`` — partial files, count mismatch, SHA mismatch, or
                       any other inconsistent state. Caller should exit 1.
    """
    files_present = {
        p: p.exists()
        for p in (
            EVAL_PATH,
            TO_LABEL_PATH,
            UNUSED_RELIGION_PATH,
            META_PATH,
        )
    }
    n_present = sum(files_present.values())

    if n_present == 0:
        return "empty", None

    if not META_PATH.exists():
        present_names = sorted(p.name for p, ok in files_present.items() if ok)
        return (
            "mismatch",
            f"output files present but no sidecar: {present_names}. "
            "Likely an interrupted prior run. Re-run with --force to redo.",
        )

    try:
        sidecar = json.loads(META_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return "mismatch", f"could not parse sidecar {META_PATH}: {exc}"

    missing = [
        p.name
        for p in (EVAL_PATH, TO_LABEL_PATH, UNUSED_RELIGION_PATH)
        if not p.exists()
    ]
    if missing:
        return (
            "mismatch",
            f"sidecar exists but jsonl files missing: {missing}. "
            "Re-run with --force to redo.",
        )

    expected = {
        EVAL_PATH: sidecar.get("n_eval_unlabeled"),
        TO_LABEL_PATH: sidecar.get("n_pairs_to_label"),
        UNUSED_RELIGION_PATH: sidecar.get("n_unused_religion"),
    }
    for path, exp in expected.items():
        actual = _count_jsonl_lines(path)
        if actual != exp:
            return (
                "mismatch",
                f"{path.name} record count {actual} != sidecar {exp}. "
                "Re-run with --force to redo.",
            )

    sidecar_sha = sidecar.get("input_sha256")
    if sidecar_sha != expected_input_sha:
        return (
            "mismatch",
            f"input SHA mismatch: pairs.jsonl is {expected_input_sha[:12]}…, "
            f"sidecar recorded {str(sidecar_sha)[:12]}…. "
            "Your pairs.jsonl differs from the snapshot in the sidecar; "
            "if intentional, re-run with --force to redo the holdout "
            "against the new input.",
        )

    return "clean", None


def _force_unlink_outputs() -> None:
    for p in _ALL_OUTPUT_PATHS:
        p.unlink(missing_ok=True)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 3a — carve a 300-pair human-eval holdout out of "
            "pairs.jsonl. Deterministic given --seed. Not per-record "
            "resumable; use --force to redo. Exit codes: 0 success, "
            "1 assertion/resume-mismatch, 2 missing input file."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="RNG seed for the per-bucket sample (default 42).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Unlink the four output files and their .tmp siblings, "
            "then redo the holdout from scratch."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute splits and run all assertions; write nothing.",
    )
    parser.add_argument(
        "--preview-eval",
        action="store_true",
        help=(
            "Always implies --dry-run. After computing splits in memory, "
            "print the first 3 records of each candidate set "
            "(eval_in_dist, eval_ood, to_label, unused_religion)."
        ),
    )
    return parser.parse_args(argv)


# -----------------------------------------------------------------------------
# main()
# -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not INPUT_PATH.exists():
        logger.error("%s not found. Run data/02_construct_pairs.py first.", INPUT_PATH)
        return 2

    input_sha = file_sha256(INPUT_PATH)

    is_dry = args.dry_run or args.preview_eval

    if not is_dry:
        if args.force:
            logger.info("--force: unlinking existing outputs.")
            _force_unlink_outputs()
        else:
            state, diag = _check_existing_state(input_sha)
            if state == "clean":
                logger.info(
                    "Already done; sidecar matches current state. "
                    "Use --force to redo."
                )
                return 0
            if state == "mismatch":
                logger.error("Resume-state mismatch: %s", diag)
                return 1

    pairs = list(jsonl_read(INPUT_PATH))
    logger.info("Loaded %d pairs from %s.", len(pairs), INPUT_PATH)

    pair_ids = [r["pair_id"] for r in pairs]
    assert len(set(pair_ids)) == len(pair_ids), "duplicate pair_id in input"

    bias_cats = sorted({r["bias_category"] for r in pairs})
    for r in pairs:
        v = r["bias_category"]
        assert (
            v == v.strip().lower()
        ), f"bias_category not normalized: {v!r} in pair_id {r['pair_id']}"

    splits = sample_holdout(
        pairs,
        in_dist_targets=IN_DIST_TARGETS,
        total_ood=OOD_TOTAL,
        seed=args.seed,
    )

    in_dist_counts = _bucket_counts(splits["eval_in_dist"])
    ood_counts = _bucket_counts(splits["eval_ood"])
    ood_targets = allocate_ood_targets(IN_DIST_TARGETS, OOD_TOTAL)

    assert sum(in_dist_counts.values()) == IN_DIST_TOTAL
    assert sum(ood_counts.values()) == OOD_TOTAL
    assert in_dist_counts == IN_DIST_TARGETS
    assert ood_counts == ood_targets
    eval_total = len(splits["eval_in_dist"]) + len(splits["eval_ood"])
    assert eval_total == IN_DIST_TOTAL + OOD_TOTAL
    assert len(splits["to_label"]) + len(splits["unused_religion"]) + eval_total == len(
        pairs
    )
    assert not any(r["bias_category"] == RELIGION_VALUE for r in splits["to_label"])
    assert all(r["bias_category"] == RELIGION_VALUE for r in splits["eval_ood"])
    assert all(r["bias_category"] != RELIGION_VALUE for r in splits["eval_in_dist"])

    eval_ids = {r["pair_id"] for r in splits["eval_in_dist"]} | {
        r["pair_id"] for r in splits["eval_ood"]
    }
    to_label_ids = {r["pair_id"] for r in splits["to_label"]}
    unused_ids = {r["pair_id"] for r in splits["unused_religion"]}
    assert eval_ids.isdisjoint(to_label_ids)
    assert eval_ids.isdisjoint(unused_ids)
    assert to_label_ids.isdisjoint(unused_ids)

    _print_report(splits)

    if args.preview_eval:
        _print_preview(splits)

    if is_dry:
        logger.info("--dry-run / --preview-eval: not writing outputs.")
        return 0

    eval_records = sorted(
        splits["eval_in_dist"] + splits["eval_ood"], key=lambda r: r["pair_id"]
    )
    to_label_records = sorted(splits["to_label"], key=lambda r: r["pair_id"])
    unused_records = sorted(splits["unused_religion"], key=lambda r: r["pair_id"])

    atomic_write_jsonl(EVAL_PATH, eval_records)
    atomic_write_jsonl(TO_LABEL_PATH, to_label_records)
    atomic_write_jsonl(UNUSED_RELIGION_PATH, unused_records)

    bias_cat_counts: dict[str, int] = {}
    for r in pairs:
        bias_cat_counts[r["bias_category"]] = (
            bias_cat_counts.get(r["bias_category"], 0) + 1
        )

    try:
        input_path_str = str(INPUT_PATH.relative_to(REPO_ROOT))
    except ValueError:
        input_path_str = str(INPUT_PATH)

    meta = {
        "seed": args.seed,
        "input_path": input_path_str,
        "input_sha256": input_sha,
        "input_n_pairs": len(pairs),
        "input_bias_categories": bias_cats,
        "input_bias_category_counts": dict(sorted(bias_cat_counts.items())),
        "n_eval_unlabeled": len(eval_records),
        "n_eval_in_dist": len(splits["eval_in_dist"]),
        "n_eval_ood_religion": len(splits["eval_ood"]),
        "n_pairs_to_label": len(to_label_records),
        "n_unused_religion": len(unused_records),
        "in_dist_targets": IN_DIST_TARGETS,
        "ood_religion_targets": ood_targets,
        "in_dist_achieved": in_dist_counts,
        "ood_religion_achieved": ood_counts,
        "stripped_fields": [
            "response_a.suspected_bias_level",
            "response_b.suspected_bias_level",
        ],
        "kept_fields_note": (
            "response_a.model and response_b.model are kept for the human "
            "labeler — matches production conditions."
        ),
        "force": args.force,
        "created_at_utc": dt.datetime.now(tz=dt.UTC).isoformat(),
    }
    atomic_write_json(META_PATH, meta)

    logger.info("Wrote %s (%d records).", EVAL_PATH, len(eval_records))
    logger.info("Wrote %s (%d records).", TO_LABEL_PATH, len(to_label_records))
    logger.info("Wrote %s (%d records).", UNUSED_RELIGION_PATH, len(unused_records))
    logger.info("Wrote sidecar to %s.", META_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
