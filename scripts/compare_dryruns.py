"""Compare two dryrun runs by per-pair Sonnet records.

Reads two JSONL files of dryrun records (one per run) and emits a
side-by-side comparison: identical, verdict-flipped, confidence-shifted
counts; the specific pair_ids that flipped; and an optional flip-rate
gate decision.

Run after ``data/04_label_pairs.py dryrun`` whenever you have two
dryrun snapshots and want to see how the labels shifted between them.

Usage:
    uv run python scripts/compare_dryruns.py \\
        --v1 data/labeled/dryrun_sonnet_v1.jsonl \\
        --v3 data/labeled/dryrun_sonnet.jsonl \\
        [--max-flip-rate 0.10] \\
        [--out data/labeled/comparison_v1_v3.json]

Exit codes: 0 PASS / 1 schema error / 3 FAIL gate.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from scripts.common import jsonl_read

logger = logging.getLogger(__name__)


def compare_dryrun_runs(
    v1: list[dict[str, Any]], v3: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare per-pair Sonnet records across two dryrun runs.

    Joins on ``pair_id``. For pairs present in both runs, classifies
    each as one of:
      - identical:         same verdict AND same confidence
      - verdict_flipped:   different verdict (regardless of confidence)
      - confidence_shifted: same verdict but different confidence

    Args:
        v1: List of records from the earlier run. Each must have at
            minimum ``pair_id``, ``verdict``, ``confidence``.
        v3: Same shape; the later run.

    Returns:
        ``{
            "n_v1": int, "n_v3": int, "n_common": int,
            "n_only_v1": int, "n_only_v3": int,
            "identical": int,
            "verdict_flipped": int,
            "confidence_shifted": int,
            "flip_pair_ids": list[str],   # pair_ids that flipped verdict
            "flip_details": list[dict],   # per-flip {pair_id, v1_verdict, v3_verdict, ...}
            "shift_details": list[dict],  # per-confidence-shift details
        }``
    """
    by_id_v1 = {r["pair_id"]: r for r in v1}
    by_id_v3 = {r["pair_id"]: r for r in v3}
    common = sorted(set(by_id_v1) & set(by_id_v3))

    identical = 0
    verdict_flipped = 0
    confidence_shifted = 0
    flip_pair_ids: list[str] = []
    flip_details: list[dict[str, Any]] = []
    shift_details: list[dict[str, Any]] = []

    for pid in common:
        a = by_id_v1[pid]
        b = by_id_v3[pid]
        same_verdict = a.get("verdict") == b.get("verdict")
        same_confidence = a.get("confidence") == b.get("confidence")

        if same_verdict and same_confidence:
            identical += 1
        elif not same_verdict:
            verdict_flipped += 1
            flip_pair_ids.append(pid)
            flip_details.append(
                {
                    "pair_id": pid,
                    "pair_category": a.get("pair_category") or b.get("pair_category"),
                    "v1_verdict": a.get("verdict"),
                    "v3_verdict": b.get("verdict"),
                    "v1_confidence": a.get("confidence"),
                    "v3_confidence": b.get("confidence"),
                    "v1_reasoning": a.get("reasoning"),
                    "v3_reasoning": b.get("reasoning"),
                }
            )
        else:
            confidence_shifted += 1
            shift_details.append(
                {
                    "pair_id": pid,
                    "pair_category": a.get("pair_category") or b.get("pair_category"),
                    "verdict": a.get("verdict"),
                    "v1_confidence": a.get("confidence"),
                    "v3_confidence": b.get("confidence"),
                }
            )

    return {
        "n_v1": len(by_id_v1),
        "n_v3": len(by_id_v3),
        "n_common": len(common),
        "n_only_v1": len(set(by_id_v1) - set(by_id_v3)),
        "n_only_v3": len(set(by_id_v3) - set(by_id_v1)),
        "identical": identical,
        "verdict_flipped": verdict_flipped,
        "confidence_shifted": confidence_shifted,
        "flip_pair_ids": flip_pair_ids,
        "flip_details": flip_details,
        "shift_details": shift_details,
    }


def decide_flip_gate(flips: int, n_common: int, max_rate: float) -> tuple[str, str]:
    """Decide whether the verdict-flip rate is within tolerance.

    Returns ``("PASS", msg)`` if ``flips / n_common <= max_rate``,
    else ``("FAIL", msg)``. Returns ``("PASS", msg)`` with a warning if
    ``n_common == 0`` (nothing to compare).
    """
    if n_common == 0:
        return ("PASS", "no overlapping pair_ids — nothing to compare")
    rate = flips / n_common
    pct = rate * 100.0
    threshold_pct = max_rate * 100.0
    if rate <= max_rate:
        return (
            "PASS",
            f"verdict-flip rate {flips}/{n_common} = {pct:.1f}% ≤ "
            f"{threshold_pct:.1f}% threshold",
        )
    return (
        "FAIL",
        f"verdict-flip rate {flips}/{n_common} = {pct:.1f}% > "
        f"{threshold_pct:.1f}% threshold",
    )


def _load_records(path: Path) -> list[dict[str, Any]]:
    records = list(jsonl_read(path))
    for r in records:
        for required in ("pair_id", "verdict", "confidence"):
            if required not in r:
                raise ValueError(
                    f"{path}: record missing required field {required!r}: {r}"
                )
    return records


def _format_summary(
    cmp_result: dict[str, Any],
    gate: tuple[str, str],
    *,
    v1_path: Path,
    v3_path: Path,
) -> str:
    """Render the human-readable summary table."""
    n = cmp_result["n_common"]
    bar = "=" * 72
    lines = [
        bar,
        f"DRYRUN COMPARISON  (v1: {v1_path.name}  →  v3: {v3_path.name})",
        bar,
        f"  v1 records:                    {cmp_result['n_v1']}",
        f"  v3 records:                    {cmp_result['n_v3']}",
        f"  pairs in common:               {n}",
        f"  pairs only in v1:              {cmp_result['n_only_v1']}",
        f"  pairs only in v3:              {cmp_result['n_only_v3']}",
        "",
        f"  identical (verdict + conf):    {cmp_result['identical']}/{n}",
        f"  verdict-flipped:               {cmp_result['verdict_flipped']}/{n}",
        f"  confidence-shifted (same v):   {cmp_result['confidence_shifted']}/{n}",
        "",
        f"  flip-rate gate: {gate[0]} — {gate[1]}",
        bar,
    ]
    if cmp_result["flip_details"]:
        lines.append("VERDICT FLIPS (pair_id  v1→v3  category):")
        for d in cmp_result["flip_details"]:
            lines.append(
                f"  {d['pair_id']}  {d['v1_verdict']}→{d['v3_verdict']}  ({d['pair_category']})"
            )
        lines.append(bar)
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two dryrun runs by per-pair Sonnet records. "
            "Reports identical / verdict-flipped / confidence-shifted "
            "counts and applies a flip-rate gate."
        )
    )
    parser.add_argument(
        "--v1",
        type=Path,
        required=True,
        help="JSONL with the earlier dryrun's per-pair records.",
    )
    parser.add_argument(
        "--v3",
        type=Path,
        required=True,
        help="JSONL with the later dryrun's per-pair records.",
    )
    parser.add_argument(
        "--max-flip-rate",
        type=float,
        default=0.10,
        help="Pass if verdict-flip rate ≤ this fraction of n_common (default 0.10).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the full comparison as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)

    if not args.v1.exists():
        logger.error("v1 records file not found: %s", args.v1)
        return 1
    if not args.v3.exists():
        logger.error("v3 records file not found: %s", args.v3)
        return 1

    try:
        v1 = _load_records(args.v1)
        v3 = _load_records(args.v3)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    cmp_result = compare_dryrun_runs(v1, v3)
    gate = decide_flip_gate(
        cmp_result["verdict_flipped"], cmp_result["n_common"], args.max_flip_rate
    )
    cmp_result["flip_gate"] = {
        "decision": gate[0],
        "message": gate[1],
        "max_flip_rate": args.max_flip_rate,
    }

    print(_format_summary(cmp_result, gate, v1_path=args.v1, v3_path=args.v3))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(cmp_result, indent=2), encoding="utf-8")
        logger.info("Wrote %s", args.out)

    return 0 if gate[0] == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
