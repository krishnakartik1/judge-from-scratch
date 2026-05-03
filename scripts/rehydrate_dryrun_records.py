"""Backfill per-pair dryrun records from a finished Anthropic batch.

The earlier dryrun reports captured aggregate metrics + the
disagreements, but not the full per-pair Sonnet outputs. To run
``compare_dryruns.py`` against a snapshot from before the labeling
code persisted dryrun JSONLs, we refetch the original batch results
from Anthropic by ``batch_id`` (Anthropic retains batch results for
~29 days) and write them as a JSONL.

This is a one-shot utility — once both runs persist their own
JSONLs, comparison can be run directly without rehydration.

Usage:
    uv run python scripts/rehydrate_dryrun_records.py \\
        --batch-id msgbatch_01TW8C7fopwdtubFGCnoUetg \\
        --pairs   data/pairs/pairs_to_label.jsonl \\
        --out     data/labeled/dryrun_sonnet_v1.jsonl

Exit codes: 0 ok / 1 schema error / 2 missing input / 5 API failure.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

from scripts.common import atomic_write_jsonl, jsonl_read, load_env

logger = logging.getLogger(__name__)


def _make_anthropic_client() -> Any:
    import anthropic  # type: ignore

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment.")
    return anthropic.Anthropic(api_key=api_key)


def rehydrate(
    client: Any,
    batch_id: str,
    pairs_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pull and parse a finished batch's per-request outputs.

    Tolerates parse failures (logs and skips) so a partial rehydration
    still produces a useful JSONL. Schema matches the live dryrun
    persistence in ``data/04_label_pairs.py`` so the two outputs are
    directly comparable.
    """
    # Local import keeps the script importable for unit tests without
    # forcing the whole stage4 module to load.
    import importlib.util

    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "stage4_label", repo_root / "data" / "04_label_pairs.py"
    )
    assert spec is not None and spec.loader is not None
    stage = importlib.util.module_from_spec(spec)
    sys.modules["stage4_label"] = stage
    spec.loader.exec_module(stage)

    out: list[dict[str, Any]] = []
    n_seen = 0
    n_succeeded = 0
    n_parse_errors = 0
    model_seen: str | None = None

    for r in stage.fetch_anthropic_results(client, batch_id):
        n_seen += 1
        if r.status != "succeeded":
            logger.warning(
                "rehydrate: pair_id=%s status=%s — skipping", r.custom_id, r.status
            )
            continue
        n_succeeded += 1
        try:
            parsed = stage.parse_model_output(r.text or "")
        except stage.ParseError as exc:
            n_parse_errors += 1
            logger.warning("rehydrate: parse error on %s: %s", r.custom_id, exc)
            continue
        pair = pairs_by_id.get(r.custom_id)
        if pair is None:
            logger.warning(
                "rehydrate: pair_id=%s not in pairs_to_label.jsonl — skipping",
                r.custom_id,
            )
            continue
        # Recover the model from the batch's first usage payload — older
        # ledger rows record this, but the BatchResult from the SDK does
        # not always echo it. Caller may pass --model to override.
        if model_seen is None and r.usage:
            model_seen = r.usage.get("model") or None
        out.append(
            {
                "pair_id": pair["pair_id"],
                "pair_category": pair["pair_category"],
                "verdict": parsed["verdict"],
                "confidence": parsed["confidence"],
                "reasoning": parsed["reasoning"],
                "model": model_seen,
                "batch_id": batch_id,
                "usage": dict(r.usage or {}),
            }
        )

    logger.info(
        "rehydrate: seen=%d succeeded=%d parse_errors=%d kept=%d",
        n_seen,
        n_succeeded,
        n_parse_errors,
        len(out),
    )
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill per-pair dryrun records from a finished Anthropic "
            "batch by batch_id. One-shot utility for comparing against "
            "snapshots taken before the labeling code persisted dryrun "
            "JSONLs."
        )
    )
    parser.add_argument(
        "--batch-id",
        required=True,
        help="Anthropic message batch ID (e.g. msgbatch_…).",
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("data/pairs/pairs_to_label.jsonl"),
        help="JSONL of pair records (default: data/pairs/pairs_to_label.jsonl).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Where to write the rehydrated JSONL.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Optional model ID to record on each row "
            "(if Anthropic's results don't carry it forward)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    load_env()

    if not args.pairs.exists():
        logger.error("pairs file not found: %s", args.pairs)
        return 2

    pairs = list(jsonl_read(args.pairs))
    pairs_by_id = {p["pair_id"]: p for p in pairs}
    logger.info("Loaded %d candidate pairs from %s.", len(pairs), args.pairs)

    try:
        client = _make_anthropic_client()
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 2

    try:
        records = rehydrate(client, args.batch_id, pairs_by_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("API failure during rehydration: %r", exc)
        return 5

    if args.model is not None:
        for r in records:
            r["model"] = args.model

    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(args.out, records)
    logger.info("Wrote %d records to %s.", len(records), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
