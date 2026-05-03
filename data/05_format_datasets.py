"""Stage 5 — Format SFT and DPO datasets for Stage 6/7 training.

Reads ``data/labeled/labeled_pairs.jsonl`` (1,937 records) and
produces two TRL-format JSONL files:

    data/formatted/sft.jsonl  — {prompt, target, pair_id, swap}
    data/formatted/dpo.jsonl  — {prompt, chosen, rejected, pair_id, swap, source}

The DPO ``rejected`` is sourced from one of two paths per pair (per
project decision #22):

    - bucket 0..6 (~70%): synthesized via Sonnet 4.6 Batch API
      ("bad rationalization" of the wrong verdict)
    - bucket 7..9 (~30%): verdict-flip on the original sonnet_reasoning

Both sources go through position-swap doubling.

Phases (subcommands):

    preflight              hard checks before any other phase
    format-sft   [--force] pure transform; no API calls
    synth-dryrun [--submit|--poll|--fetch]   first 20 pairs
    synth-full   --confirm-dryrun [--submit|--poll|--fetch]
    format-dpo   --confirm-synth [--force]
    verify                 token-length/grep/key checks
    status                 print .batches.json + cost ledger summary

Operator workflow (no flags = auto-pick lifecycle action):

    1. uv run python data/05_format_datasets.py preflight
    2. uv run python data/05_format_datasets.py format-sft
    3. uv run python data/05_format_datasets.py synth-dryrun --submit
    4. (wait minutes-hours; SLA 24h) → status
    5. uv run python data/05_format_datasets.py synth-dryrun --fetch
    6. (review data/formatted/synthesis_dryrun.md)
    7. uv run python data/05_format_datasets.py synth-full --confirm-dryrun --submit
    8. (wait) → status → synth-full --fetch
    9. uv run python data/05_format_datasets.py format-dpo --confirm-synth
    10. uv run python data/05_format_datasets.py verify

Per project decision #13 the trained judge emits only ``<reasoning>``
and ``<verdict>`` — no ``<confidence>``, no ``<|think|>``. Stage 5
strips ``<confidence>`` defensively (already absent in labels) and
asserts ``<|think|>`` is not produced by the chat template.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Sibling modules in data/
import synth_rejected as sr  # noqa: E402
from _format_helpers import (  # noqa: E402
    SYNTH_BUCKET_HI,
    TOTAL_BUCKETS,
    apply_chat,
    build_user_message,
    clean_reasoning,
    dpo_split_index,
    flip_verdict,
    format_target,
    is_synth_bucket,
    make_target,
)

from scripts.common import (
    already_processed,
    atomic_write_json,
    atomic_write_jsonl,
    file_sha256,
    jsonl_append,
    jsonl_read,
    load_env,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

LABELED_PATH = REPO_ROOT / "data" / "labeled" / "labeled_pairs.jsonl"
LABELED_META_PATH = REPO_ROOT / "data" / "labeled" / "labeled_pairs.meta.json"
JUDGE_SYSTEM_PROMPT_PATH = REPO_ROOT / "data" / "judge_system_prompt.md"
STAGE4_SCRIPT_PATH = REPO_ROOT / "data" / "04_label_pairs.py"

FORMATTED_DIR = REPO_ROOT / "data" / "formatted"
SFT_PATH = FORMATTED_DIR / "sft.jsonl"
SFT_META_PATH = FORMATTED_DIR / "sft.meta.json"
DPO_PATH = FORMATTED_DIR / "dpo.jsonl"
DPO_META_PATH = FORMATTED_DIR / "dpo.meta.json"

SYNTH_RESULTS_PATH = FORMATTED_DIR / "synthesis_results.jsonl"
SYNTH_DRYRUN_REPORT_MD = FORMATTED_DIR / "synthesis_dryrun.md"
SYNTH_DRYRUN_META = FORMATTED_DIR / "synthesis_dryrun.meta.json"
BATCHES_STATE_PATH = FORMATTED_DIR / ".batches.json"
COST_LEDGER_PATH = FORMATTED_DIR / ".synth_cost_ledger.jsonl"
PARSE_ERRORS_PATH = FORMATTED_DIR / ".synth_parse_errors.jsonl"
BATCH_ERRORS_PATH = FORMATTED_DIR / ".synth_batch_errors.jsonl"
TERMINAL_ERRORS_PATH = FORMATTED_DIR / ".synth_errors.jsonl"


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

TOKENIZER_MODEL_ID = "unsloth/gemma-4-E4B-it"
DROPPED_PAIR_ID = "96b558e0bf7cbd01"  # decision #19
EXPECTED_RECORD_COUNT = 1_937

SFT_MIN_CONFIDENCE = 3
DPO_MIN_CONFIDENCE = 4
DPO_POOL_LO = 950
DPO_POOL_HI = 1150

DRYRUN_SIZE = 20
DRYRUN_MEDIAN_TOLERANCE = 1.25  # synth median / chosen median
SYNTH_BUDGET_USD = 15.0
SYNTH_FETCH_FRACTION_REQUIRED = 0.95

# Verify-step gates
VERIFY_LENGTH_BLOCK_RATIO = 1.15  # chosen median / rejected median > → block
VERIFY_LENGTH_WARN_RATIO = 1.20  # chosen p90  / rejected p90 > → warn
VERIFY_SOURCE_DIVERGE_WARN = 1.25  # synth vs flip rejected median ratio

# Forbidden token in any record (decision #13).
THINKING_PATTERN = re.compile(r"<\|think\|>", re.IGNORECASE)
CONFIDENCE_PATTERN = re.compile(r"<confidence>", re.IGNORECASE)


# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------


class PreflightError(RuntimeError):
    """Preflight assertion failed; refuse to proceed."""


class GateError(RuntimeError):
    """Phase gate failed (e.g., synth-full without confirmed dryrun)."""


# -----------------------------------------------------------------------------
# Lazy clients
# -----------------------------------------------------------------------------


def _make_anthropic_client() -> Any:
    from anthropic import Anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise PreflightError("ANTHROPIC_API_KEY is unset. Add it to .env or export it.")
    return Anthropic(api_key=api_key)


def _load_tokenizer() -> Any:
    from transformers import AutoTokenizer

    logger.info("Loading tokenizer %s", TOKENIZER_MODEL_ID)
    return AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID)


# -----------------------------------------------------------------------------
# Pair pool helpers
# -----------------------------------------------------------------------------


def _read_labeled_pairs() -> list[dict[str, Any]]:
    if not LABELED_PATH.exists():
        raise PreflightError(
            f"Labeled pairs not found at {LABELED_PATH}. Run Stage 4 first."
        )
    return list(jsonl_read(LABELED_PATH))


def _sft_pool(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply confidence floor for SFT (>=3). Defends against null."""
    return [
        r
        for r in records
        if (c := r.get("sonnet_confidence")) is not None and c >= SFT_MIN_CONFIDENCE
    ]


def _dpo_pool(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply DPO filter: confidence >= 4 AND verdict != TIE.

    Note: the ``disagreement is not True`` clause from the original
    spec was dropped per decision #22 (cross-checker disagreement
    signals rubric divergence, not Sonnet errors). We do NOT
    additionally filter on disagreement here.
    """
    return [
        r
        for r in records
        if (c := r.get("sonnet_confidence")) is not None
        and c >= DPO_MIN_CONFIDENCE
        and r.get("sonnet_verdict") != "TIE"
    ]


def _synth_subset(dpo_pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The DPO-pool subset whose rejected source is synthesis (~70%)."""
    return [r for r in dpo_pool if is_synth_bucket(r["pair_id"])]


def _flip_subset(dpo_pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The DPO-pool subset whose rejected source is verdict-flip (~30%)."""
    return [r for r in dpo_pool if not is_synth_bucket(r["pair_id"])]


def _sha1_sorted(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stable sort by sha1(pair_id) — used to pick the dryrun subset."""
    import hashlib

    return sorted(
        records,
        key=lambda r: hashlib.sha1(r["pair_id"].encode("utf-8")).hexdigest(),
    )


def _load_judge_system_prompt() -> str:
    if not JUDGE_SYSTEM_PROMPT_PATH.exists():
        raise PreflightError(
            f"Judge system prompt missing: {JUDGE_SYSTEM_PROMPT_PATH}. "
            f"Stage 5 requires it (decision #146 / Step 0)."
        )
    text = JUDGE_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not text:
        raise PreflightError(
            f"Judge system prompt is empty: {JUDGE_SYSTEM_PROMPT_PATH}."
        )
    if THINKING_PATTERN.search(text):
        raise PreflightError(
            "Judge system prompt contains <|think|> token; that would "
            "enable Gemma 4 native thinking mode (decision #13)."
        )
    return text


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        )
        return out.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# -----------------------------------------------------------------------------
# Preflight
# -----------------------------------------------------------------------------


def run_preflight(_args: argparse.Namespace) -> int:
    load_env()
    print("== Stage 5 preflight ==")

    # 1. Judge system prompt
    judge_prompt = _load_judge_system_prompt()
    print(
        f"  judge_system_prompt.md OK "
        f"({len(judge_prompt)} chars; sha={file_sha256(JUDGE_SYSTEM_PROMPT_PATH)[:12]})"
    )

    # 2. Labeled pairs schema
    records = _read_labeled_pairs()
    if len(records) != EXPECTED_RECORD_COUNT:
        raise PreflightError(
            f"labeled_pairs.jsonl has {len(records)} records; "
            f"expected {EXPECTED_RECORD_COUNT} (decision #19)."
        )
    if any(r.get("pair_id") == DROPPED_PAIR_ID for r in records):
        raise PreflightError(
            f"Dropped pair {DROPPED_PAIR_ID!r} present in labeled_pairs "
            f"(decision #19); pipeline contract violated."
        )
    required_fields = {
        "pair_id",
        "question_text",
        "answer_choices",
        "response_a",
        "response_b",
        "sonnet_verdict",
        "sonnet_reasoning",
        "sonnet_confidence",
    }
    missing = required_fields - set(records[0].keys())
    if missing:
        raise PreflightError(f"Labeled record missing fields: {missing}")
    print(
        f"  labeled_pairs.jsonl OK "
        f"({len(records)} records; sha={file_sha256(LABELED_PATH)[:12]})"
    )

    # 3. Cross-check synth model ID against Stage 4 SONNET constant.
    stage4_text = STAGE4_SCRIPT_PATH.read_text(encoding="utf-8")
    expected = f'SONNET = "{sr.MODEL}"'
    if expected not in stage4_text:
        raise PreflightError(
            f"synth_rejected.MODEL ({sr.MODEL!r}) does not match Stage 4 "
            f"SONNET constant. Update one or the other to keep the "
            f"labeler/synth model in lockstep."
        )
    print(f"  synth model {sr.MODEL!r} matches Stage 4 SONNET constant")

    # 4. HF_TOKEN check (warning only — Unsloth gemma-4 mirrors are public).
    if not os.environ.get("HF_TOKEN"):
        logger.warning(
            "HF_TOKEN not set; tokenizer download relies on the public "
            "mirror. If %s is gated, format-sft will fail.",
            TOKENIZER_MODEL_ID,
        )

    # 5. Tokenizer warm-load (one-time HF cache hit; small).
    try:
        tok = _load_tokenizer()
    except Exception as exc:  # noqa: BLE001
        raise PreflightError(
            f"Tokenizer load failed for {TOKENIZER_MODEL_ID}: {exc!r}"
        ) from exc
    # Sanity: chat-template wrapping doesn't leak <|think|>.
    sample = apply_chat(tok, "system text", "user text")
    if THINKING_PATTERN.search(sample):
        raise PreflightError(
            "Chat template wrapped <|think|> on a no-thinking sample. "
            "Tokenizer config likely enables native thinking mode."
        )
    print(f"  tokenizer {TOKENIZER_MODEL_ID!r} loaded ({len(sample)} chars sample)")

    # 6. Anthropic client constructible (skipped if no key — surfaces in synth).
    try:
        _make_anthropic_client()
        print("  Anthropic client constructible")
    except PreflightError as exc:
        logger.warning("Anthropic client unavailable: %s", exc)

    # 7. DPO pool size sanity
    dpo_pool = _dpo_pool(records)
    if not (DPO_POOL_LO <= len(dpo_pool) <= DPO_POOL_HI):
        raise PreflightError(
            f"DPO pool size {len(dpo_pool)} outside expected "
            f"[{DPO_POOL_LO}, {DPO_POOL_HI}]. Investigate before "
            f"running Stage 5."
        )
    synth_n = len(_synth_subset(dpo_pool))
    flip_n = len(_flip_subset(dpo_pool))
    print(
        f"  DPO pool: {len(dpo_pool)} (synth={synth_n}, flip={flip_n}; "
        f"split via dpo_split_index, buckets <{SYNTH_BUCKET_HI}/{TOTAL_BUCKETS})"
    )

    # 8. Synth cost projection
    projection = sr.project_synth_cost(synth_n)
    print(f"  Synth cost projection: ${projection:.2f} (cap ${SYNTH_BUDGET_USD:.2f})")
    if projection > SYNTH_BUDGET_USD:
        raise PreflightError(
            f"Projected synth cost ${projection:.2f} > cap "
            f"${SYNTH_BUDGET_USD:.2f}. Either raise the cap (in code) "
            f"or reduce SYNTH_BUCKET_HI."
        )

    print("== preflight OK ==")
    return 0


# -----------------------------------------------------------------------------
# format-sft
# -----------------------------------------------------------------------------


def _build_sft_rows(
    records: list[dict[str, Any]],
    *,
    judge_prompt: str,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Produce position-swap-doubled SFT rows.

    Each input record yields TWO rows: one with original A/B order,
    one with A and B swapped (and the verdict + reasoning labels
    flipped accordingly).
    """
    rows: list[dict[str, Any]] = []
    for r in records:
        for swap in (False, True):
            user = build_user_message(r, swap=swap)
            prompt = apply_chat(tokenizer, judge_prompt, user)
            target = make_target(r["sonnet_reasoning"], r["sonnet_verdict"], swap=swap)
            rows.append(
                {
                    "pair_id": r["pair_id"],
                    "swap": swap,
                    "prompt": prompt,
                    "target": target,
                }
            )
    return rows


def run_format_sft(args: argparse.Namespace) -> int:
    load_env()
    if SFT_PATH.exists() and not args.force:
        print(f"sft.jsonl already exists at {SFT_PATH}. " f"Pass --force to rebuild.")
        return 0

    judge_prompt = _load_judge_system_prompt()
    records = _read_labeled_pairs()
    pool = _sft_pool(records)
    dropped = len(records) - len(pool)

    tok = _load_tokenizer()
    rows = _build_sft_rows(pool, judge_prompt=judge_prompt, tokenizer=tok)

    FORMATTED_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(SFT_PATH, rows)
    meta = {
        "stage": "5/format-sft",
        "raw_count": len(records),
        "post_confidence_filter_count": len(pool),
        "dropped_low_confidence": dropped,
        "post_position_swap_count": len(rows),
        "judge_prompt_sha": file_sha256(JUDGE_SYSTEM_PROMPT_PATH),
        "labeled_pairs_sha": file_sha256(LABELED_PATH),
        "tokenizer_model_id": TOKENIZER_MODEL_ID,
        "git_sha": _git_sha(),
    }
    atomic_write_json(SFT_META_PATH, meta)
    print(
        f"format-sft: wrote {len(rows)} rows "
        f"(raw={len(records)}, post-filter={len(pool)}, dropped={dropped}) "
        f"-> {SFT_PATH}"
    )
    return 0


# -----------------------------------------------------------------------------
# Synthesis lifecycle (dryrun + full)
# -----------------------------------------------------------------------------


def _resolve_lifecycle_action(args: argparse.Namespace, phase_key: str) -> str:
    """Pick submit/poll/fetch when no flag was passed.

    If the user explicitly passed --submit, --poll, or --fetch we
    obey. Otherwise auto-select: submit if no batch yet, poll if
    pending, fetch if ended.
    """
    explicit = [a for a in ("submit", "poll", "fetch") if getattr(args, a, False)]
    if len(explicit) > 1:
        raise GateError(f"Pass at most one of --submit/--poll/--fetch (got {explicit})")
    if explicit:
        return explicit[0]
    state = sr.read_batch_state(BATCHES_STATE_PATH)
    entry = state.get(phase_key)
    if entry is None:
        return "submit"
    status = entry.get("status", "in_progress")
    if status == "in_progress":
        return "poll"
    return "fetch"  # ended/canceled/expired


def _select_synth_pairs(
    dpo_pool: list[dict[str, Any]], *, dryrun: bool
) -> list[dict[str, Any]]:
    """Pick the dryrun set (first 20 by sha1) or the full synth bucket."""
    synth_pool = _sha1_sorted(_synth_subset(dpo_pool))
    if dryrun:
        return synth_pool[:DRYRUN_SIZE]
    return synth_pool


def _build_requests(
    pairs: list[dict[str, Any]], *, retry: bool = False
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for r in pairs:
        idx = dpo_split_index(r["pair_id"])
        # Failure-mode round-robin uses the bucket index. Multiple
        # pair_ids share a bucket; that's fine — variety still
        # comes from the per-pair content.
        mode = sr.failure_mode_for(idx)
        requests.append(sr.build_synth_request(r, failure_mode=mode, retry=retry))
    return requests


def _process_fetch_results(
    pairs_by_id: dict[str, dict[str, Any]],
    results_iter: Iterable[Any],
    *,
    record_phase: str,
    record_failure_mode: bool = True,
) -> dict[str, int]:
    """Stream SynthBatchResult rows; persist successes + sidecars.

    Returns counters: ``{succeeded, parse_error, batch_error}``.
    """
    counters: Counter[str] = Counter()
    total_cost = 0.0
    for result in results_iter:
        pair_id = result.custom_id
        pair = pairs_by_id.get(pair_id)
        if pair is None:
            logger.warning("Unknown custom_id in batch results: %r", pair_id)
            continue
        if result.status != "succeeded":
            jsonl_append(
                BATCH_ERRORS_PATH,
                {
                    "pair_id": pair_id,
                    "phase": record_phase,
                    "status": result.status,
                    "error": result.error,
                },
            )
            counters["batch_error"] += 1
            continue
        try:
            parsed = sr.validate_synth_output(
                result.text or "",
                expected_verdict=flip_verdict(pair["sonnet_verdict"]),
            )
        except sr.SynthParseError as exc:
            jsonl_append(
                PARSE_ERRORS_PATH,
                {
                    "pair_id": pair_id,
                    "phase": record_phase,
                    "raw": result.text,
                    "reason": str(exc),
                    "stop_reason": result.stop_reason,
                },
            )
            counters["parse_error"] += 1
            continue
        record = {
            "pair_id": pair_id,
            "phase": record_phase,
            "reasoning": parsed["reasoning"],
            "verdict": parsed["verdict"],
            "sonnet_verdict": pair["sonnet_verdict"],
            "stop_reason": result.stop_reason,
        }
        if record_failure_mode:
            record["failure_mode"] = sr.failure_mode_for(dpo_split_index(pair_id))
        jsonl_append(SYNTH_RESULTS_PATH, record)
        cost = sr.compute_synth_cost(result.usage)
        total_cost += cost
        jsonl_append(
            COST_LEDGER_PATH,
            {
                "pair_id": pair_id,
                "phase": record_phase,
                "usage": result.usage,
                "cost_usd": cost,
            },
        )
        counters["succeeded"] += 1
    if total_cost:
        print(f"  charged ${total_cost:.4f} this fetch pass")
    return dict(counters)


def _read_parse_error_pair_ids() -> set[str]:
    if not PARSE_ERRORS_PATH.exists():
        return set()
    return {r["pair_id"] for r in jsonl_read(PARSE_ERRORS_PATH)}


def _read_terminal_error_pair_ids() -> set[str]:
    if not TERMINAL_ERRORS_PATH.exists():
        return set()
    return {r["pair_id"] for r in jsonl_read(TERMINAL_ERRORS_PATH)}


def _maybe_promote_to_terminal(parse_errors: set[str], succeeded: set[str]) -> int:
    """Move pair_ids that failed retry into the terminal sidecar."""
    promoted = 0
    for pid in sorted(parse_errors):
        if pid in succeeded:
            continue
        # If this pair_id has 2+ parse_error entries, retry exhausted.
        count = 0
        for rec in jsonl_read(PARSE_ERRORS_PATH):
            if rec["pair_id"] == pid:
                count += 1
        if count >= 2:
            jsonl_append(
                TERMINAL_ERRORS_PATH,
                {"pair_id": pid, "reason": "parse_error after retry"},
            )
            promoted += 1
    return promoted


def _confirm_synth_budget_or_raise(n_requests: int) -> None:
    projection = sr.project_synth_cost(n_requests)
    spent = _ledger_total_spend()
    total = spent + projection
    print(
        f"  budget check: spent ${spent:.4f} + projection ${projection:.4f} "
        f"= ${total:.4f} (cap ${SYNTH_BUDGET_USD:.2f})"
    )
    if total > SYNTH_BUDGET_USD:
        raise GateError(
            f"Projected total ${total:.4f} exceeds cap "
            f"${SYNTH_BUDGET_USD:.2f}. Re-run with --yes to override "
            f"after raising the cap in code."
        )


def _ledger_total_spend() -> float:
    if not COST_LEDGER_PATH.exists():
        return 0.0
    return sum(float(r.get("cost_usd") or 0.0) for r in jsonl_read(COST_LEDGER_PATH))


# ----- dryrun ---------------------------------------------------------------


def run_synth_dryrun(args: argparse.Namespace) -> int:
    load_env()
    phase_key = "dryrun-synth"
    action = _resolve_lifecycle_action(args, phase_key)
    print(f"== synth-dryrun ({action}) ==")

    records = _read_labeled_pairs()
    dpo_pool = _dpo_pool(records)
    dryrun_pairs = _select_synth_pairs(dpo_pool, dryrun=True)
    pairs_by_id = {r["pair_id"]: r for r in dryrun_pairs}

    if action == "submit":
        if SYNTH_DRYRUN_META.exists() and not args.force:
            print(
                f"Dryrun already complete: {SYNTH_DRYRUN_META}. "
                f"--force to re-submit."
            )
            return 0
        # Skip pair_ids already present in synthesis_results.jsonl.
        seen = already_processed(SYNTH_RESULTS_PATH, "pair_id")
        todo = [r for r in dryrun_pairs if r["pair_id"] not in seen]
        if not todo:
            print("All dryrun pairs already in synthesis_results.jsonl.")
            return 0
        _confirm_synth_budget_or_raise(len(todo))
        if not args.yes:
            print(
                f"About to submit Anthropic Batch with {len(todo)} requests. "
                f"Re-run with --yes to confirm."
            )
            return 4
        client = _make_anthropic_client()
        requests = _build_requests(todo)
        batch_id = sr.submit_batch(
            client,
            requests,
            state_path=BATCHES_STATE_PATH,
            phase_key=phase_key,
        )
        print(f"Submitted batch {batch_id}; poll with `synth-dryrun --poll`.")
        return 0

    if action == "poll":
        state = sr.read_batch_state(BATCHES_STATE_PATH)
        entry = state.get(phase_key)
        if entry is None or "batch_id" not in entry:
            raise GateError(f"No batch recorded for phase {phase_key!r}. Submit first.")
        client = _make_anthropic_client()
        status = sr.poll_batch(
            client,
            entry["batch_id"],
            state_path=BATCHES_STATE_PATH,
            phase_key=phase_key,
            once=True,
        )
        print(f"  batch {entry['batch_id']} status={status}")
        return 0

    if action == "fetch":
        state = sr.read_batch_state(BATCHES_STATE_PATH)
        entry = state.get(phase_key)
        if entry is None or "batch_id" not in entry:
            raise GateError(f"No batch recorded for phase {phase_key!r}. Submit first.")
        client = _make_anthropic_client()
        results = sr.fetch_batch_results(client, entry["batch_id"])
        counters = _process_fetch_results(
            pairs_by_id,
            results,
            record_phase=phase_key,
        )
        print(f"  fetch counters: {counters}")
        # Length-bias gate.
        ok = _write_dryrun_report(dryrun_pairs)
        if not ok:
            raise GateError(
                "Dryrun length-bias gate failed; re-shape the synthesis "
                "prompt and re-submit (with --force)."
            )
        return 0

    raise GateError(f"Unknown lifecycle action {action!r}")


def _write_dryrun_report(dryrun_pairs: list[dict[str, Any]]) -> bool:
    """Render dryrun report; gate on synth-vs-chosen length parity.

    Returns True if the gate passes (and the meta sidecar gets written).
    """
    synth_records = {
        r["pair_id"]: r
        for r in jsonl_read(SYNTH_RESULTS_PATH)
        if r.get("phase") == "dryrun-synth"
    }
    written = 0
    chosen_lens: list[int] = []
    synth_lens: list[int] = []
    examples: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for pair in dryrun_pairs:
        synth = synth_records.get(pair["pair_id"])
        if synth is None:
            continue
        written += 1
        chosen_text = format_target(
            clean_reasoning(pair["sonnet_reasoning"]),
            pair["sonnet_verdict"],
        )
        rejected_text = format_target(synth["reasoning"], synth["verdict"])
        chosen_lens.append(len(chosen_text))
        synth_lens.append(len(rejected_text))
        if len(examples) < 5:
            examples.append((pair, synth))

    if not synth_lens:
        print("No dryrun synth records yet; nothing to report.")
        return False

    chosen_med = statistics.median(chosen_lens)
    synth_med = statistics.median(synth_lens)
    ratio = synth_med / chosen_med if chosen_med else 0.0

    md = ["# Synthesis Dryrun Report\n"]
    md.append(f"- Records: {written} / {len(dryrun_pairs)} target")
    md.append(f"- Chosen median chars: {chosen_med:.0f}")
    md.append(f"- Synth median chars:  {synth_med:.0f}")
    md.append(f"- Synth/chosen median ratio: {ratio:.2f}x")
    md.append(
        f"- Length gate: {'PASS' if ratio <= DRYRUN_MEDIAN_TOLERANCE else 'FAIL'} "
        f"(threshold {DRYRUN_MEDIAN_TOLERANCE:.2f}x)"
    )
    md.append("\n## First 5 synthesized rejecteds\n")
    for pair, synth in examples:
        md.append(f"### pair_id `{pair['pair_id']}`")
        md.append(
            f"- sonnet_verdict (chosen): {pair['sonnet_verdict']} "
            f"| synth verdict (rejected): {synth['verdict']}"
        )
        md.append(f"- failure_mode: {synth.get('failure_mode')}")
        md.append("```")
        md.append(format_target(synth["reasoning"], synth["verdict"]))
        md.append("```\n")

    SYNTH_DRYRUN_REPORT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"  wrote {SYNTH_DRYRUN_REPORT_MD}")

    if ratio > DRYRUN_MEDIAN_TOLERANCE:
        print(
            f"  LENGTH GATE FAILED: synth median {synth_med:.0f} chars is "
            f"{ratio:.2f}x chosen median {chosen_med:.0f}. "
            f"Refusing to write dryrun.meta.json."
        )
        return False

    atomic_write_json(
        SYNTH_DRYRUN_META,
        {
            "n_records": written,
            "n_target": len(dryrun_pairs),
            "chosen_median_chars": chosen_med,
            "synth_median_chars": synth_med,
            "synth_to_chosen_ratio": ratio,
            "gate_threshold": DRYRUN_MEDIAN_TOLERANCE,
            "synth_results_sha": file_sha256(SYNTH_RESULTS_PATH),
        },
    )
    print(f"  wrote {SYNTH_DRYRUN_META}")
    return True


# ----- full run -------------------------------------------------------------


def _verify_dryrun_complete() -> None:
    if not SYNTH_DRYRUN_META.exists():
        raise GateError(f"{SYNTH_DRYRUN_META} missing. Run synth-dryrun first.")
    meta = json.loads(SYNTH_DRYRUN_META.read_text(encoding="utf-8"))
    if meta.get("n_records") != DRYRUN_SIZE:
        raise GateError(
            f"Dryrun completed only {meta.get('n_records')} records; "
            f"expected {DRYRUN_SIZE}."
        )
    if meta.get("synth_to_chosen_ratio", 0) > DRYRUN_MEDIAN_TOLERANCE:
        raise GateError("Dryrun length gate had failed; cannot promote to full run.")


def run_synth_full(args: argparse.Namespace) -> int:
    load_env()
    if not args.confirm_dryrun:
        raise GateError("--confirm-dryrun is required for synth-full.")
    _verify_dryrun_complete()

    phase_key = "full-synth"
    action = _resolve_lifecycle_action(args, phase_key)
    print(f"== synth-full ({action}) ==")

    records = _read_labeled_pairs()
    dpo_pool = _dpo_pool(records)
    full_pairs = _select_synth_pairs(dpo_pool, dryrun=False)
    pairs_by_id = {r["pair_id"]: r for r in full_pairs}

    if action == "submit":
        seen = already_processed(SYNTH_RESULTS_PATH, "pair_id")
        terminal = _read_terminal_error_pair_ids()
        # Keep dryrun pairs out of the full submission (they're already
        # in synthesis_results.jsonl).
        todo = [
            r
            for r in full_pairs
            if r["pair_id"] not in seen and r["pair_id"] not in terminal
        ]
        if not todo:
            print("All full-synth pairs already in synthesis_results.jsonl.")
            return 0
        _confirm_synth_budget_or_raise(len(todo))
        if not args.yes:
            print(
                f"About to submit Anthropic Batch with {len(todo)} requests. "
                f"Re-run with --yes to confirm."
            )
            return 4
        client = _make_anthropic_client()
        requests = _build_requests(todo)
        batch_id = sr.submit_batch(
            client,
            requests,
            state_path=BATCHES_STATE_PATH,
            phase_key=phase_key,
        )
        print(f"Submitted batch {batch_id}; poll with `synth-full --poll`.")
        return 0

    if action == "poll":
        state = sr.read_batch_state(BATCHES_STATE_PATH)
        entry = state.get(phase_key)
        if entry is None or "batch_id" not in entry:
            raise GateError(f"No batch recorded for phase {phase_key!r}. Submit first.")
        client = _make_anthropic_client()
        status = sr.poll_batch(
            client,
            entry["batch_id"],
            state_path=BATCHES_STATE_PATH,
            phase_key=phase_key,
            once=True,
        )
        print(f"  batch {entry['batch_id']} status={status}")
        return 0

    if action == "fetch":
        state = sr.read_batch_state(BATCHES_STATE_PATH)
        entry = state.get(phase_key)
        if entry is None or "batch_id" not in entry:
            raise GateError(f"No batch recorded for phase {phase_key!r}. Submit first.")
        client = _make_anthropic_client()
        results = sr.fetch_batch_results(client, entry["batch_id"])
        counters = _process_fetch_results(pairs_by_id, results, record_phase=phase_key)
        print(f"  fetch counters: {counters}")

        # Auto-retry pass for parse errors (one shot).
        seen = already_processed(SYNTH_RESULTS_PATH, "pair_id")
        parse_errors = _read_parse_error_pair_ids()
        retry_pids = parse_errors - seen - _read_terminal_error_pair_ids()
        retry_pairs = [
            pairs_by_id[pid] for pid in sorted(retry_pids) if pid in pairs_by_id
        ]
        if retry_pairs:
            print(f"  retry pass: {len(retry_pairs)} parse-error pairs")
            _confirm_synth_budget_or_raise(len(retry_pairs))
            requests = _build_requests(retry_pairs, retry=True)
            retry_batch_id = sr.submit_batch(
                client,
                requests,
                state_path=BATCHES_STATE_PATH,
                phase_key=f"{phase_key}-retry",
            )
            sr.poll_batch(
                client,
                retry_batch_id,
                state_path=BATCHES_STATE_PATH,
                phase_key=f"{phase_key}-retry",
            )
            retry_results = sr.fetch_batch_results(client, retry_batch_id)
            retry_counters = _process_fetch_results(
                pairs_by_id, retry_results, record_phase=f"{phase_key}-retry"
            )
            print(f"  retry counters: {retry_counters}")
            promoted = _maybe_promote_to_terminal(
                _read_parse_error_pair_ids(),
                already_processed(SYNTH_RESULTS_PATH, "pair_id"),
            )
            print(f"  promoted to terminal-error sidecar: {promoted}")
        return 0

    raise GateError(f"Unknown lifecycle action {action!r}")


# -----------------------------------------------------------------------------
# format-dpo
# -----------------------------------------------------------------------------


def _build_dpo_rows(
    dpo_pool: list[dict[str, Any]],
    synth_results: dict[str, dict[str, Any]],
    *,
    judge_prompt: str,
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Assemble DPO rows + per-source counters.

    For each DPO-pool pair, choose source by ``is_synth_bucket``:
    synth if the synth result is available; otherwise flip. The
    chosen target argues for ``sonnet_verdict``; the rejected
    target argues for ``flip(sonnet_verdict)``. Position-swap
    doubles each row.
    """
    rows: list[dict[str, Any]] = []
    n_synth_used = 0
    n_flip_used = 0
    n_synth_skipped = 0
    for pair in dpo_pool:
        pair_id = pair["pair_id"]
        sonnet_verdict = pair["sonnet_verdict"]
        wrong_verdict = flip_verdict(sonnet_verdict)
        synth = synth_results.get(pair_id) if is_synth_bucket(pair_id) else None
        if is_synth_bucket(pair_id) and synth is None:
            n_synth_skipped += 1
            continue
        if synth is not None:
            rejected_reasoning = synth["reasoning"]
            source = "synth"
        else:
            rejected_reasoning = pair["sonnet_reasoning"]
            source = "flip"
        for swap in (False, True):
            user = build_user_message(pair, swap=swap)
            prompt = apply_chat(tokenizer, judge_prompt, user)
            chosen = make_target(pair["sonnet_reasoning"], sonnet_verdict, swap=swap)
            # make_target with swap=True applies both
            # ``swap_response_labels`` and ``flip_verdict``. Passing
            # ``wrong_verdict`` means: under swap, the rejected
            # verdict becomes flip(wrong) = sonnet — which is the
            # opposite of chosen's swapped verdict. Symmetric and
            # correct for both synth and flip sources.
            rejected = make_target(rejected_reasoning, wrong_verdict, swap=swap)
            rows.append(
                {
                    "pair_id": pair_id,
                    "swap": swap,
                    "source": source,
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )
        if source == "synth":
            n_synth_used += 1
        else:
            n_flip_used += 1
    return rows, {
        "synth_used": n_synth_used,
        "flip_used": n_flip_used,
        "synth_skipped": n_synth_skipped,
    }


def run_format_dpo(args: argparse.Namespace) -> int:
    load_env()
    if not args.confirm_synth:
        raise GateError("--confirm-synth is required for format-dpo.")
    if DPO_PATH.exists() and not args.force:
        print(f"dpo.jsonl already exists at {DPO_PATH}. --force to rebuild.")
        return 0

    judge_prompt = _load_judge_system_prompt()
    records = _read_labeled_pairs()
    dpo_pool = _dpo_pool(records)
    synth_results = {
        r["pair_id"]: r
        for r in jsonl_read(SYNTH_RESULTS_PATH)
        if r.get("phase")
        in ("dryrun-synth", "full-synth", "full-synth-retry", "leak-retry")
    }

    synth_target_pairs = _synth_subset(dpo_pool)
    n_synth_required = len(synth_target_pairs)
    n_synth_have = sum(1 for p in synth_target_pairs if p["pair_id"] in synth_results)
    coverage = n_synth_have / n_synth_required if n_synth_required else 1.0
    print(
        f"  synth coverage: {n_synth_have}/{n_synth_required} = "
        f"{coverage * 100:.1f}% (require >= {SYNTH_FETCH_FRACTION_REQUIRED * 100:.0f}%)"
    )
    if coverage < SYNTH_FETCH_FRACTION_REQUIRED:
        raise GateError(
            f"Synth coverage {coverage:.1%} below required "
            f"{SYNTH_FETCH_FRACTION_REQUIRED:.0%}. Investigate parse errors "
            f"or extend retry policy."
        )

    tok = _load_tokenizer()
    rows, counts = _build_dpo_rows(
        dpo_pool,
        synth_results,
        judge_prompt=judge_prompt,
        tokenizer=tok,
    )
    n_synth_used = counts["synth_used"]
    n_flip_used = counts["flip_used"]
    n_synth_skipped = counts["synth_skipped"]

    FORMATTED_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(DPO_PATH, rows)
    meta = {
        "stage": "5/format-dpo",
        "raw_pool": len(dpo_pool),
        "synth_used_pairs": n_synth_used,
        "flip_used_pairs": n_flip_used,
        "synth_skipped_pairs": n_synth_skipped,
        "post_position_swap_count": len(rows),
        "judge_prompt_sha": file_sha256(JUDGE_SYSTEM_PROMPT_PATH),
        "labeled_pairs_sha": file_sha256(LABELED_PATH),
        "synth_results_sha": (
            file_sha256(SYNTH_RESULTS_PATH) if SYNTH_RESULTS_PATH.exists() else None
        ),
        "tokenizer_model_id": TOKENIZER_MODEL_ID,
        "git_sha": _git_sha(),
    }
    atomic_write_json(DPO_META_PATH, meta)
    print(
        f"format-dpo: wrote {len(rows)} rows (synth={n_synth_used} pairs, "
        f"flip={n_flip_used} pairs, skipped={n_synth_skipped}) -> {DPO_PATH}"
    )
    return 0


# -----------------------------------------------------------------------------
# verify
# -----------------------------------------------------------------------------


def _percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sv = sorted(values)
    k = (len(sv) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sv) - 1)
    if f == c:
        return float(sv[int(k)])
    return sv[f] + (sv[c] - sv[f]) * (k - f)


def _token_lens(tok: Any, texts: Iterable[str]) -> list[int]:
    return [len(tok.encode(t, add_special_tokens=False)) for t in texts]


def run_verify(_args: argparse.Namespace) -> int:
    load_env()
    if not SFT_PATH.exists():
        raise GateError(f"{SFT_PATH} missing — run format-sft first.")
    if not DPO_PATH.exists():
        raise GateError(f"{DPO_PATH} missing — run format-dpo first.")
    judge_prompt = _load_judge_system_prompt()
    sft_rows = list(jsonl_read(SFT_PATH))
    dpo_rows = list(jsonl_read(DPO_PATH))
    failures: list[str] = []

    print("== Stage 5 verify ==")

    # 1. Row counts.
    sft_meta = json.loads(SFT_META_PATH.read_text(encoding="utf-8"))
    dpo_meta = json.loads(DPO_META_PATH.read_text(encoding="utf-8"))
    print("Row counts:")
    print(
        f"  SFT: raw={sft_meta['raw_count']} → "
        f"post-confidence-filter={sft_meta['post_confidence_filter_count']} → "
        f"post-position-swap={sft_meta['post_position_swap_count']}"
    )
    print(
        f"  DPO: pool={dpo_meta['raw_pool']} → "
        f"post-position-swap={dpo_meta['post_position_swap_count']} "
        f"(synth={dpo_meta['synth_used_pairs']}, "
        f"flip={dpo_meta['flip_used_pairs']}, "
        f"skipped={dpo_meta['synth_skipped_pairs']})"
    )

    # 2. DPO source breakdown.
    sources = Counter(r.get("source") for r in dpo_rows)
    total = sum(sources.values()) or 1
    print("DPO source breakdown:")
    for k in ("synth", "flip"):
        n = sources.get(k, 0)
        print(f"  {k}: {n} ({n / total * 100:.1f}%)")

    # 3. JSONL key validation.
    sft_required = {"prompt", "target"}
    dpo_required = {"prompt", "chosen", "rejected"}
    bad_sft = [r for r in sft_rows if not sft_required.issubset(r.keys())]
    bad_dpo = [r for r in dpo_rows if not dpo_required.issubset(r.keys())]
    if bad_sft:
        failures.append(
            f"{len(bad_sft)} SFT rows missing required keys "
            f"{sft_required - set(bad_sft[0].keys())}"
        )
    if bad_dpo:
        failures.append(
            f"{len(bad_dpo)} DPO rows missing required keys "
            f"{dpo_required - set(bad_dpo[0].keys())}"
        )

    # 4. <|think|> grep — hard block.
    think_hits = _grep_token(sft_rows, dpo_rows, judge_prompt, THINKING_PATTERN)
    if think_hits:
        failures.append(f"<|think|> appears in {think_hits} record-fields (hard block)")
    else:
        print("Token <|think|>: 0 hits (OK)")

    # 5. <confidence> grep — informational.
    conf_hits = _grep_token(
        sft_rows, dpo_rows, judge_prompt, CONFIDENCE_PATTERN, fields_only=True
    )
    print(f"Token <confidence>: {conf_hits} hits (informational; expected 0)")

    # 6. Length distribution: chosen vs rejected (DPO).
    tok = _load_tokenizer()
    chosen_lens = _token_lens(tok, (r["chosen"] for r in dpo_rows))
    rejected_lens = _token_lens(tok, (r["rejected"] for r in dpo_rows))
    chosen_med = statistics.median(chosen_lens)
    rejected_med = statistics.median(rejected_lens)
    chosen_p90 = _percentile(chosen_lens, 90)
    rejected_p90 = _percentile(rejected_lens, 90)
    print("Length distribution (tokens):")
    print(
        f"  chosen   mean={statistics.mean(chosen_lens):.1f} "
        f"median={chosen_med:.1f} p90={chosen_p90:.1f}"
    )
    print(
        f"  rejected mean={statistics.mean(rejected_lens):.1f} "
        f"median={rejected_med:.1f} p90={rejected_p90:.1f}"
    )
    if rejected_med:
        ratio_med = chosen_med / rejected_med
        print(f"  chosen/rejected median ratio: {ratio_med:.2f}x")
        if ratio_med > VERIFY_LENGTH_BLOCK_RATIO:
            failures.append(
                f"chosen median is {ratio_med:.2f}x rejected median "
                f"(threshold {VERIFY_LENGTH_BLOCK_RATIO:.2f}x) → verbosity-bias block"
            )
    if rejected_p90:
        ratio_p90 = chosen_p90 / rejected_p90
        if ratio_p90 > VERIFY_LENGTH_WARN_RATIO:
            print(
                f"  WARN: chosen p90 {ratio_p90:.2f}x rejected p90 "
                f"(threshold {VERIFY_LENGTH_WARN_RATIO:.2f}x)"
            )

    # 7. Length divergence: synth vs flip.
    synth_lens = _token_lens(
        tok, (r["rejected"] for r in dpo_rows if r.get("source") == "synth")
    )
    flip_lens = _token_lens(
        tok, (r["rejected"] for r in dpo_rows if r.get("source") == "flip")
    )
    if synth_lens and flip_lens:
        s_med = statistics.median(synth_lens)
        f_med = statistics.median(flip_lens)
        ratio = max(s_med, f_med) / min(s_med, f_med)
        print(
            f"  synth median {s_med:.1f}, flip median {f_med:.1f} "
            f"(ratio {ratio:.2f}x)"
        )
        if ratio > VERIFY_SOURCE_DIVERGE_WARN:
            print(
                f"  WARN: synth/flip median ratio {ratio:.2f}x exceeds "
                f"{VERIFY_SOURCE_DIVERGE_WARN:.2f}x — synthesis prompt may "
                f"be producing oddly-shaped output"
            )

    # 8. Three example rows from each.
    print("\nExample SFT rows:")
    for r in sft_rows[:3]:
        _print_example(r)
    print("\nExample DPO rows:")
    for r in dpo_rows[:3]:
        _print_example(r)

    if failures:
        print("\n== verify FAILED ==")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\n== verify OK ==")
    return 0


def _grep_token(
    sft_rows: list[dict[str, Any]],
    dpo_rows: list[dict[str, Any]],
    judge_prompt: str,
    pattern: re.Pattern[str],
    *,
    fields_only: bool = False,
) -> int:
    """Count records whose fields match ``pattern``.

    ``fields_only=True`` skips the system/prompt fields and only
    checks the assistant-target fields (target/chosen/rejected) —
    used for the <confidence> check where the prompt may legitimately
    discuss the tag.
    """
    hits = 0
    if not fields_only and pattern.search(judge_prompt):
        hits += 1
    for row in sft_rows:
        for field in ("target",) if fields_only else ("prompt", "target"):
            if pattern.search(row.get(field, "")):
                hits += 1
    for row in dpo_rows:
        for field in (
            ("chosen", "rejected") if fields_only else ("prompt", "chosen", "rejected")
        ):
            if pattern.search(row.get(field, "")):
                hits += 1
    return hits


def _print_example(row: dict[str, Any]) -> None:
    pid = row.get("pair_id", "?")
    swap = row.get("swap")
    src = row.get("source", "")
    label = f"pair_id={pid} swap={swap}"
    if src:
        label += f" source={src}"
    print(f"  --- {label} ---")
    prompt = row.get("prompt", "")
    print(f"  prompt[:300]: {prompt[:300]!r}")
    if "target" in row:
        print(f"  target: {row['target']}")
    if "chosen" in row:
        print(f"  chosen: {row['chosen']}")
        print(f"  rejected: {row['rejected']}")


# -----------------------------------------------------------------------------
# status
# -----------------------------------------------------------------------------


def run_status(_args: argparse.Namespace) -> int:
    state = sr.read_batch_state(BATCHES_STATE_PATH)
    print("== Stage 5 status ==")
    print("Batches:")
    if not state:
        print("  (none)")
    for phase, entry in sorted(state.items()):
        print(
            f"  {phase}: batch_id={entry.get('batch_id')} "
            f"status={entry.get('status')} n={entry.get('n_requests')} "
            f"submitted_at={entry.get('submitted_at')}"
        )
    spent = _ledger_total_spend()
    print(f"Cost ledger total: ${spent:.4f}")
    return 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="05_format_datasets")
    sub = p.add_subparsers(dest="phase", required=True)

    sub.add_parser("preflight")

    sft = sub.add_parser("format-sft")
    sft.add_argument("--force", action="store_true")

    for name in ("synth-dryrun", "synth-full"):
        sp = sub.add_parser(name)
        sp.add_argument("--submit", action="store_true")
        sp.add_argument("--poll", action="store_true")
        sp.add_argument("--fetch", action="store_true")
        sp.add_argument("--yes", action="store_true")
        sp.add_argument("--force", action="store_true")
        if name == "synth-full":
            sp.add_argument("--confirm-dryrun", action="store_true")

    dpo = sub.add_parser("format-dpo")
    dpo.add_argument("--confirm-synth", action="store_true")
    dpo.add_argument("--force", action="store_true")

    sub.add_parser("verify")
    sub.add_parser("status")
    return p


_HANDLERS = {
    "preflight": run_preflight,
    "format-sft": run_format_sft,
    "synth-dryrun": run_synth_dryrun,
    "synth-full": run_synth_full,
    "format-dpo": run_format_dpo,
    "verify": run_verify,
    "status": run_status,
}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = _build_parser().parse_args(argv)
    handler = _HANDLERS[args.phase]
    try:
        return handler(args)
    except (PreflightError, GateError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
