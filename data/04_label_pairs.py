"""Stage 4 — Claude labeling driver (Sonnet primary + GPT/DeepSeek cross-check).

Three gated phases:

    dryrun       50-pair Sonnet-vs-Opus comparison via Batch API.
                 Writes data/labeled/dryrun_report.{json,md}. Exits with
                 a PROCEED / REVIEW / ABORT decision per the gate
                 thresholds in §Constants. No primary work happens
                 inside this invocation.

    primary      Sonnet labels all 1,938 pairs in
                 data/pairs/pairs_to_label.jsonl via Batch API + prompt
                 caching. Writes data/labeled/labeled_pairs.jsonl with
                 explicit-null cross-check fields (so the merge in the
                 next phase only updates, never inserts). Resumable;
                 in-flight batches re-attach via .batches.json.

    crosscheck   500-pair triangulation: stratified across the four hard
                 buckets, run through OpenAI GPT (sync, async fan-out)
                 and DeepSeek V3.1 (Together Batch API). Per-pair patches
                 land in crosscheck_results.jsonl, then merged into
                 labeled_pairs.jsonl via merge_jsonl_patches in one
                 atomic pass.

    retry-parse-errors {primary|crosscheck}
                 Re-attempt pairs whose model output failed schema
                 parsing on the first attempt. Surviving failures need
                 manual investigation.

    status       Print .batches.json + cost ledger summary.

Gates between phases are explicit human stops:
  - primary requires --confirm-dryrun AND a PROCEED decision in
    dryrun_report.json.
  - crosscheck requires --confirm-primary AND meta.primary.status == "complete".
  - Pre-submit cost confirmation prompts the operator before every
    batch unless --yes is passed.
  - A $20 hard budget guardrail aborts (or interactively prompts) if
    cumulative spend + the next-batch projection would exceed it.

Per project decision #13 the labeling prompt at
``data/labeling_prompt.md`` uses XML tags for output
(``<reasoning>``, ``<verdict>``, ``<confidence>``) — NOT JSON.

Usage:
    uv run python data/04_label_pairs.py dryrun        [--seed 42] [--limit N] [--dry-run] [--yes]
    uv run python data/04_label_pairs.py primary       --confirm-dryrun [...]
    uv run python data/04_label_pairs.py crosscheck    --confirm-primary [...]
    uv run python data/04_label_pairs.py retry-parse-errors {primary|crosscheck} [--yes]
    uv run python data/04_label_pairs.py status

Exit codes: 0 ok / PROCEED, 1 schema/assertion violation, 2 missing
input or prompt, 3 REVIEW gate, 4 ABORT gate or BudgetExceededError,
5 API failure after retries, 130 SIGINT.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import io
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.common import (
    already_processed,
    atomic_write_json,
    atomic_write_jsonl,
    file_sha256,
    jsonl_append,
    jsonl_read,
    merge_jsonl_patches,
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
PAIRS_DIR = REPO_ROOT / "data" / "pairs"
LABELED_DIR = REPO_ROOT / "data" / "labeled"

INPUT_PATH = PAIRS_DIR / "pairs_to_label.jsonl"
PROMPT_PATH = REPO_ROOT / "data" / "labeling_prompt.md"

LABELED_PATH = LABELED_DIR / "labeled_pairs.jsonl"
META_PATH = LABELED_DIR / "labeled_pairs.meta.json"
DRYRUN_REPORT_JSON = LABELED_DIR / "dryrun_report.json"
DRYRUN_REPORT_MD = LABELED_DIR / "dryrun_report.md"
CROSSCHECK_RESULTS_PATH = LABELED_DIR / "crosscheck_results.jsonl"

BATCHES_STATE_PATH = LABELED_DIR / ".batches.json"
COST_LEDGER_PATH = LABELED_DIR / ".cost_ledger.jsonl"
PARSE_ERRORS_PATH = LABELED_DIR / ".parse_errors.jsonl"
BATCH_ERRORS_PATH = LABELED_DIR / ".batch_errors.jsonl"
REFUSALS_PATH = LABELED_DIR / ".refusals.jsonl"

# -----------------------------------------------------------------------------
# Constants — models, pricing, gates, sampling
# -----------------------------------------------------------------------------

PRICING_AS_OF = "2026-05-02"

# Per-million-token USD pricing. Anthropic + Together Batch API discount
# is applied separately via BATCH_DISCOUNT. cached_input is the read-hit
# rate (Anthropic charges 25% of input for cache hits; we use 0.10 as a
# conservative proxy that the ledger surfaces if pricing drifts).
PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.0, "cached_input": 0.30, "output": 15.0},
    "claude-opus-4-6": {"input": 15.0, "cached_input": 1.50, "output": 75.0},
    # GPT-5.4 placeholder pricing — refresh when the cross-check phase
    # resolves the actual API ID. Recorded into meta as PRICING_AS_OF.
    "gpt-5.4": {"input": 5.0, "cached_input": 0.50, "output": 20.0},
    # DeepSeek V3.1 via Together (no caching tier on Together batch).
    # Qwen3-235B-Instruct-tput pricing on Together (approx; retained
    # for the legacy Together code path which is no longer active).
    "Qwen/Qwen3-235B-A22B-Instruct-2507-tput": {
        "input": 0.20,
        "cached_input": 0.20,
        "output": 0.60,
    },
    # OpenRouter Qwen3-235B-Instruct (active third-labeler path).
    # Pricing from OpenRouter's catalog query, May 2026.
    "qwen/qwen3-235b-a22b-2507": {
        "input": 0.071,
        "cached_input": 0.071,
        "output": 0.10,
    },
}

BATCH_DISCOUNT = 0.5
BUDGET_USD_HARD = 20.0
# Budget breakdown (rough): dryrun ~$1-2 (Sonnet+Opus on 50);
# primary ~$8 (Sonnet on 1,938 with caching); crosscheck ~$3 (GPT) +
# ~$1 (DeepSeek). Headroom ~$5 for retries and pricing drift.

SONNET = "claude-sonnet-4-6"
OPUS = "claude-opus-4-6"
DEEPSEEK = "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
# DEEPSEEK constant retained only for the legacy Together-batch code
# path (now dead). The active third-labeler route is OpenRouter — see
# OPENROUTER_MODEL below. The labeled-pairs schema fields are now
# provider-neutral slot IDs (crosscheck_verdict_b / _c) so a future
# swap of the actual model doesn't require a schema migration. The
# resolved model identity for each slot is recorded in
# meta.crosscheck.labelers and in the per-call cost ledger.

OPENROUTER_MODEL = "qwen/qwen3-235b-a22b-2507"
# Active third-labeler model on OpenRouter (async fan-out, no batch
# API). Together's batch-enabled Qwen variant (-tput) was indefinitely
# stuck at 0% progress; OpenRouter routes the same family directly.
# If OpenRouter fails too, the orchestrator degrades gracefully to
# GPT-only triangulation (logs a warning, leaves crosscheck_*_c=null).
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GPT_DEFAULT = "gpt-5.4"

ANTHROPIC_MAX_TOKENS = 1024  # caps per-pair output budget; reasoning is short.
OPENAI_MAX_TOKENS = 1024
DEEPSEEK_MAX_TOKENS = 1024  # non-reasoning model — short XML answer fits

DRYRUN_PER_BUCKET = 10
DRYRUN_BUCKETS: tuple[str, ...] = (
    "subtle_bias_vs_clean",
    "tracked_bias_vs_alternate",
    "adversarial",
    "clear_bias_vs_clean",
    "both_clean_tie",
)
HARD_BUCKETS: tuple[str, ...] = (
    "subtle_bias_vs_clean",
    "tracked_bias_vs_alternate",
    "adversarial",
)

# Gate thresholds (counts on a 50-pair / 30-hard-pair sample).
DRYRUN_GATE_OVERALL_MIN = 46  # 46/50 = 92 %  — Sonnet vs Opus overall agreement floor
DRYRUN_GATE_HARD_MIN = 24  # 24/30 = 80 %    — Sonnet vs Opus hard-bucket floor
DRYRUN_GATE_HARD_ABORT = 22  # <22/30 ≈ 73 % → ABORT (failure floor)

CROSSCHECK_TARGETS: dict[str, int] = {
    "subtle_bias_vs_clean": 175,
    "both_clean_tie": 182,
    "tracked_bias_vs_alternate": 62,
    "adversarial": 81,
}
assert sum(CROSSCHECK_TARGETS.values()) == 500

REQUIRED_PROMPT_TOKEN = "answer_choices"
REQUIRED_PLACEHOLDERS: tuple[str, ...] = (
    "question_text",
    "answer_choices_block",
    "response_a_text",
    "response_b_text",
)

MIN_CACHE_TOKENS_SONNET = 1024

POLL_SLEEP_SEC = 30
POLL_MAX_CONSECUTIVE_5XX = 5
SUBMIT_RETRY_ATTEMPTS = 3
OPENAI_CONCURRENCY = 20

VALID_VERDICTS = ("A", "B", "TIE")
VALID_CONFIDENCES = (1, 2, 3, 4, 5)

# Refusal heuristics — case-insensitive substring matches that, when
# present near the start of a response, suggest the model declined to
# label rather than producing a verdict. Cross-checked against
# stop_reason where available.
REFUSAL_PATTERNS: tuple[str, ...] = (
    "i can't help",
    "i cannot help",
    "i won't",
    "i will not",
    "i'm not able to",
    "i am not able to",
    "i'm unable to",
    "i am unable to",
    "i must decline",
)


# -----------------------------------------------------------------------------
# Custom exceptions
# -----------------------------------------------------------------------------


class PromptError(RuntimeError):
    """Raised when the labeling prompt is missing or malformed."""


class ParseError(ValueError):
    """Raised when a model response cannot be parsed into the schema."""


class BudgetExceededError(RuntimeError):
    """Raised when cumulative spend + next-batch projection > budget cap."""


class GateError(RuntimeError):
    """Raised when a phase entry gate fails (e.g. dryrun not PROCEED)."""


class ModelNotFoundError(RuntimeError):
    """Raised when a runtime model verification call fails."""


# -----------------------------------------------------------------------------
# Pure functions (unit-tested via dependency injection)
# -----------------------------------------------------------------------------


class _SafeMissing(dict):
    """``dict`` subclass that raises on missing keys for ``str.format_map``."""

    def __missing__(self, key: str) -> str:
        raise KeyError(f"prompt template references unknown placeholder: {{{key}}}")


def verify_prompt_template(text: str) -> None:
    """Assert the labeling prompt satisfies the contract.

    Raises ``PromptError`` on any of:
      - missing literal substring ``REQUIRED_PROMPT_TOKEN``
      - any placeholder in ``REQUIRED_PLACEHOLDERS`` not appearing as
        ``{name}`` in the template

    Args:
        text: Full text of ``data/labeling_prompt.md``.
    """
    if REQUIRED_PROMPT_TOKEN not in text:
        raise PromptError(
            f"labeling_prompt.md is missing the required substring "
            f"{REQUIRED_PROMPT_TOKEN!r}; the prompt MUST render the "
            f"answer_choices field into the question framing."
        )
    missing = []
    for placeholder in REQUIRED_PLACEHOLDERS:
        # Match `{placeholder}` or `{placeholder:fmt}` but not `{{placeholder}}`.
        pattern = r"(?<!\{)\{" + re.escape(placeholder) + r"(?::[^{}]*)?\}(?!\})"
        if not re.search(pattern, text):
            missing.append(placeholder)
    if missing:
        raise PromptError(
            f"labeling_prompt.md is missing required placeholder(s): "
            f"{missing}. Required: {list(REQUIRED_PLACEHOLDERS)}."
        )


def split_prompt_static_prefix(text: str) -> tuple[str, str]:
    """Split the prompt into (static_prefix, per_pair_template).

    The contract requires a stable prefix that does not change between
    pairs, followed by the per-pair item block (last). The split is at
    the first heading whose body contains any required placeholder.

    Returns ``(prefix, suffix)`` where the suffix contains all required
    placeholders and the prefix contains none of them. Raises
    ``PromptError`` if no clean split exists.
    """
    lines = text.splitlines(keepends=True)
    placeholder_line: int | None = None
    for idx, line in enumerate(lines):
        if any(("{" + p) in line for p in REQUIRED_PLACEHOLDERS):
            placeholder_line = idx
            break
    if placeholder_line is None:
        raise PromptError("could not locate per-pair placeholders in template")
    # Walk back to the nearest preceding heading line (starts with `#`).
    split_at = placeholder_line
    for idx in range(placeholder_line - 1, -1, -1):
        if lines[idx].lstrip().startswith("#"):
            split_at = idx
            break
    prefix = "".join(lines[:split_at])
    suffix = "".join(lines[split_at:])
    if not prefix.strip():
        raise PromptError("prompt template has empty static prefix")
    for p in REQUIRED_PLACEHOLDERS:
        if ("{" + p) in prefix:
            raise PromptError(
                f"placeholder {{{p}}} appears in the static prefix; "
                f"per-pair content must come LAST."
            )
    return prefix, suffix


def render_prompt(template: str, pair: dict[str, Any]) -> str:
    """Substitute pair fields into the labeling-prompt template.

    Substitutions:
      - question_text         ← pair["question_text"]
      - answer_choices_block  ← "A. …\\nB. …\\nC. …" rendered from
                                 pair["answer_choices"]
      - response_a_text       ← pair["response_a"]["text"]
      - response_b_text       ← pair["response_b"]["text"]

    Uses ``str.format_map`` with a missing-key guard so any unknown
    placeholder raises ``KeyError`` rather than silently emitting an
    empty string.

    Args:
        template: Full text of ``data/labeling_prompt.md``.
        pair: One pair record from ``pairs_to_label.jsonl``.

    Returns:
        Fully rendered prompt ready for the model.
    """
    block = "\n".join(f'{c["letter"]}. {c["text"]}' for c in pair["answer_choices"])
    fields = _SafeMissing(
        question_text=pair["question_text"],
        answer_choices_block=block,
        response_a_text=pair["response_a"]["text"],
        response_b_text=pair["response_b"]["text"],
    )
    return template.format_map(fields)


def stratified_sample(
    pairs: list[dict[str, Any]],
    targets: dict[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    """Sample N pairs per ``pair_category`` deterministically.

    Sorts the input by ``pair_id`` first to make sampling reproducible
    independent of input ordering. Asserts per-bucket supply ≥ target
    BEFORE sampling so a shortfall raises clearly.

    Args:
        pairs: All input pair records.
        targets: ``{pair_category: target_count}``.
        seed: RNG seed.

    Returns:
        Flat list of picked pairs, ordered by bucket name then pair_id.
    """
    sorted_pairs = sorted(pairs, key=lambda r: r["pair_id"])
    buckets: dict[str, list[dict[str, Any]]] = {b: [] for b in targets}
    for p in sorted_pairs:
        cat = p.get("pair_category")
        if cat in buckets:
            buckets[cat].append(p)

    for bucket, target in targets.items():
        supply = len(buckets[bucket])
        if supply < target:
            raise AssertionError(
                f"stratified_sample: bucket {bucket!r} has supply {supply} "
                f"< target {target}"
            )

    # Seed FRESH per bucket so each bucket's sample is independent of
    # the others' iteration order. Without this, bumping one bucket's
    # target advances the shared RNG state and shifts other buckets'
    # picks. Using a string seed (deterministic across processes via
    # Python's SHA-512 hashing in random.seed v2) instead of a tuple,
    # which would depend on PYTHONHASHSEED.
    picked: list[dict[str, Any]] = []
    for bucket in sorted(targets):
        rng = random.Random(f"{seed}|{bucket}")
        chosen = rng.sample(buckets[bucket], targets[bucket])
        chosen.sort(key=lambda r: r["pair_id"])
        picked.extend(chosen)
    return picked


_REASONING_RE = re.compile(
    r"<\s*reasoning\s*>(.*?)<\s*/\s*reasoning\s*>", re.DOTALL | re.IGNORECASE
)
_VERDICT_RE = re.compile(
    r"<\s*verdict\s*>\s*(.*?)\s*<\s*/\s*verdict\s*>", re.DOTALL | re.IGNORECASE
)
_CONFIDENCE_RE = re.compile(
    r"<\s*confidence\s*>\s*(.*?)\s*<\s*/\s*confidence\s*>", re.DOTALL | re.IGNORECASE
)


def parse_model_output(text: str) -> dict[str, Any]:
    """Extract ``{reasoning, verdict, confidence}`` from XML-tagged output.

    The labeling prompt instructs models to produce
    ``<reasoning>…</reasoning><verdict>A|B|TIE</verdict><confidence>1..5</confidence>``
    and nothing else. This parser tolerates whitespace and surrounding
    text (logged as a soft warning), but raises ``ParseError`` on any
    of: missing tag, invalid verdict value, non-integer confidence,
    confidence out of range, empty reasoning.

    Args:
        text: The raw text returned by the model.

    Returns:
        ``{"reasoning": str, "verdict": "A"|"B"|"TIE", "confidence": int}``.
    """
    if not text or not text.strip():
        raise ParseError("empty model output")

    reasoning_match = _REASONING_RE.search(text)
    verdict_match = _VERDICT_RE.search(text)
    confidence_match = _CONFIDENCE_RE.search(text)

    if reasoning_match is None:
        raise ParseError("missing <reasoning>…</reasoning> tag")
    if verdict_match is None:
        raise ParseError("missing <verdict>…</verdict> tag")
    if confidence_match is None:
        raise ParseError("missing <confidence>…</confidence> tag")

    reasoning = reasoning_match.group(1).strip()
    verdict = verdict_match.group(1).strip().upper()
    confidence_raw = confidence_match.group(1).strip()

    if not reasoning:
        raise ParseError("empty <reasoning> body")
    if verdict not in VALID_VERDICTS:
        raise ParseError(
            f"invalid verdict {verdict!r}; expected one of {VALID_VERDICTS}"
        )
    try:
        confidence = int(confidence_raw)
    except (TypeError, ValueError) as exc:
        raise ParseError(f"non-integer confidence {confidence_raw!r}") from exc
    if confidence not in VALID_CONFIDENCES:
        raise ParseError(
            f"confidence {confidence} out of range; expected one of "
            f"{VALID_CONFIDENCES}"
        )

    # Soft-warn if the model produced text outside the three tags. The
    # prompt says "nothing else" but models drift; we tolerate it.
    last_close = max(reasoning_match.end(), verdict_match.end(), confidence_match.end())
    first_open = min(
        reasoning_match.start(), verdict_match.start(), confidence_match.start()
    )
    leading = text[:first_open].strip()
    trailing = text[last_close:].strip()
    if leading:
        logger.debug("parse_model_output: tolerating leading text: %r", leading[:100])
    if trailing:
        logger.debug("parse_model_output: tolerating trailing text: %r", trailing[:100])

    return {
        "reasoning": reasoning,
        "verdict": verdict,
        "confidence": confidence,
    }


def detect_refusal(text: str | None, stop_reason: str | None = None) -> bool:
    """Heuristic refusal detector.

    Returns True if the response looks like a refusal — either because
    the API stop_reason flagged it (``"refusal"``) or because the
    leading text matches a known refusal pattern. Used to route
    refusals to ``.refusals.jsonl`` instead of ``.parse_errors.jsonl``,
    since the right remediation differs (refusal needs a prompt or
    safety-classifier intervention, not a retry).
    """
    if stop_reason == "refusal":
        return True
    if not text:
        return False
    head = text.strip().lower()[:300]
    return any(p in head for p in REFUSAL_PATTERNS)


def compute_cost(
    usage: dict[str, int] | dict[str, Any],
    model: str,
    batch: bool,
) -> float:
    """Convert token usage into USD using ``PRICING`` and ``BATCH_DISCOUNT``.

    ``usage`` is accepted in either Anthropic shape
    (``input_tokens``, ``output_tokens``, ``cache_read_input_tokens``,
    ``cache_creation_input_tokens``) or OpenAI/Together shape
    (``prompt_tokens``, ``completion_tokens``, ``cached_tokens``).

    Args:
        usage: Token counts returned by the provider.
        model: Model ID; used to look up per-million-token pricing.
        batch: Apply ``BATCH_DISCOUNT`` if True.

    Returns:
        Cost in USD as a float. Returns 0.0 when pricing is unknown
        for ``model`` (and logs a WARNING).
    """
    if model not in PRICING:
        logger.warning(
            "compute_cost: no pricing entry for model %r; returning 0.", model
        )
        return 0.0
    rates = PRICING[model]

    input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
    output_tokens = int(
        usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    )
    # OpenAI 1.x+ nests cached_tokens under prompt_tokens_details; we
    # check the nested location first, then fall back to top-level forms
    # used by Anthropic (``cache_read_input_tokens``) and older OpenAI
    # / Together-style payloads (``cached_tokens``).
    nested = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    cached_tokens = int(
        nested
        or usage.get("cache_read_input_tokens", usage.get("cached_tokens", 0))
        or 0
    )
    # Cache-creation tokens are billed at full input rate on Anthropic;
    # we lump them into input_tokens.
    cache_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)
    fresh_input = max(input_tokens - cached_tokens, 0) + cache_creation

    cost = (
        (fresh_input / 1_000_000.0) * rates["input"]
        + (cached_tokens / 1_000_000.0) * rates["cached_input"]
        + (output_tokens / 1_000_000.0) * rates["output"]
    )
    if batch:
        cost *= BATCH_DISCOUNT
    return cost


def disagreement(verdicts: list[str | None]) -> bool | None:
    """Return ``True`` iff at least two verdicts differ; ``None`` if any are missing.

    Used to populate the per-pair ``disagreement`` field in
    ``labeled_pairs.jsonl``. TIE is treated as a distinct label —
    ``["A", "TIE", "A"]`` counts as a disagreement.
    """
    if any(v is None for v in verdicts):
        return None
    return len(set(verdicts)) > 1


def build_anthropic_request(
    pair: dict[str, Any],
    static_system: str,
    template: str,
    model: str,
    custom_id: str | None = None,
) -> dict[str, Any]:
    """Construct one Anthropic Batch API request entry with cache_control.

    Uses Anthropic's top-level ``cache_control`` parameter — the SDK
    handles breakpoint placement automatically. The static instruction
    prefix from ``data/labeling_prompt.md`` is folded into the system
    message so the cacheable prefix sits at ~1,575 tokens (above the
    1,024-token Sonnet minimum). Only the per-pair item block lands in
    the user message, so each request's fresh content is small.

    Args:
        pair: Pair record; provides per-pair fields and the default
            ``custom_id``.
        static_system: Stable system-role preamble (e.g. role
            instructions). Concatenated to the prompt's static prefix
            with a blank line in between.
        template: The full UNRENDERED labeling-prompt template
            (``data/labeling_prompt.md``). Internally split into static
            prefix + per-pair suffix; only the suffix is rendered with
            the pair's fields and sent in the user message.
        model: Anthropic model ID.
        custom_id: Optional override (defaults to ``pair["pair_id"]``).

    Returns:
        Dict in the shape Anthropic's ``messages.batches.create``
        accepts as one element of its ``requests`` list.
    """
    static_prefix, per_pair_template = split_prompt_static_prefix(template)
    block = "\n".join(f'{c["letter"]}. {c["text"]}' for c in pair["answer_choices"])
    fields = _SafeMissing(
        question_text=pair["question_text"],
        answer_choices_block=block,
        response_a_text=pair["response_a"]["text"],
        response_b_text=pair["response_b"]["text"],
    )
    per_pair_block = per_pair_template.format_map(fields)

    return {
        "custom_id": custom_id or str(pair["pair_id"]),
        "params": {
            "model": model,
            "max_tokens": ANTHROPIC_MAX_TOKENS,
            "system": [
                {
                    "type": "text",
                    "text": f"{static_system}\n\n{static_prefix}",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": per_pair_block}],
        },
    }


def summarize_results(
    records: list[dict[str, Any]], verdict_field: str, confidence_field: str
) -> dict[str, Any]:
    """Compute verdict + confidence distributions for a list of labeled records.

    Args:
        records: Labeled records.
        verdict_field: e.g. ``"sonnet_verdict"``.
        confidence_field: e.g. ``"sonnet_confidence"``.

    Returns:
        ``{"count": N, "verdicts": {A: int, B: int, TIE: int},
        "confidence": {1..5: int}}``.
    """
    verdicts: Counter[str] = Counter()
    confidences: Counter[int] = Counter()
    for r in records:
        v = r.get(verdict_field)
        c = r.get(confidence_field)
        if v is not None:
            verdicts[v] += 1
        if c is not None:
            confidences[int(c)] += 1
    return {
        "count": len(records),
        "verdicts": {k: verdicts.get(k, 0) for k in VALID_VERDICTS},
        "confidence": {k: confidences.get(k, 0) for k in VALID_CONFIDENCES},
    }


def comparison_metrics(
    sonnet_records: list[dict[str, Any]],
    opus_records: list[dict[str, Any]],
    hard_buckets: tuple[str, ...] = HARD_BUCKETS,
) -> dict[str, Any]:
    """Compute Sonnet-vs-Opus agreement on a paired sample.

    Both lists must cover the same pair_ids.

    Returns a dict with:
      - ``overall``: ``{matches, total, rate}`` over all pairs.
      - ``per_category``: ``{pair_category: {matches, total, rate}}``.
      - ``hard``: ``{matches, total, rate}`` over hard buckets only.
      - ``confidence_sonnet`` / ``confidence_opus``: Counter dicts.
      - ``disagreements``: list of ``{pair_id, pair_category,
        sonnet_verdict, opus_verdict, sonnet_reasoning, opus_reasoning,
        sonnet_confidence, opus_confidence}``.
    """
    sonnet_by_id = {r["pair_id"]: r for r in sonnet_records}
    opus_by_id = {r["pair_id"]: r for r in opus_records}
    common_ids = sorted(set(sonnet_by_id) & set(opus_by_id))

    per_category_acc: dict[str, dict[str, int]] = {}
    overall_match = 0
    overall_total = 0
    hard_match = 0
    hard_total = 0
    disagreements: list[dict[str, Any]] = []
    sonnet_conf: Counter[int] = Counter()
    opus_conf: Counter[int] = Counter()

    for pid in common_ids:
        s = sonnet_by_id[pid]
        o = opus_by_id[pid]
        category = s.get("pair_category", "unknown")
        bucket = per_category_acc.setdefault(category, {"matches": 0, "total": 0})
        bucket["total"] += 1
        overall_total += 1
        if category in hard_buckets:
            hard_total += 1
        if s.get("sonnet_verdict") == o.get("sonnet_verdict"):
            # Note: opus records also live in the "sonnet_verdict" field
            # because we reuse the labeled-record schema for both —
            # caller is responsible for tagging which model produced
            # which list. For comparison_metrics, both are compared on
            # the same field.
            bucket["matches"] += 1
            overall_match += 1
            if category in hard_buckets:
                hard_match += 1
        else:
            disagreements.append(
                {
                    "pair_id": pid,
                    "pair_category": category,
                    "sonnet_verdict": s.get("sonnet_verdict"),
                    "opus_verdict": o.get("sonnet_verdict"),
                    "sonnet_reasoning": s.get("sonnet_reasoning"),
                    "opus_reasoning": o.get("sonnet_reasoning"),
                    "sonnet_confidence": s.get("sonnet_confidence"),
                    "opus_confidence": o.get("sonnet_confidence"),
                }
            )
        if (sc := s.get("sonnet_confidence")) is not None:
            sonnet_conf[int(sc)] += 1
        if (oc := o.get("sonnet_confidence")) is not None:
            opus_conf[int(oc)] += 1

    def _rate(m: int, t: int) -> float:
        return m / t if t else 0.0

    return {
        "overall": {
            "matches": overall_match,
            "total": overall_total,
            "rate": _rate(overall_match, overall_total),
        },
        "per_category": {
            cat: {
                "matches": v["matches"],
                "total": v["total"],
                "rate": _rate(v["matches"], v["total"]),
            }
            for cat, v in sorted(per_category_acc.items())
        },
        "hard": {
            "matches": hard_match,
            "total": hard_total,
            "rate": _rate(hard_match, hard_total),
        },
        "confidence_sonnet": {k: sonnet_conf.get(k, 0) for k in VALID_CONFIDENCES},
        "confidence_opus": {k: opus_conf.get(k, 0) for k in VALID_CONFIDENCES},
        "disagreements": disagreements,
    }


def decide_dryrun_gate(overall_matches: int, hard_matches: int) -> tuple[str, str]:
    """Map raw match counts to ``(decision, message)``.

    PROCEED iff overall ≥ DRYRUN_GATE_OVERALL_MIN AND hard ≥ DRYRUN_GATE_HARD_MIN.
    ABORT iff hard < DRYRUN_GATE_HARD_ABORT.
    REVIEW otherwise (manual judgment required).
    """
    if hard_matches < DRYRUN_GATE_HARD_ABORT:
        return (
            "ABORT",
            f"Hard-bucket agreement {hard_matches}/30 < {DRYRUN_GATE_HARD_ABORT}/30 (70%). "
            f"Sonnet diverges from Opus too much on the cases that matter most.",
        )
    if (
        overall_matches >= DRYRUN_GATE_OVERALL_MIN
        and hard_matches >= DRYRUN_GATE_HARD_MIN
    ):
        return (
            "PROCEED",
            f"overall {overall_matches}/50 ≥ {DRYRUN_GATE_OVERALL_MIN}/50 "
            f"AND hard {hard_matches}/30 ≥ {DRYRUN_GATE_HARD_MIN}/30. "
            f"Switch to Sonnet 4.6 as primary.",
        )
    return (
        "REVIEW",
        f"overall {overall_matches}/50 (need ≥ {DRYRUN_GATE_OVERALL_MIN}) "
        f"hard {hard_matches}/30 (need ≥ {DRYRUN_GATE_HARD_MIN}, abort < {DRYRUN_GATE_HARD_ABORT}). "
        f"Manual review required.",
    )


def estimate_pair_chars(prompt: str) -> int:
    """Cheap projection: total characters in a rendered prompt."""
    return len(prompt)


# -----------------------------------------------------------------------------
# Dataclasses for the IO seam returns
# -----------------------------------------------------------------------------


@dataclass
class BatchResult:
    """One row from an Anthropic batch results stream."""

    custom_id: str
    status: str  # "succeeded" | "errored" | "expired" | "canceled"
    text: str | None = None
    usage: dict[str, Any] | None = None
    error: str | None = None
    stop_reason: str | None = None


# -----------------------------------------------------------------------------
# IO seams (thin, mockable wrappers around external SDKs)
# -----------------------------------------------------------------------------


def load_prompt_template(
    path: Path = PROMPT_PATH,
    count_tokens: Callable[[str, str], int] | None = None,
) -> str:
    """Read, verify, and return the labeling prompt template.

    If ``count_tokens`` is provided, it's called with ``(static_prefix,
    SONNET)`` and the result is checked against
    ``MIN_CACHE_TOKENS_SONNET``. A short prefix logs a WARNING but does
    not abort (caching simply no-ops; the cost projection in dryrun
    will surface the regression).
    """
    if not path.exists():
        raise PromptError(
            f"labeling_prompt.md not found at {path}. Author it before "
            f"running this stage."
        )
    text = path.read_text(encoding="utf-8")
    verify_prompt_template(text)
    static_prefix, _ = split_prompt_static_prefix(text)
    if count_tokens is not None:
        try:
            n = count_tokens(static_prefix, SONNET)
        except Exception as exc:  # noqa: BLE001
            raise PromptError(
                f"count_prefix_tokens failed: {exc!r}. Cannot verify "
                f"cache-eligibility; install or upgrade the Anthropic SDK."
            ) from exc
        if n < MIN_CACHE_TOKENS_SONNET:
            logger.warning(
                "Static prompt prefix is %d tokens (< %d); Sonnet "
                "prompt caching will silently no-op and primary cost "
                "will balloon. Consider lengthening the static prefix.",
                n,
                MIN_CACHE_TOKENS_SONNET,
            )
        else:
            logger.info(
                "Static prompt prefix is %d tokens (≥ %d cache threshold).",
                n,
                MIN_CACHE_TOKENS_SONNET,
            )
    return text


def count_prefix_tokens_anthropic(client: Any, prefix: str, model: str) -> int:
    """Use Anthropic's tokenizer to measure a text block."""
    resp = client.messages.count_tokens(
        model=model,
        messages=[{"role": "user", "content": prefix}],
    )
    return int(resp.input_tokens)


def verify_anthropic_models(client: Any, names: list[str]) -> None:
    """Abort if any model in ``names`` is not retrievable."""
    for name in names:
        try:
            client.models.retrieve(name)
        except Exception as exc:  # noqa: BLE001
            raise ModelNotFoundError(
                f"Anthropic model verification failed for {name!r}: {exc!r}. "
                f"Check the model ID and your account access."
            ) from exc
    logger.info("Verified Anthropic models: %s", names)


def verify_openai_model(client: Any, name: str) -> str:
    """Verify an OpenAI model is retrievable; return the resolved ID."""
    try:
        m = client.models.retrieve(name)
    except Exception as exc:  # noqa: BLE001 - SDK exception surface varies
        # Try to surface the available list for an actionable error.
        try:
            avail = [m.id for m in client.models.list().data][:25]
        except Exception:  # noqa: BLE001
            avail = []
        raise ModelNotFoundError(
            f"OpenAI model verification failed for {name!r}: {exc!r}. "
            f"Visible models (first 25): {avail}"
        ) from exc
    resolved = getattr(m, "id", name)
    logger.info("Verified OpenAI model: %s", resolved)
    return resolved


def verify_together_model(client: Any, name: str) -> str:
    """Verify a Together model is in ``models.list()``; return the ID."""
    try:
        ids = [getattr(m, "id", None) for m in client.models.list()]
    except Exception as exc:  # noqa: BLE001
        raise ModelNotFoundError(f"Together models.list() failed: {exc!r}") from exc
    if name not in ids:
        # Together model IDs are case-sensitive; surface near-matches.
        near = [m for m in ids if name.lower() in (m or "").lower()][:10]
        raise ModelNotFoundError(
            f"Together model {name!r} not in models.list(). " f"Near-matches: {near}"
        )
    logger.info("Verified Together model: %s", name)
    return name


def verify_openrouter_model(api_key: str, name: str) -> str:
    """Verify an OpenRouter model exists by hitting /models endpoint."""
    import httpx

    r = httpx.get(
        f"{OPENROUTER_BASE_URL}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30.0,
    )
    r.raise_for_status()
    ids = [m["id"] for m in r.json().get("data", [])]
    if name not in ids:
        prefix = name.split("/")[0] if "/" in name else name
        near = [m for m in ids if prefix in m][:5]
        raise ModelNotFoundError(
            f"OpenRouter model {name!r} not found. Near-matches: {near}"
        )
    logger.info("Verified OpenRouter model: %s", name)
    return name


# ------ Batch state sidecar -------------------------------------------------


def _read_batch_state() -> dict[str, dict[str, Any]]:
    if not BATCHES_STATE_PATH.exists():
        return {}
    return json.loads(BATCHES_STATE_PATH.read_text(encoding="utf-8"))


def _write_batch_state(state: dict[str, dict[str, Any]]) -> None:
    atomic_write_json(BATCHES_STATE_PATH, state)


def _update_batch_state(phase_key: str, patch: dict[str, Any]) -> None:
    state = _read_batch_state()
    entry = state.get(phase_key, {})
    entry.update(patch)
    state[phase_key] = entry
    _write_batch_state(state)


# ------ Anthropic batch IO --------------------------------------------------


def submit_anthropic_batch(
    client: Any,
    requests: list[dict[str, Any]],
    model: str,
    phase_key: str,
) -> str:
    """Submit a batch; persist batch_id; return it.

    Retries 5xx responses up to ``SUBMIT_RETRY_ATTEMPTS`` times with
    exponential backoff. 4xx aborts immediately.
    """
    last_exc: Exception | None = None
    for attempt in range(SUBMIT_RETRY_ATTEMPTS):
        try:
            batch = client.messages.batches.create(requests=requests)
            batch_id = batch.id
            _update_batch_state(
                phase_key,
                {
                    "batch_id": batch_id,
                    "model": model,
                    "provider": "anthropic",
                    "n_requests": len(requests),
                    "submitted_at": dt.datetime.now(tz=dt.UTC).isoformat(),
                    "status": "in_progress",
                },
            )
            logger.info(
                "Submitted Anthropic batch %s (%d requests, model=%s, phase=%s).",
                batch_id,
                len(requests),
                model,
                phase_key,
            )
            return batch_id
        except Exception as exc:  # noqa: BLE001
            status = getattr(exc, "status_code", None)
            if status is not None and 400 <= status < 500:
                raise
            last_exc = exc
            backoff = 2**attempt
            logger.warning(
                "submit_anthropic_batch attempt %d failed: %r; retrying in %ds",
                attempt + 1,
                exc,
                backoff,
            )
            time.sleep(backoff)
    raise RuntimeError(
        f"submit_anthropic_batch failed after {SUBMIT_RETRY_ATTEMPTS} attempts"
    ) from last_exc


def poll_anthropic_batch(
    client: Any, batch_id: str, sleep_s: float = POLL_SLEEP_SEC
) -> str:
    """Block until the batch reaches a terminal state; return the status.

    Tolerates up to ``POLL_MAX_CONSECUTIVE_5XX`` consecutive transient
    errors before raising. Updates the sidecar each poll so a SIGINT
    leaves a fresh status.
    """
    consecutive_errors = 0
    while True:
        try:
            batch = client.messages.batches.retrieve(batch_id)
            consecutive_errors = 0
        except Exception as exc:  # noqa: BLE001
            consecutive_errors += 1
            logger.warning(
                "poll_anthropic_batch transient error (%d/%d): %r",
                consecutive_errors,
                POLL_MAX_CONSECUTIVE_5XX,
                exc,
            )
            if consecutive_errors >= POLL_MAX_CONSECUTIVE_5XX:
                raise RuntimeError(
                    f"poll_anthropic_batch: {POLL_MAX_CONSECUTIVE_5XX} "
                    f"consecutive errors polling {batch_id}"
                ) from exc
            time.sleep(sleep_s)
            continue
        status = getattr(batch, "processing_status", None) or getattr(
            batch, "status", "unknown"
        )
        request_counts = getattr(batch, "request_counts", None)
        logger.info(
            "Batch %s status=%s counts=%s",
            batch_id,
            status,
            request_counts,
        )
        if status in ("ended", "canceled", "expired"):
            return status
        time.sleep(sleep_s)


def fetch_anthropic_results(client: Any, batch_id: str) -> Iterator[BatchResult]:
    """Stream parsed ``BatchResult`` rows from an ended batch."""
    for raw in client.messages.batches.results(batch_id):
        custom_id = getattr(raw, "custom_id", None) or ""
        result = getattr(raw, "result", None)
        result_type = getattr(result, "type", "unknown") if result else "unknown"
        if result_type == "succeeded":
            message = getattr(result, "message", None)
            text = ""
            if message is not None:
                blocks = getattr(message, "content", []) or []
                parts = []
                for block in blocks:
                    btype = getattr(block, "type", None)
                    if btype == "text":
                        parts.append(getattr(block, "text", "") or "")
                text = "".join(parts)
            usage_obj = getattr(message, "usage", None) if message else None
            usage = _usage_to_dict(usage_obj)
            stop_reason = getattr(message, "stop_reason", None) if message else None
            yield BatchResult(
                custom_id=custom_id,
                status="succeeded",
                text=text,
                usage=usage,
                stop_reason=stop_reason,
            )
        else:
            error = getattr(result, "error", None) if result else None
            err_msg = (
                repr(error)[:500] if error is not None else f"status={result_type}"
            )
            yield BatchResult(
                custom_id=custom_id,
                status=result_type,
                error=err_msg,
            )


def _usage_to_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    try:
        return usage.model_dump()  # type: ignore[no-any-return]
    except AttributeError:
        out = {}
        for k in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            v = getattr(usage, k, None)
            if v is not None:
                out[k] = v
        return out


# ------ Together batch IO ---------------------------------------------------


def submit_together_batch(
    client: Any,
    requests_jsonl_text: str,
    model: str,
    phase_key: str,
) -> str:
    """Upload a JSONL file and start a Together batch over it.

    Each request line is OpenAI-compatible (``custom_id`` + ``method``
    + ``url`` + ``body``). Together's SDK runs a client-side
    ``check_file`` validator that only knows fine-tune column formats
    and rejects batch-api content unconditionally — but the SDK
    accepts a ``check=False`` flag that skips it while still using
    their proper multi-step upload protocol. Use that.
    """
    import tempfile

    last_exc: Exception | None = None
    for attempt in range(SUBMIT_RETRY_ATTEMPTS):
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=f"-{phase_key}.jsonl",
                delete=False,
                encoding="utf-8",
            ) as f:
                f.write(requests_jsonl_text)
                tmp_path = Path(f.name)
            file_obj = client.files.upload(
                file=str(tmp_path),
                purpose="batch-api",
                check=False,
            )
            file_id = getattr(file_obj, "id", None) or file_obj["id"]
            batch = client.batches.create(
                input_file_id=file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            # Together wraps the BatchJob inside ``.job`` on
            # ``batches.create``'s response (asymmetric — ``retrieve``
            # returns BatchJob directly).
            job = getattr(batch, "job", None) or batch
            batch_id = getattr(job, "id", None)
            if not batch_id:
                # ValueError is in the don't-retry list — failing to
                # extract a batch id is a code/SDK-shape bug, not a
                # transient error, so we must NOT retry (which would
                # create duplicate batches at real cost).
                raise ValueError(
                    f"Together batches.create returned no batch id: {batch!r}"
                )
            _update_batch_state(
                phase_key,
                {
                    "batch_id": batch_id,
                    "input_file_id": file_id,
                    "model": model,
                    "provider": "together",
                    "submitted_at": dt.datetime.now(tz=dt.UTC).isoformat(),
                    "status": "in_progress",
                },
            )
            logger.info(
                "Submitted Together batch %s (file=%s, model=%s, phase=%s).",
                batch_id,
                file_id,
                model,
                phase_key,
            )
            return batch_id
        except (TypeError, AttributeError, KeyError, ValueError):
            # Programmer / SDK-shape errors are not transient; raise
            # immediately. (Without this, a code bug causes 3 silent
            # duplicate batch submissions before surfacing — burns
            # money and pollutes Together's queue.)
            raise
        except Exception as exc:  # noqa: BLE001
            http_status = getattr(exc, "status_code", None) or getattr(
                getattr(exc, "response", None), "status_code", None
            )
            if http_status is not None and 400 <= http_status < 500:
                raise
            last_exc = exc
            backoff = 2**attempt
            logger.warning(
                "submit_together_batch attempt %d failed: %r; retrying in %ds",
                attempt + 1,
                exc,
                backoff,
            )
            time.sleep(backoff)
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
    raise RuntimeError(
        f"submit_together_batch failed after {SUBMIT_RETRY_ATTEMPTS} attempts"
    ) from last_exc


def poll_together_batch(
    client: Any, batch_id: str, sleep_s: float = POLL_SLEEP_SEC
) -> str:
    """Block until a Together batch reaches a terminal state."""
    consecutive_errors = 0
    while True:
        try:
            batch = client.batches.retrieve(batch_id)
            consecutive_errors = 0
        except Exception as exc:  # noqa: BLE001
            consecutive_errors += 1
            logger.warning(
                "poll_together_batch transient error (%d/%d): %r",
                consecutive_errors,
                POLL_MAX_CONSECUTIVE_5XX,
                exc,
            )
            if consecutive_errors >= POLL_MAX_CONSECUTIVE_5XX:
                raise
            time.sleep(sleep_s)
            continue
        # Together's BatchJob.status is upper-snake-case
        # (VALIDATING, IN_PROGRESS, COMPLETED, FAILED, EXPIRED,
        # CANCELLED). Lowercase it so the rest of the code's
        # comparisons stay consistent with the OpenAI/Anthropic paths.
        status = (getattr(batch, "status", None) or "").lower()
        logger.info("Together batch %s status=%s", batch_id, status)
        if status in ("completed", "failed", "expired", "cancelled", "canceled"):
            return status
        time.sleep(sleep_s)


def fetch_together_results(client: Any, batch_id: str) -> Iterator[BatchResult]:
    """Stream parsed results from a Together batch."""
    batch = client.batches.retrieve(batch_id)
    output_file_id = getattr(batch, "output_file_id", None)
    if output_file_id is None:
        return
    raw = client.files.content(output_file_id)
    text = raw.read().decode("utf-8") if hasattr(raw, "read") else str(raw)
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        custom_id = row.get("custom_id", "")
        response = row.get("response") or {}
        body = response.get("body") or {}
        if response.get("status_code", 200) >= 400 or row.get("error"):
            yield BatchResult(
                custom_id=custom_id,
                status="errored",
                error=json.dumps(row.get("error") or response)[:500],
            )
            continue
        choices = body.get("choices") or []
        msg = choices[0].get("message") if choices else {}
        text_out = (msg or {}).get("content", "") if msg else ""
        usage = body.get("usage", {})
        stop_reason = choices[0].get("finish_reason") if choices else None
        yield BatchResult(
            custom_id=custom_id,
            status="succeeded",
            text=text_out,
            usage=usage,
            stop_reason=stop_reason,
        )


# ------ OpenAI batch IO -----------------------------------------------------


def submit_openai_batch(
    client: Any,
    requests_jsonl_text: str,
    model: str,
    phase_key: str,
) -> str:
    """Upload a JSONL of requests and start an OpenAI batch over it.

    Each request line has shape
    ``{"custom_id", "method": "POST", "url": "/v1/chat/completions", "body": {...}}``.
    OpenAI's batch API gives a 50 % discount over sync; SLA is 24 h
    but small batches typically complete in minutes.
    """
    last_exc: Exception | None = None
    for attempt in range(SUBMIT_RETRY_ATTEMPTS):
        try:
            buf = io.BytesIO(requests_jsonl_text.encode("utf-8"))
            buf.name = f"judge-from-scratch-{phase_key}.jsonl"
            file_obj = client.files.create(file=buf, purpose="batch")
            file_id = getattr(file_obj, "id", None) or file_obj["id"]
            batch = client.batches.create(
                input_file_id=file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            batch_id = getattr(batch, "id", None) or batch["id"]
            _update_batch_state(
                phase_key,
                {
                    "batch_id": batch_id,
                    "input_file_id": file_id,
                    "model": model,
                    "provider": "openai",
                    "submitted_at": dt.datetime.now(tz=dt.UTC).isoformat(),
                    "status": "in_progress",
                },
            )
            logger.info(
                "Submitted OpenAI batch %s (file=%s, model=%s, phase=%s).",
                batch_id,
                file_id,
                model,
                phase_key,
            )
            return batch_id
        except Exception as exc:  # noqa: BLE001
            status = getattr(exc, "status_code", None)
            if status is not None and 400 <= status < 500:
                raise
            last_exc = exc
            backoff = 2**attempt
            logger.warning(
                "submit_openai_batch attempt %d failed: %r; retrying in %ds",
                attempt + 1,
                exc,
                backoff,
            )
            time.sleep(backoff)
    raise RuntimeError(
        f"submit_openai_batch failed after {SUBMIT_RETRY_ATTEMPTS} attempts"
    ) from last_exc


def poll_openai_batch(
    client: Any, batch_id: str, sleep_s: float = POLL_SLEEP_SEC
) -> str:
    """Block until an OpenAI batch reaches a terminal state."""
    consecutive_errors = 0
    while True:
        try:
            batch = client.batches.retrieve(batch_id)
            consecutive_errors = 0
        except Exception as exc:  # noqa: BLE001
            consecutive_errors += 1
            logger.warning(
                "poll_openai_batch transient error (%d/%d): %r",
                consecutive_errors,
                POLL_MAX_CONSECUTIVE_5XX,
                exc,
            )
            if consecutive_errors >= POLL_MAX_CONSECUTIVE_5XX:
                raise
            time.sleep(sleep_s)
            continue
        status = (getattr(batch, "status", None) or "").lower()
        logger.info("OpenAI batch %s status=%s", batch_id, status)
        if status in ("completed", "failed", "expired", "cancelled", "canceled"):
            return status
        time.sleep(sleep_s)


def fetch_openai_results(client: Any, batch_id: str) -> Iterator[BatchResult]:
    """Stream parsed results from an OpenAI batch.

    Reads BOTH ``output_file_id`` (succeeded requests) and
    ``error_file_id`` (failed requests) so all 250 results show up,
    even if every one of them failed validation. Without this, a
    batch where every request errors silently yields nothing and the
    caller sees ``n_requests=0``.
    """
    batch = client.batches.retrieve(batch_id)
    for file_id_attr in ("output_file_id", "error_file_id"):
        file_id = getattr(batch, file_id_attr, None)
        if file_id is None:
            continue
        raw = client.files.content(file_id)
        text = raw.read().decode("utf-8") if hasattr(raw, "read") else str(raw)
        for line in text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            custom_id = row.get("custom_id", "")
            response = row.get("response") or {}
            body = response.get("body") or {}
            row_error = row.get("error")
            body_error = (body or {}).get("error") if isinstance(body, dict) else None
            if response.get("status_code", 200) >= 400 or row_error or body_error:
                yield BatchResult(
                    custom_id=custom_id,
                    status="errored",
                    error=json.dumps(row_error or body_error or response)[:500],
                )
                continue
            choices = body.get("choices") or []
            msg = choices[0].get("message") if choices else {}
            text_out = (msg or {}).get("content", "") if msg else ""
            usage = body.get("usage", {})
            stop_reason = choices[0].get("finish_reason") if choices else None
            yield BatchResult(
                custom_id=custom_id,
                status="succeeded",
                text=text_out,
                usage=usage,
                stop_reason=stop_reason,
            )


# ------ OpenAI single-call IO (legacy fallback, unused after batch switch) ---


async def call_openai_one(
    client: Any,
    pair: dict[str, Any],
    rendered_prompt: str,
    model: str,
    semaphore: asyncio.Semaphore,
) -> BatchResult:
    """Issue one chat completion for a pair; never raises."""
    async with semaphore:
        for attempt in range(SUBMIT_RETRY_ATTEMPTS):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": rendered_prompt}],
                    max_tokens=OPENAI_MAX_TOKENS,
                )
                msg = resp.choices[0].message
                text = msg.content if msg else ""
                usage = (
                    resp.usage.model_dump()
                    if hasattr(resp.usage, "model_dump")
                    else dict(resp.usage) if resp.usage else {}
                )
                stop_reason = resp.choices[0].finish_reason
                return BatchResult(
                    custom_id=str(pair["pair_id"]),
                    status="succeeded",
                    text=text,
                    usage=usage,
                    stop_reason=stop_reason,
                )
            except Exception as exc:  # noqa: BLE001
                status = getattr(exc, "status_code", None)
                if status is not None and 400 <= status < 500:
                    return BatchResult(
                        custom_id=str(pair["pair_id"]),
                        status="errored",
                        error=repr(exc)[:500],
                    )
                if attempt == SUBMIT_RETRY_ATTEMPTS - 1:
                    return BatchResult(
                        custom_id=str(pair["pair_id"]),
                        status="errored",
                        error=repr(exc)[:500],
                    )
                await asyncio.sleep(2**attempt)
    # unreachable
    return BatchResult(
        custom_id=str(pair["pair_id"]), status="errored", error="unreachable"
    )


# ------ Cost ledger + budget guardrail --------------------------------------


def record_cost(
    phase: str,
    model: str,
    batch_id: str | None,
    usage_totals: dict[str, Any],
    n_requests: int,
    cost_usd: float,
) -> None:
    """Append one row to ``.cost_ledger.jsonl``."""
    jsonl_append(
        COST_LEDGER_PATH,
        {
            "ts": dt.datetime.now(tz=dt.UTC).isoformat(),
            "phase": phase,
            "model": model,
            "batch_id": batch_id,
            "n_requests": n_requests,
            "usage": usage_totals,
            "cost_usd": round(cost_usd, 6),
            "pricing_as_of": PRICING_AS_OF,
        },
    )


def total_spend() -> float:
    """Sum the cost ledger; 0.0 if no ledger yet."""
    if not COST_LEDGER_PATH.exists():
        return 0.0
    return sum(float(r.get("cost_usd", 0.0)) for r in jsonl_read(COST_LEDGER_PATH))


def check_budget(spent: float, projected: float) -> None:
    """Raise ``BudgetExceededError`` (or prompt on TTY) when over the cap."""
    proposed = spent + projected
    if proposed <= BUDGET_USD_HARD:
        return
    msg = (
        f"BUDGET GUARDRAIL: spent ${spent:.2f} + projected ${projected:.2f} "
        f"= ${proposed:.2f} > cap ${BUDGET_USD_HARD:.2f}"
    )
    if not sys.stdin.isatty():
        raise BudgetExceededError(msg)
    print(msg, file=sys.stderr)
    answer = input("Type CONTINUE to override the budget cap: ")
    if answer.strip() != "CONTINUE":
        raise BudgetExceededError(msg + " (operator declined to override)")


def confirm_cost_or_exit(
    estimated_usd: float, label: str, *, yes: bool, dry_run: bool
) -> None:
    """TTY confirmation gate before paid work. Skipped if --yes or --dry-run."""
    print(
        f"[cost] {label}: estimated ${estimated_usd:.2f} (cumulative spend so far: ${total_spend():.2f})",
        file=sys.stderr,
    )
    if dry_run or yes:
        return
    if not sys.stdin.isatty():
        raise RuntimeError(
            f"non-interactive run requires --yes to confirm ${estimated_usd:.2f} for {label}"
        )
    answer = input(f"Proceed with {label} for ~${estimated_usd:.2f}? [y/N] ")
    if answer.strip().lower() not in ("y", "yes"):
        raise SystemExit(0)


def project_cost_anthropic(
    rendered_prompts: list[str], model: str, batch: bool = True
) -> float:
    """Rough char/4 token projection for an Anthropic batch."""
    fake_usage = {
        "input_tokens": sum(estimate_pair_chars(p) for p in rendered_prompts) // 4,
        "output_tokens": ANTHROPIC_MAX_TOKENS // 2 * len(rendered_prompts),
        "cache_read_input_tokens": 0,
    }
    return compute_cost(fake_usage, model, batch=batch)


# -----------------------------------------------------------------------------
# Phase orchestrators
# -----------------------------------------------------------------------------


def _ensure_dirs() -> None:
    LABELED_DIR.mkdir(parents=True, exist_ok=True)


def _read_existing_meta() -> dict[str, Any]:
    if not META_PATH.exists():
        return {}
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def _write_meta_phase(phase: str, payload: dict[str, Any]) -> None:
    meta = _read_existing_meta()
    meta[phase] = payload
    atomic_write_json(META_PATH, meta)


def _make_anthropic_client() -> Any:
    import anthropic  # type: ignore

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment.")
    return anthropic.Anthropic(api_key=api_key)


def _make_openai_async_client() -> Any:
    from openai import AsyncOpenAI  # type: ignore

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    return AsyncOpenAI(api_key=api_key)


def _make_openai_client() -> Any:
    from openai import OpenAI  # type: ignore

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    return OpenAI(api_key=api_key)


def _make_together_client() -> Any:
    import together  # type: ignore

    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY not set in environment.")
    return together.Together(api_key=api_key)


def _make_openrouter_async_client() -> Any:
    """Async OpenAI-compatible client pointed at OpenRouter.

    OpenRouter has no batch API, so this is the sync (async fan-out)
    path for the third labeler. Same pattern Stage 1 used for
    candidate generation.
    """
    from openai import AsyncOpenAI  # type: ignore

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment.")
    return AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        timeout=60.0,
        max_retries=2,
        default_headers={
            "HTTP-Referer": "https://github.com/krishnakartik1/judge-from-scratch",
            "X-Title": "Judge from Scratch",
        },
    )


def _label_record(
    pair: dict[str, Any],
    parsed: dict[str, Any] | None,
    fields_prefix: str = "sonnet",
) -> dict[str, Any]:
    """Build a labeled-pairs record from a pair + parsed model output."""
    out = dict(pair)
    if parsed is None:
        out[f"{fields_prefix}_verdict"] = None
        out[f"{fields_prefix}_reasoning"] = None
        out[f"{fields_prefix}_confidence"] = None
    else:
        out[f"{fields_prefix}_verdict"] = parsed["verdict"]
        out[f"{fields_prefix}_reasoning"] = parsed["reasoning"]
        out[f"{fields_prefix}_confidence"] = parsed["confidence"]
    # Initialize cross-check fields explicitly so the merge phase only
    # updates, never inserts. Field names are PROVIDER-NEUTRAL slot
    # IDs (b, c) rather than model-specific (gpt, deepseek): the
    # actual model used for each slot can swap (e.g. DeepSeek →
    # OpenRouter Qwen) without making the schema lie. See
    # meta.crosscheck.labelers for the resolved per-slot model.
    for k in (
        "crosscheck_verdict_b",
        "crosscheck_reasoning_b",
        "crosscheck_verdict_c",
        "crosscheck_reasoning_c",
    ):
        out.setdefault(k, None)
    out.setdefault("disagreement", None)
    return out


def _collect_dryrun_results(
    client: Any,
    batch_id: str,
    pairs_by_id: dict[str, dict[str, Any]],
    *,
    model: str,
    label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int], int]:
    """Fetch a dryrun batch and return ``(records, per_pair_jsonl, totals, n)``.

    - ``records``: parsed in-memory rows used by ``comparison_metrics``
      (shape: same as primary phase's ``_label_record`` output).
    - ``per_pair_jsonl``: lightweight rows ready for
      ``atomic_write_jsonl``; one per parsed pair, with
      ``{pair_id, pair_category, verdict, confidence, reasoning,
      model, batch_id, usage}``. ``usage`` is the per-request usage
      dict, so cache hits per pair are recoverable from disk.
    - ``totals``: summed batch usage (input/output/cache_read/cache_creation).
    - ``n``: total number of result rows seen (succeeded + errored + …).
    """
    records: list[dict[str, Any]] = []
    per_pair: list[dict[str, Any]] = []
    totals: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    n = 0
    for r in fetch_anthropic_results(client, batch_id):
        n += 1
        if r.status != "succeeded":
            continue
        if r.usage:
            for k in totals:
                totals[k] += int(r.usage.get(k, 0) or 0)
        try:
            parsed = parse_model_output(r.text or "")
        except ParseError as exc:
            logger.warning("dryrun %s parse error on %s: %s", label, r.custom_id, exc)
            continue
        pair = pairs_by_id[r.custom_id]
        records.append(_label_record(pair, parsed))
        per_pair.append(
            {
                "pair_id": pair["pair_id"],
                "pair_category": pair["pair_category"],
                "verdict": parsed["verdict"],
                "confidence": parsed["confidence"],
                "reasoning": parsed["reasoning"],
                "model": model,
                "batch_id": batch_id,
                "usage": dict(r.usage or {}),
            }
        )
    return records, per_pair, totals, n


def _process_anthropic_results(
    results: Iterator[BatchResult],
    pairs_by_id: dict[str, dict[str, Any]],
    *,
    phase: str,
    model: str,
    output_path: Path,
    fields_prefix: str = "sonnet",
) -> dict[str, int]:
    """Dispatch a stream of BatchResult rows. Returns counters."""
    counts = {
        "succeeded_parsed": 0,
        "succeeded_refusal": 0,
        "succeeded_parse_error": 0,
        "errored": 0,
        "expired": 0,
        "canceled": 0,
        "missing_pair": 0,
    }
    usage_totals: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    n_requests = 0
    for r in results:
        n_requests += 1
        pair = pairs_by_id.get(r.custom_id)
        if pair is None:
            counts["missing_pair"] += 1
            logger.warning(
                "Batch result for unknown pair_id %r — dropped.", r.custom_id
            )
            continue
        if r.status == "succeeded":
            if r.usage:
                for k in usage_totals:
                    usage_totals[k] += int(r.usage.get(k, 0) or 0)
            if detect_refusal(r.text, r.stop_reason):
                jsonl_append(
                    REFUSALS_PATH,
                    {
                        "pair_id": r.custom_id,
                        "phase": phase,
                        "model": model,
                        "raw_text": r.text or "",
                        "stop_reason": r.stop_reason,
                    },
                )
                counts["succeeded_refusal"] += 1
                continue
            try:
                parsed = parse_model_output(r.text or "")
            except ParseError as exc:
                jsonl_append(
                    PARSE_ERRORS_PATH,
                    {
                        "pair_id": r.custom_id,
                        "phase": phase,
                        "model": model,
                        "raw_text": r.text or "",
                        "error": str(exc),
                    },
                )
                counts["succeeded_parse_error"] += 1
                continue
            jsonl_append(output_path, _label_record(pair, parsed, fields_prefix))
            counts["succeeded_parsed"] += 1
        else:
            jsonl_append(
                BATCH_ERRORS_PATH,
                {
                    "pair_id": r.custom_id,
                    "phase": phase,
                    "model": model,
                    "batch_status": r.status,
                    "error": r.error or "",
                },
            )
            if r.status == "errored":
                counts["errored"] += 1
            elif r.status == "expired":
                counts["expired"] += 1
            elif r.status == "canceled":
                counts["canceled"] += 1

    cost = compute_cost(usage_totals, model, batch=True)
    record_cost(
        phase=phase,
        model=model,
        batch_id=None,
        usage_totals=usage_totals,
        n_requests=n_requests,
        cost_usd=cost,
    )
    counts["cost_usd"] = round(cost, 6)
    return counts


# -----------------------------------------------------------------------------
# Dry-run report writer
# -----------------------------------------------------------------------------


def _truncate(s: str | None, n: int = 600) -> str:
    if not s:
        return ""
    s = s.strip()
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def write_dryrun_report_md(
    out_path: Path,
    metrics: dict[str, Any],
    decision: str,
    decision_msg: str,
    sonnet_records: list[dict[str, Any]],
    opus_records: list[dict[str, Any]],
    sonnet_summary: dict[str, Any],
    opus_summary: dict[str, Any],
    sonnet_spend: float,
    opus_spend: float,
) -> None:
    """Render the human-readable dry-run report."""
    lines: list[str] = []

    lines.append(f"# Dry-run comparison: {SONNET} vs {OPUS}")
    lines.append("")
    lines.append(f"**Decision: `{decision}`**")
    lines.append("")
    lines.append(f"> {decision_msg}")
    lines.append("")
    lines.append(
        f"Generated {dt.datetime.now(tz=dt.UTC).isoformat()} • "
        f"PRICING_AS_OF={PRICING_AS_OF}"
    )
    lines.append("")

    lines.append("## 1. Overall agreement")
    lines.append("")
    o = metrics["overall"]
    lines.append(
        f"- Sonnet vs Opus verdict match: **{o['matches']}/{o['total']}** ({o['rate']*100:.1f} %)"
    )
    h = metrics["hard"]
    lines.append(
        f"- Hard buckets only: **{h['matches']}/{h['total']}** ({h['rate']*100:.1f} %) — "
        f"buckets: {list(HARD_BUCKETS)}"
    )
    lines.append("")

    lines.append("## 2. Per-pair_category agreement")
    lines.append("")
    lines.append("| pair_category | matches / total | rate |")
    lines.append("|---|---|---|")
    for cat, v in metrics["per_category"].items():
        lines.append(f"| {cat} | {v['matches']}/{v['total']} | {v['rate']*100:.1f} % |")
    lines.append("")

    lines.append("## 3. Confidence distribution")
    lines.append("")
    lines.append("| confidence | sonnet | opus |")
    lines.append("|---|---|---|")
    for c in VALID_CONFIDENCES:
        lines.append(
            f"| {c} | {metrics['confidence_sonnet'].get(c, 0)} | "
            f"{metrics['confidence_opus'].get(c, 0)} |"
        )
    lines.append("")

    lines.append("## 4. Per-model spend")
    lines.append("")
    lines.append(f"- Sonnet: **${sonnet_spend:.4f}**")
    lines.append(f"- Opus:   **${opus_spend:.4f}**")
    lines.append(f"- Total:  **${sonnet_spend + opus_spend:.4f}**")
    lines.append("")

    lines.append("## 5. First 5 labeled pairs (side-by-side)")
    lines.append("")
    sonnet_by_id = {r["pair_id"]: r for r in sonnet_records}
    opus_by_id = {r["pair_id"]: r for r in opus_records}
    common = sorted(set(sonnet_by_id) & set(opus_by_id))[:5]
    for pid in common:
        s = sonnet_by_id[pid]
        op = opus_by_id[pid]
        lines.append(f"### pair_id `{pid}` ({s.get('pair_category')})")
        lines.append("")
        lines.append(f"**Question:** {_truncate(s.get('question_text'), 300)}")
        lines.append("")
        lines.append(
            f"**Sonnet:** verdict=**{s.get('sonnet_verdict')}** "
            f"confidence={s.get('sonnet_confidence')}"
        )
        lines.append("")
        lines.append(f"> {_truncate(s.get('sonnet_reasoning'))}")
        lines.append("")
        lines.append(
            f"**Opus:** verdict=**{op.get('sonnet_verdict')}** "
            f"confidence={op.get('sonnet_confidence')}"
        )
        lines.append("")
        lines.append(f"> {_truncate(op.get('sonnet_reasoning'))}")
        lines.append("")

    lines.append(f"## 6. Disagreements ({len(metrics['disagreements'])})")
    lines.append("")
    if not metrics["disagreements"]:
        lines.append("_None._")
    for d in metrics["disagreements"]:
        lines.append(
            f"### `{d['pair_id']}` ({d['pair_category']}) — "
            f"Sonnet=**{d['sonnet_verdict']}** vs Opus=**{d['opus_verdict']}**"
        )
        lines.append(
            f"- Sonnet conf={d['sonnet_confidence']} • Opus conf={d['opus_confidence']}"
        )
        lines.append("")
        lines.append("**Sonnet reasoning:**")
        lines.append("")
        lines.append(f"> {_truncate(d['sonnet_reasoning'])}")
        lines.append("")
        lines.append("**Opus reasoning:**")
        lines.append("")
        lines.append(f"> {_truncate(d['opus_reasoning'])}")
        lines.append("")

    lines.append("## 7. Per-model summaries")
    lines.append("")
    lines.append(
        f"- Sonnet: count={sonnet_summary['count']} verdicts={sonnet_summary['verdicts']}"
    )
    lines.append(
        f"- Opus:   count={opus_summary['count']} verdicts={opus_summary['verdicts']}"
    )
    lines.append("")
    lines.append("## 8. Next step")
    lines.append("")
    if decision == "PROCEED":
        lines.append(
            "Re-invoke with `data/04_label_pairs.py primary --confirm-dryrun` "
            "to label all 1,938 pairs with Sonnet."
        )
    elif decision == "ABORT":
        lines.append(
            "Sonnet diverges too much from Opus on hard cases. Either "
            "switch primary back to Opus (override `SONNET` constant) "
            "or revise `data/labeling_prompt.md` and re-run dryrun."
        )
    else:
        lines.append(
            "Manual review required. Read the disagreements above and "
            "decide whether to override with `--force-gate` or revise "
            "the prompt."
        )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Phase: dryrun
# -----------------------------------------------------------------------------


def run_dryrun_phase(args: argparse.Namespace) -> int:
    """50-pair Sonnet vs Opus comparison."""
    _ensure_dirs()

    if args.dry_run:
        logger.info("--dry-run: skipping Anthropic client construction.")
        client: Any = None
        template = load_prompt_template()
    else:
        client = _make_anthropic_client()
        template = load_prompt_template(
            count_tokens=lambda prefix, model: count_prefix_tokens_anthropic(
                client, prefix, model
            ),
        )

    static_system = "You are a careful labeler for a bias-evaluation training dataset."

    pairs = list(jsonl_read(INPUT_PATH))
    targets = {b: DRYRUN_PER_BUCKET for b in DRYRUN_BUCKETS}
    sample = stratified_sample(pairs, targets, seed=args.seed)
    if args.limit:
        sample = sample[: args.limit]
    pairs_by_id = {p["pair_id"]: p for p in sample}

    rendered = [render_prompt(template, p) for p in sample]
    sonnet_proj = project_cost_anthropic(rendered, SONNET, batch=True)
    opus_proj = project_cost_anthropic(rendered, OPUS, batch=True)
    total_proj = sonnet_proj + opus_proj
    logger.info(
        "Dry-run sample: %d pairs. Sonnet est ~$%.4f, Opus est ~$%.4f.",
        len(sample),
        sonnet_proj,
        opus_proj,
    )

    if args.dry_run:
        for r in rendered[: min(3, len(rendered))]:
            print("--- rendered prompt (first 800 chars) ---")
            print(r[:800])
            print("--- end ---")
        return 0

    verify_anthropic_models(client, [SONNET, OPUS])

    confirm_cost_or_exit(
        total_proj, "dryrun (Sonnet + Opus, 50 pairs)", yes=args.yes, dry_run=False
    )
    check_budget(total_spend(), total_proj)

    sonnet_requests = [
        build_anthropic_request(p, static_system, template, SONNET) for p in sample
    ]
    opus_requests = [
        build_anthropic_request(p, static_system, template, OPUS) for p in sample
    ]

    sonnet_batch_id = submit_anthropic_batch(
        client, sonnet_requests, SONNET, "dryrun_sonnet"
    )
    opus_batch_id = submit_anthropic_batch(client, opus_requests, OPUS, "dryrun_opus")

    poll_anthropic_batch(client, sonnet_batch_id)
    poll_anthropic_batch(client, opus_batch_id)
    _update_batch_state("dryrun_sonnet", {"status": "ended"})
    _update_batch_state("dryrun_opus", {"status": "ended"})

    sonnet_records, sonnet_per_pair, sonnet_usage, sonnet_n = _collect_dryrun_results(
        client, sonnet_batch_id, pairs_by_id, model=SONNET, label="sonnet"
    )
    sonnet_cost = compute_cost(sonnet_usage, SONNET, batch=True)
    record_cost(
        "dryrun_sonnet", SONNET, sonnet_batch_id, sonnet_usage, sonnet_n, sonnet_cost
    )

    opus_records, opus_per_pair, opus_usage, opus_n = _collect_dryrun_results(
        client, opus_batch_id, pairs_by_id, model=OPUS, label="opus"
    )
    opus_cost = compute_cost(opus_usage, OPUS, batch=True)
    record_cost("dryrun_opus", OPUS, opus_batch_id, opus_usage, opus_n, opus_cost)

    # Persist per-pair records as JSONL for audit + post-hoc comparison.
    atomic_write_jsonl(LABELED_DIR / "dryrun_sonnet.jsonl", sonnet_per_pair)
    atomic_write_jsonl(LABELED_DIR / "dryrun_opus.jsonl", opus_per_pair)
    logger.info(
        "Wrote dryrun_sonnet.jsonl (%d) and dryrun_opus.jsonl (%d).",
        len(sonnet_per_pair),
        len(opus_per_pair),
    )

    sonnet_cache_hits = sum(
        1 for r in sonnet_per_pair if r["usage"].get("cache_read_input_tokens", 0) > 0
    )
    opus_cache_hits = sum(
        1 for r in opus_per_pair if r["usage"].get("cache_read_input_tokens", 0) > 0
    )
    logger.info(
        "Cache hits — Sonnet: %d/%d  Opus: %d/%d",
        sonnet_cache_hits,
        len(sonnet_per_pair),
        opus_cache_hits,
        len(opus_per_pair),
    )

    metrics = comparison_metrics(sonnet_records, opus_records)
    decision, msg = decide_dryrun_gate(
        metrics["overall"]["matches"], metrics["hard"]["matches"]
    )
    sonnet_summary = summarize_results(
        sonnet_records, "sonnet_verdict", "sonnet_confidence"
    )
    opus_summary = summarize_results(
        opus_records, "sonnet_verdict", "sonnet_confidence"
    )

    report = {
        "decision": decision,
        "decision_msg": msg,
        "metrics": metrics,
        "sonnet_summary": sonnet_summary,
        "opus_summary": opus_summary,
        "sonnet_spend_usd": round(sonnet_cost, 6),
        "opus_spend_usd": round(opus_cost, 6),
        "pricing_as_of": PRICING_AS_OF,
        "seed": args.seed,
    }
    atomic_write_json(DRYRUN_REPORT_JSON, report)
    write_dryrun_report_md(
        DRYRUN_REPORT_MD,
        metrics,
        decision,
        msg,
        sonnet_records,
        opus_records,
        sonnet_summary,
        opus_summary,
        sonnet_cost,
        opus_cost,
    )
    _write_meta_phase(
        "dryrun",
        {
            "status": "complete",
            "decision": decision,
            "completed_at": dt.datetime.now(tz=dt.UTC).isoformat(),
            "sonnet_spend_usd": round(sonnet_cost, 6),
            "opus_spend_usd": round(opus_cost, 6),
        },
    )

    print("=" * 72)
    print(f"DRYRUN DECISION: {decision}")
    print(msg)
    print(
        f"Reports: {DRYRUN_REPORT_MD.relative_to(REPO_ROOT)} (human) "
        f"and {DRYRUN_REPORT_JSON.relative_to(REPO_ROOT)} (machine)."
    )
    print("=" * 72)

    if decision == "PROCEED":
        return 0
    if decision == "ABORT":
        return 4
    return 3


# -----------------------------------------------------------------------------
# Phase: primary
# -----------------------------------------------------------------------------


def run_primary_phase(args: argparse.Namespace) -> int:
    """Sonnet on 1,938 pairs."""
    _ensure_dirs()

    if not args.confirm_dryrun:
        logger.error("primary requires --confirm-dryrun")
        return 1

    if not DRYRUN_REPORT_JSON.exists():
        raise GateError(f"missing {DRYRUN_REPORT_JSON}; run `dryrun` first.")
    report = json.loads(DRYRUN_REPORT_JSON.read_text(encoding="utf-8"))
    if report.get("decision") != "PROCEED":
        if not args.force_gate:
            raise GateError(
                f"dryrun decision was {report.get('decision')!r} "
                f"(not PROCEED). Use --force-gate to override."
            )
        if sys.stdin.isatty():
            answer = input(
                f"Override dryrun decision {report.get('decision')!r}? Type YES: "
            )
            if answer.strip() != "YES":
                raise GateError("operator declined to override gate")

    if args.dry_run:
        logger.info("--dry-run: skipping Anthropic client construction.")
        client: Any = None
        template = load_prompt_template()
    else:
        client = _make_anthropic_client()
        template = load_prompt_template(
            count_tokens=lambda prefix, model: count_prefix_tokens_anthropic(
                client, prefix, model
            )
        )
    static_system = "You are a careful labeler for a bias-evaluation training dataset."

    pairs = list(jsonl_read(INPUT_PATH))
    pair_ids = [p["pair_id"] for p in pairs]
    if len(set(pair_ids)) != len(pair_ids):
        raise AssertionError("duplicate pair_id in input")

    seen = already_processed(LABELED_PATH, "pair_id")
    for sidecar in (PARSE_ERRORS_PATH, REFUSALS_PATH, BATCH_ERRORS_PATH):
        if sidecar.exists():
            seen |= {r["pair_id"] for r in jsonl_read(sidecar) if "pair_id" in r}

    todo = [p for p in pairs if p["pair_id"] not in seen]
    if args.limit:
        todo = todo[: args.limit]

    if not todo:
        logger.info("All pairs already accounted for; nothing to do.")
    else:
        rendered = [render_prompt(template, p) for p in todo]
        proj = project_cost_anthropic(rendered, SONNET, batch=True)
        logger.info("Primary: %d pairs to label; estimated $%.4f.", len(todo), proj)

        if args.dry_run:
            for r in rendered[: min(3, len(rendered))]:
                print("--- rendered prompt (first 800 chars) ---")
                print(r[:800])
                print("--- end ---")
            return 0

        verify_anthropic_models(client, [SONNET])

        # Re-attach to in-flight batch if present.
        state = _read_batch_state()
        existing = state.get("primary_sonnet")
        if existing and existing.get("status") not in ("ended", None):
            batch_id = existing["batch_id"]
            logger.info("Re-attaching to in-flight Sonnet batch %s.", batch_id)
        else:
            confirm_cost_or_exit(
                proj,
                f"primary Sonnet labeling ({len(todo)} pairs)",
                yes=args.yes,
                dry_run=False,
            )
            check_budget(total_spend(), proj)
            requests = [
                build_anthropic_request(p, static_system, template, SONNET)
                for p in todo
            ]
            batch_id = submit_anthropic_batch(
                client, requests, SONNET, "primary_sonnet"
            )

        poll_anthropic_batch(client, batch_id)
        _update_batch_state("primary_sonnet", {"status": "ended"})
        pairs_by_id = {p["pair_id"]: p for p in todo}
        results = fetch_anthropic_results(client, batch_id)
        counts = _process_anthropic_results(
            results,
            pairs_by_id,
            phase="primary",
            model=SONNET,
            output_path=LABELED_PATH,
        )
        logger.info("Primary batch fetch counts: %s", counts)

    # Reconciliation
    labeled = list(jsonl_read(LABELED_PATH)) if LABELED_PATH.exists() else []
    parse_errors = (
        list(jsonl_read(PARSE_ERRORS_PATH)) if PARSE_ERRORS_PATH.exists() else []
    )
    refusals = list(jsonl_read(REFUSALS_PATH)) if REFUSALS_PATH.exists() else []
    batch_errors = (
        list(jsonl_read(BATCH_ERRORS_PATH)) if BATCH_ERRORS_PATH.exists() else []
    )
    accounted = len(labeled) + len(parse_errors) + len(refusals) + len(batch_errors)
    summary = summarize_results(labeled, "sonnet_verdict", "sonnet_confidence")

    print("=" * 72)
    print("PRIMARY LABELING COMPLETE")
    print(f"  labeled (success):  {len(labeled)}")
    print(f"  parse errors:       {len(parse_errors)}")
    print(f"  refusals:           {len(refusals)}")
    print(f"  batch errors:       {len(batch_errors)}")
    print(f"  accounted:          {accounted}")
    print(f"  input pairs:        {len(pairs)}")
    if accounted != len(pairs):
        print(
            f"  WARNING: {len(pairs) - accounted} pair(s) unaccounted "
            "for. Re-run primary to retry, or inspect sidecars."
        )
    print(f"  verdicts:           {summary['verdicts']}")
    print(f"  confidence:         {summary['confidence']}")
    print(f"  total spend so far: ${total_spend():.4f}")
    print("=" * 72)
    print("Next: read summary, then run `crosscheck --confirm-primary`.")

    _write_meta_phase(
        "primary",
        {
            "status": "complete" if accounted == len(pairs) else "partial",
            "model": SONNET,
            "count": len(labeled),
            "verdicts": summary["verdicts"],
            "confidence": summary["confidence"],
            "spend_usd": round(total_spend(), 6),
            "prompt_sha": file_sha256(PROMPT_PATH),
            "output_sha": file_sha256(LABELED_PATH) if LABELED_PATH.exists() else None,
            "completed_at": dt.datetime.now(tz=dt.UTC).isoformat(),
            "pricing_as_of": PRICING_AS_OF,
        },
    )
    return 0


# -----------------------------------------------------------------------------
# Phase: crosscheck
# -----------------------------------------------------------------------------


def run_crosscheck_phase(args: argparse.Namespace) -> int:
    """500-pair triangulation via GPT + DeepSeek."""
    _ensure_dirs()

    if not args.confirm_primary:
        logger.error("crosscheck requires --confirm-primary")
        return 1

    meta = _read_existing_meta()
    primary = meta.get("primary") or {}
    if primary.get("status") != "complete":
        raise GateError(
            f"primary phase status is {primary.get('status')!r} (not 'complete'). "
            f"Re-run primary to completion first."
        )

    current_prompt_sha = file_sha256(PROMPT_PATH)
    if primary.get("prompt_sha") != current_prompt_sha:
        raise GateError(
            "prompt SHA changed since primary completed; cross-check would "
            "use a different prompt than primary, invalidating triangulation. "
            "Either revert the prompt or re-run primary."
        )

    # Integrity check: file SHA must match either primary's or
    # crosscheck's recorded SHA (whichever is more recent). This way
    # re-running crosscheck after a prior merge still passes the
    # check, but out-of-band edits between phases are caught.
    if LABELED_PATH.exists():
        expected_shas = [
            (meta.get("crosscheck") or {}).get("output_sha"),
            primary.get("output_sha"),
        ]
        expected_shas = [s for s in expected_shas if s]
        if expected_shas:
            actual = file_sha256(LABELED_PATH)
            if actual not in expected_shas:
                raise GateError(
                    "labeled_pairs.jsonl SHA matches neither primary nor "
                    "crosscheck recorded SHAs (out-of-band edit?). "
                    "Re-verify before running cross-check."
                )

    template = load_prompt_template()
    labeled = list(jsonl_read(LABELED_PATH))
    sample = stratified_sample(labeled, CROSSCHECK_TARGETS, seed=args.seed)
    if args.limit:
        sample = sample[: args.limit]
    sample_by_id = {p["pair_id"]: p for p in sample}

    # Resume: load any pair_ids already cross-checked.
    done_ids: set[str] = set()
    if CROSSCHECK_RESULTS_PATH.exists():
        for r in jsonl_read(CROSSCHECK_RESULTS_PATH):
            done_ids.add(str(r["pair_id"]))

    todo = [p for p in sample if p["pair_id"] not in done_ids]
    logger.info(
        "Cross-check: target %d pairs; %d already done; %d to run.",
        len(sample),
        len(done_ids),
        len(todo),
    )

    if todo:
        rendered = {p["pair_id"]: render_prompt(template, p) for p in todo}

        gpt_proj = sum(
            compute_cost(
                {
                    "prompt_tokens": estimate_pair_chars(t) // 4,
                    "completion_tokens": OPENAI_MAX_TOKENS // 2,
                },
                GPT_DEFAULT,
                batch=True,
            )
            for t in rendered.values()
        )
        openrouter_proj = sum(
            compute_cost(
                {
                    "prompt_tokens": estimate_pair_chars(t) // 4,
                    "completion_tokens": DEEPSEEK_MAX_TOKENS // 2,
                },
                OPENROUTER_MODEL,
                batch=False,  # OpenRouter has no batch discount
            )
            for t in rendered.values()
        )
        proj = gpt_proj + openrouter_proj
        logger.info(
            "Cross-check projection: GPT ~$%.4f, OpenRouter(%s) ~$%.4f, total ~$%.4f",
            gpt_proj,
            OPENROUTER_MODEL,
            openrouter_proj,
            proj,
        )

        if args.dry_run:
            for pid, t in list(rendered.items())[:2]:
                print(f"--- rendered prompt for {pid} (first 800 chars) ---")
                print(t[:800])
                print("--- end ---")
            return 0

        oai_client = _make_openai_client()
        gpt_model = verify_openai_model(oai_client, GPT_DEFAULT)

        confirm_cost_or_exit(
            proj,
            f"cross-check ({len(todo)} pairs × GPT + OpenRouter)",
            yes=args.yes,
            dry_run=False,
        )
        check_budget(total_spend(), proj)

        # GPT (OpenAI batch, resumable) is our primary cross-check
        # signal. OpenRouter is the third labeler — wrapped in
        # try/except so a verify or fan-out failure degrades the
        # cross-check to GPT-only triangulation rather than killing
        # the whole phase. The disagreement field handles the
        # "third labeler missing" case gracefully (treats it as a
        # 2-labeler comparison).
        gpt_results = _run_openai_crosscheck(oai_client, todo, rendered, gpt_model)

        deepseek_model: str | None = None
        deepseek_results: list[BatchResult] = []
        try:
            or_client = _make_openrouter_async_client()
            deepseek_model = verify_openrouter_model(
                os.environ["OPENROUTER_API_KEY"], OPENROUTER_MODEL
            )
            deepseek_results = asyncio.run(
                _run_openrouter_crosscheck(or_client, todo, rendered, deepseek_model)
            )
        except Exception as exc:  # noqa: BLE001 — degrade gracefully
            logger.warning(
                "OpenRouter cross-check failed (%r); proceeding with "
                "slot-B-only triangulation. Pairs will have "
                "crosscheck_verdict_c=null and disagreement computed "
                "over sonnet + slot B only.",
                exc,
            )

        gpt_by_id = {r.custom_id: r for r in gpt_results}
        deepseek_by_id = {r.custom_id: r for r in deepseek_results}

        for pair in todo:
            pid = pair["pair_id"]
            b_r = gpt_by_id.get(pid)
            c_r = deepseek_by_id.get(pid)
            # Slot B = OpenAI batch (GPT). Slot C = third labeler
            # (currently OpenRouter Qwen, was Together DeepSeek). The
            # field names are provider-neutral; the actual model is
            # recorded in meta.crosscheck.labelers.
            patch: dict[str, Any] = {"pair_id": pid}
            patch["crosscheck_verdict_b"] = None
            patch["crosscheck_reasoning_b"] = None
            patch["crosscheck_verdict_c"] = None
            patch["crosscheck_reasoning_c"] = None

            if b_r is not None and b_r.status == "succeeded":
                try:
                    parsed = parse_model_output(b_r.text or "")
                    patch["crosscheck_verdict_b"] = parsed["verdict"]
                    patch["crosscheck_reasoning_b"] = parsed["reasoning"]
                except ParseError as exc:
                    jsonl_append(
                        PARSE_ERRORS_PATH,
                        {
                            "pair_id": pid,
                            "phase": "crosscheck",
                            "slot": "b",
                            "model": gpt_model,
                            "raw_text": b_r.text or "",
                            "error": str(exc),
                        },
                    )
            if c_r is not None and c_r.status == "succeeded":
                try:
                    parsed = parse_model_output(c_r.text or "")
                    patch["crosscheck_verdict_c"] = parsed["verdict"]
                    patch["crosscheck_reasoning_c"] = parsed["reasoning"]
                except ParseError as exc:
                    jsonl_append(
                        PARSE_ERRORS_PATH,
                        {
                            "pair_id": pid,
                            "phase": "crosscheck",
                            "slot": "c",
                            "model": deepseek_model,
                            "raw_text": c_r.text or "",
                            "error": str(exc),
                        },
                    )

            sonnet_v = sample_by_id[pid].get("sonnet_verdict")
            # Compute disagreement over the LABELERS THAT FIRED.
            # The pure disagreement() helper returns None if any
            # verdict is missing — too strict when slot C fails (or
            # is skipped). Here we count it as a disagreement iff
            # ≥ 2 non-null verdicts disagree.
            non_null = [
                v
                for v in (
                    sonnet_v,
                    patch["crosscheck_verdict_b"],
                    patch["crosscheck_verdict_c"],
                )
                if v is not None
            ]
            patch["disagreement"] = (
                None if len(non_null) < 2 else len(set(non_null)) > 1
            )
            jsonl_append(CROSSCHECK_RESULTS_PATH, patch)

    if not CROSSCHECK_RESULTS_PATH.exists():
        logger.error("no crosscheck_results.jsonl produced; aborting merge.")
        return 1

    merge_stats = merge_jsonl_patches(LABELED_PATH, CROSSCHECK_RESULTS_PATH, "pair_id")
    logger.info("Merge stats: %s", merge_stats)
    CROSSCHECK_RESULTS_PATH.unlink(missing_ok=True)

    final = list(jsonl_read(LABELED_PATH))
    n_b = sum(1 for r in final if r.get("crosscheck_verdict_b") is not None)
    n_c = sum(1 for r in final if r.get("crosscheck_verdict_c") is not None)
    n_disagreement = sum(1 for r in final if r.get("disagreement") is True)

    print("=" * 72)
    print("CROSS-CHECK COMPLETE")
    print(f"  records with crosscheck_verdict_b (slot B):  {n_b}")
    print(f"  records with crosscheck_verdict_c (slot C):  {n_c}")
    print(f"  records flagged disagreement:                {n_disagreement}")
    print(f"  total spend so far:                          ${total_spend():.4f}")
    print("=" * 72)

    _write_meta_phase(
        "crosscheck",
        {
            "status": "complete",
            "n_crosscheck_b": n_b,
            "n_crosscheck_c": n_c,
            "n_disagreement": n_disagreement,
            "spend_usd": round(total_spend(), 6),
            "completed_at": dt.datetime.now(tz=dt.UTC).isoformat(),
            "output_sha": file_sha256(LABELED_PATH),
            "pricing_as_of": PRICING_AS_OF,
            # Provider-resolved model identity for each slot — this
            # is the canonical place to look up "what is _b/_c?"
            "labelers": {
                "b": {"role": "openai_batch", "model": gpt_model},
                "c": {
                    "role": "openrouter_async" if deepseek_model else "skipped",
                    "model": deepseek_model,
                },
            },
        },
    )
    return 0


OPENAI_BATCH_CHUNK_SIZE = 250
"""Pairs per OpenAI batch chunk.

OpenAI's per-org enqueued-token limit is ~900k. Each request is
~1,775 input + 1,024 max_tokens reserved = ~2,800 enqueued tokens,
so 250 pairs ≈ 700k enqueued — comfortably under the cap. Multiple
chunks are submitted SEQUENTIALLY, waiting for each to leave the
queue before submitting the next.
"""


def _run_openai_crosscheck(
    client: Any,
    pairs: list[dict[str, Any]],
    rendered: dict[str, str],
    model: str,
) -> list[BatchResult]:
    """Submit GPT crosscheck via OpenAI Batch API, chunked.

    OpenAI's batch API caps enqueued tokens per org at ~900k; a 500-
    pair single batch overshoots that. We chunk into ``OPENAI_BATCH_CHUNK_SIZE``-pair
    sub-batches and submit them sequentially. Each chunk gets its own
    key in ``.batches.json`` so a crash mid-fanout can resume.

    Caching: OpenAI auto-applies prompt caching for prompts ≥ 1,024
    tokens; cached_tokens come back nested under
    ``usage.prompt_tokens_details.cached_tokens`` and are picked up
    by ``compute_cost``.
    """
    chunks = [
        pairs[i : i + OPENAI_BATCH_CHUNK_SIZE]
        for i in range(0, len(pairs), OPENAI_BATCH_CHUNK_SIZE)
    ]
    logger.info(
        "OpenAI crosscheck: %d pairs split into %d chunk(s) of up to %d each.",
        len(pairs),
        len(chunks),
        OPENAI_BATCH_CHUNK_SIZE,
    )

    all_results: list[BatchResult] = []
    usage_totals: dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
    }

    state = _read_batch_state()

    for idx, chunk in enumerate(chunks):
        phase_key = f"crosscheck_gpt_chunk_{idx:03d}"
        existing = state.get(phase_key) or {}
        resumed = False
        # Resume path: if a completed batch from a previous run is on
        # disk, re-fetch instead of re-submitting (saves real money).
        if existing.get("status") == "completed" and existing.get("batch_id"):
            batch_id = existing["batch_id"]
            resumed = True
            logger.info(
                "OpenAI chunk %d/%d: re-using completed batch %s (resume).",
                idx + 1,
                len(chunks),
                batch_id,
            )
        else:
            lines = [
                json.dumps(
                    {
                        "custom_id": p["pair_id"],
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model,
                            "messages": [
                                {"role": "user", "content": rendered[p["pair_id"]]}
                            ],
                            # GPT-5 family uses max_completion_tokens;
                            # legacy max_tokens param is rejected.
                            "max_completion_tokens": OPENAI_MAX_TOKENS,
                        },
                    }
                )
                for p in chunk
            ]
            requests_jsonl = "\n".join(lines) + "\n"
            logger.info(
                "OpenAI chunk %d/%d: submitting %d pairs.",
                idx + 1,
                len(chunks),
                len(chunk),
            )
            batch_id = submit_openai_batch(client, requests_jsonl, model, phase_key)
            final_status = poll_openai_batch(client, batch_id)
            _update_batch_state(phase_key, {"status": final_status})
            if final_status != "completed":
                logger.warning(
                    "OpenAI chunk batch %s ended with status=%s",
                    batch_id,
                    final_status,
                )
        chunk_results = list(fetch_openai_results(client, batch_id))
        for r in chunk_results:
            if r.usage:
                usage_totals["prompt_tokens"] += int(
                    r.usage.get("prompt_tokens", 0) or 0
                )
                usage_totals["completion_tokens"] += int(
                    r.usage.get("completion_tokens", 0) or 0
                )
                details = r.usage.get("prompt_tokens_details") or {}
                usage_totals["cached_tokens"] += int(
                    details.get("cached_tokens", r.usage.get("cached_tokens", 0)) or 0
                )
        all_results.extend(chunk_results)
        chunk_cost = compute_cost(
            {
                "prompt_tokens": sum(
                    int((r.usage or {}).get("prompt_tokens", 0) or 0)
                    for r in chunk_results
                ),
                "completion_tokens": sum(
                    int((r.usage or {}).get("completion_tokens", 0) or 0)
                    for r in chunk_results
                ),
                "cached_tokens": sum(
                    int(
                        ((r.usage or {}).get("prompt_tokens_details") or {}).get(
                            "cached_tokens",
                            (r.usage or {}).get("cached_tokens", 0),
                        )
                        or 0
                    )
                    for r in chunk_results
                ),
            },
            model,
            batch=True,
        )
        if not resumed:
            # Only ledger if we actually paid this run; resumed chunks
            # were already billed when first submitted.
            record_cost(
                phase_key,
                model,
                batch_id,
                usage_totals,
                len(chunk_results),
                chunk_cost,
            )
        else:
            logger.info(
                "OpenAI chunk %d/%d: skipping ledger entry (resumed batch).",
                idx + 1,
                len(chunks),
            )

    return all_results


async def _run_openrouter_crosscheck(
    client: Any,
    pairs: list[dict[str, Any]],
    rendered: dict[str, str],
    model: str,
) -> list[BatchResult]:
    """Async fan-out via OpenRouter (no batch API).

    Reuses the existing ``call_openai_one`` helper since OpenRouter
    is OpenAI-compatible. Concurrency is bounded by
    ``OPENAI_CONCURRENCY``. There's no batch discount or prompt
    caching here, but OpenRouter's per-request latency is predictable
    (typically seconds) so 500 calls finish in ~5-10 minutes.
    """
    sem = asyncio.Semaphore(OPENAI_CONCURRENCY)
    tasks = [
        call_openai_one(client, p, rendered[p["pair_id"]], model, sem) for p in pairs
    ]
    results: list[BatchResult] = []
    usage_totals: dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
    }
    n = 0
    for coro in asyncio.as_completed(tasks):
        r = await coro
        results.append(r)
        n += 1
        if r.usage:
            usage_totals["prompt_tokens"] += int(r.usage.get("prompt_tokens", 0) or 0)
            usage_totals["completion_tokens"] += int(
                r.usage.get("completion_tokens", 0) or 0
            )
            details = r.usage.get("prompt_tokens_details") or {}
            usage_totals["cached_tokens"] += int(
                details.get("cached_tokens", r.usage.get("cached_tokens", 0)) or 0
            )
    cost = compute_cost(usage_totals, model, batch=False)
    record_cost("crosscheck_openrouter", model, None, usage_totals, n, cost)
    return results


def _run_together_crosscheck(
    client: Any,
    pairs: list[dict[str, Any]],
    rendered: dict[str, str],
    model: str,
) -> list[BatchResult]:
    # Together's batch validator requires the full OpenAI-compatible
    # shape (method + url + body). Without method/url it errors out
    # with "Could not detect a format for the line".
    lines = []
    for p in pairs:
        lines.append(
            json.dumps(
                {
                    "custom_id": p["pair_id"],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [
                            {"role": "user", "content": rendered[p["pair_id"]]}
                        ],
                        "max_tokens": DEEPSEEK_MAX_TOKENS,
                    },
                }
            )
        )
    requests_jsonl = "\n".join(lines) + "\n"

    state = _read_batch_state()
    existing = state.get("crosscheck_deepseek") or {}
    resumed = False
    if existing.get("status") == "completed" and existing.get("batch_id"):
        batch_id = existing["batch_id"]
        resumed = True
        logger.info("Together: re-using completed batch %s (resume).", batch_id)
    else:
        batch_id = submit_together_batch(
            client, requests_jsonl, model, "crosscheck_deepseek"
        )
        final_status = poll_together_batch(client, batch_id)
        _update_batch_state("crosscheck_deepseek", {"status": final_status})
        if final_status != "completed":
            logger.warning(
                "Together batch %s ended with status=%s", batch_id, final_status
            )

    results = list(fetch_together_results(client, batch_id))
    usage_totals: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
    for r in results:
        if r.usage:
            for k in usage_totals:
                usage_totals[k] += int(r.usage.get(k, 0) or 0)
    if not resumed:
        cost = compute_cost(usage_totals, model, batch=True)
        record_cost(
            "crosscheck_deepseek", model, batch_id, usage_totals, len(results), cost
        )
    else:
        logger.info("Together: skipping ledger entry (resumed batch).")
    return results


# -----------------------------------------------------------------------------
# Phase: retry-parse-errors
# -----------------------------------------------------------------------------


def run_retry_parse_errors(args: argparse.Namespace) -> int:
    """Re-attempt pairs whose model output failed parsing."""
    _ensure_dirs()
    phase = args.target_phase
    if not PARSE_ERRORS_PATH.exists():
        logger.info("No parse errors to retry.")
        return 0
    errors = list(jsonl_read(PARSE_ERRORS_PATH))
    relevant = [e for e in errors if e.get("phase") == phase]
    if not relevant:
        logger.info("No parse errors for phase=%s.", phase)
        return 0

    logger.info(
        "Re-attempting %d parse-failed pair(s) for phase=%s.", len(relevant), phase
    )

    pairs = list(jsonl_read(INPUT_PATH))
    by_id = {p["pair_id"]: p for p in pairs}
    todo = [by_id[e["pair_id"]] for e in relevant if e["pair_id"] in by_id]

    if phase == "primary":
        client = _make_anthropic_client()
        template = load_prompt_template(
            count_tokens=lambda prefix, model: count_prefix_tokens_anthropic(
                client, prefix, model
            )
        )
        static_system = (
            "You are a careful labeler for a bias-evaluation training dataset."
        )
        rendered = [render_prompt(template, p) for p in todo]
        proj = project_cost_anthropic(rendered, SONNET, batch=True)
        confirm_cost_or_exit(
            proj,
            f"retry-parse-errors primary ({len(todo)} pairs)",
            yes=args.yes,
            dry_run=False,
        )
        check_budget(total_spend(), proj)
        verify_anthropic_models(client, [SONNET])
        requests = [
            build_anthropic_request(p, static_system, template, SONNET) for p in todo
        ]
        batch_id = submit_anthropic_batch(client, requests, SONNET, "retry_primary")
        poll_anthropic_batch(client, batch_id)
        _update_batch_state("retry_primary", {"status": "ended"})
        pairs_by_id = {p["pair_id"]: p for p in todo}
        results = fetch_anthropic_results(client, batch_id)
        counts = _process_anthropic_results(
            results,
            pairs_by_id,
            phase="retry_primary",
            model=SONNET,
            output_path=LABELED_PATH,
        )
        logger.info("Retry counts: %s", counts)
        # Remove successfully-retried pair_ids from .parse_errors.jsonl.
        labeled_ids = already_processed(LABELED_PATH, "pair_id")
        remaining = [
            e
            for e in errors
            if e["pair_id"] not in labeled_ids or e.get("phase") != phase
        ]
        atomic_write_json_lines(PARSE_ERRORS_PATH, remaining)
        return 0

    logger.error("retry for phase=%r not yet implemented", phase)
    return 1


def atomic_write_json_lines(path: Path, records: list[dict[str, Any]]) -> None:
    """Local helper: atomically rewrite a JSONL file."""
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with tmp.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False))
            fh.write("\n")
    tmp.replace(path)


# -----------------------------------------------------------------------------
# Phase: status
# -----------------------------------------------------------------------------


def run_status(_: argparse.Namespace) -> int:
    """Print sidecar state and ledger summary."""
    print("=" * 72)
    print("STATUS")
    print("=" * 72)
    print(f"Spend ledger total: ${total_spend():.4f} (cap ${BUDGET_USD_HARD:.2f})")
    state = _read_batch_state()
    print(f"Batch state ({len(state)} entries):")
    for k, v in state.items():
        print(
            f"  {k}: model={v.get('model')} status={v.get('status')} batch_id={v.get('batch_id')}"
        )
    meta = _read_existing_meta()
    print(f"Meta phases: {sorted(meta.keys())}")
    for path, name in (
        (LABELED_PATH, "labeled_pairs.jsonl"),
        (PARSE_ERRORS_PATH, ".parse_errors.jsonl"),
        (REFUSALS_PATH, ".refusals.jsonl"),
        (BATCH_ERRORS_PATH, ".batch_errors.jsonl"),
    ):
        if path.exists():
            n = sum(1 for _ in jsonl_read(path))
            print(f"  {name}: {n} records")
        else:
            print(f"  {name}: (not present)")
    return 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 4 — Claude labeling driver (Sonnet primary + cross-check)."
    )
    sub = parser.add_subparsers(dest="phase", required=True)

    common_kwargs = dict(action="store_true")

    p_dry = sub.add_parser("dryrun", help="50-pair Sonnet vs Opus comparison.")
    p_dry.add_argument("--seed", type=int, default=42)
    p_dry.add_argument("--limit", type=int, default=None)
    p_dry.add_argument("--dry-run", **common_kwargs)
    p_dry.add_argument("--yes", **common_kwargs)

    p_pri = sub.add_parser("primary", help="Sonnet labels all 1,938 pairs.")
    p_pri.add_argument("--confirm-dryrun", **common_kwargs)
    p_pri.add_argument("--limit", type=int, default=None)
    p_pri.add_argument("--dry-run", **common_kwargs)
    p_pri.add_argument("--yes", **common_kwargs)
    p_pri.add_argument("--force-gate", **common_kwargs)
    p_pri.add_argument("--force", **common_kwargs)
    p_pri.add_argument("--force-and-discard-crosscheck", **common_kwargs)

    p_cc = sub.add_parser("crosscheck", help="500-pair GPT + DeepSeek triangulation.")
    p_cc.add_argument("--confirm-primary", **common_kwargs)
    p_cc.add_argument("--seed", type=int, default=42)
    p_cc.add_argument("--limit", type=int, default=None)
    p_cc.add_argument("--dry-run", **common_kwargs)
    p_cc.add_argument("--yes", **common_kwargs)

    p_retry = sub.add_parser(
        "retry-parse-errors", help="Re-attempt pairs whose output failed parsing."
    )
    p_retry.add_argument("target_phase", choices=("primary", "crosscheck"))
    p_retry.add_argument("--yes", **common_kwargs)

    sub.add_parser("status", help="Print ledger + batch state summary.")
    return parser


def main(argv: list[str] | None = None) -> int:
    from scripts.common import load_env

    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_env()

    if not INPUT_PATH.exists():
        logger.error("%s not found. Run data/03a_holdout_eval.py first.", INPUT_PATH)
        return 2

    handler = {
        "dryrun": run_dryrun_phase,
        "primary": run_primary_phase,
        "crosscheck": run_crosscheck_phase,
        "retry-parse-errors": run_retry_parse_errors,
        "status": run_status,
    }[args.phase]
    try:
        return handler(args)
    except KeyboardInterrupt:
        logger.warning(
            "SIGINT received; in-flight batch state preserved in .batches.json."
        )
        return 130
    except (PromptError, GateError) as exc:
        logger.error("%s", exc)
        return 2 if isinstance(exc, PromptError) else 1
    except BudgetExceededError as exc:
        logger.error("%s", exc)
        return 4
    except ModelNotFoundError as exc:
        logger.error("%s", exc)
        return 5


if __name__ == "__main__":
    sys.exit(main())
