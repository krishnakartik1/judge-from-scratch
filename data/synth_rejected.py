"""Synthesize DPO rejected responses via the Anthropic Batch API.

Mirrors Stage 4's batch lifecycle (submit → poll → fetch → resume
via ``.batches.json``) but produces "bad rationalizations" instead
of bias judgments. Each call asks Sonnet 4.6 to land on the WRONG
verdict using one of four failure modes (round-robined across
pairs for reproducibility).

Public surface used by ``data/05_format_datasets.py``:

- ``MODEL`` — module constant for cross-checking against Stage 4
- ``FAILURE_MODES`` — the four failure-mode tags
- ``failure_mode_for(idx)`` — round-robin selector
- ``build_synth_request(pair, ...)`` — one Batch API request entry
- ``validate_synth_output(text, expected_verdict)`` — regex parse
- ``compute_synth_cost(usage)`` — Batch-discounted USD per call
- ``submit_batch`` / ``poll_batch`` / ``fetch_batch`` — lifecycle
- ``read_batch_state`` / ``write_batch_state`` — sidecar I/O

The synthesis prompt itself lives at module scope for visibility;
edits to the prompt invalidate Anthropic's cache breakpoint and
should be flagged in the cost ledger.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _format_helpers import (  # sibling module in data/
    VALID_VERDICTS,
    flip_verdict,
    format_answer_choices,
)

from scripts.common import atomic_write_json

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

MODEL = "claude-sonnet-4-6"

# Batch API: Anthropic discounts input/output by 50%.
BATCH_DISCOUNT = 0.5

# Per-million-token USD pricing for Sonnet 4.6 (mirrors Stage 4).
PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {
        "input": 3.0,
        "cached_input": 0.30,
        "output": 15.0,
    },
}

ANTHROPIC_MAX_TOKENS = 350  # ~120 reasoning + verdict tag + slack

POLL_SLEEP_SEC = 30
POLL_MAX_CONSECUTIVE_5XX = 5
SUBMIT_RETRY_ATTEMPTS = 3

# Validation regex for synthesized output. DOTALL so multi-line
# reasoning matches; ANCHORED so trailing junk fails.
_OUTPUT_RE = re.compile(
    r"^\s*<reasoning>(?P<reasoning>.+?)</reasoning>\s*"
    r"<verdict>(?P<verdict>A|B|TIE)</verdict>\s*$",
    re.DOTALL,
)


FAILURE_MODES: tuple[str, ...] = (
    "verbose_hedging",
    "surface_engagement",
    "stereotype_aligned",
    "length_burying",
)

_FAILURE_MODE_DIRECTIVES: dict[str, str] = {
    "verbose_hedging": (
        "Dilute the analysis with caveats and qualifiers that obscure "
        "the conclusion. Sound thoughtful but reach a flimsy verdict."
    ),
    "surface_engagement": (
        "Engage with the question's surface (who said what) without "
        "analyzing the reasoning chain or stereotype reliance."
    ),
    "stereotype_aligned": (
        "Invoke a demographic stereotype confidently as the basis for "
        "the wrong verdict. Reasoning sounds plausible but rests on "
        "an unsupported group-level assumption."
    ),
    "length_burying": (
        "Write a longer-than-needed response that buries the verdict "
        "inside meandering analysis. Stay under 120 reasoning tokens."
    ),
}


SYNTH_SYSTEM_PROMPT = """\
You are generating HARD NEGATIVE training data for a bias-judge model.
Given a question, two responses, and the CORRECT verdict, write a
plausible-sounding but WRONG analysis that lands on a different
verdict. The output is a rejected response in a DPO pair — its job
is to teach the trained judge to discriminate against this exact
failure mode.

Output format (exact, no extra text):
<reasoning>...2-5 sentences exhibiting the specified failure mode...</reasoning>
<verdict>WRONG_VERDICT</verdict>

Failure modes:
- verbose_hedging: dilute the analysis with caveats and qualifiers
  that obscure the conclusion.
- surface_engagement: respond to the question's surface (who said
  what) without analyzing the reasoning chain or stereotype reliance.
- stereotype_aligned: invoke a stereotype confidently as the basis
  for the wrong verdict.
- length_burying: write a longer-than-needed response that buries
  the verdict inside meandering analysis.

Stay under ~120 reasoning tokens regardless of failure mode (length
parity matters — verbose != overlong).

The "WRONG_VERDICT" you must pick is provided per request. NEVER
emit the correct verdict. NEVER add preambles, disclaimers, or text
outside the two tags. NEVER emit <think>, <thinking>, or
<|think|> blocks.\
"""


_USER_TEMPLATE = """\
Failure mode: {failure_mode}
{directive}

Correct verdict (do NOT pick this): {sonnet_verdict}
You must pick (the WRONG verdict): {wrong_verdict}

**Question:** {question_text}

**Answer choices:**
{answer_choices_block}

**Response A:** {response_a_text}

**Response B:** {response_b_text}\
"""


_RETRY_SUFFIX = (
    "\n\nCRITICAL: output must be exactly "
    "<reasoning>...</reasoning><verdict>{wrong_verdict}</verdict>. "
    "The verdict MUST be {wrong_verdict}, not {sonnet_verdict}. "
    "No preamble, no thinking blocks, no explanation outside the tags."
)


# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------


class SynthParseError(ValueError):
    """Synthesized output failed regex / verdict validation."""


# -----------------------------------------------------------------------------
# BatchResult (mirrors Stage 4)
# -----------------------------------------------------------------------------


@dataclass
class SynthBatchResult:
    """One row from an Anthropic synth batch results stream."""

    custom_id: str
    status: str  # "succeeded" | "errored" | "expired" | "canceled"
    text: str | None = None
    usage: dict[str, Any] | None = None
    error: str | None = None
    stop_reason: str | None = None


# -----------------------------------------------------------------------------
# Pure helpers
# -----------------------------------------------------------------------------


def failure_mode_for(idx: int) -> str:
    """Round-robin failure mode selection by sha1-sorted index."""
    return FAILURE_MODES[idx % len(FAILURE_MODES)]


def build_synth_request(
    pair: dict[str, Any],
    *,
    failure_mode: str,
    custom_id: str | None = None,
    retry: bool = False,
) -> dict[str, Any]:
    """Construct one Batch API request entry with cache_control.

    The system block holds the (cached) synthesis instructions; the
    user block holds the per-pair item plus the failure-mode
    directive. ``retry=True`` appends the stricter suffix used in
    the second-pass batch for parse-error recovery.
    """
    if failure_mode not in FAILURE_MODES:
        raise ValueError(
            f"unknown failure_mode {failure_mode!r}; "
            f"expected one of {FAILURE_MODES}"
        )
    sonnet_verdict = pair["sonnet_verdict"]
    wrong_verdict = flip_verdict(sonnet_verdict)
    user_block = _USER_TEMPLATE.format(
        failure_mode=failure_mode,
        directive=_FAILURE_MODE_DIRECTIVES[failure_mode],
        sonnet_verdict=sonnet_verdict,
        wrong_verdict=wrong_verdict,
        question_text=pair["question_text"],
        answer_choices_block=format_answer_choices(pair["answer_choices"]),
        response_a_text=pair["response_a"]["text"],
        response_b_text=pair["response_b"]["text"],
    )
    if retry:
        user_block += _RETRY_SUFFIX.format(
            wrong_verdict=wrong_verdict,
            sonnet_verdict=sonnet_verdict,
        )

    return {
        "custom_id": custom_id or str(pair["pair_id"]),
        "params": {
            "model": MODEL,
            "max_tokens": ANTHROPIC_MAX_TOKENS,
            "system": [
                {
                    "type": "text",
                    "text": SYNTH_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": user_block}],
        },
    }


def validate_synth_output(text: str, expected_verdict: str) -> dict[str, str]:
    """Parse + validate a synthesized rejected response.

    Returns ``{"reasoning": ..., "verdict": ...}`` on success.
    Raises ``SynthParseError`` if the regex fails OR if the verdict
    doesn't match ``expected_verdict`` (Sonnet sometimes
    self-corrects to the right answer — known failure mode).
    """
    if expected_verdict not in VALID_VERDICTS:
        raise ValueError(
            f"expected_verdict {expected_verdict!r} not in {VALID_VERDICTS}"
        )
    m = _OUTPUT_RE.match(text or "")
    if m is None:
        raise SynthParseError(f"synth output did not match regex: {text[:200]!r}")
    verdict = m.group("verdict")
    if verdict != expected_verdict:
        raise SynthParseError(
            f"synth verdict {verdict!r} != expected {expected_verdict!r} "
            f"(model self-corrected to the right answer)"
        )
    reasoning = m.group("reasoning").strip()
    if not reasoning:
        raise SynthParseError("empty reasoning block")
    return {"reasoning": reasoning, "verdict": verdict}


def compute_synth_cost(usage: dict[str, Any] | None) -> float:
    """USD cost for one Sonnet 4.6 Batch call from a usage dict."""
    if not usage:
        return 0.0
    pricing = PRICING[MODEL]
    in_tok = int(usage.get("input_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_create = int(usage.get("cache_creation_input_tokens") or 0)
    out_tok = int(usage.get("output_tokens") or 0)
    # Anthropic's `input_tokens` excludes cached/cache-creation; sum
    # to get the billable input total before the discount.
    fresh_input_cost = in_tok / 1_000_000 * pricing["input"]
    cache_read_cost = cache_read / 1_000_000 * pricing["cached_input"]
    cache_create_cost = cache_create / 1_000_000 * pricing["input"] * 1.25
    output_cost = out_tok / 1_000_000 * pricing["output"]
    raw = fresh_input_cost + cache_read_cost + cache_create_cost + output_cost
    return raw * BATCH_DISCOUNT


def project_synth_cost(n_requests: int, system_tokens: int = 280) -> float:
    """Conservative cost projection for ``n_requests`` synth calls.

    Assumes:
      - System block is ~280 tokens (the cached prefix), cache-read
        on all but the first call of each cache window. We treat
        this as a single cache-creation + (n-1) cache-reads.
      - User block ~1.5k tokens average (pair + directive).
      - Output ~250 tokens average (max 350).
    """
    if n_requests <= 0:
        return 0.0
    pricing = PRICING[MODEL]
    user_tokens = 1500
    output_tokens = 250
    cache_create_cost = system_tokens / 1_000_000 * pricing["input"] * 1.25
    cache_read_cost_per = system_tokens / 1_000_000 * pricing["cached_input"]
    fresh_per = user_tokens / 1_000_000 * pricing["input"]
    out_per = output_tokens / 1_000_000 * pricing["output"]
    per_call_after_first = cache_read_cost_per + fresh_per + out_per
    raw = (
        cache_create_cost
        + fresh_per
        + out_per
        + per_call_after_first * (n_requests - 1)
    )
    return raw * BATCH_DISCOUNT


# -----------------------------------------------------------------------------
# Batch state sidecar
# -----------------------------------------------------------------------------


def read_batch_state(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_batch_state(path: Path, state: dict[str, dict[str, Any]]) -> None:
    atomic_write_json(path, state)


def update_batch_state(path: Path, phase_key: str, patch: dict[str, Any]) -> None:
    state = read_batch_state(path)
    entry = state.get(phase_key, {})
    entry.update(patch)
    state[phase_key] = entry
    write_batch_state(path, state)


# -----------------------------------------------------------------------------
# Anthropic Batch lifecycle (mirrors Stage 4)
# -----------------------------------------------------------------------------


def submit_batch(
    client: Any,
    requests: list[dict[str, Any]],
    *,
    state_path: Path,
    phase_key: str,
) -> str:
    """Submit a batch; persist batch_id; return it.

    Retries 5xx with exponential backoff. 4xx aborts immediately.
    """
    last_exc: Exception | None = None
    for attempt in range(SUBMIT_RETRY_ATTEMPTS):
        try:
            batch = client.messages.batches.create(requests=requests)
            batch_id = batch.id
            update_batch_state(
                state_path,
                phase_key,
                {
                    "batch_id": batch_id,
                    "model": MODEL,
                    "provider": "anthropic",
                    "n_requests": len(requests),
                    "submitted_at": dt.datetime.now(tz=dt.UTC).isoformat(),
                    "status": "in_progress",
                },
            )
            logger.info(
                "Submitted synth batch %s (%d requests, phase=%s).",
                batch_id,
                len(requests),
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
                "submit_batch attempt %d failed: %r; retry in %ds",
                attempt + 1,
                exc,
                backoff,
            )
            time.sleep(backoff)
    raise RuntimeError(
        f"submit_batch failed after {SUBMIT_RETRY_ATTEMPTS} attempts"
    ) from last_exc


def poll_batch(
    client: Any,
    batch_id: str,
    *,
    state_path: Path,
    phase_key: str,
    sleep_s: float = POLL_SLEEP_SEC,
    once: bool = False,
) -> str:
    """Poll until terminal (or once if ``once=True``); return status.

    Status values per Anthropic API: ``in_progress``, ``ended``,
    ``canceled``, ``expired``. We treat any of the last three as
    terminal and return immediately.
    """
    consecutive_errors = 0
    while True:
        try:
            batch = client.messages.batches.retrieve(batch_id)
            consecutive_errors = 0
        except Exception as exc:  # noqa: BLE001
            consecutive_errors += 1
            logger.warning(
                "poll_batch transient error (%d/%d): %r",
                consecutive_errors,
                POLL_MAX_CONSECUTIVE_5XX,
                exc,
            )
            if consecutive_errors >= POLL_MAX_CONSECUTIVE_5XX:
                raise RuntimeError(
                    f"poll_batch: {POLL_MAX_CONSECUTIVE_5XX} consecutive "
                    f"errors polling {batch_id}"
                ) from exc
            time.sleep(sleep_s)
            continue
        status = getattr(batch, "processing_status", None) or getattr(
            batch, "status", "unknown"
        )
        request_counts = getattr(batch, "request_counts", None)
        update_batch_state(state_path, phase_key, {"status": status})
        logger.info("Batch %s status=%s counts=%s", batch_id, status, request_counts)
        if status in ("ended", "canceled", "expired"):
            return status
        if once:
            return status
        time.sleep(sleep_s)


def fetch_batch_results(client: Any, batch_id: str) -> Iterator[SynthBatchResult]:
    """Stream parsed ``SynthBatchResult`` rows from an ended batch."""
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
                    if getattr(block, "type", None) == "text":
                        parts.append(getattr(block, "text", "") or "")
                text = "".join(parts)
            usage_obj = getattr(message, "usage", None) if message else None
            usage = _usage_to_dict(usage_obj)
            stop_reason = getattr(message, "stop_reason", None) if message else None
            yield SynthBatchResult(
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
            yield SynthBatchResult(
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
