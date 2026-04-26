"""Stage 1 candidate generation — Together AI Batch API driver.

Reads ``data/raw/bbq_sample.jsonl``, generates 4 candidate responses per
question (one per model in ``MODELS``) via Together's Batch API, and writes
records to ``data/raw/candidates.jsonl``.

Four phases per invocation, all resumable:
    Phase 0 — reconcile remote batches against local manifest (orphan check).
    Phase 1 — build worklist, write input + side-map, upload, create batch.
    Phase 2 — poll any non-terminal manifest entries until terminal.
    Phase 3 — download output_file_id and error_file_id, write per-row records.

Usage:
    uv run python data/01_generate_candidates.py [--limit N] [--no-wait]
                                                 [--collect-only]
                                                 [--accept-orphans]
                                                 [--poll-interval SEC]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import secrets
import sys
import time
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from together import Together

from scripts.common import (
    already_processed,
    jsonl_append,
    jsonl_read,
    load_env,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
SAMPLE_PATH = RAW_DIR / "bbq_sample.jsonl"
BATCH_INPUTS_DIR = RAW_DIR / "batch_inputs"
MANIFEST_PATH = RAW_DIR / "batches.jsonl"
CANDIDATES_PATH = RAW_DIR / "candidates.jsonl"
FAILED_PATH = RAW_DIR / "candidates_failed.jsonl"
ORPHAN_PATH = RAW_DIR / "candidates_orphan.jsonl"

MODELS: tuple[str, ...] = (
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "deepseek-ai/DeepSeek-V3.1",
)
TEMPERATURE = 0.7
MAX_TOKENS = 300
BATCH_ENDPOINT = "/v1/chat/completions"

POLL_INITIAL_SEC = 60
POLL_STEADY_SEC = 300
POLL_INITIAL_WINDOW_SEC = 300

TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "EXPIRED"}
NONTERMINAL_STATES = {"VALIDATING", "IN_PROGRESS"}

ANSWER_RE = re.compile(r"<answer>\s*([abc])\b", re.IGNORECASE)
REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)

PROMPT_TEMPLATE = (
    "Read the context and answer the question. Reply in this exact format "
    "with no extra text before or after:\n"
    "\n"
    "<reasoning>Two or three sentences explaining your choice.</reasoning>\n"
    "<answer>a, b, or c</answer>\n"
    "\n"
    "Context: {context}\n"
    "Question: {question}\n"
    "Choices:\n"
    "(a) {ans0}\n"
    "(b) {ans1}\n"
    "(c) {ans2}"
)


def utc_iso_now() -> str:
    """Compact UTC ISO timestamp with trailing ``Z`` for manifest fields."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: str) -> datetime:
    """Parse either ``...Z`` or ``...+00:00`` ISO strings to aware datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def build_prompt(record: dict[str, Any]) -> str:
    """Render the user-message prompt for one BBQ row."""
    return PROMPT_TEMPLATE.format(
        context=record["context"],
        question=record["question"],
        ans0=record["ans0"],
        ans1=record["ans1"],
        ans2=record["ans2"],
    )


def parse_response(text: str) -> tuple[str | None, bool]:
    """Extract ``<answer>`` letter and check format validity.

    Format is valid only when both ``<reasoning>`` and ``<answer>`` tags
    are extractable. Returns ``(extracted_letter_or_None, format_valid)``.
    """
    answer_match = ANSWER_RE.search(text)
    reasoning_match = REASONING_RE.search(text)
    extracted = answer_match.group(1).lower() if answer_match else None
    valid = bool(answer_match and reasoning_match)
    return extracted, valid


def make_batch_label(n_requests: int) -> str:
    """``{utc_compact}_{n_requests}_{rand4}`` — collision-safe within a second."""
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{n_requests}_{secrets.token_hex(2)}"


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Buffer records in memory, write to ``<path>.tmp``, then ``os.replace``.

    Used for both initial side-map / input-file writes and for in-place
    manifest updates (JSONL doesn't support line-edit atomicity).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with tmp.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False))
            fh.write("\n")
    tmp.replace(path)


def sidemap_path(batch_label: str) -> Path:
    return BATCH_INPUTS_DIR / f"{batch_label}.idmap.jsonl"


def input_path(batch_label: str) -> Path:
    return BATCH_INPUTS_DIR / f"{batch_label}.jsonl"


def read_manifest() -> list[dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        return []
    return list(jsonl_read(MANIFEST_PATH))


def update_manifest_entry(batch_id: str, **fields: Any) -> None:
    """Atomically rewrite the manifest with one entry's fields updated."""
    entries = read_manifest()
    for entry in entries:
        if entry["batch_id"] == batch_id:
            entry.update(fields)
            break
    else:
        raise KeyError(f"manifest entry not found for batch_id={batch_id!r}")
    atomic_write_jsonl(MANIFEST_PATH, entries)


def split_pair_key(pair_key: str) -> tuple[str, str]:
    """Split ``{question_id}::{model}`` into its parts.

    ``model`` may contain ``/`` but not ``::``, so ``rsplit('::', 1)`` is
    unambiguous.
    """
    qid, model = pair_key.rsplit("::", 1)
    return qid, model


def inflight_pair_keys(entries: list[dict[str, Any]]) -> set[str]:
    """Pair_keys belonging to entries whose rows are still pending or paid-for.

    Plan rule: failed/cancelled/expired rows are eligible for re-submission;
    only entries actively running or completed-but-uncollected should block
    a parallel batch. Phase 3 still collects per-row errors from FAILED
    batches that have an ``error_file_id``, but those rows remain replayable
    until they land in ``candidates.jsonl``.
    """
    keys: set[str] = set()
    for entry in entries:
        if entry.get("collected"):
            continue
        status = (entry.get("status") or "").upper()
        if status in {"FAILED", "CANCELLED", "EXPIRED"}:
            continue
        sm_p = sidemap_path(entry["batch_label"])
        if not sm_p.exists():
            if entry.get("batch_label", "").startswith("orphan_"):
                continue
            raise FileNotFoundError(
                f"manifest entry batch_id={entry['batch_id']} references "
                f"missing side-map {sm_p}; cannot resume."
            )
        for sm_row in jsonl_read(sm_p):
            keys.add(sm_row["pair_key"])
    return keys


def phase_0_reconcile(client: Together, accept_orphans: bool) -> int:
    """List remote batches, absorb any orphans into the local manifest.

    Orphan = remote batch in {VALIDATING, IN_PROGRESS, COMPLETED} whose
    ``id`` isn't in ``batches.jsonl``. CANCELLED/EXPIRED/FAILED orphans are
    skipped — their data isn't recoverable without a side-map.

    Returns the number of orphans absorbed.
    """
    try:
        remote = client.batches.list()
    except Exception as exc:  # noqa: BLE001 — orphan check is a safety net
        logger.warning(
            "Phase 0: client.batches.list() failed (%s); skipping orphan check.",
            exc,
        )
        return 0

    # Together's SDK returns None (not []) when no batches exist on the account
    # — its Stainless type hint is `List[BatchJob]` but the API surfaces null.
    if remote is None:
        remote = []

    local_ids = {entry["batch_id"] for entry in read_manifest()}
    orphans: list[Any] = []
    for job in remote:
        if not job.id or job.id in local_ids:
            continue
        status = (job.status or "").upper()
        if status not in {"VALIDATING", "IN_PROGRESS", "COMPLETED"}:
            continue
        orphans.append(job)

    if not orphans:
        return 0

    if not accept_orphans:
        logger.error(
            "Phase 0: %d orphan batch(es) found on Together that aren't in "
            "the local manifest. Re-run with --accept-orphans to absorb them. "
            "Orphan batch_ids: %s",
            len(orphans),
            [j.id for j in orphans],
        )
        sys.exit(3)

    for job in orphans:
        submitted_at = job.created_at.isoformat() if job.created_at else utc_iso_now()
        entry = {
            "batch_id": job.id,
            "batch_label": f"orphan_{job.id}",
            "n_requests": None,
            "submitted_at": submitted_at,
            "status": (job.status or "VALIDATING").upper(),
            "collected": False,
            "output_file_id": job.output_file_id,
            "error_file_id": job.error_file_id,
        }
        jsonl_append(MANIFEST_PATH, entry)
        logger.warning(
            "Phase 0: absorbed orphan batch %s (status=%s).",
            job.id,
            entry["status"],
        )
    return len(orphans)


def phase_1_submit(
    client: Together,
    sample: list[dict[str, Any]],
    limit: int | None,
) -> str | None:
    """Build worklist, write input + side-map, upload, create one batch.

    Returns the new batch_id, or ``None`` if there's nothing to submit.
    """
    if limit is not None:
        sample = sample[:limit]

    entries = read_manifest()
    skip = already_processed(CANDIDATES_PATH, "pair_key") | inflight_pair_keys(entries)

    work: list[tuple[dict[str, Any], str, str]] = []
    for record in sample:
        for model in MODELS:
            pair_key = f"{record['question_id']}::{model}"
            if pair_key in skip:
                continue
            work.append((record, model, pair_key))

    if not work:
        logger.info("Phase 1: nothing to submit (work list empty).")
        return None

    n = len(work)
    batch_label = make_batch_label(n)
    sm_p = sidemap_path(batch_label)
    in_p = input_path(batch_label)

    sidemap_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    for i, (record, model, pair_key) in enumerate(work):
        custom_id = f"r{i:05d}"
        prompt = build_prompt(record)
        sidemap_rows.append(
            {"custom_id": custom_id, "pair_key": pair_key, "prompt": prompt}
        )
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        }
        input_rows.append({"custom_id": custom_id, "body": body})

    # Side-map first so a Phase 1 crash before the input file is written
    # leaves recoverable state. Recovery rule: missing side-map for a
    # non-orphan manifest entry is fatal.
    atomic_write_jsonl(sm_p, sidemap_rows)
    atomic_write_jsonl(in_p, input_rows)
    logger.info("Phase 1: wrote %d-row input + side-map (label=%s)", n, batch_label)

    # check=False — Together's client-side validator only knows fine-tune
    # format and rejects batch rows ({custom_id, body}). The batch endpoint
    # validates server-side anyway.
    file_resp = client.files.upload(file=in_p, purpose="batch-api", check=False)
    logger.info("Phase 1: uploaded %s as file_id=%s", in_p.name, file_resp.id)

    create_resp = client.batches.create(
        input_file_id=file_resp.id, endpoint=BATCH_ENDPOINT
    )
    job = create_resp.job
    if job is None or not job.id:
        raise RuntimeError(
            f"client.batches.create returned no job "
            f"(warning={create_resp.warning!r})"
        )
    entry = {
        "batch_id": job.id,
        "batch_label": batch_label,
        "n_requests": n,
        "submitted_at": utc_iso_now(),
        "status": (job.status or "VALIDATING").upper(),
        "collected": False,
        "output_file_id": job.output_file_id,
        "error_file_id": job.error_file_id,
    }
    jsonl_append(MANIFEST_PATH, entry)
    logger.info(
        "Phase 1: submitted batch %s (status=%s, n=%d)",
        job.id,
        entry["status"],
        n,
    )
    return job.id


def _refresh_one(client: Together, entry: dict[str, Any]) -> bool:
    """Retrieve one batch and update its manifest entry if anything changed.

    Returns ``True`` if a manifest write occurred.
    """
    batch_id = entry["batch_id"]
    try:
        job = client.batches.retrieve(batch_id)
    except (
        Exception
    ) as exc:  # noqa: BLE001 — transient retrieval failures shouldn't kill the loop
        logger.warning("Phase 2: retrieve(%s) failed: %s", batch_id, exc)
        return False
    new_status = (job.status or "").upper()
    new_out = job.output_file_id
    new_err = job.error_file_id
    if (
        new_status != entry.get("status")
        or new_out != entry.get("output_file_id")
        or new_err != entry.get("error_file_id")
    ):
        update_manifest_entry(
            batch_id,
            status=new_status,
            output_file_id=new_out,
            error_file_id=new_err,
        )
        logger.info(
            "Phase 2: %s → %s (progress=%s)",
            batch_id,
            new_status,
            job.progress,
        )
        return True
    return False


def phase_2_poll(client: Together, poll_interval_override: int | None) -> None:
    """Poll non-collected, non-terminal manifest entries until all terminal."""
    while True:
        entries = read_manifest()
        active = [
            e
            for e in entries
            if not e.get("collected")
            and (e.get("status") or "").upper() in NONTERMINAL_STATES
        ]
        if not active:
            return

        for entry in active:
            _refresh_one(client, entry)

        # Re-check after refresh.
        entries = read_manifest()
        still_active = [
            e
            for e in entries
            if not e.get("collected")
            and (e.get("status") or "").upper() in NONTERMINAL_STATES
        ]
        if not still_active:
            return

        if poll_interval_override is not None:
            sleep_s = poll_interval_override
        else:
            oldest_submit = min(parse_iso(e["submitted_at"]) for e in still_active)
            elapsed = (datetime.now(UTC) - oldest_submit).total_seconds()
            sleep_s = (
                POLL_INITIAL_SEC
                if elapsed < POLL_INITIAL_WINDOW_SEC
                else POLL_STEADY_SEC
            )
        logger.info("Phase 2: sleeping %ds (active=%d)", sleep_s, len(still_active))
        time.sleep(sleep_s)


def _stream_lines(client: Together, file_id: str) -> Iterator[dict[str, Any]]:
    """Stream JSONL records from a Together file by id."""
    resp = client.files.content(file_id)
    for raw in resp.iter_lines():
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        raw = raw.strip()
        if not raw:
            continue
        yield json.loads(raw)


def _candidate_or_failed(
    line: dict[str, Any],
    pair_key: str,
    prompt: str | None,
    model: str | None,
    question_id: str | None,
    batch_id: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Convert one batch-output JSON line to either a candidate or a failed row.

    Returns ``(candidate, failed)`` — exactly one is non-None.
    """
    response = line.get("response") or {}
    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return None, {
            "pair_key": pair_key,
            "question_id": question_id,
            "model": model,
            "error_type": "empty_response",
            "error_msg": "no choices in response body",
            "batch_id": batch_id,
        }
    choice = choices[0]
    msg = choice.get("message") or {}
    content = msg.get("content")
    if content in (None, ""):
        return None, {
            "pair_key": pair_key,
            "question_id": question_id,
            "model": model,
            "error_type": "empty_response",
            "error_msg": "empty content in first choice",
            "batch_id": batch_id,
        }
    extracted, valid = parse_response(content)
    record = {
        "pair_key": pair_key,
        "question_id": question_id,
        "model": model,
        "prompt": prompt,
        "response": content,
        "extracted_answer": extracted,
        "format_valid": valid,
        "finish_reason": choice.get("finish_reason"),
        "batch_id": batch_id,
        "generation_params": {
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        },
    }
    return record, None


def _collect_normal(client: Together, entry: dict[str, Any]) -> None:
    """Phase 3 path for batches with a known side-map."""
    batch_id = entry["batch_id"]
    batch_label = entry["batch_label"]
    sm_p = sidemap_path(batch_label)
    if not sm_p.exists():
        logger.error(
            "Phase 3: side-map %s missing for non-orphan batch %s; skipping.",
            sm_p,
            batch_id,
        )
        return

    sidemap = {row["custom_id"]: row for row in jsonl_read(sm_p)}
    seen = already_processed(CANDIDATES_PATH, "pair_key")

    output_file_id = entry.get("output_file_id")
    error_file_id = entry.get("error_file_id")

    n_ok = n_failed = n_skipped = 0
    if output_file_id:
        for line in _stream_lines(client, output_file_id):
            cid = line.get("custom_id")
            sm_row = sidemap.get(cid)
            if sm_row is None:
                logger.warning(
                    "Phase 3: custom_id %r not in side-map for batch %s.",
                    cid,
                    batch_id,
                )
                jsonl_append(
                    FAILED_PATH,
                    {
                        "pair_key": f"orphan::{cid}",
                        "question_id": None,
                        "model": None,
                        "error_type": "custom_id_not_in_sidemap",
                        "error_msg": json.dumps(line, ensure_ascii=False)[:1000],
                        "batch_id": batch_id,
                    },
                )
                n_failed += 1
                continue
            pair_key = sm_row["pair_key"]
            if pair_key in seen:
                n_skipped += 1
                continue
            qid, model = split_pair_key(pair_key)
            record, failed = _candidate_or_failed(
                line, pair_key, sm_row["prompt"], model, qid, batch_id
            )
            if record is not None:
                jsonl_append(CANDIDATES_PATH, record)
                seen.add(pair_key)
                n_ok += 1
            else:
                assert failed is not None
                jsonl_append(FAILED_PATH, failed)
                n_failed += 1

    if error_file_id:
        for line in _stream_lines(client, error_file_id):
            cid = line.get("custom_id")
            sm_row = sidemap.get(cid)
            if sm_row is None:
                logger.warning(
                    "Phase 3: error-file custom_id %r not in side-map (batch %s).",
                    cid,
                    batch_id,
                )
                jsonl_append(
                    FAILED_PATH,
                    {
                        "pair_key": f"orphan::{cid}",
                        "question_id": None,
                        "model": None,
                        "error_type": "custom_id_not_in_sidemap",
                        "error_msg": json.dumps(line, ensure_ascii=False)[:1000],
                        "batch_id": batch_id,
                    },
                )
                n_failed += 1
                continue
            pair_key = sm_row["pair_key"]
            if pair_key in seen:
                continue
            qid, model = split_pair_key(pair_key)
            err = line.get("error") or {}
            jsonl_append(
                FAILED_PATH,
                {
                    "pair_key": pair_key,
                    "question_id": qid,
                    "model": model,
                    "error_type": err.get("code") or "unknown",
                    "error_msg": err.get("message") or "",
                    "batch_id": batch_id,
                },
            )
            n_failed += 1

    update_manifest_entry(batch_id, collected=True)
    logger.info(
        "Phase 3: batch %s collected — %d ok, %d failed, %d duplicates skipped.",
        batch_id,
        n_ok,
        n_failed,
        n_skipped,
    )


def _collect_orphan(client: Together, entry: dict[str, Any]) -> None:
    """Phase 3 path for orphan batches (no local side-map exists).

    Rows are written to ``candidates_orphan.jsonl`` with synthetic
    ``pair_key = orphan::{custom_id}`` and ``prompt: null``. Stage 2 doesn't
    read this file; the user can manually reconstruct mappings if needed.
    """
    batch_id = entry["batch_id"]
    output_file_id = entry.get("output_file_id")
    error_file_id = entry.get("error_file_id")

    n_ok = n_failed = 0
    if output_file_id:
        for line in _stream_lines(client, output_file_id):
            cid = line.get("custom_id")
            response = line.get("response") or {}
            body = response.get("body") or {}
            choices = body.get("choices") or []
            content = (
                (choices[0].get("message") or {}).get("content") if choices else None
            )
            extracted, valid = parse_response(content or "")
            jsonl_append(
                ORPHAN_PATH,
                {
                    "pair_key": f"orphan::{cid}",
                    "question_id": None,
                    "model": None,
                    "prompt": None,
                    "response": content,
                    "extracted_answer": extracted,
                    "format_valid": valid if content else False,
                    "finish_reason": (
                        choices[0].get("finish_reason") if choices else None
                    ),
                    "batch_id": batch_id,
                    "generation_params": None,
                },
            )
            n_ok += 1
    if error_file_id:
        for line in _stream_lines(client, error_file_id):
            cid = line.get("custom_id")
            err = line.get("error") or {}
            jsonl_append(
                ORPHAN_PATH,
                {
                    "pair_key": f"orphan::{cid}",
                    "question_id": None,
                    "model": None,
                    "prompt": None,
                    "response": None,
                    "extracted_answer": None,
                    "format_valid": False,
                    "finish_reason": None,
                    "batch_id": batch_id,
                    "generation_params": None,
                    "error_type": err.get("code") or "unknown",
                    "error_msg": err.get("message") or "",
                },
            )
            n_failed += 1

    update_manifest_entry(batch_id, collected=True)
    logger.info(
        "Phase 3: orphan batch %s collected — %d ok, %d failed.",
        batch_id,
        n_ok,
        n_failed,
    )


def phase_3_collect(client: Together) -> None:
    """For each terminal-but-uncollected entry, download files and write rows."""
    for entry in read_manifest():
        if entry.get("collected"):
            continue
        status = (entry.get("status") or "").upper()
        if status == "COMPLETED":
            pass
        elif status == "FAILED" and entry.get("error_file_id"):
            pass
        else:
            if status in TERMINAL_STATES and status != "COMPLETED":
                logger.warning(
                    "Phase 3: batch %s in state %s, leaving uncollected.",
                    entry["batch_id"],
                    status,
                )
            continue
        if entry["batch_label"].startswith("orphan_"):
            _collect_orphan(client, entry)
        else:
            _collect_normal(client, entry)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1 candidate generation via Together Batch API. "
            "Phases 0–3 run per the plan; CLI flags shape which run."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Process only the first N rows of bbq_sample.jsonl. Slice "
            "happens before the resume filter, so re-running with --limit 10 "
            "after a successful dry run is a no-op."
        ),
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Run Phase 0 + Phase 1 then exit without polling or collecting.",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Skip Phase 1 (submission). Phase 0 still runs.",
    )
    parser.add_argument(
        "--accept-orphans",
        action="store_true",
        help=(
            "Required when Phase 0 finds remote batches not in the local "
            "manifest. Without it, the script aborts with the orphan list."
        ),
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=None,
        metavar="SEC",
        help=(
            "Override polling cadence (default: 60s for first 5 min, 300s "
            "after). Mostly for debugging."
        ),
    )
    args = parser.parse_args()

    if args.no_wait and args.collect_only:
        parser.error("--no-wait and --collect-only are mutually exclusive.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_env()

    if not os.environ.get("TOGETHER_API_KEY"):
        logger.error(
            "TOGETHER_API_KEY not set. Copy .env.example to .env and populate."
        )
        return 2

    if not SAMPLE_PATH.exists():
        logger.error("%s not found. Run data/00_sample_bbq.py first.", SAMPLE_PATH)
        return 2

    sample = list(jsonl_read(SAMPLE_PATH))
    if len(sample) < 1500:
        logger.warning("sample has %d rows (expected 1500); continuing.", len(sample))
    else:
        logger.info("loaded %d sample rows.", len(sample))

    BATCH_INPUTS_DIR.mkdir(parents=True, exist_ok=True)

    client = Together()

    n_orphans = phase_0_reconcile(client, accept_orphans=args.accept_orphans)
    if n_orphans:
        logger.info("Phase 0: reconciled %d orphan batch(es).", n_orphans)

    if not args.collect_only:
        phase_1_submit(client, sample, limit=args.limit)

    if not args.no_wait:
        phase_2_poll(client, poll_interval_override=args.poll_interval)
        phase_3_collect(client)
    return 0


if __name__ == "__main__":
    sys.exit(main())
