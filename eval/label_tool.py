"""Stage 5 — interactive CLI to hand-label the 300-pair eval set.

Reads ``data/pairs/eval_set_unlabeled.jsonl`` (Stage 3a output) and
mutates it in place by filling in ``human_verdict`` (``"A"``/``"B"``/
``"TIE"``), ``confidence`` (1–5), and ``notes`` (free-form, single
line). The schema was pre-shaped by Stage 3a with these three fields
set to ``null`` precisely so this tool can fill them in.

Per-label persistence is atomic via :func:`scripts.common.atomic_write_jsonl`
(tmp+rename), so a crash between pairs leaves either the prior or the
new file on disk, never a half-written one. On startup the tool also
writes a one-shot timestamped backup of the input alongside it, and
acquires an exclusive ``flock`` on a sibling ``.lock`` file so two
concurrent sessions can't clobber each other's work.

Usage:
    uv run python -m eval.label_tool [--input PATH]
                                     [--slice {in_dist,ood_religion}]
                                     [--random-order] [--seed N]
                                     [--review]

Verdict canonical strings match Appendix C of the fine-tuning primer:
``"A"`` (response_a less biased / cleaner), ``"B"`` (response_b less
biased / cleaner), ``"TIE"`` (equally biased, equally clean, or
genuinely ambiguous).

Filesystem assumption: local POSIX. ``os.replace`` (which the atomic
writer relies on) is not atomic on NFS / sshfs / Dropbox-synced dirs;
do not point ``--input`` at a network mount.

Note on ``print()``: this module uses ``print()`` for the interactive
labeling UI. Global project rules prefer ``logging`` for production
code; the TUI is the documented exception. ``logger`` is still used
for diagnostics (lock contention, resume status, backups, errors).
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import logging
import random
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any, Literal

from scripts.common import atomic_write_jsonl, jsonl_read

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "data" / "pairs" / "eval_set_unlabeled.jsonl"

Verdict = Literal["A", "B", "TIE"]
SliceName = Literal["in_dist", "ood_religion"]

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]

_VERDICT_HELP = (
    "  a / A / b / B / t / T / tie / TIE  → record a verdict\n"
    "  q                                  → quit (in-flight pair NOT saved)\n"
    "  s                                  → skip this pair (re-shows next session)\n"
    "  d                                  → (review mode only) clear this label\n"
    "  Enter (review mode only)           → keep current label as-is\n"
    "  ?                                  → show this help"
)


# -----------------------------------------------------------------------------
# Pure parsing helpers (unit-tested)
# -----------------------------------------------------------------------------


def parse_verdict(raw: str) -> Verdict | None:
    """Canonicalize user input into ``"A"``/``"B"``/``"TIE"`` or ``None``.

    Accepts ``a``/``A``/``b``/``B``/``t``/``T``/``tie``/``TIE`` and any
    surrounding whitespace. Anything else returns ``None``; callers are
    expected to re-prompt.

    Args:
        raw: Raw user input string from the verdict prompt.

    Returns:
        Canonical verdict string, or ``None`` for unrecognized input.
    """
    s = raw.strip().lower()
    if s in {"a"}:
        return "A"
    if s in {"b"}:
        return "B"
    if s in {"t", "tie"}:
        return "TIE"
    return None


def parse_confidence(raw: str) -> int | None:
    """Parse a 1–5 integer confidence; ``None`` on anything else.

    Args:
        raw: Raw user input string from the confidence prompt.

    Returns:
        ``int`` in ``[1, 5]`` on success, else ``None``.
    """
    s = raw.strip()
    if not s:
        return None
    try:
        n = int(s)
    except ValueError:
        return None
    if 1 <= n <= 5:
        return n
    return None


# -----------------------------------------------------------------------------
# Queue construction (pure; unit-tested)
# -----------------------------------------------------------------------------


def build_queue(
    records: list[dict[str, Any]],
    *,
    slice_filter: SliceName | None,
    review: bool,
    randomize: bool,
    seed: int | None,
) -> list[int]:
    """Compute the ordered list of indices into ``records`` to visit.

    Order of operations:
        1. filter by ``slice_filter`` (skip records of the wrong slice);
        2. if ``randomize``, shuffle the *full* filtered list with
           ``random.Random(seed)`` — applied **before** the
           labeled/unlabeled filter so that ``--seed`` produces a stable
           order across resume sessions for whatever records remain;
        3. drop entries that aren't eligible for the current mode
           (default mode keeps unlabeled; review mode keeps labeled).

    Args:
        records: All records, in their original file order.
        slice_filter: ``"in_dist"`` / ``"ood_religion"`` / ``None``.
        review: ``True`` to build a review queue (labeled records only).
        randomize: ``True`` to shuffle.
        seed: RNG seed (used only if ``randomize`` is ``True``).

    Returns:
        List of integer indices into ``records`` in visit order.
    """
    candidates = [
        i
        for i, r in enumerate(records)
        if slice_filter is None or r["eval_slice"] == slice_filter
    ]

    if randomize:
        rng = random.Random(seed)
        rng.shuffle(candidates)

    queue: list[int] = []
    for i in candidates:
        is_labeled = records[i].get("human_verdict") is not None
        if review and not is_labeled:
            continue
        if not review and is_labeled:
            continue
        queue.append(i)
    return queue


# -----------------------------------------------------------------------------
# Display helpers
# -----------------------------------------------------------------------------


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(f"{prefix}{line}" for line in text.split("\n"))


def progress_line(
    records: list[dict[str, Any]],
    slice_filter: SliceName | None,
) -> str:
    """One-line progress summary, slice-scoped if a slice is set."""
    if slice_filter is not None:
        sliced = [r for r in records if r["eval_slice"] == slice_filter]
        labeled = sum(1 for r in sliced if r.get("human_verdict") is not None)
        return (
            f"{labeled} / {len(sliced)} {slice_filter} labeled. "
            f"{len(sliced) - labeled} remaining."
        )

    total = len(records)
    labeled = sum(1 for r in records if r.get("human_verdict") is not None)
    in_dist = sum(
        1
        for r in records
        if r["eval_slice"] == "in_dist" and r.get("human_verdict") is not None
    )
    ood = sum(
        1
        for r in records
        if r["eval_slice"] == "ood_religion" and r.get("human_verdict") is not None
    )
    return (
        f"{labeled} / {total} labeled ({in_dist} in_dist + {ood} ood_religion). "
        f"{total - labeled} remaining."
    )


def format_pair(
    record: dict[str, Any],
    header_line: str,
    *,
    color: bool = True,
) -> str:
    """Render a pair as a multi-line display block.

    Indents response/question text by two spaces and otherwise prints
    text verbatim — the terminal handles wrapping. Embedded markup like
    ``<answer>C</answer>`` and intentional whitespace in the responses
    is preserved as-is.

    Args:
        record: A single pair record.
        header_line: One-line counter / progress text shown next to the
            slice tag (e.g. ``"42 / 240 in_dist labeled. 198 remaining."``).
        color: If ``True``, wrap the slice tag in ANSI bold escapes.

    Returns:
        A single multi-line string ready to pass to an ``OutputFn``.
    """
    bold_open, bold_close = ("\033[1m", "\033[0m") if color else ("", "")
    slice_tag = f"[ {record['eval_slice'].upper()} ]"

    bar_eq = "=" * 80
    bar_dash = "-" * 80

    parts = [
        bar_eq,
        f"{bold_open}{slice_tag}{bold_close}   {header_line}",
        (
            f"pair_id: {record['pair_id']}     "
            f"bias: {record['bias_category']}     "
            f"pair_category: {record['pair_category']}"
        ),
        bar_dash,
        "QUESTION:",
        _indent(record["question_text"]),
        "",
        f"RESPONSE A  ({record['response_a']['model']}):",
        _indent(record["response_a"]["text"]),
        "",
        f"RESPONSE B  ({record['response_b']['model']}):",
        _indent(record["response_b"]["text"]),
        bar_dash,
    ]
    return "\n".join(parts)


# -----------------------------------------------------------------------------
# Lock + backup
# -----------------------------------------------------------------------------


def acquire_lock(input_path: Path) -> IO[str]:
    """Acquire an exclusive non-blocking ``flock`` on a sibling lock file.

    The lock file is ``<input_path>.lock``. We deliberately do *not*
    flock the input file itself: ``atomic_write_jsonl`` replaces the
    input via tmp+rename, which swaps inodes — a flock held on the old
    inode would silently stop guarding the live file.

    Args:
        input_path: Path of the JSONL artifact being labeled.

    Returns:
        Open file handle on the lock file. Caller keeps it open for the
        session lifetime; closing/exiting releases the lock.

    Raises:
        BlockingIOError: Another session already holds the lock.
    """
    lock_path = input_path.with_name(input_path.name + ".lock")
    fh = lock_path.open("a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        raise
    return fh


def make_backup(input_path: Path) -> Path:
    """Copy the input to a timestamped ``.bak-<UTC-ISO>`` sibling.

    Skips the copy if a backup with the exact same timestamp already
    exists (impossible in practice within one second, but cheap to
    guard).

    Args:
        input_path: Path of the JSONL artifact being labeled.

    Returns:
        Path of the backup file.
    """
    ts = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    bak = input_path.with_name(f"{input_path.name}.bak-{ts}")
    if not bak.exists():
        shutil.copy2(input_path, bak)
    return bak


# -----------------------------------------------------------------------------
# Interactive prompts
# -----------------------------------------------------------------------------


def _prompt_verdict(
    input_fn: InputFn,
    output_fn: OutputFn,
    *,
    review_mode: bool,
) -> str:
    """Loop until the user gives a recognizable verdict-prompt response.

    Returns one of: ``"A"``, ``"B"``, ``"TIE"``, ``"QUIT"``, ``"SKIP"``,
    ``"KEEP"`` (review-mode Enter), ``"CLEAR"`` (review-mode ``d``).
    """
    if review_mode:
        prompt = "verdict [a/b/t, Enter=keep, d=clear, q=quit, s=skip, ?=help]: "
    else:
        prompt = "verdict [a/b/t, q=quit, s=skip, ?=help]: "

    while True:
        raw = input_fn(prompt)
        s = raw.strip().lower()
        if s == "":
            if review_mode:
                return "KEEP"
            output_fn("(empty input — type a, b, t, q, s, or ?)")
            continue
        if s == "q":
            return "QUIT"
        if s == "s":
            return "SKIP"
        if s == "?":
            output_fn(_VERDICT_HELP)
            continue
        if review_mode and s == "d":
            return "CLEAR"
        v = parse_verdict(s)
        if v is not None:
            return v
        output_fn(f"(invalid: {raw!r}. type ? for help.)")


def _prompt_confidence(
    input_fn: InputFn,
    output_fn: OutputFn,
    *,
    default: int | None,
) -> int:
    """Loop until the user gives a 1–5 integer (or accepts a default)."""
    if default is not None:
        prompt = f"confidence [1-5, Enter={default}]: "
    else:
        prompt = "confidence [1-5]: "
    while True:
        raw = input_fn(prompt)
        if raw.strip() == "" and default is not None:
            return default
        c = parse_confidence(raw)
        if c is not None:
            return c
        output_fn(f"(invalid: {raw!r}. enter an integer 1-5.)")


def _prompt_notes(input_fn: InputFn, *, default: str | None) -> str | None:
    """Single-line free-form notes; Enter keeps default or yields ``None``."""
    if default is not None:
        prompt = f"notes (Enter to keep {default!r}): "
    else:
        prompt = "notes (optional, single line): "
    raw = input_fn(prompt)
    if raw.strip() == "":
        return default
    return raw.rstrip("\n")


# -----------------------------------------------------------------------------
# Session loop
# -----------------------------------------------------------------------------


def run_label_session(
    records: list[dict[str, Any]],
    queue: list[int],
    input_path: Path,
    *,
    review_mode: bool,
    slice_filter: SliceName | None = None,
    color: bool = True,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
) -> None:
    """Drive the interactive labeling loop.

    Mutates ``records`` in place and persists the full list to
    ``input_path`` via :func:`scripts.common.atomic_write_jsonl` after
    every committed change. The ``input_fn``/``output_fn`` injection
    is purely for testability — production callers leave the defaults.

    Args:
        records: All records loaded from ``input_path``.
        queue: Visit-order indices into ``records`` (from
            :func:`build_queue`).
        input_path: Where to atomically rewrite after each label.
        review_mode: ``True`` for the ``--review`` path (labeled
            records, edit-or-keep ergonomics).
        slice_filter: Used only for the per-iteration progress line.
        color: ANSI bold on/off (caller usually sets this from
            ``sys.stdout.isatty()``).
        input_fn: Replacement for ``input``; useful in tests.
        output_fn: Replacement for ``print``; useful in tests.
    """
    if not queue:
        if review_mode:
            output_fn("No labeled records yet — nothing to review.")
        else:
            output_fn("All matching pairs are already labeled.")
        return

    pending = list(queue)
    while pending:
        idx = pending.pop(0)
        record = records[idx]

        header = progress_line(records, slice_filter)
        output_fn(format_pair(record, header, color=color))

        if review_mode:
            output_fn(
                f"current: verdict={record.get('human_verdict')}  "
                f"confidence={record.get('confidence')}  "
                f"notes={record.get('notes')!r}"
            )

        try:
            action = _prompt_verdict(input_fn, output_fn, review_mode=review_mode)
        except (EOFError, KeyboardInterrupt):
            output_fn("\n(interrupted — in-flight pair not saved.)")
            return

        if action == "QUIT":
            output_fn("(quitting — in-flight pair not saved.)")
            return
        if action == "SKIP":
            continue
        if action == "KEEP":
            continue
        if action == "CLEAR":
            record["human_verdict"] = None
            record["confidence"] = None
            record["notes"] = None
            atomic_write_jsonl(input_path, records)
            continue

        # action is a Verdict literal — collect confidence and notes.
        verdict: Verdict = action  # type: ignore[assignment]
        try:
            existing_c = record.get("confidence") if review_mode else None
            existing_n = record.get("notes") if review_mode else None
            confidence = _prompt_confidence(input_fn, output_fn, default=existing_c)
            notes = _prompt_notes(input_fn, default=existing_n)
        except (EOFError, KeyboardInterrupt):
            output_fn("\n(interrupted — in-flight pair not saved.)")
            return

        record["human_verdict"] = verdict
        record["confidence"] = confidence
        record["notes"] = notes
        atomic_write_jsonl(input_path, records)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 5 — interactive CLI to hand-label the 300-pair eval "
            "set. Resumable across sessions; mutates the input JSONL "
            "in place; takes a startup backup; refuses to start if "
            "another session is already labeling the same file."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        metavar="PATH",
        help=(
            "Path to the JSONL holdout (default: "
            "data/pairs/eval_set_unlabeled.jsonl)."
        ),
    )
    parser.add_argument(
        "--slice",
        choices=("in_dist", "ood_religion"),
        default=None,
        dest="slice_filter",
        help="Restrict the work queue to one eval slice.",
    )
    parser.add_argument(
        "--random-order",
        action="store_true",
        help="Shuffle the work queue (use --seed for reproducibility).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help=(
            "RNG seed for --random-order. Shuffle is applied to the full "
            "filtered set BEFORE labeled records are dropped, so the same "
            "seed yields a stable visit order across resume sessions."
        ),
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help=(
            "Iterate over already-labeled records and let me edit them. "
            "Composes with --slice and --random-order."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    input_path: Path = args.input
    if not input_path.exists():
        logger.error("Input not found: %s", input_path)
        return 2

    try:
        lock_handle = acquire_lock(input_path)
    except BlockingIOError:
        logger.error(
            "Another labeling session appears to be running against %s "
            "(lock file: %s). Refusing to start.",
            input_path,
            input_path.with_name(input_path.name + ".lock"),
        )
        return 1

    try:
        bak = make_backup(input_path)
        logger.info("Wrote startup backup to %s", bak)

        records = list(jsonl_read(input_path))
        logger.info("Loaded %d records from %s", len(records), input_path)

        queue = build_queue(
            records,
            slice_filter=args.slice_filter,
            review=args.review,
            randomize=args.random_order,
            seed=args.seed,
        )

        line = progress_line(records, args.slice_filter)
        if args.review:
            print(f"Review mode: {line}")
            print(f"Queue: {len(queue)} labeled record(s) to review.")
        else:
            print(f"Resuming: {line}")
            print(f"Queue: {len(queue)} unlabeled pair(s) this session.")

        run_label_session(
            records,
            queue,
            input_path,
            review_mode=args.review,
            slice_filter=args.slice_filter,
            color=sys.stdout.isatty(),
        )
    finally:
        lock_handle.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
