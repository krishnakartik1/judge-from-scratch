"""Shared helpers for REVAL Judge pipeline scripts.

Provides:
    - load_env: load .env into os.environ and surface required keys.
    - jsonl_read: stream records from a JSONL file.
    - jsonl_append: append a single record to a JSONL file.
    - already_processed: collect keys already written to a JSONL artifact,
      enabling resumable pipeline stages.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env(dotenv_path: Path | str | None = None) -> None:
    """Load environment variables from a .env file at the repo root.

    Args:
        dotenv_path: Optional explicit path to a .env file. Defaults to
            ``<repo_root>/.env``.
    """
    path = Path(dotenv_path) if dotenv_path is not None else REPO_ROOT / ".env"
    if not path.exists():
        logger.warning("No .env file at %s; relying on process environment.", path)
        return
    load_dotenv(dotenv_path=path, override=False)
    logger.debug("Loaded environment from %s", path)


def jsonl_read(path: Path | str) -> Iterator[dict[str, Any]]:
    """Yield records from a JSONL file one at a time.

    Skips blank lines. Raises ``json.JSONDecodeError`` on malformed lines so
    pipeline failures surface loudly rather than silently dropping data.

    Args:
        path: Path to the JSONL file.

    Yields:
        Parsed dict for each non-empty line.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError:
                logger.error("Malformed JSON in %s at line %d", p, line_num)
                raise


def jsonl_append(path: Path | str, record: dict[str, Any]) -> None:
    """Append a single record as a JSON line.

    Creates parent directories if missing. Each call opens, writes, and
    closes the file so a crash mid-pipeline still leaves a valid JSONL
    on disk.

    Args:
        path: Destination JSONL file.
        record: A JSON-serializable dict.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False))
        fh.write("\n")


def already_processed(path: Path | str, key_field: str) -> set[str]:
    """Collect values of ``key_field`` already present in a JSONL artifact.

    Used by resumable pipeline stages: read this set first, then skip any
    input record whose key is in it. If the file does not yet exist,
    returns an empty set.

    Records missing ``key_field`` are skipped with a warning rather than
    raising — partial artifacts from earlier crashes shouldn't block a
    resume.

    Args:
        path: JSONL artifact written by a previous run of the stage.
        key_field: Name of the field whose value uniquely identifies a
            record (e.g., "question_id", "pair_id").

    Returns:
        Set of stringified key values already written.
    """
    p = Path(path)
    if not p.exists():
        return set()

    seen: set[str] = set()
    for record in jsonl_read(p):
        if key_field not in record:
            logger.warning(
                "Record in %s missing key_field %r; skipping for resume set.",
                p,
                key_field,
            )
            continue
        seen.add(str(record[key_field]))
    return seen
