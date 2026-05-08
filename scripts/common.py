"""Shared helpers for judge-from-scratch pipeline scripts.

Provides:
    - load_env: load .env into os.environ and surface required keys.
    - jsonl_read: stream records from a JSONL file.
    - jsonl_append: append a single record to a JSONL file.
    - already_processed: collect keys already written to a JSONL artifact,
      enabling resumable pipeline stages.
    - atomic_write_json: tmp+rename write of a single JSON payload.
    - atomic_write_jsonl: tmp+rename write of a list of JSONL records.
    - merge_jsonl_patches: merge per-key patches into a base JSONL via
      atomic tmp+rename.
    - file_sha256: hex SHA-256 of a file's contents.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Iterator
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


def atomic_write_json(path: Path | str, payload: dict[str, Any]) -> None:
    """Write a JSON payload via tmp+rename (all-or-nothing).

    The tmp file lives at ``path.with_name(path.name + ".tmp")`` so a
    concurrent reader of ``path`` never sees a half-written file.

    Args:
        path: Destination JSON file.
        payload: A JSON-serializable dict; written with ``indent=2``.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    tmp.replace(p)


def atomic_write_jsonl(path: Path | str, records: Iterable[dict[str, Any]]) -> None:
    """Write a list of dicts as JSONL via tmp+rename (all-or-nothing).

    Same tmp-path convention as :func:`atomic_write_json`. Use this when
    the entire output is known up-front and a partial file would be
    worse than no file (e.g. deterministic one-shot pipeline stages).

    Args:
        path: Destination JSONL file.
        records: Iterable of JSON-serializable dicts; each is written
            as one line.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with tmp.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")
    tmp.replace(p)


def merge_jsonl_patches(
    base_path: Path | str,
    patches_path: Path | str,
    key_field: str,
) -> dict[str, int]:
    """Merge per-key patches into a base JSONL via atomic tmp+rename.

    Reads ``base_path`` once, applies any matching patch from
    ``patches_path`` (matched on ``key_field``), and writes the result
    via tmp+rename so a crash mid-write leaves the base untouched.

    Behavior:
        - Output preserves the base file's record order.
        - Patches whose key is NOT present in the base are dropped with a
          ``WARNING`` log line listing each unmatched key (never silently
          inserted, never lost without a log).
        - When multiple patches share the same key, the last one wins
          (logged at ``WARNING`` once per key).
        - Each base record is updated via ``dict.update(patch)`` — top-level
          keys in the patch overwrite top-level keys in the base; nested
          dicts are not deep-merged.

    Args:
        base_path: JSONL file to patch (read-only until the final rename).
        patches_path: JSONL file of patch records; each must contain
            ``key_field``.
        key_field: Name of the field used to match patch → base.

    Returns:
        ``{"base": N, "patches_applied": K, "patches_unmatched": U,
        "duplicate_patches": D}`` for caller-side reconciliation.
    """
    base = Path(base_path)
    patches = Path(patches_path)

    patch_map: dict[str, dict[str, Any]] = {}
    duplicates: dict[str, int] = {}
    for record in jsonl_read(patches):
        if key_field not in record:
            logger.warning(
                "Patch in %s missing key_field %r; skipping.",
                patches,
                key_field,
            )
            continue
        key = str(record[key_field])
        if key in patch_map:
            duplicates[key] = duplicates.get(key, 1) + 1
        patch_map[key] = record

    for key, count in duplicates.items():
        logger.warning(
            "merge_jsonl_patches: %d patches for key %r; last one wins.",
            count,
            key,
        )

    tmp = base.with_name(base.name + ".tmp")
    if tmp.exists():
        tmp.unlink()

    base_count = 0
    applied = 0
    matched_keys: set[str] = set()
    with tmp.open("w", encoding="utf-8") as fh:
        for record in jsonl_read(base):
            base_count += 1
            if key_field in record:
                key = str(record[key_field])
                if key in patch_map:
                    record = {**record, **patch_map[key]}
                    matched_keys.add(key)
                    applied += 1
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")

    unmatched = sorted(set(patch_map.keys()) - matched_keys)
    if unmatched:
        sample = unmatched[:10]
        logger.warning(
            "merge_jsonl_patches: %d patch key(s) not present in base; "
            "dropped. First %d: %s",
            len(unmatched),
            len(sample),
            sample,
        )

    tmp.replace(base)

    return {
        "base": base_count,
        "patches_applied": applied,
        "patches_unmatched": len(unmatched),
        "duplicate_patches": len(duplicates),
    }


def file_sha256(path: Path | str) -> str:
    """Return the hex-encoded SHA-256 of a file's contents.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex digest string (64 chars).
    """
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


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
