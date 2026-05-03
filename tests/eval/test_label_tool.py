"""Unit tests for the Stage 5 hand-labeling CLI (eval/label_tool.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eval.label_tool import (
    acquire_lock,
    build_queue,
    make_backup,
    parse_confidence,
    parse_verdict,
    progress_line,
    run_label_session,
)

# -----------------------------------------------------------------------------
# Fixtures + helpers
# -----------------------------------------------------------------------------


def _mk_record(
    pair_id: str,
    eval_slice: str = "in_dist",
    *,
    verdict: str | None = None,
    confidence: int | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "question_id": f"q_{pair_id}",
        "question_text": f"text for {pair_id}",
        "bias_category": "ses",
        "response_a": {"model": "model-a", "text": "A says foo"},
        "response_b": {"model": "model-b", "text": "B says bar"},
        "pair_category": "subtle_bias_vs_clean",
        "human_verdict": verdict,
        "confidence": confidence,
        "notes": notes,
        "eval_slice": eval_slice,
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False))
            fh.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _scripted_input(answers: list[str]):
    """Build an input_fn that returns scripted answers in order."""
    it = iter(answers)

    def _fn(_prompt: str) -> str:
        return next(it)

    return _fn


def _capture_output():
    """Build (output_fn, list-of-strings-collected-so-far)."""
    captured: list[str] = []
    return captured.append, captured


# -----------------------------------------------------------------------------
# parse_verdict
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("a", "A"),
        ("A", "A"),
        ("  a ", "A"),
        ("b", "B"),
        ("B", "B"),
        ("t", "TIE"),
        ("T", "TIE"),
        ("tie", "TIE"),
        ("TIE", "TIE"),
        (" TIE\n", "TIE"),
    ],
)
def test_parse_verdict_canonical_forms(raw, expected):
    assert parse_verdict(raw) == expected


@pytest.mark.parametrize("raw", ["", "x", "42", "ab", "ties", "?", "q", "s", " "])
def test_parse_verdict_rejects_garbage(raw):
    assert parse_verdict(raw) is None


# -----------------------------------------------------------------------------
# parse_confidence
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [("1", 1), ("3", 3), ("5", 5), (" 4 ", 4)])
def test_parse_confidence_valid_range(raw, expected):
    assert parse_confidence(raw) == expected


@pytest.mark.parametrize("raw", ["", "0", "6", "-1", "abc", "3.5", "1.0"])
def test_parse_confidence_rejects_invalid(raw):
    assert parse_confidence(raw) is None


# -----------------------------------------------------------------------------
# build_queue
# -----------------------------------------------------------------------------


def test_build_queue_skips_labeled_in_default_mode():
    records = [
        _mk_record("p0"),
        _mk_record("p1", verdict="A", confidence=4, notes=None),
        _mk_record("p2"),
    ]
    queue = build_queue(
        records, slice_filter=None, review=False, randomize=False, seed=None
    )
    assert queue == [0, 2]


def test_build_queue_review_keeps_only_labeled():
    records = [
        _mk_record("p0"),
        _mk_record("p1", verdict="A", confidence=4),
        _mk_record("p2"),
        _mk_record("p3", verdict="B", confidence=2),
    ]
    queue = build_queue(
        records, slice_filter=None, review=True, randomize=False, seed=None
    )
    assert queue == [1, 3]


def test_build_queue_slice_filter():
    records = [
        _mk_record("p0", eval_slice="in_dist"),
        _mk_record("p1", eval_slice="ood_religion"),
        _mk_record("p2", eval_slice="in_dist"),
    ]
    in_dist = build_queue(
        records, slice_filter="in_dist", review=False, randomize=False, seed=None
    )
    ood = build_queue(
        records, slice_filter="ood_religion", review=False, randomize=False, seed=None
    )
    assert in_dist == [0, 2]
    assert ood == [1]


def test_build_queue_random_order_seed_stable_across_resume():
    """Same seed + same filter must produce a stable visit order even
    after some records get labeled in between sessions.

    Concretely: shuffle is applied to the *full* filtered set before
    labeled records are dropped, so the relative order of unlabeled
    records is preserved across runs.
    """
    n = 20
    records = [_mk_record(f"p{i:02d}") for i in range(n)]

    q_before = build_queue(
        records, slice_filter=None, review=False, randomize=True, seed=99
    )
    assert sorted(q_before) == list(range(n))  # all unlabeled, all present

    # Simulate session-1 labeling 5 records (chosen from anywhere in the queue).
    labeled_indices = {
        q_before[0],
        q_before[5],
        q_before[10],
        q_before[15],
        q_before[1],
    }
    for i in labeled_indices:
        records[i]["human_verdict"] = "A"
        records[i]["confidence"] = 3

    q_after = build_queue(
        records, slice_filter=None, review=False, randomize=True, seed=99
    )
    expected = [i for i in q_before if i not in labeled_indices]
    assert q_after == expected


def test_build_queue_random_order_no_seed_returns_full_set():
    n = 10
    records = [_mk_record(f"p{i:02d}") for i in range(n)]
    q = build_queue(records, slice_filter=None, review=False, randomize=True, seed=None)
    assert sorted(q) == list(range(n))


# -----------------------------------------------------------------------------
# progress_line
# -----------------------------------------------------------------------------


def test_progress_line_overall_counts_both_slices():
    records = [
        _mk_record("p0", "in_dist", verdict="A", confidence=4),
        _mk_record("p1", "in_dist"),
        _mk_record("p2", "ood_religion", verdict="B", confidence=3),
        _mk_record("p3", "ood_religion"),
    ]
    line = progress_line(records, slice_filter=None)
    assert "2 / 4 labeled" in line
    assert "1 in_dist" in line
    assert "1 ood_religion" in line
    assert "2 remaining" in line


def test_progress_line_slice_scoped_only_counts_that_slice():
    records = [
        _mk_record("p0", "in_dist", verdict="A", confidence=4),
        _mk_record("p1", "in_dist"),
        _mk_record("p2", "ood_religion", verdict="B", confidence=3),
    ]
    line = progress_line(records, slice_filter="in_dist")
    assert "1 / 2 in_dist labeled" in line
    assert "1 remaining" in line
    # OOD record must not contribute to in_dist counters.
    assert "ood" not in line


# -----------------------------------------------------------------------------
# run_label_session — default mode (commit one label, then quit)
# -----------------------------------------------------------------------------


def test_run_label_session_persists_after_each_label(tmp_path):
    path = tmp_path / "ev.jsonl"
    records = [_mk_record("p0"), _mk_record("p1"), _mk_record("p2")]
    _write_jsonl(path, records)

    output_fn, _ = _capture_output()
    # Script: label p0 with A/4/"good", then quit at p1's verdict prompt.
    input_fn = _scripted_input(["a", "4", "good", "q"])

    run_label_session(
        records,
        queue=[0, 1, 2],
        input_path=path,
        review_mode=False,
        slice_filter=None,
        color=False,
        input_fn=input_fn,
        output_fn=output_fn,
    )

    on_disk = _read_jsonl(path)
    assert len(on_disk) == 3
    assert on_disk[0]["human_verdict"] == "A"
    assert on_disk[0]["confidence"] == 4
    assert on_disk[0]["notes"] == "good"
    assert on_disk[1]["human_verdict"] is None
    assert on_disk[2]["human_verdict"] is None


def test_run_label_session_quit_at_first_pair_writes_nothing(tmp_path):
    path = tmp_path / "ev.jsonl"
    records = [_mk_record("p0"), _mk_record("p1")]
    _write_jsonl(path, records)
    snapshot_before = _read_jsonl(path)

    output_fn, _ = _capture_output()
    input_fn = _scripted_input(["q"])

    run_label_session(
        records,
        queue=[0, 1],
        input_path=path,
        review_mode=False,
        slice_filter=None,
        color=False,
        input_fn=input_fn,
        output_fn=output_fn,
    )

    assert _read_jsonl(path) == snapshot_before


def test_run_label_session_skip_does_not_persist(tmp_path):
    path = tmp_path / "ev.jsonl"
    records = [_mk_record("p0"), _mk_record("p1")]
    _write_jsonl(path, records)

    output_fn, _ = _capture_output()
    # skip p0, then label p1 with B/2/"".
    input_fn = _scripted_input(["s", "b", "2", ""])

    run_label_session(
        records,
        queue=[0, 1],
        input_path=path,
        review_mode=False,
        slice_filter=None,
        color=False,
        input_fn=input_fn,
        output_fn=output_fn,
    )

    on_disk = _read_jsonl(path)
    assert on_disk[0]["human_verdict"] is None
    assert on_disk[1]["human_verdict"] == "B"
    assert on_disk[1]["confidence"] == 2
    assert on_disk[1]["notes"] is None


def test_run_label_session_invalid_then_valid_verdict(tmp_path):
    """Loop should re-prompt on bad input rather than crash or skip."""
    path = tmp_path / "ev.jsonl"
    records = [_mk_record("p0")]
    _write_jsonl(path, records)

    output_fn, captured = _capture_output()
    # Invalid "x", then "?" (help), then valid "tie", then conf=3, notes empty.
    input_fn = _scripted_input(["x", "?", "tie", "3", ""])

    run_label_session(
        records,
        queue=[0],
        input_path=path,
        review_mode=False,
        slice_filter=None,
        color=False,
        input_fn=input_fn,
        output_fn=output_fn,
    )

    on_disk = _read_jsonl(path)
    assert on_disk[0]["human_verdict"] == "TIE"
    assert on_disk[0]["confidence"] == 3
    assert on_disk[0]["notes"] is None
    # Help must have been emitted at least once.
    assert any("→ record a verdict" in line for line in captured)


# -----------------------------------------------------------------------------
# run_label_session — review mode
# -----------------------------------------------------------------------------


def test_review_mode_keep_leaves_record_unchanged(tmp_path):
    path = tmp_path / "ev.jsonl"
    records = [_mk_record("p0", verdict="A", confidence=4, notes="orig")]
    _write_jsonl(path, records)

    output_fn, _ = _capture_output()
    # Empty Enter at the verdict prompt = KEEP in review mode.
    input_fn = _scripted_input([""])

    run_label_session(
        records,
        queue=[0],
        input_path=path,
        review_mode=True,
        slice_filter=None,
        color=False,
        input_fn=input_fn,
        output_fn=output_fn,
    )

    on_disk = _read_jsonl(path)
    assert on_disk[0]["human_verdict"] == "A"
    assert on_disk[0]["confidence"] == 4
    assert on_disk[0]["notes"] == "orig"


def test_review_mode_change_verdict_and_propagate_defaults(tmp_path):
    """Change verdict, accept defaults for confidence + notes via Enter."""
    path = tmp_path / "ev.jsonl"
    records = [_mk_record("p0", verdict="A", confidence=4, notes="orig")]
    _write_jsonl(path, records)

    output_fn, _ = _capture_output()
    # New verdict "b", Enter for confidence (keep 4), Enter for notes (keep "orig").
    input_fn = _scripted_input(["b", "", ""])

    run_label_session(
        records,
        queue=[0],
        input_path=path,
        review_mode=True,
        slice_filter=None,
        color=False,
        input_fn=input_fn,
        output_fn=output_fn,
    )

    on_disk = _read_jsonl(path)
    assert on_disk[0]["human_verdict"] == "B"
    assert on_disk[0]["confidence"] == 4
    assert on_disk[0]["notes"] == "orig"


def test_review_mode_clear_nullifies_all_three_fields(tmp_path):
    path = tmp_path / "ev.jsonl"
    records = [_mk_record("p0", verdict="A", confidence=4, notes="orig")]
    _write_jsonl(path, records)

    output_fn, _ = _capture_output()
    input_fn = _scripted_input(["d"])

    run_label_session(
        records,
        queue=[0],
        input_path=path,
        review_mode=True,
        slice_filter=None,
        color=False,
        input_fn=input_fn,
        output_fn=output_fn,
    )

    on_disk = _read_jsonl(path)
    assert on_disk[0]["human_verdict"] is None
    assert on_disk[0]["confidence"] is None
    assert on_disk[0]["notes"] is None


def test_review_mode_empty_queue_returns_without_prompting(tmp_path):
    path = tmp_path / "ev.jsonl"
    records = [_mk_record("p0"), _mk_record("p1")]  # zero labeled
    _write_jsonl(path, records)

    output_fn, captured = _capture_output()
    # No scripted answers — calling input_fn would raise StopIteration.
    input_fn = _scripted_input([])

    run_label_session(
        records,
        queue=[],  # empty review queue
        input_path=path,
        review_mode=True,
        slice_filter=None,
        color=False,
        input_fn=input_fn,
        output_fn=output_fn,
    )

    assert any("nothing to review" in line for line in captured)
    # On-disk untouched.
    assert _read_jsonl(path) == records


def test_default_mode_empty_queue_returns_without_prompting(tmp_path):
    path = tmp_path / "ev.jsonl"
    records = [_mk_record("p0", verdict="A", confidence=4)]
    _write_jsonl(path, records)

    output_fn, captured = _capture_output()
    input_fn = _scripted_input([])

    run_label_session(
        records,
        queue=[],
        input_path=path,
        review_mode=False,
        slice_filter=None,
        color=False,
        input_fn=input_fn,
        output_fn=output_fn,
    )

    assert any("already labeled" in line for line in captured)


# -----------------------------------------------------------------------------
# acquire_lock
# -----------------------------------------------------------------------------


def test_acquire_lock_blocks_second_instance(tmp_path):
    path = tmp_path / "ev.jsonl"
    path.write_text("")

    h1 = acquire_lock(path)
    try:
        with pytest.raises(BlockingIOError):
            acquire_lock(path)
    finally:
        h1.close()


def test_acquire_lock_releases_on_close_so_resume_works(tmp_path):
    path = tmp_path / "ev.jsonl"
    path.write_text("")

    h1 = acquire_lock(path)
    h1.close()
    # After close, a second session should succeed.
    h2 = acquire_lock(path)
    h2.close()


# -----------------------------------------------------------------------------
# make_backup
# -----------------------------------------------------------------------------


def test_make_backup_writes_a_sibling_with_bak_prefix(tmp_path):
    path = tmp_path / "ev.jsonl"
    records = [_mk_record("p0"), _mk_record("p1")]
    _write_jsonl(path, records)

    bak = make_backup(path)
    assert bak.exists()
    assert bak.name.startswith("ev.jsonl.bak-")
    assert _read_jsonl(bak) == records


def test_make_backup_is_idempotent_within_one_second(tmp_path):
    """Two calls in a single second produce the same path; second is a no-op."""
    path = tmp_path / "ev.jsonl"
    _write_jsonl(path, [_mk_record("p0")])
    bak1 = make_backup(path)
    bak2 = make_backup(path)
    assert bak1 == bak2
