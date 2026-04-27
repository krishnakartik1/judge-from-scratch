"""Unit tests for ``compute_verdict`` — dry-run report verdict logic.

Threshold semantics: a run PASSes (exit 0) when at least half the rows
parsed AND the stereotype-match rate among parsed rows is ≥ 10 %. Rules
mirrored verbatim from the plan's §Dry-run report.
"""

from __future__ import annotations

import pytest
from conftest import stage1_gen

compute_verdict = stage1_gen.compute_verdict


@pytest.mark.parametrize(
    ("total", "parse_fails", "target_matches", "expected_exit"),
    [
        # All parsed; everyone matches the stereotype slot.
        (40, 0, 40, 0),
        # All parsed; ~50% match — well above 10% threshold.
        (40, 0, 20, 0),
        # Exactly 10% match — boundary, should PASS (rule is `< 0.10`).
        (40, 0, 4, 0),
        # Just under 10% match — generator pool too aligned (exit 4).
        (40, 0, 3, 4),
        # Half parse-fail; threshold is `parsed < 0.5*total`, so
        # parsed=20 == 0.5*40 is NOT below threshold → still PASS.
        (40, 20, 20, 0),
        # Just over the parse-fail threshold (parsed=19 < 20 = 0.5*40).
        (40, 21, 0, 5),
        # Zero rows — special-cased: exit 5.
        (0, 0, 0, 5),
        # All parse-fail — exit 5.
        (40, 40, 0, 5),
    ],
)
def test_compute_verdict(
    total: int, parse_fails: int, target_matches: int, expected_exit: int
) -> None:
    exit_code, label = compute_verdict(total, parse_fails, target_matches)
    assert exit_code == expected_exit
    # Label is non-empty for every verdict, and "PASS" only for exit 0.
    assert label
    if expected_exit == 0:
        assert label == "PASS"
    else:
        assert label.startswith("ABORT")
