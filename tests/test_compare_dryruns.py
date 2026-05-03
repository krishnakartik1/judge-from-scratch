"""Unit tests for scripts/compare_dryruns.py."""

from __future__ import annotations

from typing import Any

import pytest

from scripts.compare_dryruns import compare_dryrun_runs, decide_flip_gate


def _mk(
    pair_id: str,
    verdict: str,
    confidence: int,
    pair_category: str = "subtle_bias_vs_clean",
) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "pair_category": pair_category,
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": f"reasoning for {pair_id} ({verdict})",
    }


# -----------------------------------------------------------------------------
# compare_dryrun_runs
# -----------------------------------------------------------------------------


def test_compare_all_identical():
    v1 = [_mk("a", "A", 4), _mk("b", "TIE", 5), _mk("c", "B", 3)]
    v3 = [_mk("a", "A", 4), _mk("b", "TIE", 5), _mk("c", "B", 3)]
    out = compare_dryrun_runs(v1, v3)
    assert out["identical"] == 3
    assert out["verdict_flipped"] == 0
    assert out["confidence_shifted"] == 0
    assert out["flip_pair_ids"] == []


def test_compare_one_verdict_flip():
    v1 = [_mk("a", "A", 4), _mk("b", "TIE", 5)]
    v3 = [_mk("a", "B", 4), _mk("b", "TIE", 5)]
    out = compare_dryrun_runs(v1, v3)
    assert out["identical"] == 1
    assert out["verdict_flipped"] == 1
    assert out["confidence_shifted"] == 0
    assert out["flip_pair_ids"] == ["a"]
    assert out["flip_details"][0]["v1_verdict"] == "A"
    assert out["flip_details"][0]["v3_verdict"] == "B"


def test_compare_confidence_shift_only():
    v1 = [_mk("a", "A", 4), _mk("b", "TIE", 5)]
    v3 = [_mk("a", "A", 5), _mk("b", "TIE", 5)]
    out = compare_dryrun_runs(v1, v3)
    assert out["identical"] == 1
    assert out["verdict_flipped"] == 0
    assert out["confidence_shifted"] == 1
    assert out["flip_pair_ids"] == []
    assert out["shift_details"][0]["v1_confidence"] == 4
    assert out["shift_details"][0]["v3_confidence"] == 5


def test_compare_verdict_flip_takes_priority_over_confidence():
    # Same pair: different verdict AND different confidence.
    # Should count as verdict_flipped, not confidence_shifted.
    v1 = [_mk("a", "A", 4)]
    v3 = [_mk("a", "B", 5)]
    out = compare_dryrun_runs(v1, v3)
    assert out["verdict_flipped"] == 1
    assert out["confidence_shifted"] == 0


def test_compare_mismatched_pair_ids():
    v1 = [_mk("a", "A", 4), _mk("b", "B", 3)]
    v3 = [_mk("a", "A", 4), _mk("c", "TIE", 5)]
    out = compare_dryrun_runs(v1, v3)
    assert out["n_common"] == 1
    assert out["n_only_v1"] == 1  # "b" only in v1
    assert out["n_only_v3"] == 1  # "c" only in v3
    assert out["identical"] == 1


def test_compare_empty_inputs():
    out = compare_dryrun_runs([], [])
    assert out["n_common"] == 0
    assert out["identical"] == 0
    assert out["verdict_flipped"] == 0


def test_compare_50_pair_realistic():
    # 50 pairs: 45 identical, 3 verdict-flipped, 2 confidence-shifted
    v1 = []
    v3 = []
    for i in range(45):
        v1.append(_mk(f"id_{i}", "A", 4))
        v3.append(_mk(f"id_{i}", "A", 4))
    for i in range(45, 48):
        v1.append(_mk(f"id_{i}", "A", 4))
        v3.append(_mk(f"id_{i}", "B", 4))
    for i in range(48, 50):
        v1.append(_mk(f"id_{i}", "A", 4))
        v3.append(_mk(f"id_{i}", "A", 5))
    out = compare_dryrun_runs(v1, v3)
    assert out["identical"] == 45
    assert out["verdict_flipped"] == 3
    assert out["confidence_shifted"] == 2
    assert sorted(out["flip_pair_ids"]) == ["id_45", "id_46", "id_47"]


# -----------------------------------------------------------------------------
# decide_flip_gate
# -----------------------------------------------------------------------------


def test_decide_flip_gate_pass_under_threshold():
    decision, msg = decide_flip_gate(flips=3, n_common=50, max_rate=0.10)
    assert decision == "PASS"
    assert "6.0%" in msg
    assert "10.0%" in msg


def test_decide_flip_gate_pass_at_threshold():
    decision, _ = decide_flip_gate(flips=5, n_common=50, max_rate=0.10)
    assert decision == "PASS"


def test_decide_flip_gate_fail_just_above():
    decision, _ = decide_flip_gate(flips=6, n_common=50, max_rate=0.10)
    assert decision == "FAIL"


def test_decide_flip_gate_fail_far_above():
    decision, msg = decide_flip_gate(flips=20, n_common=50, max_rate=0.10)
    assert decision == "FAIL"
    assert "40.0%" in msg


def test_decide_flip_gate_zero_common_passes_with_note():
    decision, msg = decide_flip_gate(flips=0, n_common=0, max_rate=0.10)
    assert decision == "PASS"
    assert "nothing to compare" in msg


def test_decide_flip_gate_custom_threshold():
    decision, _ = decide_flip_gate(flips=10, n_common=50, max_rate=0.25)
    assert decision == "PASS"
    decision, _ = decide_flip_gate(flips=15, n_common=50, max_rate=0.25)
    assert decision == "FAIL"


# -----------------------------------------------------------------------------
# CLI smoke test (exit codes via main)
# -----------------------------------------------------------------------------


def test_main_exit_0_on_pass(tmp_path):
    from scripts.common import jsonl_append
    from scripts.compare_dryruns import main

    v1_path = tmp_path / "v1.jsonl"
    v3_path = tmp_path / "v3.jsonl"
    for r in [_mk("a", "A", 4), _mk("b", "B", 3)]:
        jsonl_append(v1_path, r)
        jsonl_append(v3_path, r)

    rc = main(["--v1", str(v1_path), "--v3", str(v3_path)])
    assert rc == 0


def test_main_exit_3_on_fail(tmp_path):
    from scripts.common import jsonl_append
    from scripts.compare_dryruns import main

    v1_path = tmp_path / "v1.jsonl"
    v3_path = tmp_path / "v3.jsonl"
    # 5 pairs, 2 flipped → 40% > 10% threshold → FAIL
    for i in range(5):
        jsonl_append(v1_path, _mk(f"id_{i}", "A", 4))
    for i, verdict in enumerate(["A", "A", "A", "B", "B"]):
        jsonl_append(v3_path, _mk(f"id_{i}", verdict, 4))
    rc = main(["--v1", str(v1_path), "--v3", str(v3_path)])
    assert rc == 3


def test_main_exit_1_on_missing_required_field(tmp_path):
    from scripts.common import jsonl_append
    from scripts.compare_dryruns import main

    v1_path = tmp_path / "v1.jsonl"
    v3_path = tmp_path / "v3.jsonl"
    jsonl_append(v1_path, {"pair_id": "a", "verdict": "A"})  # missing confidence
    jsonl_append(v3_path, _mk("a", "A", 4))
    rc = main(["--v1", str(v1_path), "--v3", str(v3_path)])
    assert rc == 1


def test_main_writes_out_file(tmp_path):
    from scripts.common import jsonl_append
    from scripts.compare_dryruns import main

    v1_path = tmp_path / "v1.jsonl"
    v3_path = tmp_path / "v3.jsonl"
    out_path = tmp_path / "comparison.json"
    for r in [_mk("a", "A", 4)]:
        jsonl_append(v1_path, r)
        jsonl_append(v3_path, r)

    rc = main(["--v1", str(v1_path), "--v3", str(v3_path), "--out", str(out_path)])
    assert rc == 0
    assert out_path.exists()


@pytest.mark.parametrize(
    "rate,flips,n,expected",
    [
        (0.10, 0, 50, "PASS"),
        (0.10, 5, 50, "PASS"),
        (0.10, 6, 50, "FAIL"),
        (0.20, 10, 50, "PASS"),
        (0.20, 11, 50, "FAIL"),
        (0.0, 0, 50, "PASS"),
        (0.0, 1, 50, "FAIL"),
    ],
)
def test_decide_flip_gate_table(rate, flips, n, expected):
    decision, _ = decide_flip_gate(flips=flips, n_common=n, max_rate=rate)
    assert decision == expected
