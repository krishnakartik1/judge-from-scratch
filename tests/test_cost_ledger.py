"""Unit tests for ``train/modal/_cost_ledger.py``."""

from __future__ import annotations

from pathlib import Path

import pytest

from train.modal._cost_ledger import (
    MODAL_GPU_RATES_USD_PER_SEC,
    BudgetExceededError,
    check_budget,
    compute_modal_cost,
    project_cost,
    record_cost,
    total_spend,
)

# ---------------------------------------------------------------------------
# compute_modal_cost / project_cost


def test_compute_modal_cost_a100() -> None:
    """A100 at $1.86/h means 60 s costs $0.031."""
    cost = compute_modal_cost("A100", wallclock_s=60.0)
    assert cost == pytest.approx(1.86 / 3600 * 60, rel=1e-9)


def test_compute_modal_cost_a100_80gb_pricier() -> None:
    """80GB variant must cost more per second than 40GB."""
    assert compute_modal_cost("A100-80GB", 100) > compute_modal_cost("A100-40GB", 100)


def test_compute_modal_cost_unknown_gpu_falls_back_to_cpu() -> None:
    """Unknown GPU specs should not crash — fall back to CPU pricing."""
    cpu_rate = MODAL_GPU_RATES_USD_PER_SEC["CPU"]
    assert compute_modal_cost("InventedGPU-9000", 60) == pytest.approx(cpu_rate * 60)


def test_compute_modal_cost_negative_wallclock_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        compute_modal_cost("A100", -1)


def test_project_cost_uses_worst_case_timeout() -> None:
    """Projection equals timeout × rate (hard upper bound)."""
    timeout = 1800.0  # 30 min A100 dryrun
    projected = project_cost("A100", timeout)
    assert projected == pytest.approx(1.86 / 3600 * 1800, rel=1e-9)


# ---------------------------------------------------------------------------
# total_spend / record_cost


def test_total_spend_zero_when_no_ledger(tmp_path: Path) -> None:
    assert total_spend(tmp_path / "missing.jsonl") == 0.0


def test_record_cost_appends_and_total_sums(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    row1 = record_cost("stage6", "smoke_test", "A100", 60.0, ledger_path=ledger)
    row2 = record_cost("stage6", "train_sft_dryrun", "A100", 300.0, ledger_path=ledger)

    assert row1["est_cost_usd"] > 0
    assert row2["est_cost_usd"] > 0
    expected = row1["est_cost_usd"] + row2["est_cost_usd"]
    assert total_spend(ledger) == pytest.approx(expected)


def test_record_cost_jsonl_one_record_per_line(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    record_cost("stage6", "f1", "CPU", 10.0, ledger_path=ledger)
    record_cost("stage6", "f2", "A100", 20.0, ledger_path=ledger)
    lines = ledger.read_text().strip().split("\n")
    assert len(lines) == 2


def test_record_cost_includes_required_fields(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    row = record_cost(
        "stage6",
        "train_sft_full",
        "A100-80GB",
        wallclock_s=14400.0,
        modal_run_id="ap-abc123",
        notes="full SFT run",
        ledger_path=ledger,
    )
    for key in (
        "timestamp",
        "stage",
        "function",
        "gpu",
        "wallclock_s",
        "rate_usd_per_sec",
        "est_cost_usd",
        "modal_run_id",
        "notes",
    ):
        assert key in row, f"Missing field: {key}"


# ---------------------------------------------------------------------------
# check_budget


def test_check_budget_passes_when_under_cap(
    capsys: pytest.CaptureFixture[str],
) -> None:
    check_budget(projected_usd=1.0, cap_usd=10.0, spent_usd=2.0)
    out = capsys.readouterr().out
    assert "projected $1.00" in out
    assert "spent so far $2.00" in out


def test_check_budget_raises_at_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BUDGET_OVERRIDE", raising=False)
    monkeypatch.setattr("builtins.input", lambda _: "no")
    with pytest.raises(BudgetExceededError, match="cap"):
        check_budget(projected_usd=10.0, cap_usd=5.0, spent_usd=0.0)


def test_check_budget_continues_on_continue_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BUDGET_OVERRIDE", raising=False)
    monkeypatch.setattr("builtins.input", lambda _: "CONTINUE")
    # No raise expected.
    check_budget(projected_usd=10.0, cap_usd=5.0, spent_usd=0.0)


def test_check_budget_force_skips_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """force=True must not call input()."""

    def boom(_prompt: str) -> str:
        raise AssertionError("input() was called despite force=True")

    monkeypatch.setattr("builtins.input", boom)
    monkeypatch.delenv("BUDGET_OVERRIDE", raising=False)
    check_budget(projected_usd=10.0, cap_usd=5.0, force=True)


def test_check_budget_env_override_skips_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BUDGET_OVERRIDE=1 in env must not call input()."""

    def boom(_prompt: str) -> str:
        raise AssertionError("input() was called despite BUDGET_OVERRIDE=1")

    monkeypatch.setattr("builtins.input", boom)
    monkeypatch.setenv("BUDGET_OVERRIDE", "1")
    check_budget(projected_usd=10.0, cap_usd=5.0)


def test_check_budget_uses_total_spend_when_no_spent_arg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default ``spent_usd=None`` should read the ledger."""
    ledger = tmp_path / "ledger.jsonl"
    record_cost("stage6", "f1", "A100", 100.0, ledger_path=ledger)
    monkeypatch.setattr("train.modal._cost_ledger.LEDGER_PATH", ledger)
    monkeypatch.delenv("BUDGET_OVERRIDE", raising=False)
    monkeypatch.setattr("builtins.input", lambda _: "no")

    # Spent ~$0.05; project $5; cap $4 → total $5.05 > $4 → must raise.
    with pytest.raises(BudgetExceededError):
        check_budget(projected_usd=5.0, cap_usd=4.0)
