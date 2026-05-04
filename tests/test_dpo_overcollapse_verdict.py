"""Unit tests for ``compute_overcollapse_verdict`` — DPO health check.

The detector reads ``trainer.state.log_history`` and classifies the
back half of training as OK / WARN / ABORT based on margin and
log-prob deltas. Thresholds documented in
``Critical risk #6`` of the Stage 7 plan.
"""

from __future__ import annotations

from train.modal.dpo import compute_overcollapse_verdict


def _hist(points: list[dict[str, float]]) -> list[dict[str, float]]:
    """Wrap raw metric dicts in the shape trainer.state.log_history uses."""
    return [{**p, "step": i + 1} for i, p in enumerate(points)]


def test_insufficient_data_when_zero_metric_points() -> None:
    out = compute_overcollapse_verdict([{"step": 1, "loss": 1.0}])
    assert out["verdict"] == "INSUFFICIENT_DATA"
    assert out["n_points"] == 0


def test_ok_when_margins_grow_and_accuracy_high() -> None:
    """Healthy DPO: margin separates, accuracy comfortably > 0.55."""
    history = _hist(
        [
            {
                "rewards/margins": 0.0,
                "logps/chosen": -50.0,
                "logps/rejected": -50.0,
                "rewards/accuracies": 0.50,
            },
            {
                "rewards/margins": 1.5,
                "logps/chosen": -50.5,
                "logps/rejected": -52.0,
                "rewards/accuracies": 0.75,
            },
        ]
    )
    out = compute_overcollapse_verdict(history)
    assert out["verdict"] == "OK"
    assert out["margin_delta"] > 0.5
    assert out["final_accuracy"] >= 0.55


def test_abort_when_margin_does_not_separate() -> None:
    history = _hist(
        [
            {
                "rewards/margins": 0.5,
                "logps/chosen": -50.0,
                "logps/rejected": -50.5,
                "rewards/accuracies": 0.60,
            },
            {
                "rewards/margins": 0.55,  # delta 0.05, below 0.10 threshold
                "logps/chosen": -50.0,
                "logps/rejected": -50.6,
                "rewards/accuracies": 0.61,
            },
        ]
    )
    out = compute_overcollapse_verdict(history)
    assert out["verdict"] == "ABORT"
    assert "margin_delta" in out["reason"]


def test_abort_when_both_logps_collapse_with_weak_separation() -> None:
    """Classic over-collapse: chosen and rejected both crash, margin anemic."""
    history = _hist(
        [
            {
                "rewards/margins": 0.10,
                "logps/chosen": -50.0,
                "logps/rejected": -50.5,
                "rewards/accuracies": 0.55,
            },
            {
                "rewards/margins": 0.30,  # passes margin>0.10 floor
                "logps/chosen": -53.5,  # delta -3.5
                "logps/rejected": -54.0,  # delta -3.5
                "rewards/accuracies": 0.56,
            },
        ]
    )
    out = compute_overcollapse_verdict(history)
    assert out["verdict"] == "ABORT"
    assert "collapsing" in out["reason"]


def test_abort_when_accuracy_below_coin_flip() -> None:
    history = _hist(
        [
            {
                "rewards/margins": 0.0,
                "logps/chosen": -50.0,
                "logps/rejected": -50.0,
                "rewards/accuracies": 0.50,
            },
            {
                "rewards/margins": 1.0,
                "logps/chosen": -50.0,
                "logps/rejected": -51.0,
                "rewards/accuracies": 0.45,  # below 0.50
            },
        ]
    )
    out = compute_overcollapse_verdict(history)
    assert out["verdict"] == "ABORT"


def test_warn_when_chosen_logp_falls_fast() -> None:
    """Margin grows but model is winning by depressing chosen too."""
    history = _hist(
        [
            {
                "rewards/margins": 0.0,
                "logps/chosen": -50.0,
                "logps/rejected": -50.0,
                "rewards/accuracies": 0.50,
            },
            {
                "rewards/margins": 0.30,  # >0.10 (passes floor) but <0.50 (not OK)
                "logps/chosen": -52.5,  # delta -2.5 → fast fall
                "logps/rejected": -52.8,
                "rewards/accuracies": 0.62,
            },
        ]
    )
    out = compute_overcollapse_verdict(history)
    assert out["verdict"] == "WARN"


def test_includes_raw_deltas_for_dryrun_report() -> None:
    history = _hist(
        [
            {
                "rewards/margins": 0.0,
                "logps/chosen": -50.0,
                "logps/rejected": -50.0,
                "rewards/accuracies": 0.50,
            },
            {
                "rewards/margins": 1.0,
                "logps/chosen": -50.0,
                "logps/rejected": -51.0,
                "rewards/accuracies": 0.70,
            },
        ]
    )
    out = compute_overcollapse_verdict(history)
    for key in (
        "margin_delta",
        "chosen_logp_delta",
        "rejected_logp_delta",
        "final_accuracy",
        "n_points",
    ):
        assert key in out
