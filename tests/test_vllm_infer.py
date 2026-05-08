"""Unit tests for ``eval/modal/vllm_infer.py`` — pure helpers only.

No Modal calls, no GPU, no network. Tests cover:

- :func:`run_type_to_partition` ↔ :func:`partition_to_run_type`
  bijection across all 7 valid run_types.
- Cache row write/read round-trip with strict ``Prediction``
  reconstruction (catches the ``Prediction(**rest)`` TypeError the
  plan reviewer flagged).
- Local prompt rendering via a stub tokenizer; ``<|think|>``
  injection raises (decision #13 enforcement).
- :func:`validate_model_name` rejects unknowns.
- :func:`_do_relocate_legacy_cache` semantics on a stub layout.
- Cost-ledger call shapes round-trip through the real pure-Python
  ``_cost_ledger`` module without Modal imports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eval.eval_harness import Prediction
from eval.modal.vllm_infer import (
    CONSISTENCY_RUN_INDEXES,
    CONSISTENCY_TEMPERATURE,
    EXTRA_CACHE_KEYS,
    PREDICTION_KEYS,
    VALID_RUN_TYPES,
    _do_relocate_legacy_cache,
    build_cache_row,
    partition_to_run_type,
    render_prompts_for_pass,
    run_type_to_partition,
    to_prediction,
    validate_model_name,
)

# ---------------------------------------------------------------------------
# run_type ↔ partition-key bijection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("run_type", VALID_RUN_TYPES)
def test_run_type_round_trip(run_type: str) -> None:
    """Every valid run_type round-trips: rt → partition → rt."""
    triple = run_type_to_partition(run_type)
    assert partition_to_run_type(*triple) == run_type


def test_run_type_partition_table() -> None:
    """The bijection table matches the documented schema exactly."""
    assert run_type_to_partition("original") == (0.0, False, 0)
    assert run_type_to_partition("swapped") == (0.0, True, 0)
    for i in CONSISTENCY_RUN_INDEXES:
        assert run_type_to_partition(f"consistency_{i}") == (
            CONSISTENCY_TEMPERATURE,
            False,
            i,
        )


def test_run_type_invalid() -> None:
    with pytest.raises(ValueError):
        run_type_to_partition("unknown")
    with pytest.raises(ValueError):
        run_type_to_partition("consistency_5")  # out of range
    with pytest.raises(ValueError):
        run_type_to_partition("consistency_x")
    with pytest.raises(ValueError):
        partition_to_run_type(0.5, False, 0)


# ---------------------------------------------------------------------------
# Cache row write/read round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("run_type", VALID_RUN_TYPES)
def test_cache_row_roundtrip_to_prediction(run_type: str) -> None:
    """Write→JSON→read→to_prediction yields a Prediction matching inputs.

    Regression for the ``Prediction(**rest)`` TypeError: the row
    schema includes ``model`` and ``run_type``, which must be
    stripped before the dataclass constructor sees the dict.
    """
    raw_output = "<reasoning>r</reasoning><verdict>A</verdict>"
    row = build_cache_row(
        model_name="baseline",
        run_type=run_type,
        pair_id="abc123",
        raw_output=raw_output,
        prompt_hash_str="deadbeef",
    )
    # All schema keys present
    for k in PREDICTION_KEYS + EXTRA_CACHE_KEYS:
        assert k in row, f"missing key {k!r}"

    # Round-trip through JSON
    serialized = json.dumps(row)
    restored = json.loads(serialized)
    pred = to_prediction(restored)

    # Verify the 8 dataclass fields match the original row
    assert isinstance(pred, Prediction)
    assert pred.pair_id == "abc123"
    assert pred.verdict == "A"
    assert pred.reasoning == "r"
    assert pred.raw_output == raw_output
    assert pred.prompt_hash == "deadbeef"
    expected_t, expected_s, expected_idx = run_type_to_partition(run_type)
    assert pred.temperature == expected_t
    assert pred.swapped == expected_s
    assert pred.run_index == expected_idx


def test_to_prediction_missing_field_raises() -> None:
    """Cache row missing a Prediction field surfaces a clear error."""
    bad = {"model": "baseline", "run_type": "original", "pair_id": "x"}
    with pytest.raises(ValueError, match="missing fields"):
        to_prediction(bad)


def test_build_cache_row_parse_failure() -> None:
    """Malformed model output produces a PARSE_FAIL verdict, not crash."""
    row = build_cache_row(
        model_name="sft",
        run_type="original",
        pair_id="p1",
        raw_output="(no tags here at all)",
        prompt_hash_str="abc",
    )
    assert row["verdict"] == "PARSE_FAIL"
    assert row["reasoning"] is None


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


class _StubTokenizer:
    """Whitespace tokenizer with chat-template stub for prompt-rendering tests."""

    def __init__(self, *, leak_thinking: bool = False) -> None:
        self._leak_thinking = leak_thinking

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        body = "\n".join(f"{m['role']}: {m['content']}" for m in conversation)
        if self._leak_thinking:
            return f"<|think|>\n{body}"
        return body

    def encode(self, text: str) -> list[int]:
        return [hash(tok) for tok in text.split()]


def _make_record(pair_id: str = "p1") -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "question_text": "Why?",
        "answer_choices": [
            {"letter": "A", "text": "first"},
            {"letter": "B", "text": "second"},
        ],
        "response_a": {"text": "alpha"},
        "response_b": {"text": "beta"},
    }


def test_render_prompts_for_pass_clean() -> None:
    records = [_make_record("p1"), _make_record("p2")]
    tok = _StubTokenizer()
    prompts, meta = render_prompts_for_pass(records, tok, "system text", swap=False)
    assert len(prompts) == 2
    assert all("alpha" in p and "beta" in p for p in prompts)
    assert {m["pair_id"] for m in meta} == {"p1", "p2"}


def test_render_prompts_swap_swaps_text() -> None:
    records = [_make_record("p1")]
    tok = _StubTokenizer()
    prompts_normal, _ = render_prompts_for_pass(records, tok, "sys", swap=False)
    prompts_swapped, _ = render_prompts_for_pass(records, tok, "sys", swap=True)
    # In normal: A→alpha B→beta; in swapped: A→beta B→alpha. Both
    # contain both strings, so check ordering by the raw template.
    assert prompts_normal[0].index("alpha") < prompts_normal[0].index("beta")
    assert prompts_swapped[0].index("beta") < prompts_swapped[0].index("alpha")


def test_render_prompts_rejects_thinking_token() -> None:
    """Decision #13: ``<|think|>`` in the rendered prompt must hard-fail."""
    records = [_make_record()]
    tok = _StubTokenizer(leak_thinking=True)
    with pytest.raises(AssertionError, match="think"):
        render_prompts_for_pass(records, tok, "sys", swap=False)


# ---------------------------------------------------------------------------
# Model name validation
# ---------------------------------------------------------------------------


def test_validate_model_name_accepts_known() -> None:
    for name in ("baseline", "sft", "dpo"):
        validate_model_name(name)


def test_validate_model_name_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        validate_model_name("ppo")
    with pytest.raises(ValueError):
        validate_model_name("")


# ---------------------------------------------------------------------------
# Legacy cache relocation
# ---------------------------------------------------------------------------


def test_relocate_legacy_cache_moves_files(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    legacy_dir = cache_dir / "legacy_unsloth"
    cache_dir.mkdir()
    (cache_dir / "baseline_predictions.jsonl").write_text("[]\n")
    (cache_dir / "sft_predictions.jsonl").write_text("[]\n")

    result = _do_relocate_legacy_cache(cache_dir, legacy_dir)

    assert result["count"] == 2
    assert sorted(result["moved"]) == [
        "baseline_predictions.jsonl",
        "sft_predictions.jsonl",
    ]
    assert legacy_dir.is_dir()
    assert (legacy_dir / "baseline_predictions.jsonl").exists()
    assert (legacy_dir / "sft_predictions.jsonl").exists()
    # Top-level cache_dir no longer has the predictions files
    remaining = [p.name for p in cache_dir.glob("*_predictions.jsonl") if p.is_file()]
    assert remaining == []


def test_relocate_legacy_cache_idempotent_empty(tmp_path: Path) -> None:
    """Re-running on an empty top-level cache is a no-op."""
    cache_dir = tmp_path / "cache"
    legacy_dir = cache_dir / "legacy_unsloth"
    cache_dir.mkdir()
    legacy_dir.mkdir()

    result = _do_relocate_legacy_cache(cache_dir, legacy_dir)
    assert result == {"moved": [], "count": 0}


def test_relocate_legacy_cache_creates_legacy_dir(tmp_path: Path) -> None:
    """First run with no pre-existing legacy_unsloth/ creates it."""
    cache_dir = tmp_path / "cache"
    legacy_dir = cache_dir / "legacy_unsloth"
    cache_dir.mkdir()
    (cache_dir / "x_predictions.jsonl").write_text("a\n")
    assert not legacy_dir.exists()

    result = _do_relocate_legacy_cache(cache_dir, legacy_dir)
    assert result["count"] == 1
    assert legacy_dir.is_dir()
    assert (legacy_dir / "x_predictions.jsonl").exists()


def test_relocate_legacy_cache_does_not_recurse(tmp_path: Path) -> None:
    """Files already in legacy_unsloth/ are not re-moved."""
    cache_dir = tmp_path / "cache"
    legacy_dir = cache_dir / "legacy_unsloth"
    cache_dir.mkdir()
    legacy_dir.mkdir()
    (legacy_dir / "old_predictions.jsonl").write_text("x\n")

    result = _do_relocate_legacy_cache(cache_dir, legacy_dir)
    assert result["count"] == 0
    # File still there
    assert (legacy_dir / "old_predictions.jsonl").exists()


# ---------------------------------------------------------------------------
# Cost-ledger call shapes (real module, no Modal imports needed)
# ---------------------------------------------------------------------------


def test_cost_ledger_signatures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact call shapes ``run_all`` uses must round-trip cleanly."""
    from train.modal._cost_ledger import (
        STAGE8_BUDGET_CAP_USD,
        check_budget,
        project_cost,
        record_cost,
        total_spend,
    )

    # STAGE8 cap exists and is sane.
    assert STAGE8_BUDGET_CAP_USD > 0

    # project_cost(gpu, timeout_s) — kwarg name must be timeout_s, not
    # duration_s; the original v2 plan got this wrong.
    cost = project_cost("A100-80GB", timeout_s=1800 * 3)
    assert cost > 0

    # record_cost(stage, function, gpu, wallclock_s, ...) signature
    # used by run_all. Use a tmp ledger to avoid polluting the real one.
    tmp_ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr("train.modal._cost_ledger.LEDGER_PATH", tmp_ledger)
    row = record_cost(
        stage="stage8",
        function="vllm_infer.baseline",
        gpu="A100-80GB",
        wallclock_s=42.5,
        notes="unit test",
        ledger_path=tmp_ledger,
    )
    assert row["stage"] == "stage8"
    assert row["function"] == "vllm_infer.baseline"
    assert row["gpu"] == "A100-80GB"
    assert row["wallclock_s"] == 42.5

    # check_budget pre-flight: the worst-case projection should fit
    # under the Stage 8 cap when no prior spend exists.
    spent = total_spend(ledger_path=tmp_ledger)
    check_budget(
        projected_usd=cost,
        cap_usd=STAGE8_BUDGET_CAP_USD,
        spent_usd=spent,
        force=True,  # avoid stdin prompt regardless
        label="unit-test",
    )
