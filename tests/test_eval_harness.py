"""Tests for the pure-Python pieces of ``eval/eval_harness.py``.

CPU-only — no GPU, no model loads, no network. ``StubTokenizer`` is the
shared fake; it implements the parts of the tokenizer surface the
metrics actually call (``encode``).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from eval.eval_harness import (
    EVAL_SLICES,
    IN_DIST_CATEGORIES,
    PARSE_FAIL,
    Prediction,
    PredictionCache,
    aggregate_metrics,
    assert_no_thinking_in_prompt,
    compute_kappa,
    kappa_by_category_in_dist,
    kappa_by_slice,
    load_eval_set,
    parse_failure_rate,
    parse_output,
    position_bias_rate,
    prompt_hash,
    render_markdown_table,
    seed_for,
    select_dryrun_subset,
    self_consistency_rate,
    slug,
    verbosity_bias_score,
)


class StubTokenizer:
    """Whitespace tokenizer used in metric tests."""

    def encode(self, text: str) -> list[int]:
        return [hash(token) for token in text.split()]


def _make_record(
    pair_id: str,
    *,
    human_verdict: str = "A",
    eval_slice: str = "in_dist",
    pair_category: str = "clear_bias_vs_clean",
    response_a_text: str = "short",
    response_b_text: str = "longer response with more tokens here",
) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "human_verdict": human_verdict,
        "eval_slice": eval_slice,
        "pair_category": pair_category,
        "response_a": {"model": "stub-a", "text": response_a_text},
        "response_b": {"model": "stub-b", "text": response_b_text},
    }


def _make_pred(
    pair_id: str,
    *,
    verdict: str = "A",
    swapped: bool = False,
    temperature: float = 0.0,
    run_index: int = 0,
    reasoning: str | None = "stub reasoning",
    raw_output: str = "<reasoning>stub reasoning</reasoning><verdict>A</verdict>",
) -> Prediction:
    return Prediction(
        pair_id=pair_id,
        verdict=verdict,  # type: ignore[arg-type]
        reasoning=reasoning,
        raw_output=raw_output,
        prompt_hash="hash-" + pair_id,
        temperature=temperature,
        swapped=swapped,
        run_index=run_index,
    )


# --- parse_output --------------------------------------------------------


def test_parse_output_happy() -> None:
    text = "<reasoning>both fine</reasoning><verdict>A</verdict>"
    verdict, reasoning = parse_output(text)
    assert verdict == "A"
    assert reasoning == "both fine"


def test_parse_output_multiline_reasoning() -> None:
    text = "<reasoning>line1\nline2\nline3</reasoning><verdict>TIE</verdict>"
    verdict, reasoning = parse_output(text)
    assert verdict == "TIE"
    assert reasoning is not None and "line2" in reasoning


def test_parse_output_missing_verdict() -> None:
    assert parse_output("<reasoning>no verdict</reasoning>") == (PARSE_FAIL, None)


def test_parse_output_invalid_verdict_letter() -> None:
    assert parse_output("<reasoning>ok</reasoning><verdict>C</verdict>") == (
        PARSE_FAIL,
        None,
    )


def test_parse_output_whitespace_between_tags() -> None:
    text = "<reasoning>ok</reasoning>\n  \n<verdict>B</verdict>"
    verdict, reasoning = parse_output(text)
    assert verdict == "B"
    assert reasoning == "ok"


# --- assert_no_thinking_in_prompt ----------------------------------------


def test_assert_no_thinking_passes_clean_prompt() -> None:
    assert_no_thinking_in_prompt("You are a judge. No thinking blocks.")


def test_assert_no_thinking_raises_on_think_token() -> None:
    with pytest.raises(AssertionError, match=r"<\|think\|>"):
        assert_no_thinking_in_prompt("System with <|think|> token.")


# --- load_eval_set -------------------------------------------------------


def test_load_eval_set_reads_records(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    record = _make_record("p1", human_verdict="B")
    path.write_text(json.dumps(record) + "\n")
    records = load_eval_set(path)
    assert len(records) == 1
    assert records[0]["pair_id"] == "p1"


def test_load_eval_set_aborts_on_null_verdict(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    bad = _make_record("p2")
    bad["human_verdict"] = None
    path.write_text(json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match="human_verdict"):
        load_eval_set(path)


def test_load_eval_set_aborts_on_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_eval_set(path)


# --- select_dryrun_subset ------------------------------------------------


def _make_dryrun_pool() -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    for i in range(40):
        pool.append(_make_record(f"in-{i:03d}", eval_slice="in_dist"))
    for i in range(30):
        pool.append(_make_record(f"ood-{i:03d}", eval_slice="ood_religion"))
    return pool


def test_select_dryrun_subset_stratifies_25_25() -> None:
    subset = select_dryrun_subset(_make_dryrun_pool())
    assert len(subset) == 50
    in_dist = [r for r in subset if r["eval_slice"] == "in_dist"]
    ood = [r for r in subset if r["eval_slice"] == "ood_religion"]
    assert len(in_dist) == 25
    assert len(ood) == 25


def test_select_dryrun_subset_deterministic() -> None:
    pool = _make_dryrun_pool()
    a = select_dryrun_subset(pool)
    b = select_dryrun_subset(pool)
    assert [r["pair_id"] for r in a] == [r["pair_id"] for r in b]


def test_select_dryrun_subset_too_few_records_raises() -> None:
    pool = [_make_record(f"in-{i:03d}", eval_slice="in_dist") for i in range(10)] + [
        _make_record(f"ood-{i:03d}", eval_slice="ood_religion") for i in range(30)
    ]
    with pytest.raises(ValueError, match="in_dist"):
        select_dryrun_subset(pool)


# --- prompt_hash, seed_for, slug -----------------------------------------


def test_prompt_hash_changes_with_either_input() -> None:
    a = prompt_hash("sys", "user")
    b = prompt_hash("sys", "USER")
    c = prompt_hash("SYS", "user")
    assert len({a, b, c}) == 3


def test_seed_for_deterministic_and_distinct_per_run() -> None:
    s0 = seed_for("pair-x", 0)
    s0_again = seed_for("pair-x", 0)
    s1 = seed_for("pair-x", 1)
    s2 = seed_for("pair-x", 2)
    assert s0 == s0_again
    assert len({s0, s1, s2}) == 3
    assert all(0 <= s < 2**32 for s in (s0, s1, s2))


def test_slug_filesystem_safe() -> None:
    assert slug("After SFT+DPO") == "after-sft-dpo"
    assert slug("base/path:thing") == "base-path-thing"
    assert slug("!!!") == "unknown"


# --- PredictionCache -----------------------------------------------------


def test_prediction_cache_roundtrip(tmp_path: Path) -> None:
    cache = PredictionCache(tmp_path / "cache.jsonl")
    pred = _make_pred("p1")
    cache.put(pred)
    fetched = cache.get(
        pred.pair_id,
        pred.prompt_hash,
        pred.temperature,
        pred.swapped,
        pred.run_index,
    )
    assert fetched == pred


def test_prediction_cache_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    cache_a = PredictionCache(path)
    pred = _make_pred("p1")
    cache_a.put(pred)

    cache_b = PredictionCache(path)
    fetched = cache_b.get(
        pred.pair_id,
        pred.prompt_hash,
        pred.temperature,
        pred.swapped,
        pred.run_index,
    )
    assert fetched == pred


def test_prediction_cache_skips_malformed_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "cache.jsonl"
    good = _make_pred("p1")
    with open(path, "w") as f:
        f.write(json.dumps(asdict(good)) + "\n")
        f.write("{not valid json\n")

    with caplog.at_level("WARNING"):
        cache = PredictionCache(path)
    assert len(cache) == 1
    assert any("malformed" in r.message for r in caplog.records)


def test_prediction_cache_distinguishes_swap_and_run(tmp_path: Path) -> None:
    cache = PredictionCache(tmp_path / "cache.jsonl")
    p_normal = _make_pred("p1", swapped=False, run_index=0)
    p_swapped = _make_pred("p1", swapped=True, run_index=0, verdict="B")
    p_run1 = _make_pred(
        "p1", swapped=False, run_index=1, temperature=0.3, verdict="TIE"
    )
    cache.put(p_normal)
    cache.put(p_swapped)
    cache.put(p_run1)
    assert cache.get("p1", p_normal.prompt_hash, 0.0, False, 0) == p_normal
    assert cache.get("p1", p_normal.prompt_hash, 0.0, True, 0) == p_swapped
    assert cache.get("p1", p_normal.prompt_hash, 0.3, False, 1) == p_run1
    assert cache.get("p1", p_normal.prompt_hash, 0.0, False, 1) is None


# --- compute_kappa, kappa_by_slice, kappa_by_category_in_dist ------------


def test_compute_kappa_perfect_agreement() -> None:
    records = [
        _make_record("p1", human_verdict="A"),
        _make_record("p2", human_verdict="B"),
        _make_record("p3", human_verdict="TIE"),
    ]
    preds = [
        _make_pred("p1", verdict="A"),
        _make_pred("p2", verdict="B"),
        _make_pred("p3", verdict="TIE"),
    ]
    assert compute_kappa(records, preds) == pytest.approx(1.0)


def test_compute_kappa_excludes_parse_fail() -> None:
    records = [
        _make_record("p1", human_verdict="A"),
        _make_record("p2", human_verdict="B"),
        _make_record("p3", human_verdict="TIE"),
    ]
    preds = [
        _make_pred("p1", verdict="A"),
        _make_pred("p2", verdict=PARSE_FAIL, raw_output="garbage", reasoning=None),
        _make_pred("p3", verdict="TIE"),
    ]
    assert compute_kappa(records, preds) == pytest.approx(1.0)


def test_compute_kappa_returns_nan_when_all_parse_fail() -> None:
    records = [_make_record("p1"), _make_record("p2")]
    preds = [
        _make_pred("p1", verdict=PARSE_FAIL, raw_output="g", reasoning=None),
        _make_pred("p2", verdict=PARSE_FAIL, raw_output="g", reasoning=None),
    ]
    assert math.isnan(compute_kappa(records, preds))


def test_compute_kappa_constant_column_does_not_crash() -> None:
    # All-TIE on both sides — sklearn returns NaN (po=pe=1) without
    # crashing thanks to the explicit ``labels=`` kwarg.
    records = [_make_record(f"p{i}", human_verdict="TIE") for i in range(5)]
    preds = [_make_pred(f"p{i}", verdict="TIE") for i in range(5)]
    result = compute_kappa(records, preds)
    assert math.isnan(result) or result == pytest.approx(1.0)


def test_kappa_by_slice_separates_in_dist_and_ood() -> None:
    records = [
        _make_record("p1", eval_slice="in_dist", human_verdict="A"),
        _make_record("p2", eval_slice="in_dist", human_verdict="B"),
        _make_record("p3", eval_slice="ood_religion", human_verdict="A"),
        _make_record("p4", eval_slice="ood_religion", human_verdict="B"),
    ]
    preds = [
        _make_pred("p1", verdict="A"),
        _make_pred("p2", verdict="B"),
        _make_pred("p3", verdict="B"),  # disagree
        _make_pred("p4", verdict="A"),  # disagree
    ]
    out = kappa_by_slice(records, preds)
    assert set(out.keys()) == set(EVAL_SLICES)
    assert out["in_dist"] == pytest.approx(1.0)
    assert out["ood_religion"] < 0


def test_kappa_by_category_in_dist_skips_ood_records() -> None:
    records = [
        _make_record(
            "p1",
            eval_slice="in_dist",
            pair_category="clear_bias_vs_clean",
            human_verdict="A",
        ),
        _make_record(
            "p2",
            eval_slice="in_dist",
            pair_category="clear_bias_vs_clean",
            human_verdict="B",
        ),
        _make_record(
            "p3",
            eval_slice="ood_religion",
            pair_category="clear_bias_vs_clean",
            human_verdict="A",
        ),
    ]
    preds = [
        _make_pred("p1", verdict="A"),
        _make_pred("p2", verdict="B"),
        _make_pred("p3", verdict="B"),  # disagreement, but OOD: excluded
    ]
    out = kappa_by_category_in_dist(records, preds)
    assert set(out.keys()) == set(IN_DIST_CATEGORIES)
    assert out["clear_bias_vs_clean"] == pytest.approx(1.0)
    assert math.isnan(out["subtle_bias_vs_clean"])


# --- position_bias_rate (9-cell coverage + parse-fail) -------------------


_POSITION_CASES = [
    ("A", "A", True),
    ("A", "B", False),
    ("A", "TIE", True),
    ("B", "A", False),
    ("B", "B", True),
    ("B", "TIE", True),
    ("TIE", "A", True),
    ("TIE", "B", True),
    ("TIE", "TIE", False),
]


@pytest.mark.parametrize(("normal", "swapped", "is_flip"), _POSITION_CASES)
def test_position_bias_per_cell(normal: str, swapped: str, is_flip: bool) -> None:
    records = [_make_record("p1", eval_slice="in_dist")]
    n_preds = [_make_pred("p1", verdict=normal, swapped=False)]
    s_preds = [_make_pred("p1", verdict=swapped, swapped=True)]
    out = position_bias_rate(records, n_preds, s_preds)
    assert out["in_dist"] == pytest.approx(1.0 if is_flip else 0.0)


def test_position_bias_excludes_parse_fail() -> None:
    records = [
        _make_record("p1", eval_slice="in_dist"),
        _make_record("p2", eval_slice="in_dist"),
    ]
    n_preds = [
        _make_pred("p1", verdict="A", swapped=False),
        _make_pred("p2", verdict=PARSE_FAIL, swapped=False, reasoning=None),
    ]
    s_preds = [
        _make_pred("p1", verdict="B", swapped=True),
        _make_pred("p2", verdict="A", swapped=True),
    ]
    out = position_bias_rate(records, n_preds, s_preds)
    # p1 mirrors A→B (no flip); p2 excluded by PARSE_FAIL.
    assert out["in_dist"] == pytest.approx(0.0)


def test_position_bias_nan_when_no_pairs_compared() -> None:
    out = position_bias_rate([], [], [])
    assert math.isnan(out["in_dist"])
    assert math.isnan(out["ood_religion"])


# --- verbosity_bias_score ------------------------------------------------


def test_verbosity_bias_neutral_zero() -> None:
    tokenizer = StubTokenizer()
    records = [
        _make_record(
            "p1",
            response_a_text="alpha beta gamma",  # 3 tokens
            response_b_text="delta epsilon zeta",  # 3 tokens
        )
    ]
    preds = [_make_pred("p1", verdict="A")]
    assert verbosity_bias_score(records, preds, tokenizer) == pytest.approx(0.0)


def test_verbosity_bias_favors_long_positive() -> None:
    tokenizer = StubTokenizer()
    records = [
        _make_record(
            "p1",
            response_a_text="short",  # 1 token
            response_b_text="this response is much much longer here",  # 7 tokens
        )
    ]
    preds = [_make_pred("p1", verdict="B")]  # picked the longer one
    assert verbosity_bias_score(records, preds, tokenizer) == pytest.approx(6.0)


def test_verbosity_bias_skips_tie_and_parse_fail() -> None:
    tokenizer = StubTokenizer()
    records = [
        _make_record("p1", response_a_text="x", response_b_text="x x x"),
        _make_record("p2", response_a_text="y", response_b_text="y y"),
    ]
    preds = [
        _make_pred("p1", verdict="TIE"),
        _make_pred("p2", verdict=PARSE_FAIL, reasoning=None),
    ]
    assert math.isnan(verbosity_bias_score(records, preds, tokenizer))


# --- self_consistency_rate -----------------------------------------------


def test_self_consistency_all_agree_returns_one() -> None:
    records = [_make_record("p1"), _make_record("p2")]
    runs = [
        [
            _make_pred("p1", verdict="A", run_index=i),
            _make_pred("p2", verdict="B", run_index=i),
        ]
        for i in range(2)
    ]
    assert self_consistency_rate(records, runs) == pytest.approx(1.0)


def test_self_consistency_disagreement_drops_rate() -> None:
    records = [_make_record("p1"), _make_record("p2")]
    runs = [
        [
            _make_pred("p1", verdict="A", run_index=0),
            _make_pred("p2", verdict="B", run_index=0),
        ],
        [
            _make_pred("p1", verdict="A", run_index=1),
            _make_pred("p2", verdict="A", run_index=1),
        ],
    ]
    # p1 consistent (A,A); p2 inconsistent (B,A)
    assert self_consistency_rate(records, runs) == pytest.approx(0.5)


def test_self_consistency_excludes_parse_fail() -> None:
    records = [_make_record("p1"), _make_record("p2")]
    runs = [
        [
            _make_pred("p1", verdict="A", run_index=0),
            _make_pred("p2", verdict="B", run_index=0),
        ],
        [
            _make_pred("p1", verdict="A", run_index=1),
            _make_pred(
                "p2",
                verdict=PARSE_FAIL,
                run_index=1,
                reasoning=None,
            ),
        ],
    ]
    # p2 excluded entirely; p1 consistent → 1/1 = 1.0
    assert self_consistency_rate(records, runs) == pytest.approx(1.0)


def test_self_consistency_requires_two_runs() -> None:
    with pytest.raises(ValueError, match="at least 2 runs"):
        self_consistency_rate([], [[]])


def test_self_consistency_aggregate_metrics_uses_baseline_for_one_run() -> None:
    """``aggregate_metrics`` wraps the 1-sampled-run case with the T=0 baseline.

    p1: T=0 verdict A, T=0.3 verdict A → consistent.
    p2: T=0 verdict B, T=0.3 verdict A → inconsistent (1/2 = 0.5).
    """
    tokenizer = StubTokenizer()
    records = [
        _make_record("p1", human_verdict="A"),
        _make_record("p2", human_verdict="B"),
    ]
    normal = [_make_pred("p1", verdict="A"), _make_pred("p2", verdict="B")]
    swapped = [
        _make_pred("p1", verdict="B", swapped=True),
        _make_pred("p2", verdict="A", swapped=True),
    ]
    consistency = [
        [
            _make_pred("p1", verdict="A", temperature=0.3),
            _make_pred("p2", verdict="A", temperature=0.3),
        ]
    ]
    metrics = aggregate_metrics(records, normal, swapped, consistency, tokenizer)
    assert metrics["self_consistency_t03"] == pytest.approx(0.5)


# --- parse_failure_rate --------------------------------------------------


def test_parse_failure_rate_mixed() -> None:
    preds = [
        _make_pred("p1", verdict="A"),
        _make_pred("p2", verdict=PARSE_FAIL, reasoning=None),
        _make_pred("p3", verdict="B"),
        _make_pred("p4", verdict=PARSE_FAIL, reasoning=None),
    ]
    assert parse_failure_rate(preds) == pytest.approx(0.5)


def test_parse_failure_rate_empty_returns_nan() -> None:
    assert math.isnan(parse_failure_rate([]))


# --- render_markdown_table -----------------------------------------------


def _full_metrics() -> dict[str, float]:
    return {
        "kappa_in_dist": 0.5,
        "kappa_ood_religion": 0.4,
        "kappa_clear": 0.7,
        "kappa_subtle": 0.3,
        "kappa_tracked": 0.2,
        "kappa_tie": 0.1,
        "position_bias_in_dist": 0.15,
        "position_bias_ood": 0.20,
        "verbosity_bias": 5.0,
        "self_consistency_t03": 0.92,
        "parse_failure_rate": 0.01,
    }


def test_render_markdown_table_three_columns() -> None:
    table = render_markdown_table(
        {
            "Base": dict(_full_metrics(), kappa_in_dist=0.30),
            "SFT": dict(_full_metrics(), kappa_in_dist=0.55),
            "SFT+DPO": dict(_full_metrics(), kappa_in_dist=0.70),
        }
    )
    lines = table.splitlines()
    assert lines[0] == "| Metric | Base | SFT | SFT+DPO |"
    assert lines[1] == "|---|---|---|---|"
    assert lines[2] == "| Overall κ (in-dist) | 0.300 | 0.550 | 0.700 |"
    # 11 metric rows + header + separator = 13 lines total
    assert len(lines) == 13


def test_render_markdown_table_handles_nan() -> None:
    metrics = dict(_full_metrics(), kappa_in_dist=float("nan"))
    table = render_markdown_table({"Base": metrics})
    assert "—" in table.splitlines()[2]


def test_render_markdown_table_empty_returns_placeholder() -> None:
    assert render_markdown_table({}) == "(no results)"


# --- aggregate_metrics integration --------------------------------------


def test_aggregate_metrics_returns_eleven_keys() -> None:
    tokenizer = StubTokenizer()
    records = [
        _make_record(
            "p1",
            eval_slice="in_dist",
            pair_category="clear_bias_vs_clean",
            human_verdict="A",
        ),
        _make_record(
            "p2",
            eval_slice="ood_religion",
            pair_category="subtle_bias_vs_clean",
            human_verdict="B",
        ),
    ]
    normal = [_make_pred("p1", verdict="A"), _make_pred("p2", verdict="B")]
    swapped = [
        _make_pred("p1", verdict="B", swapped=True),
        _make_pred("p2", verdict="A", swapped=True),
    ]
    consistency = [
        [
            _make_pred("p1", verdict="A", temperature=0.3, run_index=0),
            _make_pred("p2", verdict="B", temperature=0.3, run_index=0),
        ]
    ]
    metrics = aggregate_metrics(records, normal, swapped, consistency, tokenizer)
    assert set(metrics.keys()) == {
        "kappa_in_dist",
        "kappa_ood_religion",
        "kappa_clear",
        "kappa_subtle",
        "kappa_tracked",
        "kappa_tie",
        "position_bias_in_dist",
        "position_bias_ood",
        "verbosity_bias",
        "self_consistency_t03",
        "parse_failure_rate",
    }
