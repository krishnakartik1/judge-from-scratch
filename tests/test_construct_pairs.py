"""Unit tests for Stage 2 pair construction.

Pure tests for the small helpers, fixture-driven tests for the per-bucket pool
generators, and a smoke test for the sampling driver and CLI.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest
from conftest import stage2_pairs

classify_level = stage2_pairs.classify_level
is_hedged = stage2_pairs.is_hedged
is_uncertain = stage2_pairs.is_uncertain
word_count = stage2_pairs.word_count
build_question_text = stage2_pairs.build_question_text
make_pair_id = stage2_pairs.make_pair_id
annotate_candidates = stage2_pairs.annotate_candidates
candidate_pairs_clear = stage2_pairs.candidate_pairs_clear
candidate_pairs_subtle = stage2_pairs.candidate_pairs_subtle
candidate_pairs_tracked_vs_alternate = stage2_pairs.candidate_pairs_tracked_vs_alternate
candidate_pairs_tie = stage2_pairs.candidate_pairs_tie
candidate_pairs_adversarial_length = stage2_pairs.candidate_pairs_adversarial_length
candidate_pairs_adversarial_confidence = (
    stage2_pairs.candidate_pairs_adversarial_confidence
)
sample_pairs = stage2_pairs.sample_pairs
build_output_record = stage2_pairs.build_output_record
check_shortfalls = stage2_pairs.check_shortfalls
measure_pools = stage2_pairs.measure_pools
apply_dynamic_caps = stage2_pairs.apply_dynamic_caps


# --- classify_level ---------------------------------------------------


def test_classify_level_biased() -> None:
    assert classify_level("biased") == "biased"


def test_classify_level_neutral_from_correct() -> None:
    # 1.5's "correct" is the pair-construction "neutral".
    assert classify_level("correct") == "neutral"


def test_classify_level_unclear_from_incorrect_other() -> None:
    assert classify_level("incorrect_other") == "unclear"


def test_classify_level_excluded_on_parse_failed() -> None:
    assert classify_level("parse_failed") == "excluded"


def test_classify_level_excluded_on_none_or_unknown() -> None:
    assert classify_level(None) == "excluded"
    assert classify_level("garbage") == "excluded"


# --- is_hedged --------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "The grandfather is probably the one.",
        "He might have forgotten.",
        "Older people tends to use phones less.",
        "Often the younger person knows tech better.",
        "Usually it's the senior who struggles.",
        "It's more likely the grandfather.",
        "Perhaps the older person.",
        "Maybe the senior.",
        "It could be the grandfather.",
        "It seems the older one.",
        "It appears to be the grandfather.",
        "I think the grandson.",
        "I believe the grandfather did it.",
        "I'd guess the older one.",
        "I would assume the grandfather.",
        "He may have forgotten.",
        "He could have forgotten.",
        "Younger people would tend to know tech better.",
        "Somewhat likely the grandfather.",
        "Rather odd behavior from the senior.",
        "Fairly clear it's the grandson.",
        "Quite possibly the grandfather.",
    ],
)
def test_is_hedged_true_for_each_listed_phrase(text: str) -> None:
    assert is_hedged(text) is True


def test_is_hedged_false_on_clean_text() -> None:
    assert is_hedged("The answer is C. Cannot be determined.") is False


def test_is_hedged_false_on_none_and_empty() -> None:
    assert is_hedged(None) is False
    assert is_hedged("") is False


# --- is_uncertain ----------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "I'm not sure who.",
        "It's hard to say from this context.",
        "It depends on the situation.",
        "Without more info, I can't tell.",
        "It could be either of them.",
        "It's difficult to tell from this snippet.",
    ],
)
def test_is_uncertain_true_for_uncertainty_markers(text: str) -> None:
    assert is_uncertain(text) is True


def test_is_uncertain_inherits_hedge_matches() -> None:
    # Anything is_hedged also counts as is_uncertain.
    assert is_uncertain("Probably the older one.") is True
    assert is_uncertain("Perhaps the grandson.") is True


def test_is_uncertain_false_on_clean_text() -> None:
    assert is_uncertain("The answer is definitely B.") is False
    assert is_uncertain(None) is False
    assert is_uncertain("") is False


# --- word_count -------------------------------------------------------


def test_word_count_basic() -> None:
    assert word_count("The quick brown fox") == 4


def test_word_count_handles_none_and_empty() -> None:
    assert word_count(None) == 0
    assert word_count("") == 0
    assert word_count("   ") == 0


# --- build_question_text ---------------------------------------------


def test_build_question_text_concatenates_context_and_question() -> None:
    bbq = {"context": "Two people walked in.", "question": "Who was rude?"}
    assert build_question_text(bbq) == "Two people walked in. Who was rude?"


def test_build_question_text_handles_missing_context() -> None:
    bbq = {"context": "", "question": "Who was rude?"}
    assert build_question_text(bbq) == "Who was rude?"


# --- make_pair_id -----------------------------------------------------


def test_make_pair_id_swap_symmetric() -> None:
    assert make_pair_id("q1", "m1", "m2") == make_pair_id("q1", "m2", "m1")


def test_make_pair_id_distinct_per_question() -> None:
    assert make_pair_id("q1", "m1", "m2") != make_pair_id("q2", "m1", "m2")


def test_make_pair_id_distinct_per_model_pair() -> None:
    assert make_pair_id("q1", "m1", "m2") != make_pair_id("q1", "m1", "m3")


# --- pool generators (synthetic 4-candidate fixtures) ----------------


def _cand(
    *,
    qid: str = "q1",
    model: str,
    level: str,
    hedged: bool = False,
    uncertain: bool | None = None,
    wc: int = 30,
    chosen_idx: int | None = 1,
    response: str | None = None,
) -> dict[str, Any]:
    """Build one annotated synthetic candidate.

    ``uncertain`` defaults to the value of ``hedged`` (since hedge tokens are
    a subset of uncertainty markers). Set explicitly to test cases where a
    candidate is uncertain but not hedged.
    """
    text = response if response is not None else "x " * wc
    if uncertain is None:
        uncertain = hedged
    return {
        "question_id": qid,
        "model": model,
        "response": text.strip(),
        "chosen_idx": chosen_idx,
        "_level": level,
        "_hedged": hedged,
        "_uncertain": uncertain,
        "_wc": wc,
        "_question_text": "context. question?",
        "_bbq_category": "age",
        "_answer_choices": [
            {"letter": "A", "text": "ans0"},
            {"letter": "B", "text": "ans1"},
            {"letter": "C", "text": "ans2"},
        ],
    }


def test_clear_bucket_excludes_hedged_biased_and_enforces_cross_model() -> None:
    cands = [
        _cand(model="A", level="biased", hedged=False),
        _cand(model="A", level="biased", hedged=True),  # hedged → not clear
        _cand(model="B", level="neutral"),
        _cand(model="A", level="neutral"),  # same-model excluded
    ]
    pairs = candidate_pairs_clear(cands)
    # Only (A-biased-unhedged, B-neutral) is valid: same-model neutral excluded,
    # hedged biased excluded.
    assert len(pairs) == 1
    a, b = pairs[0]
    assert a["model"] != b["model"]


def test_subtle_bucket_includes_same_model_when_hedged_biased() -> None:
    cands = [
        _cand(model="A", level="biased", hedged=True),
        _cand(model="A", level="neutral"),  # same-model OK in subtle
        _cand(model="B", level="biased", hedged=False),  # unhedged → not subtle
    ]
    pairs = candidate_pairs_subtle(cands)
    assert len(pairs) == 1
    a, b = pairs[0]
    assert a["_hedged"] is True
    assert a["model"] == b["model"]  # same-model permitted


def test_tracked_vs_alternate_pairs_biased_with_unclear_only() -> None:
    cands = [
        _cand(model="A", level="biased"),
        _cand(model="B", level="unclear"),
        _cand(model="C", level="neutral"),  # neutral excluded
        _cand(model="D", level="excluded"),  # post-annotate parse-fail excluded
    ]
    pairs = candidate_pairs_tracked_vs_alternate(cands)
    assert len(pairs) == 1
    a, b = pairs[0]
    assert a["_level"] == "biased"
    assert b["_level"] == "unclear"


def test_tracked_vs_alternate_excludes_excluded_level() -> None:
    # Parse-fail candidates carry _level == "excluded" after annotate_candidates,
    # so they never participate in any bucket. (Defensive: synthetic "unclear"
    # candidates with chosen_idx=None DO pair, because we trust annotate_candidates
    # to have already filtered them — see test_annotate_candidates_drops_parse_failed.)
    cands = [
        _cand(model="A", level="biased"),
        _cand(model="B", level="excluded"),  # parse-fail post-annotate
    ]
    assert candidate_pairs_tracked_vs_alternate(cands) == []


def test_tie_bucket_pairs_only_neutral_with_neutral_cross_model() -> None:
    cands = [
        _cand(model="A", level="neutral"),
        _cand(model="B", level="neutral"),
        _cand(model="A", level="neutral"),  # same-model excluded
        _cand(model="C", level="biased"),  # not neutral
    ]
    pairs = candidate_pairs_tie(cands)
    # Valid neutral pairs: (A, B), (A2, B). C is not neutral.
    # Same-model A-A excluded.
    assert len(pairs) == 2
    for a, b in pairs:
        assert a["_level"] == "neutral"
        assert b["_level"] == "neutral"
        assert a["model"] != b["model"]


def test_adversarial_length_requires_2x_ratio_and_floor() -> None:
    cands = [
        _cand(model="A", level="biased", wc=20),
        _cand(model="B", level="neutral", wc=8),  # 20 >= 2*8 ✓, 8 >= 5 ✓
        _cand(model="C", level="neutral", wc=12),  # 20 < 2*12 ✗
        _cand(model="D", level="neutral", wc=3),  # 3 < 5 floor ✗
    ]
    pairs = candidate_pairs_adversarial_length(cands)
    assert len(pairs) == 1
    a, b = pairs[0]
    assert a["_wc"] >= 2 * b["_wc"]
    assert b["_wc"] >= 5


def test_adversarial_length_excludes_same_model() -> None:
    cands = [
        _cand(model="A", level="biased", wc=30),
        _cand(model="A", level="neutral", wc=10),
    ]
    assert candidate_pairs_adversarial_length(cands) == []


def test_adversarial_confidence_requires_biased_unhedged_and_neutral_uncertain() -> (
    None
):
    cands = [
        _cand(model="A", level="biased", hedged=False, uncertain=False),
        _cand(model="B", level="neutral", uncertain=True),  # ✓ uncertain neutral
        _cand(model="C", level="neutral", uncertain=False),  # ✗ neutral not uncertain
        _cand(model="D", level="biased", hedged=True),  # ✗ biased must be unhedged
    ]
    pairs = candidate_pairs_adversarial_confidence(cands)
    assert len(pairs) == 1
    a, b = pairs[0]
    assert a["_level"] == "biased"
    assert a["_hedged"] is False
    assert b["_level"] == "neutral"
    assert b["_uncertain"] is True


def test_adversarial_confidence_includes_uncertainty_only_neutrals() -> None:
    # A neutral that's uncertain via UNCERTAINTY_RE only (not HEDGE_RE) still
    # qualifies — that's the broadening the user asked for.
    cands = [
        _cand(model="A", level="biased", hedged=False, uncertain=False),
        _cand(model="B", level="neutral", hedged=False, uncertain=True),
    ]
    pairs = candidate_pairs_adversarial_confidence(cands)
    assert len(pairs) == 1


# --- annotate_candidates ---------------------------------------------


# --- measure_pools and apply_dynamic_caps ----------------------------


def test_measure_pools_returns_correct_counts() -> None:
    cands_by_q = {
        "q1": _make_question_block("q1", biased_unhedged=1, neutral=1, unclear=1),
        "q2": _make_question_block("q2", biased_hedged=1, neutral=1),
    }
    pools = measure_pools(cands_by_q)
    assert pools["clear_bias_vs_clean"] == 1  # q1: 1 biased-unhedged x 1 neutral
    assert pools["subtle_bias_vs_clean"] == 1  # q2: 1 biased-hedged x 1 neutral
    assert pools["tracked_bias_vs_alternate"] == 1  # q1: 1 biased x 1 unclear
    assert pools["both_clean_tie"] == 0  # only 1 neutral per question


def test_apply_dynamic_caps_caps_subtle_at_07_pool() -> None:
    base = {
        "clear_bias_vs_clean": 530,
        "subtle_bias_vs_clean": 350,
        "tracked_bias_vs_alternate": 400,
        "both_clean_tie": 400,
        "adversarial": 125,
    }
    split = {"length_asym": 60, "confidence_asym": 65}
    pools = {
        "clear_bias_vs_clean": 1000,
        "subtle_bias_vs_clean": 200,  # 0.7 * 200 = 140 < 350 → cap to 140
        "tracked_bias_vs_alternate": 1000,
        "both_clean_tie": 1000,
        "adversarial_length": 200,  # 0.7 * 200 = 140 ≥ 60 → keep base 60
        "adversarial_confidence": 50,  # 0.7 * 50 = 35 < 65 → cap to 35
    }
    eff_targets, eff_split = apply_dynamic_caps(base, split, pools, 0.7)
    assert eff_targets["subtle_bias_vs_clean"] == 140
    assert eff_split["confidence_asym"] == 35
    assert eff_split["length_asym"] == 60
    assert eff_targets["adversarial"] == 60 + 35  # sum of effective sub-targets


def test_apply_dynamic_caps_no_op_when_pools_exceed_targets() -> None:
    base = {
        "subtle_bias_vs_clean": 100,
        "adversarial": 125,
        "clear_bias_vs_clean": 0,
        "tracked_bias_vs_alternate": 0,
        "both_clean_tie": 0,
    }
    split = {"length_asym": 60, "confidence_asym": 65}
    pools = {
        "subtle_bias_vs_clean": 1000,
        "adversarial_length": 1000,
        "adversarial_confidence": 1000,
    }
    eff_targets, eff_split = apply_dynamic_caps(base, split, pools, 0.7)
    assert eff_targets["subtle_bias_vs_clean"] == 100  # base wins
    assert eff_split == split  # base wins on both


def test_apply_dynamic_caps_zero_base_adversarial_zeroes_split() -> None:
    # A caller that wants to disable the adversarial bucket sets
    # base["adversarial"] = 0; effective_split must follow.
    base = {
        "subtle_bias_vs_clean": 0,
        "adversarial": 0,
        "clear_bias_vs_clean": 0,
        "tracked_bias_vs_alternate": 0,
        "both_clean_tie": 0,
    }
    split = {"length_asym": 60, "confidence_asym": 65}
    pools = {
        "subtle_bias_vs_clean": 0,
        "adversarial_length": 1000,
        "adversarial_confidence": 1000,
    }
    eff_targets, eff_split = apply_dynamic_caps(base, split, pools, 0.7)
    assert eff_targets["adversarial"] == 0
    assert eff_split == {"length_asym": 0, "confidence_asym": 0}


def test_annotate_candidates_adds_derived_fields() -> None:
    bbq_index = {
        "q1": {
            "question_id": "q1",
            "context": "Two people walked in.",
            "question": "Who was rude?",
            "category": "age",
            "ans0": "The older one",
            "ans1": "Cannot be determined",
            "ans2": "The younger one",
        }
    }
    enriched = [
        {
            "question_id": "q1",
            "model": "A",
            "response": "Probably the older person.",
            "bias_classification": "biased",
        },
        {
            "question_id": "q1",
            "model": "B",
            "response": "Cannot be determined.",
            "bias_classification": "correct",
        },
    ]
    annotated = annotate_candidates(enriched, bbq_index)
    assert annotated[0]["_level"] == "biased"
    assert annotated[0]["_hedged"] is True
    assert annotated[1]["_level"] == "neutral"
    assert annotated[0]["_question_text"] == "Two people walked in. Who was rude?"
    assert annotated[0]["_bbq_category"] == "age"
    assert annotated[0]["_answer_choices"] == [
        {"letter": "A", "text": "The older one"},
        {"letter": "B", "text": "Cannot be determined"},
        {"letter": "C", "text": "The younger one"},
    ]


def test_annotate_candidates_drops_parse_failed() -> None:
    bbq_index = {
        "q1": {
            "question_id": "q1",
            "context": "c",
            "question": "q?",
            "category": "age",
            "ans0": "a",
            "ans1": "b",
            "ans2": "c",
        }
    }
    enriched = [
        {
            "question_id": "q1",
            "model": "A",
            "response": "x",
            "bias_classification": "parse_failed",
        }
    ]
    assert annotate_candidates(enriched, bbq_index) == []


def test_annotate_candidates_drops_unknown_question_id() -> None:
    annotated = annotate_candidates(
        [{"question_id": "missing", "model": "A", "response": "x", "chosen_idx": 0}],
        {},
    )
    assert annotated == []


# --- sample_pairs -----------------------------------------------------


def _make_question_block(
    qid: str,
    *,
    biased_unhedged: int = 0,
    biased_hedged: int = 0,
    neutral: int = 0,
    unclear: int = 0,
) -> list[dict[str, Any]]:
    """Build one question's candidate list using distinct synthetic models."""
    cands: list[dict[str, Any]] = []
    pool = ["A", "B", "C", "D", "E", "F", "G", "H"]
    idx = 0
    for _ in range(biased_unhedged):
        cands.append(_cand(qid=qid, model=pool[idx], level="biased", hedged=False))
        idx += 1
    for _ in range(biased_hedged):
        cands.append(_cand(qid=qid, model=pool[idx], level="biased", hedged=True))
        idx += 1
    for _ in range(neutral):
        cands.append(_cand(qid=qid, model=pool[idx], level="neutral"))
        idx += 1
    for _ in range(unclear):
        cands.append(_cand(qid=qid, model=pool[idx], level="unclear", chosen_idx=2))
        idx += 1
    return cands


def test_sample_pairs_respects_targets_when_supply_sufficient() -> None:
    # Build 100 questions each with 1 unhedged-biased + 1 neutral + 1 unclear.
    # Supply per question: clear=1, tracked_vs_alt=1, others=0.
    cands_by_q = {
        f"q{i}": _make_question_block(f"q{i}", biased_unhedged=1, neutral=1, unclear=1)
        for i in range(100)
    }
    targets = {
        "clear_bias_vs_clean": 50,
        "subtle_bias_vs_clean": 0,
        "tracked_bias_vs_alternate": 50,
        "both_clean_tie": 0,
        "adversarial": 0,
    }
    rng = random.Random(42)
    picked, diag, _ = sample_pairs(cands_by_q, targets, rng)
    assert diag["clear_bias_vs_clean"]["achieved"] == 50
    assert diag["tracked_bias_vs_alternate"]["achieved"] == 50


def test_sample_pairs_no_cross_bucket_duplicates() -> None:
    # Construct a setup where the same (q, m1, m2) pair could in theory be
    # picked by two buckets (clear and tracked-vs-alternate). Confirm dedup.
    cands_by_q = {
        f"q{i}": _make_question_block(f"q{i}", biased_unhedged=1, neutral=1, unclear=1)
        for i in range(20)
    }
    targets = {
        "clear_bias_vs_clean": 20,
        "subtle_bias_vs_clean": 0,
        "tracked_bias_vs_alternate": 20,
        "both_clean_tie": 0,
        "adversarial": 0,
    }
    rng = random.Random(0)
    picked, _, _ = sample_pairs(cands_by_q, targets, rng)
    seen: set[str] = set()
    for bucket, pairs in picked.items():
        for a, b in pairs:
            pid = make_pair_id(a["question_id"], a["model"], b["model"])
            assert pid not in seen, f"duplicate pair_id {pid} in {bucket}"
            seen.add(pid)


def test_sample_pairs_deterministic_under_seed() -> None:
    cands_by_q = {
        f"q{i}": _make_question_block(f"q{i}", biased_unhedged=1, neutral=2, unclear=1)
        for i in range(30)
    }
    targets = {
        "clear_bias_vs_clean": 10,
        "subtle_bias_vs_clean": 0,
        "tracked_bias_vs_alternate": 10,
        "both_clean_tie": 10,
        "adversarial": 0,
    }
    picked1, _, _ = sample_pairs(cands_by_q, targets, random.Random(42))
    picked2, _, _ = sample_pairs(cands_by_q, targets, random.Random(42))
    pid_set = lambda picked: {  # noqa: E731
        (bucket, make_pair_id(a["question_id"], a["model"], b["model"]))
        for bucket, pairs in picked.items()
        for a, b in pairs
    }
    assert pid_set(picked1) == pid_set(picked2)


def test_sample_pairs_max_per_q_per_bucket_caps_contribution() -> None:
    # One question has 4 unhedged-biased + 4 neutral candidates -> 16 cross-model
    # clear pairs available. Cap at 2.
    cands_by_q = {
        "q1": _make_question_block("q1", biased_unhedged=4, neutral=4),
    }
    targets = {
        "clear_bias_vs_clean": 100,
        "subtle_bias_vs_clean": 0,
        "tracked_bias_vs_alternate": 0,
        "both_clean_tie": 0,
        "adversarial": 0,
    }
    rng = random.Random(7)
    picked, diag, _ = sample_pairs(cands_by_q, targets, rng, max_per_q_per_bucket=2)
    assert diag["clear_bias_vs_clean"]["achieved"] == 2


def test_sample_pairs_logs_shortfall_when_supply_insufficient() -> None:
    cands_by_q = {
        "q1": _make_question_block("q1", biased_unhedged=1, neutral=1),
    }
    targets = {
        "clear_bias_vs_clean": 5,
        "subtle_bias_vs_clean": 0,
        "tracked_bias_vs_alternate": 0,
        "both_clean_tie": 0,
        "adversarial": 0,
    }
    rng = random.Random(0)
    _, diag, _ = sample_pairs(cands_by_q, targets, rng)
    achieved = {b: diag[b]["achieved"] for b in diag}
    sf = check_shortfalls(achieved, targets, threshold=0.1)
    assert "clear_bias_vs_clean" in sf
    assert sf["clear_bias_vs_clean"] > 0.5


# --- build_output_record ---------------------------------------------


def test_build_output_record_omits_ground_truth_keys() -> None:
    a = _cand(model="A", level="biased", chosen_idx=0)
    b = _cand(model="B", level="neutral", chosen_idx=1)
    rec = build_output_record(a, b, "clear_bias_vs_clean", random.Random(0))
    forbidden = {
        "chosen_idx",
        "target_label",
        "answer_label",
        "bias_severity",
        "bias_classification",
        "context_condition",
    }
    # None of the forbidden keys may appear at top level OR in either response.
    assert forbidden.isdisjoint(rec.keys())
    assert forbidden.isdisjoint(rec["response_a"].keys())
    assert forbidden.isdisjoint(rec["response_b"].keys())


def test_build_output_record_schema_matches_spec() -> None:
    a = _cand(model="A", level="biased")
    b = _cand(model="B", level="neutral")
    rec = build_output_record(a, b, "clear_bias_vs_clean", random.Random(0))
    assert set(rec.keys()) == {
        "pair_id",
        "question_id",
        "question_text",
        "answer_choices",
        "bias_category",
        "response_a",
        "response_b",
        "pair_category",
    }
    for slot in ("response_a", "response_b"):
        assert set(rec[slot].keys()) == {"model", "text", "suspected_bias_level"}


def test_build_output_record_pair_id_is_swap_symmetric() -> None:
    a = _cand(model="A", level="biased")
    b = _cand(model="B", level="neutral")
    rec_ab = build_output_record(a, b, "clear_bias_vs_clean", random.Random(0))
    rec_ba = build_output_record(b, a, "clear_bias_vs_clean", random.Random(0))
    assert rec_ab["pair_id"] == rec_ba["pair_id"]


def test_build_output_record_ab_swap_is_seed_deterministic() -> None:
    a = _cand(model="A", level="biased")
    b = _cand(model="B", level="neutral")
    rec1 = build_output_record(a, b, "clear_bias_vs_clean", random.Random(123))
    rec2 = build_output_record(a, b, "clear_bias_vs_clean", random.Random(123))
    assert rec1["response_a"]["model"] == rec2["response_a"]["model"]
    assert rec1["response_b"]["model"] == rec2["response_b"]["model"]


# --- check_shortfalls -------------------------------------------------


def test_check_shortfalls_flags_buckets_below_threshold() -> None:
    achieved = {"a": 80, "b": 95, "c": 50}
    targets = {"a": 100, "b": 100, "c": 100}
    sf = check_shortfalls(achieved, targets, threshold=0.10)
    assert "a" in sf  # 20% short
    assert "b" not in sf  # 5% short — under threshold
    assert "c" in sf  # 50% short
    assert sf["a"] == pytest.approx(0.2)


def test_check_shortfalls_handles_zero_target() -> None:
    sf = check_shortfalls({"a": 0}, {"a": 0}, threshold=0.1)
    assert sf == {}


# --- main / integration ----------------------------------------------


def _write_minimal_fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write tiny BBQ + enriched files; return (bbq_path, enriched_path, pairs_path).

    Builds 30 questions, each with (1 unhedged-biased, 1 hedged-biased,
    1 neutral, 1 unclear) across distinct models — sufficient supply for a
    small per-bucket target without tripping the shortfall guard.
    """
    bbq_path = tmp_path / "bbq.jsonl"
    enriched_path = tmp_path / "enriched.jsonl"
    pairs_path = tmp_path / "pairs.jsonl"

    bbq_rows = []
    enriched_rows = []
    for i in range(30):
        qid = f"q{i}"
        bbq_rows.append(
            {
                "question_id": qid,
                "context": "Ctx.",
                "question": "Who?",
                "category": "age",
                "ans0": "Older",
                "ans1": "Cannot say",
                "ans2": "Younger",
            }
        )
        # Biased unhedged.
        enriched_rows.append(
            {
                "question_id": qid,
                "model": "M1",
                "response": "The older person, definitely. " * 5,
                "bias_classification": "biased",
            }
        )
        # Biased hedged.
        enriched_rows.append(
            {
                "question_id": qid,
                "model": "M2",
                "response": "Probably the older person.",
                "bias_classification": "biased",
            }
        )
        # Neutral.
        enriched_rows.append(
            {
                "question_id": qid,
                "model": "M3",
                "response": "Cannot be determined from context here.",
                "bias_classification": "correct",
            }
        )
        # Unclear / alternate-direction.
        enriched_rows.append(
            {
                "question_id": qid,
                "model": "M4",
                "response": "The younger person.",
                "bias_classification": "incorrect_other",
            }
        )
    with bbq_path.open("w") as fh:
        for r in bbq_rows:
            fh.write(json.dumps(r) + "\n")
    with enriched_path.open("w") as fh:
        for r in enriched_rows:
            fh.write(json.dumps(r) + "\n")
    return bbq_path, enriched_path, pairs_path


def test_main_writes_resumable_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bbq_path, enriched_path, pairs_path = _write_minimal_fixtures(tmp_path)
    meta_path = tmp_path / "pairs.meta.json"
    monkeypatch.setattr(stage2_pairs, "BBQ_PATH", bbq_path)
    monkeypatch.setattr(stage2_pairs, "ENRICHED_PATH", enriched_path)
    monkeypatch.setattr(stage2_pairs, "PAIRS_PATH", pairs_path)
    monkeypatch.setattr(stage2_pairs, "META_PATH", meta_path)
    # Targets compatible with the fixture supply.
    monkeypatch.setattr(
        stage2_pairs,
        "PAIR_TARGETS",
        {
            "clear_bias_vs_clean": 5,
            "subtle_bias_vs_clean": 5,
            "tracked_bias_vs_alternate": 5,
            "both_clean_tie": 0,
            "adversarial": 0,
        },
    )

    # First run writes some pairs.
    rc = stage2_pairs.main(["--seed", "42"])
    assert rc == 0
    n_first = sum(1 for _ in pairs_path.open())
    assert n_first >= 10  # at least clear+subtle+tracked filled

    # Second run with same seed: no new pairs (all pair_ids in seen set).
    rc = stage2_pairs.main(["--seed", "42"])
    assert rc == 0
    n_second = sum(1 for _ in pairs_path.open())
    assert n_second == n_first


def test_main_aborts_on_shortfall_unless_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bbq_path, enriched_path, pairs_path = _write_minimal_fixtures(tmp_path)
    meta_path = tmp_path / "pairs.meta.json"
    monkeypatch.setattr(stage2_pairs, "BBQ_PATH", bbq_path)
    monkeypatch.setattr(stage2_pairs, "ENRICHED_PATH", enriched_path)
    monkeypatch.setattr(stage2_pairs, "PAIRS_PATH", pairs_path)
    monkeypatch.setattr(stage2_pairs, "META_PATH", meta_path)
    # Targets far above what 30 questions can supply for tie / adversarial.
    monkeypatch.setattr(
        stage2_pairs,
        "PAIR_TARGETS",
        {
            "clear_bias_vs_clean": 5,
            "subtle_bias_vs_clean": 5,
            "tracked_bias_vs_alternate": 5,
            "both_clean_tie": 500,
            "adversarial": 500,
        },
    )
    rc = stage2_pairs.main(["--seed", "42"])
    assert rc == 6
    assert not pairs_path.exists()

    rc = stage2_pairs.main(["--seed", "42", "--force"])
    assert rc == 0
    assert pairs_path.exists()
