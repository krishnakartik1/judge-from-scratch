"""Unit tests for Stage 1.5 enrichment helpers.

Pure tests for ``classify`` (no I/O), one tmp_path test for
``load_bbq_index``, and a small fixture-based check for
``enrich_record``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import stage1_enrich

classify = stage1_enrich.classify
load_bbq_index = stage1_enrich.load_bbq_index
enrich_record = stage1_enrich.enrich_record


# --- classify --------------------------------------------------------


def test_classify_parse_failed_on_truncation() -> None:
    # finish_reason != "stop" → parse_failed regardless of chosen_idx.
    assert classify(
        chosen_idx=0,
        finish_reason="length",
        answer_label=0,
        target_label=2,
        context_condition="ambig",
    ) == ("parse_failed", None)


def test_classify_parse_failed_on_null_chosen_idx() -> None:
    assert classify(
        chosen_idx=None,
        finish_reason="stop",
        answer_label=1,
        target_label=2,
        context_condition="disambig",
    ) == ("parse_failed", None)


def test_classify_parse_failed_dominates_correct() -> None:
    # Pin the spec'd behaviour: a truncated response that *would* be
    # correct still bucket as parse_failed. A future contributor who
    # sees this as wasted data must change the spec, not the code.
    assert classify(
        chosen_idx=2,
        finish_reason="length",
        answer_label=2,
        target_label=0,
        context_condition="disambig",
    ) == ("parse_failed", None)


def test_classify_correct() -> None:
    assert classify(
        chosen_idx=1,
        finish_reason="stop",
        answer_label=1,
        target_label=2,
        context_condition="ambig",
    ) == ("correct", None)


def test_classify_correct_beats_biased_when_labels_coincide() -> None:
    # When answer_label == target_label, the model picking that index
    # is genuinely correct — priority order says so.
    assert classify(
        chosen_idx=2,
        finish_reason="stop",
        answer_label=2,
        target_label=2,
        context_condition="disambig",
    ) == ("correct", None)


def test_classify_biased_ambig() -> None:
    assert classify(
        chosen_idx=2,
        finish_reason="stop",
        answer_label=1,
        target_label=2,
        context_condition="ambig",
    ) == ("biased", "biased_ambig")


def test_classify_biased_disambig() -> None:
    # Disambig + biased is the worse failure mode — model overrode
    # disambiguating context. Severity must be the literal string
    # "biased_disambig" — pinning so a regression that returns
    # "disambig" or similar fails.
    assert classify(
        chosen_idx=0,
        finish_reason="stop",
        answer_label=1,
        target_label=0,
        context_condition="disambig",
    ) == ("biased", "biased_disambig")


def test_classify_incorrect_other() -> None:
    # Wrong answer that's neither correct nor stereotype-aligned.
    assert classify(
        chosen_idx=0,
        finish_reason="stop",
        answer_label=1,
        target_label=2,
        context_condition="ambig",
    ) == ("incorrect_other", None)


# --- load_bbq_index --------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_load_bbq_index_happy_path(tmp_path: Path) -> None:
    p = tmp_path / "bbq.jsonl"
    _write_jsonl(
        p,
        [
            {"question_id": "age::0::1::neg", "answer_label": 1},
            {"question_id": "age::0::2::neg", "answer_label": 2},
        ],
    )
    idx = load_bbq_index(p)
    assert set(idx.keys()) == {"age::0::1::neg", "age::0::2::neg"}
    assert idx["age::0::1::neg"]["answer_label"] == 1


def test_load_bbq_index_raises_on_duplicate(tmp_path: Path) -> None:
    p = tmp_path / "bbq.jsonl"
    _write_jsonl(
        p,
        [
            {"question_id": "age::0::1::neg", "answer_label": 1},
            {"question_id": "age::0::1::neg", "answer_label": 2},
        ],
    )
    with pytest.raises(ValueError, match="Duplicate question_id"):
        load_bbq_index(p)


# --- enrich_record ---------------------------------------------------


def _fake_candidate(**overrides) -> dict:
    base = {
        "pair_key": "age::0::1::neg::test/model",
        "question_id": "age::0::1::neg",
        "model": "test/model",
        "prompt": "p",
        "response": "r",
        "chosen_letter": "B",
        "chosen_idx": 1,
        "generation_params": {"temperature": 0.7, "max_tokens": 300},
        "finish_reason": "stop",
    }
    base.update(overrides)
    return base


def _fake_bbq(**overrides) -> dict:
    base = {
        "question_id": "age::0::1::neg",
        "answer_label": 1,
        "target_label": 2,
        "context_condition": "ambig",
        "stereotyped_groups": ["nonOld"],
        "question_polarity": "neg",
    }
    base.update(overrides)
    return base


def test_enrich_record_adds_seven_fields_and_preserves_original() -> None:
    cand = _fake_candidate()
    bbq = _fake_bbq()
    enriched = enrich_record(cand, bbq)

    # Original Stage 1 fields preserved unchanged.
    for key, value in cand.items():
        assert enriched[key] == value

    # Seven new fields populated.
    expected_new = {
        "answer_label",
        "target_label",
        "context_condition",
        "stereotyped_groups",
        "question_polarity",
        "bias_classification",
        "bias_severity",
    }
    assert expected_new <= set(enriched.keys())

    # bias_classification == "correct" for chosen_idx == answer_label.
    assert enriched["bias_classification"] == "correct"
    assert enriched["bias_severity"] is None


def test_enrich_record_stereotyped_groups_round_trips_as_list() -> None:
    bbq = _fake_bbq(stereotyped_groups=["old", "nonOld"])
    enriched = enrich_record(_fake_candidate(), bbq)
    assert enriched["stereotyped_groups"] == ["old", "nonOld"]
    assert isinstance(enriched["stereotyped_groups"], list)
    # Confirm JSON round-trip preserves list shape (no implicit str cast).
    rt = json.loads(json.dumps(enriched))
    assert rt["stereotyped_groups"] == ["old", "nonOld"]


def test_enrich_record_biased_disambig_severity() -> None:
    cand = _fake_candidate(chosen_idx=2, chosen_letter="C")
    bbq = _fake_bbq(answer_label=1, target_label=2, context_condition="disambig")
    enriched = enrich_record(cand, bbq)
    assert enriched["bias_classification"] == "biased"
    assert enriched["bias_severity"] == "biased_disambig"
