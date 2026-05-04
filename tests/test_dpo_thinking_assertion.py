"""Unit tests for ``assert_no_thinking`` in the DPO trainer.

DPO rows have prompt/chosen/rejected fields (not prompt/target like
SFT). The guard must scan all three for the ``<|think|>`` token —
decision #13 disables Gemma 4 native thinking.
"""

from __future__ import annotations

import pytest

from train.modal.dpo import assert_no_thinking


def test_assert_no_thinking_passes_on_clean_dpo_row() -> None:
    row = {
        "pair_id": "abc",
        "swap": False,
        "source": "synth",
        "prompt": "<bos><|turn>system\nYou are a judge<turn|>\n<|turn>model\n",
        "chosen": "<reasoning>good</reasoning><verdict>A</verdict>",
        "rejected": "<reasoning>bad</reasoning><verdict>B</verdict>",
    }
    assert_no_thinking(row)  # must not raise


def test_assert_no_thinking_raises_on_chosen_field() -> None:
    row = {
        "pair_id": "abc",
        "prompt": "P",
        "chosen": "<reasoning><|think|>r</reasoning><verdict>A</verdict>",
        "rejected": "R",
    }
    with pytest.raises(AssertionError, match="chosen"):
        assert_no_thinking(row)


def test_assert_no_thinking_raises_on_rejected_field() -> None:
    row = {
        "pair_id": "abc",
        "prompt": "P",
        "chosen": "C",
        "rejected": "<reasoning><|think|>r</reasoning><verdict>B</verdict>",
    }
    with pytest.raises(AssertionError, match="rejected"):
        assert_no_thinking(row)


def test_assert_no_thinking_raises_on_prompt_field() -> None:
    row = {
        "pair_id": "abc",
        "prompt": "<bos><|turn>system\n<|think|>You are a judge<turn|>",
        "chosen": "C",
        "rejected": "R",
    }
    with pytest.raises(AssertionError, match="prompt"):
        assert_no_thinking(row)


def test_assert_no_thinking_ignores_non_string_fields() -> None:
    row = {
        "pair_id": "abc",
        "swap": True,
        "score": 0.5,
        "tags": ["a", "b"],
        "prompt": "P",
        "chosen": "C",
        "rejected": "R",
    }
    assert_no_thinking(row)  # must not raise


def test_assert_no_thinking_error_message_cites_decision_13() -> None:
    row = {"prompt": "P", "chosen": "ok", "rejected": "<|think|>here"}
    with pytest.raises(AssertionError) as excinfo:
        assert_no_thinking(row)
    assert "rejected" in str(excinfo.value)
    assert "decision #13" in str(excinfo.value)
