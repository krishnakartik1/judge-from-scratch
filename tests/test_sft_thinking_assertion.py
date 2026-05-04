"""Unit tests for ``assert_no_thinking`` — the <|think|> token guard.

Decision #13: native thinking mode is disabled. Stage 5 hard-blocks
``<|think|>`` at format time; the Stage 6 SFT script re-asserts on
the first row before training as a backstop against drift.
"""

from __future__ import annotations

import pytest

from train.modal.sft import assert_no_thinking


def test_assert_no_thinking_passes_on_clean_row() -> None:
    row = {
        "pair_id": "abc",
        "swap": False,
        "prompt": "<bos><|turn>system\nYou are a judge<turn|>\n",
        "target": "<reasoning>r</reasoning><verdict>A</verdict>",
    }
    assert_no_thinking(row)  # must not raise


def test_assert_no_thinking_raises_on_prompt_field() -> None:
    row = {
        "pair_id": "abc",
        "prompt": "<bos><|turn>system\n<|think|>You are a judge<turn|>",
        "target": "X",
    }
    with pytest.raises(AssertionError, match="<\\|think\\|>"):
        assert_no_thinking(row)


def test_assert_no_thinking_raises_on_target_field() -> None:
    row = {
        "pair_id": "abc",
        "prompt": "P",
        "target": "<reasoning><|think|>r</reasoning><verdict>A</verdict>",
    }
    with pytest.raises(AssertionError, match="target"):
        assert_no_thinking(row)


def test_assert_no_thinking_ignores_non_string_fields() -> None:
    """Non-string values (swap=bool, counts=int, lists) cannot
    contain a <|think|> token; the guard should not blow up on them."""
    row = {
        "pair_id": "abc",
        "swap": True,
        "epoch": 0,
        "tags": ["a", "b"],
        "prompt": "P",
        "target": "T",
    }
    assert_no_thinking(row)  # must not raise


def test_assert_no_thinking_error_message_names_field() -> None:
    """The error must identify which field contains the token so
    the operator knows where to look."""
    row = {"prompt": "P", "target": "<|think|>here"}
    with pytest.raises(AssertionError) as excinfo:
        assert_no_thinking(row)
    assert "target" in str(excinfo.value)
    assert "decision #13" in str(excinfo.value)
