"""Unit tests for ``train/modal/sft.py`` pure helpers.

Covers ``make_text`` (the prompt+target+EOS concatenation) and
``select_probe_pair_ids`` (deterministic sha1-sorted holdout
selection). Both run on CPU without Modal/CUDA.
"""

from __future__ import annotations

import pytest

from train.modal.sft import make_text, select_probe_pair_ids

# ---------------------------------------------------------------------------
# make_text


def test_make_text_concatenates_prompt_target_eos() -> None:
    row = {
        "prompt": "<bos><|turn>user\nHi<turn|>\n<|turn>model\n",
        "target": "<reasoning>r</reasoning><verdict>A</verdict>",
    }
    out = make_text(row, eos_token="<turn|>")
    assert out == (
        "<bos><|turn>user\nHi<turn|>\n<|turn>model\n"
        "<reasoning>r</reasoning><verdict>A</verdict><turn|>"
    )


def test_make_text_eos_must_be_turn_close() -> None:
    """The Gemma 4 E4B EOS literally is ``<turn|>``; the SFT script
    relies on this exact string. Caller-supplied — function does
    not enforce, but must propagate verbatim."""
    row = {"prompt": "P", "target": "T"}
    assert make_text(row, eos_token="<turn|>") == "PT<turn|>"


def test_make_text_does_not_strip_target() -> None:
    """Target may end in '</verdict>' — the SFT trainer needs the
    full body before EOS, no auto-trim."""
    row = {"prompt": "PROMPT_", "target": "BODY  "}
    assert make_text(row, eos_token="EOS") == "PROMPT_BODY  EOS"


def test_make_text_rejects_non_string_fields() -> None:
    with pytest.raises(ValueError, match="prompt/target must be strings"):
        make_text({"prompt": 1, "target": "T"}, eos_token="X")
    with pytest.raises(ValueError, match="prompt/target must be strings"):
        make_text({"prompt": "P", "target": None}, eos_token="X")


def test_make_text_with_real_stage5_shape() -> None:
    """Smoke check using a row that mirrors Stage 5's exact suffix."""
    row = {
        "prompt": (
            "<bos><|turn>system\nYou are a judge<turn|>\n"
            "<|turn>user\nQ<turn|>\n<|turn>model\n"
        ),
        "target": "<reasoning>x</reasoning><verdict>TIE</verdict>",
    }
    out = make_text(row, "<turn|>")
    # Must end with the assistant turn close, twice nowhere else.
    assert out.endswith("</verdict><turn|>")
    assert out.count("<|turn>model\n") == 1


# ---------------------------------------------------------------------------
# select_probe_pair_ids


def _mk_rows(*pair_ids: str) -> list[dict[str, str]]:
    return [{"pair_id": p} for p in pair_ids]


def test_select_probe_pair_ids_deterministic() -> None:
    rows = _mk_rows("a", "b", "c", "d", "e", "f", "g")
    first = select_probe_pair_ids(rows, n=5)
    second = select_probe_pair_ids(rows, n=5)
    assert first == second
    assert len(first) == 5


def test_select_probe_pair_ids_independent_of_input_order() -> None:
    rows = _mk_rows("a", "b", "c", "d", "e", "f", "g")
    rows_reversed = list(reversed(rows))
    assert select_probe_pair_ids(rows, n=5) == select_probe_pair_ids(rows_reversed, n=5)


def test_select_probe_pair_ids_uses_sha1_not_natural_order() -> None:
    """Sort key is sha1(pair_id), so alphabetical/numeric ordering
    in the source file does not bias the selection."""
    rows = _mk_rows("aaa", "bbb", "ccc", "ddd", "eee")
    selected = select_probe_pair_ids(rows, n=3)
    # If we accidentally sorted by pair_id directly the result would
    # be ['aaa', 'bbb', 'ccc']. Confirm we got something different.
    assert selected != ["aaa", "bbb", "ccc"]


def test_select_probe_pair_ids_raises_when_too_few_rows() -> None:
    with pytest.raises(ValueError, match="at least 5 rows"):
        select_probe_pair_ids(_mk_rows("a", "b"), n=5)
