"""Unit tests for ``build_prompt`` template rendering."""

from __future__ import annotations

from conftest import stage1_gen

build_prompt = stage1_gen.build_prompt


RECORD = {
    "context": "Two friends, X and Y, were at a meeting.",
    "question": "Who arrived late?",
    "ans0": "X",
    "ans1": "Y",
    "ans2": "Cannot be determined",
}


def test_renders_context_question_and_choices() -> None:
    out = build_prompt(RECORD)
    assert "Context: Two friends, X and Y, were at a meeting." in out
    assert "Question: Who arrived late?" in out
    assert "A. X" in out
    assert "B. Y" in out
    assert "C. Cannot be determined" in out


def test_no_template_placeholders_leak() -> None:
    out = build_prompt(RECORD)
    for placeholder in ("{context}", "{question}", "{ans0}", "{ans1}", "{ans2}"):
        assert placeholder not in out


def test_includes_tag_format_instruction() -> None:
    out = build_prompt(RECORD)
    assert "<answer>A</answer>" in out
    assert "<answer>B</answer>" in out
    assert "<answer>C</answer>" in out
