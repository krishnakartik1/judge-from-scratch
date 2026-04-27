"""Unit tests for ``parse_chosen``.

Covers the failure modes flagged by 7B-class generators: tagged form,
untagged "The answer is X.", markdown bold, smart quotes, parens-style,
multi-letter prose, lowercase tags, whitespace inside tags, and inputs
with no extractable letter.
"""

from __future__ import annotations

import pytest
from conftest import stage1_gen

parse_chosen = stage1_gen.parse_chosen


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Tagged form is the preferred path.
        ("Reasoning. <answer>A</answer>", ("A", 0)),
        ("<answer>b</answer>", ("B", 1)),
        ("Some prose. < answer > c </answer>", ("C", 2)),
        # Tag wins even when fallback letters precede it.
        ("A or B? <answer>C</answer>", ("C", 2)),
        # Fallback path: anchored letters in the trailing 120 chars.
        ("After deliberation, the answer is A.", ("A", 0)),
        ("Final reasoning here. **B**", ("B", 1)),
        ('Considering both, Answer: "C"', ("C", 2)),
        # Smart quotes from Phi-3-style tokenizers.
        ("Answer: ‘C’", ("C", 2)),
        # Parens style.
        ("My pick is (A).", ("A", 0)),
        # Multi-letter prose — tail-only scan picks the LAST mention.
        ("Between A and C, I'd say A.", ("A", 0)),
        # Empty / None inputs.
        ("", (None, None)),
        (None, (None, None)),
        # No extractable letter.
        ("I cannot determine the correct option.", (None, None)),
        # Mid-prose letter with no anchor in the tail.
        ("As I argued earlier, " + "x" * 200, (None, None)),
        # Mistral head-anchored pattern: response begins with `<space><letter>.`
        # and the rationale that follows pushes the letter past the
        # tail-only fallback's 120-char window.
        (
            " C. Not enough information - The context does not provide enough"
            " information to determine who was reckless between the elderly"
            " father and the daughter.",
            ("C", 2),
        ),
        # Same head-anchored pattern, no leading space.
        ("A. The first option is correct because of x, y, z.", ("A", 0)),
        # Head-anchored guard: lone letter without trailing dot must NOT match.
        # Mistral's signal is `<letter>.`; "A is" or "C, I" is ambiguous prose.
        ("A is what I would choose for this question.", (None, None)),
    ],
)
def test_parse_chosen(
    text: str | None, expected: tuple[str | None, int | None]
) -> None:
    assert parse_chosen(text) == expected
