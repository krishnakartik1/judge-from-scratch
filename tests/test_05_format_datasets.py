"""Unit tests for Stage 5 (SFT/DPO dataset formatting).

Covers ``data/_format_helpers.py`` first; driver-level and synthesis
client tests live in the same file once those modules exist.

No real network calls. The chat-template test uses a stub tokenizer
to avoid pulling in unsloth/torch in unit tests.
"""

from __future__ import annotations

from collections import Counter

import pytest
from conftest import format_helpers as fh
from conftest import stage5_format as s5
from conftest import synth_rejected as sr

# ---------------------------------------------------------------------------
# flip_verdict
# ---------------------------------------------------------------------------


def test_flip_verdict_swaps_a_and_b() -> None:
    assert fh.flip_verdict("A") == "B"
    assert fh.flip_verdict("B") == "A"


def test_flip_verdict_keeps_tie() -> None:
    assert fh.flip_verdict("TIE") == "TIE"


def test_flip_verdict_raises_on_unknown() -> None:
    with pytest.raises(ValueError):
        fh.flip_verdict("C")


# ---------------------------------------------------------------------------
# strip_confidence
# ---------------------------------------------------------------------------


def test_strip_confidence_removes_tag() -> None:
    text = "Some reasoning.\n<confidence>4</confidence>"
    assert fh.strip_confidence(text) == "Some reasoning."


def test_strip_confidence_noop_when_absent() -> None:
    text = "Some reasoning."
    assert fh.strip_confidence(text) == "Some reasoning."


def test_strip_confidence_handles_trailing_block() -> None:
    # Realistic shape — Sonnet emits <confidence> as a sibling block
    # at end-of-output. Trailing newline should be stripped too.
    text = "Real reasoning text.\n<confidence>4</confidence>"
    assert fh.strip_confidence(text) == "Real reasoning text."


# ---------------------------------------------------------------------------
# strip_thinking_blocks
# ---------------------------------------------------------------------------


def test_strip_thinking_blocks_html() -> None:
    text = "<thinking>internal scratch</thinking>Answer follows."
    assert fh.strip_thinking_blocks(text) == "Answer follows."


def test_strip_thinking_blocks_short() -> None:
    text = "<think>x</think>Final."
    assert fh.strip_thinking_blocks(text) == "Final."


def test_strip_thinking_blocks_multiline() -> None:
    text = "<thinking>line one\nline two</thinking>Final answer."
    assert fh.strip_thinking_blocks(text) == "Final answer."


def test_strip_thinking_blocks_noop() -> None:
    text = "Plain reasoning, no thinking blocks."
    assert fh.strip_thinking_blocks(text) == text


# ---------------------------------------------------------------------------
# format_target
# ---------------------------------------------------------------------------


def test_format_target_exact_shape() -> None:
    out = fh.format_target("All clean.", "B")
    assert out == "<reasoning>All clean.</reasoning><verdict>B</verdict>"


def test_format_target_tie() -> None:
    out = fh.format_target("Equivalent.", "TIE")
    assert out == "<reasoning>Equivalent.</reasoning><verdict>TIE</verdict>"


def test_format_target_rejects_bad_verdict() -> None:
    with pytest.raises(ValueError):
        fh.format_target("text", "C")


# ---------------------------------------------------------------------------
# format_answer_choices
# ---------------------------------------------------------------------------


def test_format_answer_choices_three() -> None:
    choices = [
        {"letter": "A", "text": "brother"},
        {"letter": "B", "text": "pregnant person"},
        {"letter": "C", "text": "Not enough information"},
    ]
    out = fh.format_answer_choices(choices)
    assert out == "A. brother\nB. pregnant person\nC. Not enough information"


def test_format_answer_choices_empty_raises() -> None:
    with pytest.raises(ValueError):
        fh.format_answer_choices([])


def test_format_answer_choices_missing_letter_raises() -> None:
    with pytest.raises(ValueError):
        fh.format_answer_choices([{"text": "x"}])


# ---------------------------------------------------------------------------
# swap_response_labels
# ---------------------------------------------------------------------------


def test_swap_response_labels_basic() -> None:
    text = "Response A is biased; Response B is clean."
    out = fh.swap_response_labels(text)
    assert out == "Response B is biased; Response A is clean."


def test_swap_response_labels_lowercase() -> None:
    text = "I prefer response A over response B."
    out = fh.swap_response_labels(text)
    assert out == "I prefer response B over response A."


def test_swap_response_labels_no_double_substitution() -> None:
    text = "Response A versus Response B versus Response A."
    out = fh.swap_response_labels(text)
    assert out == "Response B versus Response A versus Response B."


def test_swap_response_labels_preserves_punctuation() -> None:
    text = "Response A's reasoning beats Response B's."
    out = fh.swap_response_labels(text)
    assert out == "Response B's reasoning beats Response A's."


def test_swap_response_labels_noop_when_absent() -> None:
    text = "No labels referenced here."
    assert fh.swap_response_labels(text) == text


# ---------------------------------------------------------------------------
# build_user_message
# ---------------------------------------------------------------------------


_PAIR = {
    "question_text": "Who is more likely to be on time?",
    "answer_choices": [
        {"letter": "A", "text": "the student"},
        {"letter": "B", "text": "the engineer"},
        {"letter": "C", "text": "Cannot determine"},
    ],
    "response_a": {"text": "Engineer because of habits.", "model": "x"},
    "response_b": {"text": "Cannot determine from context.", "model": "y"},
}


def test_build_user_message_layout() -> None:
    out = fh.build_user_message(_PAIR, swap=False)
    assert "**Question:** Who is more likely" in out
    assert "**Answer choices:**" in out
    assert "A. the student" in out
    assert "B. the engineer" in out
    assert "C. Cannot determine" in out
    assert "**Response A:** Engineer because of habits." in out
    assert "**Response B:** Cannot determine from context." in out
    assert "<reasoning>" in out and "<verdict>" in out


def test_build_user_message_swap_swaps_contents_not_labels() -> None:
    out = fh.build_user_message(_PAIR, swap=True)
    assert "**Response A:** Cannot determine from context." in out
    assert "**Response B:** Engineer because of habits." in out
    assert "A. the student" in out  # answer_choices unchanged


# ---------------------------------------------------------------------------
# clean_reasoning + make_target
# ---------------------------------------------------------------------------


def test_clean_reasoning_strips_both() -> None:
    text = "<thinking>scratch</thinking>Real reasoning.<confidence>4</confidence>"
    assert fh.clean_reasoning(text) == "Real reasoning."


def test_make_target_no_swap() -> None:
    out = fh.make_target("Response A is biased.", "A", swap=False)
    assert out == "<reasoning>Response A is biased.</reasoning><verdict>A</verdict>"


def test_make_target_swap_flips_verdict_and_labels() -> None:
    out = fh.make_target("Response A is biased.", "A", swap=True)
    # A->B verdict, "Response A" -> "Response B" in reasoning.
    assert out == "<reasoning>Response B is biased.</reasoning><verdict>B</verdict>"


def test_make_target_swap_tie_keeps_verdict_swaps_labels() -> None:
    out = fh.make_target("Response A and Response B are equivalent.", "TIE", swap=True)
    assert (
        out == "<reasoning>Response B and Response A are equivalent.</reasoning>"
        "<verdict>TIE</verdict>"
    )


def test_make_target_strips_confidence_under_swap() -> None:
    text = "Reasoning text. <confidence>5</confidence>"
    out = fh.make_target(text, "B", swap=True)
    assert "<confidence>" not in out
    assert "<verdict>A</verdict>" in out


# ---------------------------------------------------------------------------
# apply_chat (stub tokenizer; no unsloth import)
# ---------------------------------------------------------------------------


class _StubTokenizer:
    """Mimics the subset of HF tokenizer surface used by apply_chat."""

    def __init__(self, *, leak_think: bool = False) -> None:
        self.leak_think = leak_think

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        parts: list[str] = []
        for turn in conversation:
            parts.append(
                f"<start_of_turn>{turn['role']}\n{turn['content']}<end_of_turn>"
            )
        parts.append("<start_of_turn>model\n")
        if self.leak_think:
            parts.append("<|think|>")
        return "".join(parts)


def test_apply_chat_includes_generation_prompt() -> None:
    tok = _StubTokenizer()
    out = fh.apply_chat(tok, "system text", "user text")
    assert "<start_of_turn>system" in out
    assert "<start_of_turn>user" in out
    assert out.endswith("<start_of_turn>model\n")


def test_apply_chat_blocks_think_token() -> None:
    tok = _StubTokenizer(leak_think=True)
    with pytest.raises(AssertionError):
        fh.apply_chat(tok, "sys", "user")


# ---------------------------------------------------------------------------
# dpo_split_index + is_synth_bucket
# ---------------------------------------------------------------------------


def test_dpo_split_index_deterministic() -> None:
    a = fh.dpo_split_index("pair_xyz")
    b = fh.dpo_split_index("pair_xyz")
    assert a == b
    assert 0 <= a < fh.TOTAL_BUCKETS


def test_dpo_split_index_distribution_smoke() -> None:
    # Synthetic 1000 pair_ids -> distribution should be roughly uniform
    # across the 10 buckets (within 30% tolerance for the small sample).
    counts: Counter[int] = Counter()
    for i in range(1000):
        counts[fh.dpo_split_index(f"pair_{i:05d}")] += 1
    for bucket in range(fh.TOTAL_BUCKETS):
        assert 70 <= counts[bucket] <= 130, (bucket, counts)


def test_is_synth_bucket_matches_split() -> None:
    pid = "fixed_id_123"
    assert fh.is_synth_bucket(pid) == (fh.dpo_split_index(pid) < fh.SYNTH_BUCKET_HI)


# ===========================================================================
# Synthesis client (data/synth_rejected.py)
# ===========================================================================


def test_failure_mode_round_robin() -> None:
    seen = [sr.failure_mode_for(i) for i in range(8)]
    # Deterministic round-robin over the 4 modes (twice).
    assert seen[:4] == list(sr.FAILURE_MODES)
    assert seen[4:] == list(sr.FAILURE_MODES)


# ---------------------------------------------------------------------------
# build_synth_request
# ---------------------------------------------------------------------------


def _make_pair(verdict: str = "A", pair_id: str = "pid_abc") -> dict:
    return {
        "pair_id": pair_id,
        "question_text": "Who arrived late?",
        "answer_choices": [
            {"letter": "A", "text": "the older guest"},
            {"letter": "B", "text": "the younger guest"},
            {"letter": "C", "text": "Cannot determine"},
        ],
        "response_a": {
            "text": "The older guest, since older people are slower.",
            "model": "x",
        },
        "response_b": {"text": "Cannot determine from the context.", "model": "y"},
        "sonnet_verdict": verdict,
    }


def test_build_synth_request_shape() -> None:
    req = sr.build_synth_request(_make_pair("A"), failure_mode="stereotype_aligned")
    assert req["custom_id"] == "pid_abc"
    params = req["params"]
    assert params["model"] == sr.MODEL
    assert params["max_tokens"] == sr.ANTHROPIC_MAX_TOKENS
    sys_blocks = params["system"]
    assert len(sys_blocks) == 1
    assert sys_blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert sr.SYNTH_SYSTEM_PROMPT == sys_blocks[0]["text"]
    user_block = params["messages"][0]["content"]
    assert "stereotype_aligned" in user_block
    assert "Correct verdict (do NOT pick this): A" in user_block
    # Wrong verdict for sonnet_verdict=A is B.
    assert "You must pick (the WRONG verdict): B" in user_block
    assert "Who arrived late?" in user_block


def test_build_synth_request_tie_pair_keeps_tie_as_correct() -> None:
    # TIE pairs are excluded from DPO pool by the driver, but if one
    # somehow reached this function, flip_verdict would return TIE
    # and the request would land on TIE — surfaced via test, not a
    # silent contract.
    req = sr.build_synth_request(_make_pair("TIE"), failure_mode="verbose_hedging")
    user_block = req["params"]["messages"][0]["content"]
    assert "Correct verdict (do NOT pick this): TIE" in user_block
    assert "You must pick (the WRONG verdict): TIE" in user_block


def test_build_synth_request_retry_appends_suffix() -> None:
    base = sr.build_synth_request(_make_pair("A"), failure_mode="length_burying")
    retry = sr.build_synth_request(
        _make_pair("A"), failure_mode="length_burying", retry=True
    )
    base_user = base["params"]["messages"][0]["content"]
    retry_user = retry["params"]["messages"][0]["content"]
    assert retry_user.startswith(base_user)
    assert "CRITICAL: output must be exactly" in retry_user
    assert "MUST be B" in retry_user


def test_build_synth_request_unknown_failure_mode_raises() -> None:
    with pytest.raises(ValueError):
        sr.build_synth_request(_make_pair("A"), failure_mode="bogus")


def test_build_synth_request_custom_id_override() -> None:
    req = sr.build_synth_request(
        _make_pair("A", pair_id="orig"),
        failure_mode="surface_engagement",
        custom_id="override_id",
    )
    assert req["custom_id"] == "override_id"


# ---------------------------------------------------------------------------
# validate_synth_output
# ---------------------------------------------------------------------------


def test_validate_synth_output_valid() -> None:
    text = "<reasoning>plausible but wrong</reasoning><verdict>B</verdict>"
    parsed = sr.validate_synth_output(text, expected_verdict="B")
    assert parsed == {"reasoning": "plausible but wrong", "verdict": "B"}


def test_validate_synth_output_with_whitespace() -> None:
    text = "  <reasoning>multi\nline reasoning</reasoning>\n<verdict>A</verdict>\n"
    parsed = sr.validate_synth_output(text, expected_verdict="A")
    assert parsed["verdict"] == "A"
    assert "multi\nline" in parsed["reasoning"]


def test_validate_synth_output_self_correction_fails() -> None:
    # Sonnet returned the CORRECT verdict instead of the wrong one.
    text = "<reasoning>caught the bias</reasoning><verdict>B</verdict>"
    with pytest.raises(sr.SynthParseError):
        sr.validate_synth_output(text, expected_verdict="A")


def test_validate_synth_output_missing_tags() -> None:
    with pytest.raises(sr.SynthParseError):
        sr.validate_synth_output("just plain text", expected_verdict="A")


def test_validate_synth_output_trailing_junk() -> None:
    text = (
        "<reasoning>r</reasoning><verdict>A</verdict>"
        "Sorry I shouldn't have said that."
    )
    with pytest.raises(sr.SynthParseError):
        sr.validate_synth_output(text, expected_verdict="A")


def test_validate_synth_output_empty_reasoning() -> None:
    text = "<reasoning>   </reasoning><verdict>A</verdict>"
    with pytest.raises(sr.SynthParseError):
        sr.validate_synth_output(text, expected_verdict="A")


def test_validate_synth_output_unknown_expected() -> None:
    with pytest.raises(ValueError):
        sr.validate_synth_output("text", expected_verdict="C")


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------


def test_compute_synth_cost_zero_on_empty() -> None:
    assert sr.compute_synth_cost(None) == 0.0
    assert sr.compute_synth_cost({}) == 0.0


def test_compute_synth_cost_applies_batch_discount() -> None:
    usage = {
        "input_tokens": 100_000,
        "output_tokens": 10_000,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    cost = sr.compute_synth_cost(usage)
    # 100k @ $3/M + 10k @ $15/M = $0.30 + $0.15 = $0.45, * 0.5 batch.
    expected = (100_000 / 1_000_000 * 3.0 + 10_000 / 1_000_000 * 15.0) * 0.5
    assert abs(cost - expected) < 1e-9


def test_compute_synth_cost_cache_reads_cheaper_than_fresh() -> None:
    fresh = sr.compute_synth_cost({"input_tokens": 1000, "output_tokens": 0})
    cached = sr.compute_synth_cost(
        {"input_tokens": 0, "cache_read_input_tokens": 1000, "output_tokens": 0}
    )
    assert cached < fresh


def test_project_synth_cost_scales() -> None:
    one = sr.project_synth_cost(1)
    hundred = sr.project_synth_cost(100)
    assert one > 0
    assert hundred > one
    # Per-call cost after the first should be far lower than the first
    # (cache-creation overhead is amortized).
    per_call_avg = hundred / 100
    assert per_call_avg < one


# ===========================================================================
# Stage 5 driver (data/05_format_datasets.py)
# ===========================================================================


def _labeled_pair(
    pair_id: str = "p1",
    verdict: str = "A",
    confidence: int = 5,
) -> dict:
    return {
        "pair_id": pair_id,
        "question_text": "Who arrived late?",
        "answer_choices": [
            {"letter": "A", "text": "alice"},
            {"letter": "B", "text": "bob"},
            {"letter": "C", "text": "Cannot determine"},
        ],
        "response_a": {"text": "Alice was late.", "model": "x"},
        "response_b": {"text": "Cannot determine.", "model": "y"},
        "sonnet_verdict": verdict,
        "sonnet_reasoning": "Response A leans on a stereotype; Response B is clean.",
        "sonnet_confidence": confidence,
    }


def test_sft_pool_drops_low_confidence() -> None:
    records = [
        _labeled_pair(pair_id="hi", confidence=4),
        _labeled_pair(pair_id="ok", confidence=3),
        _labeled_pair(pair_id="lo", confidence=2),
        _labeled_pair(pair_id="null", confidence=None),  # type: ignore[arg-type]
    ]
    pool = s5._sft_pool(records)
    assert {p["pair_id"] for p in pool} == {"hi", "ok"}


def test_dpo_pool_filters_confidence_and_tie() -> None:
    records = [
        _labeled_pair(pair_id="hi", verdict="A", confidence=5),
        _labeled_pair(pair_id="conf3", verdict="A", confidence=3),  # too low
        _labeled_pair(pair_id="tie", verdict="TIE", confidence=5),  # TIE
        _labeled_pair(pair_id="ok", verdict="B", confidence=4),
    ]
    pool = s5._dpo_pool(records)
    assert {p["pair_id"] for p in pool} == {"hi", "ok"}


def test_synth_and_flip_subsets_partition_pool() -> None:
    pool = [_labeled_pair(pair_id=f"p_{i:04d}") for i in range(200)]
    s = s5._synth_subset(pool)
    f = s5._flip_subset(pool)
    assert len(s) + len(f) == len(pool)
    assert set(p["pair_id"] for p in s).isdisjoint(p["pair_id"] for p in f)


def test_select_synth_pairs_dryrun_is_subset_of_full() -> None:
    pool = [_labeled_pair(pair_id=f"p_{i:04d}") for i in range(100)]
    full = s5._select_synth_pairs(pool, dryrun=False)
    dryrun = s5._select_synth_pairs(pool, dryrun=True)
    assert len(dryrun) == s5.DRYRUN_SIZE
    full_ids = [p["pair_id"] for p in full]
    dryrun_ids = [p["pair_id"] for p in dryrun]
    # Strict subset and same prefix order.
    assert dryrun_ids == full_ids[: s5.DRYRUN_SIZE]


# ---------------------------------------------------------------------------
# _build_sft_rows (with stub tokenizer)
# ---------------------------------------------------------------------------


def test_build_sft_rows_doubles_via_swap() -> None:
    records = [_labeled_pair(pair_id="p1", verdict="A")]
    rows = s5._build_sft_rows(records, judge_prompt="JP", tokenizer=_StubTokenizer())
    assert len(rows) == 2  # one per swap value
    no_swap = next(r for r in rows if r["swap"] is False)
    swap = next(r for r in rows if r["swap"] is True)
    assert "<verdict>A</verdict>" in no_swap["target"]
    assert "<verdict>B</verdict>" in swap["target"]
    # Response A/B labels swapped in the swap row's target.
    assert "Response B leans on a stereotype" in swap["target"]
    assert "Response A is clean" in swap["target"]


def test_build_sft_rows_strips_confidence_and_thinking() -> None:
    pair = _labeled_pair()
    pair["sonnet_reasoning"] = (
        "<thinking>scratch</thinking>Real reasoning here.<confidence>3</confidence>"
    )
    rows = s5._build_sft_rows([pair], judge_prompt="JP", tokenizer=_StubTokenizer())
    for r in rows:
        assert "<thinking>" not in r["target"]
        assert "<confidence>" not in r["target"]
        assert "Real reasoning here." in r["target"]


# ---------------------------------------------------------------------------
# _percentile
# ---------------------------------------------------------------------------


def test_percentile_basic() -> None:
    assert s5._percentile([], 50) == 0.0
    assert s5._percentile([42], 50) == 42.0
    assert s5._percentile([1, 2, 3, 4, 5], 50) == 3.0
    assert s5._percentile([1, 2, 3, 4, 5], 100) == 5.0
    assert s5._percentile([1, 2, 3, 4, 5], 0) == 1.0


# ---------------------------------------------------------------------------
# _resolve_lifecycle_action
# ---------------------------------------------------------------------------


class _Args:
    def __init__(self, **kw) -> None:
        self.submit = kw.get("submit", False)
        self.poll = kw.get("poll", False)
        self.fetch = kw.get("fetch", False)


def test_lifecycle_action_explicit_wins(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(s5, "BATCHES_STATE_PATH", tmp_path / "b.json")
    assert s5._resolve_lifecycle_action(_Args(submit=True), "x") == "submit"
    assert s5._resolve_lifecycle_action(_Args(poll=True), "x") == "poll"
    assert s5._resolve_lifecycle_action(_Args(fetch=True), "x") == "fetch"


def test_lifecycle_action_auto_no_state_means_submit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(s5, "BATCHES_STATE_PATH", tmp_path / "b.json")
    assert s5._resolve_lifecycle_action(_Args(), "x") == "submit"


def test_lifecycle_action_auto_in_progress_means_poll(tmp_path, monkeypatch) -> None:
    state_path = tmp_path / "b.json"
    monkeypatch.setattr(s5, "BATCHES_STATE_PATH", state_path)
    state_path.write_text('{"x": {"batch_id": "abc", "status": "in_progress"}}')
    assert s5._resolve_lifecycle_action(_Args(), "x") == "poll"


def test_lifecycle_action_auto_ended_means_fetch(tmp_path, monkeypatch) -> None:
    state_path = tmp_path / "b.json"
    monkeypatch.setattr(s5, "BATCHES_STATE_PATH", state_path)
    state_path.write_text('{"x": {"batch_id": "abc", "status": "ended"}}')
    assert s5._resolve_lifecycle_action(_Args(), "x") == "fetch"


def test_lifecycle_action_multiple_explicit_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(s5, "BATCHES_STATE_PATH", tmp_path / "b.json")
    with pytest.raises(s5.GateError):
        s5._resolve_lifecycle_action(_Args(submit=True, poll=True), "x")


# ---------------------------------------------------------------------------
# _grep_token (verify helper)
# ---------------------------------------------------------------------------


def test_grep_token_counts_hits() -> None:
    sft = [{"prompt": "p1 <|think|>", "target": "t1"}]
    dpo = [{"prompt": "p2", "chosen": "<|think|>", "rejected": "r"}]
    n = s5._grep_token(sft, dpo, judge_prompt="JP", pattern=s5.THINKING_PATTERN)
    assert n == 2  # one in SFT prompt, one in DPO chosen


def test_grep_token_fields_only_skips_prompt() -> None:
    sft = [{"prompt": "<confidence>", "target": "<confidence>5</confidence>"}]
    dpo = [{"prompt": "<confidence>", "chosen": "c", "rejected": "r"}]
    n = s5._grep_token(
        sft, dpo, judge_prompt="JP", pattern=s5.CONFIDENCE_PATTERN, fields_only=True
    )
    assert n == 1  # only the SFT target hit


# ---------------------------------------------------------------------------
# Batch state sidecar (synth_rejected helpers used by the driver)
# ---------------------------------------------------------------------------


def test_batch_state_round_trip(tmp_path) -> None:
    p = tmp_path / "b.json"
    sr.update_batch_state(p, "phase1", {"batch_id": "abc", "status": "in_progress"})
    sr.update_batch_state(p, "phase1", {"status": "ended"})
    state = sr.read_batch_state(p)
    assert state["phase1"]["batch_id"] == "abc"
    assert state["phase1"]["status"] == "ended"


# ---------------------------------------------------------------------------
# _build_dpo_rows — chosen/rejected verdict invariants under swap
# ---------------------------------------------------------------------------


def test_build_dpo_rows_synth_invariants() -> None:
    pool = [_labeled_pair(pair_id="p1", verdict="A", confidence=5)]
    synth_results = {
        "p1": {
            "pair_id": "p1",
            "reasoning": "Response B is the bias-laden one.",
            "verdict": "B",
            "phase": "full-synth",
        }
    }
    # Force this pair into the synth bucket regardless of hash.
    rows, counts = s5._build_dpo_rows(
        pool, synth_results, judge_prompt="JP", tokenizer=_StubTokenizer()
    )
    if counts["synth_used"] == 0:
        # Pair_id 'p1' may hash to flip bucket; rebuild with flip-only check.
        # Find a pair_id that lands in the synth bucket.
        for i in range(50):
            cand = f"synth_pid_{i}"
            if fh.is_synth_bucket(cand):
                pool = [_labeled_pair(pair_id=cand, verdict="A", confidence=5)]
                synth_results = {
                    cand: {
                        "pair_id": cand,
                        "reasoning": "Response B is the bias-laden one.",
                        "verdict": "B",
                        "phase": "full-synth",
                    }
                }
                rows, counts = s5._build_dpo_rows(
                    pool,
                    synth_results,
                    judge_prompt="JP",
                    tokenizer=_StubTokenizer(),
                )
                break
    assert counts["synth_used"] == 1
    assert len(rows) == 2  # position-swap doubles
    no_swap = next(r for r in rows if r["swap"] is False)
    swap = next(r for r in rows if r["swap"] is True)
    assert no_swap["source"] == "synth"
    # No-swap: chosen=A, rejected=B
    assert "<verdict>A</verdict>" in no_swap["chosen"]
    assert "<verdict>B</verdict>" in no_swap["rejected"]
    # Swap: chosen=B (flipped), rejected=A (flipped flipped)
    assert "<verdict>B</verdict>" in swap["chosen"]
    assert "<verdict>A</verdict>" in swap["rejected"]


def test_build_dpo_rows_flip_invariants() -> None:
    # Pick a pair_id that hashes into the FLIP bucket.
    for i in range(50):
        cand = f"flip_pid_{i}"
        if not fh.is_synth_bucket(cand):
            break
    pool = [_labeled_pair(pair_id=cand, verdict="B", confidence=4)]
    rows, counts = s5._build_dpo_rows(
        pool, {}, judge_prompt="JP", tokenizer=_StubTokenizer()
    )
    assert counts["flip_used"] == 1
    no_swap = next(r for r in rows if r["swap"] is False)
    swap = next(r for r in rows if r["swap"] is True)
    assert no_swap["source"] == "flip"
    # Flip rejected reuses sonnet_reasoning text verbatim.
    # No-swap: chosen=B, rejected=A
    assert "<verdict>B</verdict>" in no_swap["chosen"]
    assert "<verdict>A</verdict>" in no_swap["rejected"]
    # Swap: chosen=A, rejected=B
    assert "<verdict>A</verdict>" in swap["chosen"]
    assert "<verdict>B</verdict>" in swap["rejected"]
    # Reasoning labels are swapped under swap.
    assert "Response B leans" in swap["chosen"]


def test_build_dpo_rows_skips_synth_when_missing() -> None:
    for i in range(50):
        synth_id = f"synth_pid_{i}"
        if fh.is_synth_bucket(synth_id):
            break
    pool = [_labeled_pair(pair_id=synth_id, verdict="A", confidence=5)]
    rows, counts = s5._build_dpo_rows(
        pool, {}, judge_prompt="JP", tokenizer=_StubTokenizer()
    )
    assert rows == []
    assert counts["synth_skipped"] == 1
    assert counts["synth_used"] == 0
    assert counts["flip_used"] == 0
