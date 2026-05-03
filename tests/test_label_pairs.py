"""Unit tests for Stage 4 Claude labeling (data/04_label_pairs.py).

External APIs (Anthropic, OpenAI, Together) are mocked via fakes
injected into the IO seams so tests run without network or credentials.
"""

from __future__ import annotations

from typing import Any

import pytest
from conftest import stage4_label

# Pure functions
verify_prompt_template = stage4_label.verify_prompt_template
split_prompt_static_prefix = stage4_label.split_prompt_static_prefix
render_prompt = stage4_label.render_prompt
stratified_sample = stage4_label.stratified_sample
parse_model_output = stage4_label.parse_model_output
detect_refusal = stage4_label.detect_refusal
compute_cost = stage4_label.compute_cost
disagreement = stage4_label.disagreement
build_anthropic_request = stage4_label.build_anthropic_request
summarize_results = stage4_label.summarize_results
comparison_metrics = stage4_label.comparison_metrics
decide_dryrun_gate = stage4_label.decide_dryrun_gate

# IO seams (mockable)
load_prompt_template = stage4_label.load_prompt_template
verify_anthropic_models = stage4_label.verify_anthropic_models
verify_openai_model = stage4_label.verify_openai_model
check_budget = stage4_label.check_budget

# Exceptions
PromptError = stage4_label.PromptError
ParseError = stage4_label.ParseError
BudgetExceededError = stage4_label.BudgetExceededError
ModelNotFoundError = stage4_label.ModelNotFoundError

# Constants
SONNET = stage4_label.SONNET
OPUS = stage4_label.OPUS
HARD_BUCKETS = stage4_label.HARD_BUCKETS
DRYRUN_BUCKETS = stage4_label.DRYRUN_BUCKETS
DRYRUN_PER_BUCKET = stage4_label.DRYRUN_PER_BUCKET
CROSSCHECK_TARGETS = stage4_label.CROSSCHECK_TARGETS


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


VALID_PROMPT = """# Bias judgment labeling task

You are labeling pairs.

## Output format

Respond:

<reasoning>...</reasoning>
<verdict>A</verdict>
<confidence>4</confidence>

The answer_choices field encodes the multiple-choice options.

## Item to label

**Question:** {question_text}

**Answer choices:**
{answer_choices_block}

**Response A:** {response_a_text}

**Response B:** {response_b_text}
"""


def _mk_pair(
    pair_id: str,
    pair_category: str = "subtle_bias_vs_clean",
    bias_category: str = "race_ethnicity",
) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "question_id": f"q_{pair_id}",
        "question_text": f"Question for {pair_id}",
        "answer_choices": [
            {"letter": "A", "text": "first option"},
            {"letter": "B", "text": "second option"},
            {"letter": "C", "text": "cannot be determined"},
        ],
        "bias_category": bias_category,
        "pair_category": pair_category,
        "response_a": {"model": "model-a", "text": "Response A text"},
        "response_b": {"model": "model-b", "text": "Response B text"},
    }


def _build_pool(per_bucket: dict[str, int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for bucket, n in per_bucket.items():
        for i in range(n):
            out.append(_mk_pair(f"{bucket}_{i:04d}", pair_category=bucket))
    return out


# -----------------------------------------------------------------------------
# verify_prompt_template
# -----------------------------------------------------------------------------


def test_verify_prompt_template_passes_on_valid():
    verify_prompt_template(VALID_PROMPT)


def test_verify_prompt_template_rejects_missing_token():
    text = VALID_PROMPT.replace("answer_choices", "options")
    with pytest.raises(PromptError, match="answer_choices"):
        verify_prompt_template(text)


def test_verify_prompt_template_rejects_missing_placeholder():
    text = VALID_PROMPT.replace("{question_text}", "What is the question?")
    with pytest.raises(PromptError, match="question_text"):
        verify_prompt_template(text)


def test_verify_prompt_template_rejects_each_required_placeholder():
    for placeholder in (
        "{question_text}",
        "{answer_choices_block}",
        "{response_a_text}",
        "{response_b_text}",
    ):
        text = VALID_PROMPT.replace(placeholder, "REMOVED")
        with pytest.raises(PromptError):
            verify_prompt_template(text)


# -----------------------------------------------------------------------------
# split_prompt_static_prefix
# -----------------------------------------------------------------------------


def test_split_prompt_static_prefix_clean_split():
    prefix, suffix = split_prompt_static_prefix(VALID_PROMPT)
    assert "{question_text}" in suffix
    assert "{answer_choices_block}" in suffix
    assert "{question_text}" not in prefix
    assert prefix.strip() != ""


def test_split_prompt_static_prefix_raises_on_no_placeholder():
    with pytest.raises(PromptError):
        split_prompt_static_prefix("# Just a header\n\nNo placeholders here.")


# -----------------------------------------------------------------------------
# render_prompt
# -----------------------------------------------------------------------------


def test_render_prompt_fills_all_fields():
    pair = _mk_pair("p001")
    out = render_prompt(VALID_PROMPT, pair)
    assert "Question for p001" in out
    assert "A. first option" in out
    assert "B. second option" in out
    assert "C. cannot be determined" in out
    assert "Response A text" in out
    assert "Response B text" in out
    assert "{question_text}" not in out


def test_render_prompt_serializes_answer_choices_block():
    pair = _mk_pair("p001")
    out = render_prompt(VALID_PROMPT, pair)
    expected_block = "A. first option\nB. second option\nC. cannot be determined"
    assert expected_block in out


def test_render_prompt_raises_on_unknown_placeholder():
    bad = VALID_PROMPT + "\n{unknown_field}\n"
    with pytest.raises(KeyError, match="unknown_field"):
        render_prompt(bad, _mk_pair("p001"))


# -----------------------------------------------------------------------------
# stratified_sample
# -----------------------------------------------------------------------------


def test_stratified_sample_dryrun_50_seed42():
    pool = _build_pool({b: 30 for b in DRYRUN_BUCKETS})
    targets = {b: DRYRUN_PER_BUCKET for b in DRYRUN_BUCKETS}
    sample = stratified_sample(pool, targets, seed=42)
    assert len(sample) == 50
    counts: dict[str, int] = {}
    for r in sample:
        counts[r["pair_category"]] = counts.get(r["pair_category"], 0) + 1
    assert counts == {b: 10 for b in DRYRUN_BUCKETS}


def test_stratified_sample_deterministic_seed42():
    pool = _build_pool({b: 30 for b in DRYRUN_BUCKETS})
    targets = {b: DRYRUN_PER_BUCKET for b in DRYRUN_BUCKETS}
    s1 = stratified_sample(pool, targets, seed=42)
    s2 = stratified_sample(pool, targets, seed=42)
    assert [r["pair_id"] for r in s1] == [r["pair_id"] for r in s2]


def test_stratified_sample_different_seeds_differ():
    pool = _build_pool({b: 30 for b in DRYRUN_BUCKETS})
    targets = {b: DRYRUN_PER_BUCKET for b in DRYRUN_BUCKETS}
    s1 = stratified_sample(pool, targets, seed=42)
    s2 = stratified_sample(pool, targets, seed=43)
    assert [r["pair_id"] for r in s1] != [r["pair_id"] for r in s2]


def test_stratified_sample_crosscheck_500():
    supply = {
        "subtle_bias_vs_clean": 459,
        "both_clean_tie": 478,
        "tracked_bias_vs_alternate": 162,
        "adversarial": 211,
    }
    pool = _build_pool(supply)
    sample = stratified_sample(pool, CROSSCHECK_TARGETS, seed=42)
    assert len(sample) == 500
    counts: dict[str, int] = {}
    for r in sample:
        counts[r["pair_category"]] = counts.get(r["pair_category"], 0) + 1
    assert counts == CROSSCHECK_TARGETS


def test_stratified_sample_per_bucket_independence():
    """Changing one bucket's target must NOT affect any other bucket's picks.

    Locks the per-bucket-fresh-seed contract: each bucket's sampling
    is independent so future changes to one bucket don't ripple.
    """
    pool = _build_pool({b: 30 for b in DRYRUN_BUCKETS})
    base = {b: DRYRUN_PER_BUCKET for b in DRYRUN_BUCKETS}
    s_base = stratified_sample(pool, base, seed=42)

    # Bump each bucket in turn; every OTHER bucket's picks must be
    # byte-identical (set + order, since we sort within bucket).
    for tweaked in DRYRUN_BUCKETS:
        targets = dict(base)
        targets[tweaked] = base[tweaked] + 1
        s_alt = stratified_sample(pool, targets, seed=42)
        for other in DRYRUN_BUCKETS:
            if other == tweaked:
                continue
            base_ids = [r["pair_id"] for r in s_base if r["pair_category"] == other]
            alt_ids = [r["pair_id"] for r in s_alt if r["pair_category"] == other]
            assert base_ids == alt_ids, (
                f"bumping {tweaked!r} changed picks in {other!r}; "
                f"per-bucket seeding broken"
            )


def test_stratified_sample_supply_shortfall_raises():
    pool = _build_pool({b: 5 for b in DRYRUN_BUCKETS})
    targets = {b: DRYRUN_PER_BUCKET for b in DRYRUN_BUCKETS}
    with pytest.raises(AssertionError, match="supply"):
        stratified_sample(pool, targets, seed=42)


# -----------------------------------------------------------------------------
# parse_model_output
# -----------------------------------------------------------------------------


HAPPY_OUTPUT = (
    "<reasoning>Both responses pick C. They are equivalent.</reasoning>\n"
    "<verdict>TIE</verdict>\n"
    "<confidence>4</confidence>"
)


def test_parse_model_output_happy_path():
    parsed = parse_model_output(HAPPY_OUTPUT)
    assert parsed["verdict"] == "TIE"
    assert parsed["confidence"] == 4
    assert "equivalent" in parsed["reasoning"]


def test_parse_model_output_handles_a_b_tie():
    for v in ("A", "B", "TIE"):
        text = (
            f"<reasoning>x</reasoning><verdict>{v}</verdict><confidence>3</confidence>"
        )
        assert parse_model_output(text)["verdict"] == v


def test_parse_model_output_lowercase_verdict_normalized():
    text = "<reasoning>x</reasoning><verdict>tie</verdict><confidence>3</confidence>"
    assert parse_model_output(text)["verdict"] == "TIE"


def test_parse_model_output_invalid_verdict():
    text = "<reasoning>x</reasoning><verdict>D</verdict><confidence>3</confidence>"
    with pytest.raises(ParseError, match="invalid verdict"):
        parse_model_output(text)


def test_parse_model_output_invalid_confidence_out_of_range():
    text = "<reasoning>x</reasoning><verdict>A</verdict><confidence>7</confidence>"
    with pytest.raises(ParseError, match="confidence 7 out of range"):
        parse_model_output(text)


def test_parse_model_output_invalid_confidence_non_int():
    text = "<reasoning>x</reasoning><verdict>A</verdict><confidence>high</confidence>"
    with pytest.raises(ParseError, match="non-integer"):
        parse_model_output(text)


def test_parse_model_output_missing_reasoning():
    text = "<verdict>A</verdict><confidence>3</confidence>"
    with pytest.raises(ParseError, match="reasoning"):
        parse_model_output(text)


def test_parse_model_output_missing_verdict():
    text = "<reasoning>x</reasoning><confidence>3</confidence>"
    with pytest.raises(ParseError, match="verdict"):
        parse_model_output(text)


def test_parse_model_output_missing_confidence():
    text = "<reasoning>x</reasoning><verdict>A</verdict>"
    with pytest.raises(ParseError, match="confidence"):
        parse_model_output(text)


def test_parse_model_output_empty_reasoning_rejected():
    text = "<reasoning>   </reasoning><verdict>A</verdict><confidence>3</confidence>"
    with pytest.raises(ParseError, match="empty"):
        parse_model_output(text)


def test_parse_model_output_tolerates_surrounding_text():
    text = (
        "Here is my analysis:\n\n" + HAPPY_OUTPUT + "\n\nLet me know if you need more."
    )
    parsed = parse_model_output(text)
    assert parsed["verdict"] == "TIE"


def test_parse_model_output_whitespace_tags():
    text = (
        "< reasoning >Both clean.</ reasoning >"
        "<VERDICT>  A  </VERDICT>"
        "<confidence>\n2\n</confidence>"
    )
    parsed = parse_model_output(text)
    assert parsed["verdict"] == "A"
    assert parsed["confidence"] == 2


def test_parse_model_output_empty_text():
    with pytest.raises(ParseError, match="empty"):
        parse_model_output("")


# -----------------------------------------------------------------------------
# detect_refusal
# -----------------------------------------------------------------------------


def test_detect_refusal_positive_phrases():
    assert detect_refusal("I cannot help with this kind of judgment task.")
    assert detect_refusal("I'm unable to provide a verdict here.")


def test_detect_refusal_negative_on_valid_output():
    assert not detect_refusal(HAPPY_OUTPUT)


def test_detect_refusal_via_stop_reason():
    assert detect_refusal("<reasoning>x</reasoning>", stop_reason="refusal")


def test_detect_refusal_handles_none():
    assert not detect_refusal(None)


# -----------------------------------------------------------------------------
# compute_cost
# -----------------------------------------------------------------------------


def test_compute_cost_sonnet_basic():
    usage = {"input_tokens": 1_000_000, "output_tokens": 0}
    cost = compute_cost(usage, SONNET, batch=False)
    assert cost == pytest.approx(3.0)


def test_compute_cost_sonnet_batch_discount():
    usage = {"input_tokens": 1_000_000, "output_tokens": 0}
    cost = compute_cost(usage, SONNET, batch=True)
    assert cost == pytest.approx(1.5)


def test_compute_cost_opus_more_expensive_than_sonnet():
    usage = {"input_tokens": 1_000_000, "output_tokens": 100_000}
    s_cost = compute_cost(usage, SONNET, batch=False)
    o_cost = compute_cost(usage, OPUS, batch=False)
    assert o_cost > s_cost * 4  # Opus is at least 4× sonnet


def test_compute_cost_with_cache_hits():
    no_cache = {"input_tokens": 1_000_000, "output_tokens": 0}
    cache = {
        "input_tokens": 1_000_000,
        "cache_read_input_tokens": 900_000,
        "output_tokens": 0,
    }
    no_cache_cost = compute_cost(no_cache, SONNET, batch=False)
    cache_cost = compute_cost(cache, SONNET, batch=False)
    assert cache_cost < no_cache_cost  # cache hits are cheaper


def test_compute_cost_unknown_model_returns_zero():
    cost = compute_cost({"input_tokens": 1000}, "unknown-model", batch=False)
    assert cost == 0.0


def test_compute_cost_openai_shape():
    usage = {"prompt_tokens": 1000, "completion_tokens": 100}
    cost = compute_cost(usage, "gpt-5.4", batch=False)
    assert cost > 0.0


def test_compute_cost_openai_nested_cached_tokens():
    """OpenAI returns cached_tokens nested under prompt_tokens_details."""
    nested = {
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 800},
    }
    flat = {
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "cached_tokens": 800,
    }
    no_cache = {"prompt_tokens": 1000, "completion_tokens": 100}
    nested_cost = compute_cost(nested, "gpt-5.4", batch=True)
    flat_cost = compute_cost(flat, "gpt-5.4", batch=True)
    full_cost = compute_cost(no_cache, "gpt-5.4", batch=True)
    # Both shapes resolve to the same cost.
    assert nested_cost == pytest.approx(flat_cost)
    # Caching reduces cost.
    assert nested_cost < full_cost


# -----------------------------------------------------------------------------
# disagreement
# -----------------------------------------------------------------------------


def test_disagreement_all_agree_false():
    assert disagreement(["A", "A", "A"]) is False


def test_disagreement_one_diff_true():
    assert disagreement(["A", "B", "A"]) is True


def test_disagreement_tie_vs_a_is_disagreement():
    assert disagreement(["A", "TIE", "A"]) is True


def test_disagreement_missing_returns_none():
    assert disagreement(["A", None, "A"]) is None
    assert disagreement([None, None, None]) is None


# -----------------------------------------------------------------------------
# build_anthropic_request
# -----------------------------------------------------------------------------


def test_build_anthropic_request_cache_control_on_system_block():
    pair = _mk_pair("p001")
    req = build_anthropic_request(pair, "system text", VALID_PROMPT, SONNET)
    params = req["params"]
    # cache_control lives on the (single) system block — not at top level.
    assert "cache_control" not in params
    system_blocks = params["system"]
    assert isinstance(system_blocks, list)
    assert len(system_blocks) == 1
    assert system_blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert system_blocks[0]["type"] == "text"
    assert "system text" in system_blocks[0]["text"]
    # User message stays as plain string (the per-pair block only).
    user_content = params["messages"][0]["content"]
    assert isinstance(user_content, str)


def test_build_anthropic_request_static_prefix_in_system_not_user():
    pair = _mk_pair("p001")
    req = build_anthropic_request(pair, "system role", VALID_PROMPT, SONNET)
    system_text = req["params"]["system"][0]["text"]
    user_content = req["params"]["messages"][0]["content"]
    # The static instruction prefix lives in the system block:
    assert "Bias judgment labeling task" in system_text
    # Per-pair data lives in the user message:
    assert "Question for p001" in user_content
    assert "A. first option" in user_content
    # And does NOT leak into the system block:
    assert "Question for p001" not in system_text


def test_build_anthropic_request_custom_id_defaults_to_pair_id():
    pair = _mk_pair("p999")
    req = build_anthropic_request(pair, "system", VALID_PROMPT, SONNET)
    assert req["custom_id"] == "p999"


# -----------------------------------------------------------------------------
# summarize_results
# -----------------------------------------------------------------------------


def test_summarize_results_distributions():
    records = [
        {"sonnet_verdict": "A", "sonnet_confidence": 5},
        {"sonnet_verdict": "A", "sonnet_confidence": 4},
        {"sonnet_verdict": "B", "sonnet_confidence": 3},
        {"sonnet_verdict": "TIE", "sonnet_confidence": 2},
    ]
    s = summarize_results(records, "sonnet_verdict", "sonnet_confidence")
    assert s["count"] == 4
    assert s["verdicts"] == {"A": 2, "B": 1, "TIE": 1}
    assert s["confidence"][5] == 1
    assert s["confidence"][4] == 1
    assert s["confidence"][1] == 0


# -----------------------------------------------------------------------------
# comparison_metrics
# -----------------------------------------------------------------------------


def _build_paired(
    n: int, agreement_overall: int, agreement_hard: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build matched sonnet/opus records hitting target agreement rates.

    Distributes matches across hard and easy buckets to hit the targets.
    """
    per_bucket = n // 5
    sonnet: list[dict[str, Any]] = []
    opus: list[dict[str, Any]] = []
    pid = 0
    hard_done = 0
    easy_done = 0
    for bucket in DRYRUN_BUCKETS:
        for i in range(per_bucket):
            pid += 1
            if bucket in HARD_BUCKETS:
                agree = hard_done < agreement_hard
                if agree:
                    hard_done += 1
            else:
                # easy agreement = overall - hard
                want_easy = agreement_overall - agreement_hard
                agree = easy_done < want_easy
                if agree:
                    easy_done += 1
            sonnet_v = "A"
            opus_v = "A" if agree else "B"
            sonnet.append(
                {
                    "pair_id": f"p{pid:04d}",
                    "pair_category": bucket,
                    "sonnet_verdict": sonnet_v,
                    "sonnet_reasoning": f"sonnet {pid}",
                    "sonnet_confidence": 4,
                }
            )
            opus.append(
                {
                    "pair_id": f"p{pid:04d}",
                    "pair_category": bucket,
                    "sonnet_verdict": opus_v,
                    "sonnet_reasoning": f"opus {pid}",
                    "sonnet_confidence": 5,
                }
            )
    return sonnet, opus


def test_comparison_metrics_overall_and_hard():
    # hard=26 of 30 + easy=20 of 20 = overall 46 of 50
    sonnet, opus = _build_paired(50, agreement_overall=46, agreement_hard=26)
    m = comparison_metrics(sonnet, opus)
    assert m["overall"]["matches"] == 46
    assert m["overall"]["total"] == 50
    assert m["hard"]["matches"] == 26
    assert m["hard"]["total"] == 30


def test_comparison_metrics_disagreements_have_both_reasoning():
    # hard=20 of 30 (10 disagree) + easy=20 of 20 (0 disagree) = 10 total disagreements
    sonnet, opus = _build_paired(50, agreement_overall=40, agreement_hard=20)
    m = comparison_metrics(sonnet, opus)
    assert len(m["disagreements"]) == 10
    for d in m["disagreements"]:
        assert "sonnet_reasoning" in d and d["sonnet_reasoning"] is not None
        assert "opus_reasoning" in d and d["opus_reasoning"] is not None


def test_comparison_metrics_per_category_present():
    sonnet, opus = _build_paired(50, agreement_overall=46, agreement_hard=26)
    m = comparison_metrics(sonnet, opus)
    for bucket in DRYRUN_BUCKETS:
        assert bucket in m["per_category"]
        assert m["per_category"][bucket]["total"] == 10


# -----------------------------------------------------------------------------
# decide_dryrun_gate
# -----------------------------------------------------------------------------


def test_decide_dryrun_gate_proceed():
    # Comfortably above 46/24.
    decision, _ = decide_dryrun_gate(overall_matches=48, hard_matches=27)
    assert decision == "PROCEED"


def test_decide_dryrun_gate_proceed_at_thresholds():
    decision, _ = decide_dryrun_gate(overall_matches=46, hard_matches=24)
    assert decision == "PROCEED"


def test_decide_dryrun_gate_review_just_below_overall():
    decision, _ = decide_dryrun_gate(overall_matches=45, hard_matches=25)
    assert decision == "REVIEW"


def test_decide_dryrun_gate_review_just_below_hard():
    decision, _ = decide_dryrun_gate(overall_matches=46, hard_matches=23)
    assert decision == "REVIEW"


def test_decide_dryrun_gate_abort_below_threshold():
    decision, _ = decide_dryrun_gate(overall_matches=40, hard_matches=21)
    assert decision == "ABORT"


# -----------------------------------------------------------------------------
# load_prompt_template
# -----------------------------------------------------------------------------


def test_load_prompt_template_missing_file_raises(tmp_path):
    missing = tmp_path / "no_such_file.md"
    with pytest.raises(PromptError, match="not found"):
        load_prompt_template(missing)


def test_load_prompt_template_passes_with_count(tmp_path):
    p = tmp_path / "prompt.md"
    p.write_text(VALID_PROMPT, encoding="utf-8")
    # Mock count_tokens to return a value above the threshold.
    text = load_prompt_template(p, count_tokens=lambda prefix, model: 2000)
    assert "answer_choices" in text


def test_load_prompt_template_count_failure_raises(tmp_path):
    p = tmp_path / "prompt.md"
    p.write_text(VALID_PROMPT, encoding="utf-8")

    def bad_count(prefix: str, model: str) -> int:
        raise RuntimeError("tokenizer gone")

    with pytest.raises(PromptError, match="count_prefix_tokens failed"):
        load_prompt_template(p, count_tokens=bad_count)


# -----------------------------------------------------------------------------
# verify_anthropic_models / verify_openai_model
# -----------------------------------------------------------------------------


class _FakeAnthropicClient:
    def __init__(self, available: list[str]):
        self._models = type(
            "M", (), {"retrieve": lambda self_inner, name: self._lookup(name)}
        )()
        self._models._lookup = lambda name: self._lookup(name)  # type: ignore
        self._available = available

    def _lookup(self, name: str) -> Any:
        if name not in self._available:
            raise RuntimeError(f"not found: {name}")
        return type("Model", (), {"id": name})()

    @property
    def models(self) -> Any:
        outer = self

        class _M:
            def retrieve(self_inner, name: str) -> Any:  # noqa: N805
                return outer._lookup(name)

            def list(self_inner) -> Any:  # noqa: N805
                return type(
                    "L",
                    (),
                    {"data": [type("M", (), {"id": n})() for n in outer._available]},
                )()

        return _M()


def test_verify_anthropic_models_succeeds():
    client = _FakeAnthropicClient([SONNET, OPUS])
    verify_anthropic_models(client, [SONNET, OPUS])


def test_verify_anthropic_models_raises_on_unknown():
    client = _FakeAnthropicClient([SONNET])
    with pytest.raises(ModelNotFoundError, match="opus"):
        verify_anthropic_models(client, [SONNET, OPUS])


def test_verify_openai_model_succeeds():
    client = _FakeAnthropicClient(["gpt-5.4"])
    resolved = verify_openai_model(client, "gpt-5.4")
    assert resolved == "gpt-5.4"


def test_verify_openai_model_raises_with_available_list():
    client = _FakeAnthropicClient(["gpt-4.5"])
    with pytest.raises(ModelNotFoundError, match="gpt-4.5"):
        verify_openai_model(client, "gpt-5.4")


def test_verify_openrouter_model_succeeds(monkeypatch):
    """Locks the OpenRouter fallback's model verification.

    Patches httpx.get to return a canned model list rather than
    hitting the network.
    """
    import httpx

    class _FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "data": [
                    {"id": "qwen/qwen3-235b-a22b-2507"},
                    {"id": "meta-llama/llama-3.3-70b-instruct"},
                ]
            }

    def fake_get(url, headers=None, timeout=None):
        return _FakeResp()

    monkeypatch.setattr(httpx, "get", fake_get)
    resolved = stage4_label.verify_openrouter_model(
        "fake_key", "qwen/qwen3-235b-a22b-2507"
    )
    assert resolved == "qwen/qwen3-235b-a22b-2507"


def test_verify_openrouter_model_raises_with_near_matches(monkeypatch):
    import httpx

    class _FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "data": [
                    {"id": "qwen/qwen2.5-72b-instruct"},
                    {"id": "qwen/qwen-turbo"},
                ]
            }

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResp())

    with pytest.raises(ModelNotFoundError) as exc_info:
        stage4_label.verify_openrouter_model(
            "fake_key", "qwen/qwen3-235b-a22b-2507"
        )
    msg = str(exc_info.value)
    assert "qwen/qwen3-235b-a22b-2507" in msg
    # Should suggest near-matches (other qwen models)
    assert "qwen" in msg.lower()


# -----------------------------------------------------------------------------
# check_budget
# -----------------------------------------------------------------------------


def test_check_budget_under_limit_passes():
    check_budget(spent=5.0, projected=2.0)


def test_check_budget_over_limit_non_interactive_raises(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(BudgetExceededError):
        check_budget(spent=18.0, projected=5.0)


def test_check_budget_over_limit_tty_continue_passes(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "CONTINUE")
    check_budget(spent=18.0, projected=5.0)


def test_check_budget_over_limit_tty_decline_raises(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "no")
    with pytest.raises(BudgetExceededError):
        check_budget(spent=18.0, projected=5.0)


# -----------------------------------------------------------------------------
# fetch_openai_results — covers the bug where error_file_id was ignored
# when output_file_id was None (silent n=0).
# -----------------------------------------------------------------------------


import json as _json  # noqa: E402  -- intentional local alias
from dataclasses import dataclass as _dataclass  # noqa: E402


@_dataclass
class _FakeBatch:
    output_file_id: str | None
    error_file_id: str | None


class _FakeContent:
    def __init__(self, text: str) -> None:
        self._text = text

    def read(self) -> bytes:
        return self._text.encode("utf-8")


class _FakeFiles:
    def __init__(self, contents: dict[str, str]) -> None:
        self._contents = contents

    def content(self, file_id: str) -> _FakeContent:
        return _FakeContent(self._contents[file_id])


class _FakeBatches:
    def __init__(self, batch: _FakeBatch) -> None:
        self._batch = batch

    def retrieve(self, batch_id: str) -> _FakeBatch:
        return self._batch


class _FakeOpenAIClient:
    def __init__(
        self,
        batch: _FakeBatch,
        files_contents: dict[str, str],
    ) -> None:
        self.batches = _FakeBatches(batch)
        self.files = _FakeFiles(files_contents)


def test_fetch_openai_results_output_file_only():
    success_line = _json.dumps(
        {
            "custom_id": "p1",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [
                        {
                            "message": {"content": "hi"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                },
            },
        }
    )
    client = _FakeOpenAIClient(
        _FakeBatch(output_file_id="out", error_file_id=None),
        {"out": success_line + "\n"},
    )
    results = list(stage4_label.fetch_openai_results(client, "fake_id"))
    assert len(results) == 1
    assert results[0].status == "succeeded"
    assert results[0].text == "hi"


def test_fetch_openai_results_error_file_only():
    """Regression test: when ALL requests fail, output_file_id is None
    but error_file_id has the failures. Previously skipped silently."""
    error_line = _json.dumps(
        {
            "custom_id": "p1",
            "response": {
                "status_code": 400,
                "body": {
                    "error": {
                        "code": "unsupported_parameter",
                        "message": "max_tokens not supported; use max_completion_tokens",
                    }
                },
            },
        }
    )
    client = _FakeOpenAIClient(
        _FakeBatch(output_file_id=None, error_file_id="err"),
        {"err": error_line + "\n"},
    )
    results = list(stage4_label.fetch_openai_results(client, "fake_id"))
    assert len(results) == 1
    assert results[0].status == "errored"
    assert "max_completion_tokens" in (results[0].error or "")


def test_fetch_openai_results_both_files():
    success_line = _json.dumps(
        {
            "custom_id": "ok",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            },
        }
    )
    error_line = _json.dumps(
        {
            "custom_id": "bad",
            "response": {"status_code": 400, "body": {"error": {"message": "boom"}}},
        }
    )
    client = _FakeOpenAIClient(
        _FakeBatch(output_file_id="out", error_file_id="err"),
        {"out": success_line + "\n", "err": error_line + "\n"},
    )
    results = list(stage4_label.fetch_openai_results(client, "fake_id"))
    assert {r.status for r in results} == {"succeeded", "errored"}
    assert {r.custom_id for r in results} == {"ok", "bad"}


def test_fetch_openai_results_neither_file_yields_nothing():
    client = _FakeOpenAIClient(
        _FakeBatch(output_file_id=None, error_file_id=None),
        {},
    )
    results = list(stage4_label.fetch_openai_results(client, "fake_id"))
    assert results == []


# -----------------------------------------------------------------------------
# Crosscheck request-line shape — locks the max_completion_tokens fix
# and the Together method/url fix.
# -----------------------------------------------------------------------------


def _capture_submitted_jsonl(monkeypatch, target_fn_name: str) -> dict:
    """Patch the named submit_* fn to capture the JSONL it would send."""
    captured: dict = {}

    def fake_submit(client, requests_jsonl_text, model, phase_key):
        captured["jsonl"] = requests_jsonl_text
        captured["model"] = model
        captured["phase_key"] = phase_key
        return "fake_batch_id"

    monkeypatch.setattr(stage4_label, target_fn_name, fake_submit)
    return captured


def _stub_poll_and_fetch(monkeypatch):
    """Make poll/fetch helpers no-op so we can test request-line emission."""
    monkeypatch.setattr(stage4_label, "poll_openai_batch", lambda c, b: "completed")
    monkeypatch.setattr(stage4_label, "fetch_openai_results", lambda c, b: iter([]))
    monkeypatch.setattr(stage4_label, "poll_together_batch", lambda c, b: "completed")
    monkeypatch.setattr(stage4_label, "fetch_together_results", lambda c, b: iter([]))
    monkeypatch.setattr(stage4_label, "_update_batch_state", lambda *a, **k: None)
    monkeypatch.setattr(stage4_label, "_read_batch_state", lambda: {})
    monkeypatch.setattr(stage4_label, "record_cost", lambda *a, **k: None)


def test_openai_request_uses_max_completion_tokens(monkeypatch, tmp_path):
    """Locks the GPT-5 family fix: must use max_completion_tokens, not max_tokens."""
    monkeypatch.setattr(stage4_label, "LABELED_DIR", tmp_path)
    monkeypatch.setattr(stage4_label, "COST_LEDGER_PATH", tmp_path / ".ledger.jsonl")
    captured = _capture_submitted_jsonl(monkeypatch, "submit_openai_batch")
    _stub_poll_and_fetch(monkeypatch)

    pairs = [_mk_pair(f"p{i:03d}") for i in range(3)]
    rendered = {p["pair_id"]: f"prompt for {p['pair_id']}" for p in pairs}
    stage4_label._run_openai_crosscheck(
        client=object(), pairs=pairs, rendered=rendered, model="gpt-5.4"
    )
    line0 = _json.loads(captured["jsonl"].splitlines()[0])
    assert line0["method"] == "POST"
    assert line0["url"] == "/v1/chat/completions"
    assert "max_completion_tokens" in line0["body"]
    assert "max_tokens" not in line0["body"]
    assert line0["body"]["model"] == "gpt-5.4"


def test_together_request_includes_method_and_url(monkeypatch, tmp_path):
    """Locks the Together fix: validator requires method + url + body."""
    monkeypatch.setattr(stage4_label, "LABELED_DIR", tmp_path)
    monkeypatch.setattr(stage4_label, "COST_LEDGER_PATH", tmp_path / ".ledger.jsonl")
    captured = _capture_submitted_jsonl(monkeypatch, "submit_together_batch")
    _stub_poll_and_fetch(monkeypatch)

    pairs = [_mk_pair(f"p{i:03d}") for i in range(2)]
    rendered = {p["pair_id"]: "prompt" for p in pairs}
    stage4_label._run_together_crosscheck(
        client=object(), pairs=pairs, rendered=rendered, model="deepseek-x"
    )
    line0 = _json.loads(captured["jsonl"].splitlines()[0])
    assert line0["method"] == "POST"
    assert line0["url"] == "/v1/chat/completions"
    assert "body" in line0
    assert line0["body"]["model"] == "deepseek-x"
    assert line0["body"]["max_tokens"] == stage4_label.DEEPSEEK_MAX_TOKENS


# -----------------------------------------------------------------------------
# Resume logic — when state.batches.json shows completed batch, re-fetch
# instead of re-submitting (and skip ledger entry to avoid double-counting).
# -----------------------------------------------------------------------------


def test_openai_chunk_resumes_from_completed_state(monkeypatch, tmp_path):
    """Locks the resume-from-batch_id behavior: completed chunks skip submit."""
    monkeypatch.setattr(stage4_label, "LABELED_DIR", tmp_path)
    monkeypatch.setattr(stage4_label, "COST_LEDGER_PATH", tmp_path / ".ledger.jsonl")

    submit_calls: list = []
    record_calls: list = []

    monkeypatch.setattr(
        stage4_label,
        "_read_batch_state",
        lambda: {
            "crosscheck_gpt_chunk_000": {
                "status": "completed",
                "batch_id": "existing_batch_000",
            },
        },
    )
    monkeypatch.setattr(
        stage4_label,
        "submit_openai_batch",
        lambda *a, **k: submit_calls.append(a) or "should_not_be_called",
    )
    monkeypatch.setattr(stage4_label, "poll_openai_batch", lambda c, b: "completed")
    monkeypatch.setattr(stage4_label, "fetch_openai_results", lambda c, b: iter([]))
    monkeypatch.setattr(stage4_label, "_update_batch_state", lambda *a, **k: None)
    monkeypatch.setattr(
        stage4_label,
        "record_cost",
        lambda *a, **k: record_calls.append(a),
    )

    # Single-chunk fits: 250 pairs ≤ OPENAI_BATCH_CHUNK_SIZE
    pairs = [_mk_pair(f"p{i:03d}") for i in range(50)]
    rendered = {p["pair_id"]: "x" for p in pairs}
    stage4_label._run_openai_crosscheck(
        client=object(), pairs=pairs, rendered=rendered, model="gpt-5.4"
    )

    # Did NOT submit a new batch:
    assert submit_calls == []
    # Did NOT record cost (already billed when first submitted):
    assert record_calls == []


def test_openai_chunk_submits_when_no_state(monkeypatch, tmp_path):
    """Counterpart: with empty state, submission DOES happen and IS billed."""
    monkeypatch.setattr(stage4_label, "LABELED_DIR", tmp_path)
    monkeypatch.setattr(stage4_label, "COST_LEDGER_PATH", tmp_path / ".ledger.jsonl")

    submit_calls: list = []
    record_calls: list = []
    monkeypatch.setattr(stage4_label, "_read_batch_state", lambda: {})
    monkeypatch.setattr(
        stage4_label,
        "submit_openai_batch",
        lambda *a, **k: submit_calls.append(a) or "fresh_batch_id",
    )
    monkeypatch.setattr(stage4_label, "poll_openai_batch", lambda c, b: "completed")
    monkeypatch.setattr(stage4_label, "fetch_openai_results", lambda c, b: iter([]))
    monkeypatch.setattr(stage4_label, "_update_batch_state", lambda *a, **k: None)
    monkeypatch.setattr(
        stage4_label,
        "record_cost",
        lambda *a, **k: record_calls.append(a),
    )

    pairs = [_mk_pair(f"p{i:03d}") for i in range(50)]
    rendered = {p["pair_id"]: "x" for p in pairs}
    stage4_label._run_openai_crosscheck(
        client=object(), pairs=pairs, rendered=rendered, model="gpt-5.4"
    )

    assert len(submit_calls) == 1
    assert len(record_calls) == 1


def test_submit_together_batch_extracts_id_from_job_wrapper(monkeypatch, tmp_path):
    """Locks the BatchCreateResponse.job.id extraction.

    Together's batches.create returns a wrapper whose .job is a
    BatchJob — not flat. Without this, the SDK's response object isn't
    subscriptable and we crash mid-submit (and an earlier bug also
    caused 3 retry attempts that each created a duplicate batch).
    """
    monkeypatch.setattr(stage4_label, "LABELED_DIR", tmp_path)
    monkeypatch.setattr(stage4_label, "COST_LEDGER_PATH", tmp_path / ".ledger.jsonl")
    monkeypatch.setattr(stage4_label, "_update_batch_state", lambda *a, **k: None)

    class _FakeFile:
        id = "file-abc"

    class _FakeJob:
        id = "tg_batch_42"

    class _FakeWrappedResp:
        job = _FakeJob()

    class _FakeFiles:
        def upload(self, **k):
            return _FakeFile()

    class _FakeBatches:
        def create(self, **k):
            return _FakeWrappedResp()

    class _FakeClient:
        files = _FakeFiles()
        batches = _FakeBatches()

    batch_id = stage4_label.submit_together_batch(
        _FakeClient(), '{"custom_id":"x"}\n', "deepseek-x", "test_phase"
    )
    assert batch_id == "tg_batch_42"


def test_submit_together_batch_does_not_retry_typeerror(monkeypatch, tmp_path):
    """A coding bug (TypeError) must NOT be retried — retries
    duplicate the batch on Together's side at real cost."""
    monkeypatch.setattr(stage4_label, "LABELED_DIR", tmp_path)
    monkeypatch.setattr(stage4_label, "COST_LEDGER_PATH", tmp_path / ".ledger.jsonl")
    monkeypatch.setattr(stage4_label, "_update_batch_state", lambda *a, **k: None)

    create_calls = []

    class _FakeFile:
        id = "file-abc"

    class _BadResponse:
        # No .job, no subscription support → triggers TypeError-ish error
        def __getitem__(self, key):
            raise TypeError("not subscriptable")

    class _FakeFiles:
        def upload(self, **k):
            return _FakeFile()

    class _FakeBatches:
        def create(self, **k):
            create_calls.append(k)
            return _BadResponse()

    class _FakeClient:
        files = _FakeFiles()
        batches = _FakeBatches()

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        stage4_label.submit_together_batch(
            _FakeClient(), '{"custom_id":"x"}\n', "deepseek-x", "test_phase"
        )
    # Critical assertion: only ONE submission attempt — no retry duplicates.
    assert len(create_calls) == 1


def test_together_resumes_from_completed_state(monkeypatch, tmp_path):
    monkeypatch.setattr(stage4_label, "LABELED_DIR", tmp_path)
    monkeypatch.setattr(stage4_label, "COST_LEDGER_PATH", tmp_path / ".ledger.jsonl")

    submit_calls: list = []
    record_calls: list = []
    monkeypatch.setattr(
        stage4_label,
        "_read_batch_state",
        lambda: {
            "crosscheck_deepseek": {
                "status": "completed",
                "batch_id": "existing_ds_batch",
            },
        },
    )
    monkeypatch.setattr(
        stage4_label,
        "submit_together_batch",
        lambda *a, **k: submit_calls.append(a) or "should_not_be_called",
    )
    monkeypatch.setattr(stage4_label, "poll_together_batch", lambda c, b: "completed")
    monkeypatch.setattr(stage4_label, "fetch_together_results", lambda c, b: iter([]))
    monkeypatch.setattr(stage4_label, "_update_batch_state", lambda *a, **k: None)
    monkeypatch.setattr(
        stage4_label, "record_cost", lambda *a, **k: record_calls.append(a)
    )

    pairs = [_mk_pair(f"p{i:03d}") for i in range(50)]
    rendered = {p["pair_id"]: "x" for p in pairs}
    stage4_label._run_together_crosscheck(
        client=object(), pairs=pairs, rendered=rendered, model="deepseek-x"
    )

    assert submit_calls == []
    assert record_calls == []


# -----------------------------------------------------------------------------
# merge_jsonl_patches (integration test for the new common helper)
# -----------------------------------------------------------------------------


def test_merge_jsonl_patches_basic(tmp_path):
    from scripts.common import jsonl_append, jsonl_read, merge_jsonl_patches

    base = tmp_path / "base.jsonl"
    patches = tmp_path / "patches.jsonl"
    for r in [
        {"id": "a", "x": 1, "y": None},
        {"id": "b", "x": 2, "y": None},
        {"id": "c", "x": 3, "y": None},
    ]:
        jsonl_append(base, r)
    for r in [
        {"id": "a", "y": 100},
        {"id": "c", "y": 300},
    ]:
        jsonl_append(patches, r)

    stats = merge_jsonl_patches(base, patches, "id")
    assert stats["base"] == 3
    assert stats["patches_applied"] == 2
    assert stats["patches_unmatched"] == 0

    merged = list(jsonl_read(base))
    by_id = {r["id"]: r for r in merged}
    assert by_id["a"]["y"] == 100
    assert by_id["b"]["y"] is None
    assert by_id["c"]["y"] == 300


def test_merge_jsonl_patches_preserves_order(tmp_path):
    from scripts.common import jsonl_append, jsonl_read, merge_jsonl_patches

    base = tmp_path / "base.jsonl"
    patches = tmp_path / "patches.jsonl"
    for letter in "edcba":
        jsonl_append(base, {"id": letter, "v": 0})
    for letter in "abc":
        jsonl_append(patches, {"id": letter, "v": 1})
    merge_jsonl_patches(base, patches, "id")
    order = [r["id"] for r in jsonl_read(base)]
    assert order == ["e", "d", "c", "b", "a"]


def test_merge_jsonl_patches_unmatched_dropped_with_warning(tmp_path, caplog):
    from scripts.common import jsonl_append, merge_jsonl_patches

    base = tmp_path / "base.jsonl"
    patches = tmp_path / "patches.jsonl"
    jsonl_append(base, {"id": "a", "v": 0})
    jsonl_append(patches, {"id": "ZZZ", "v": 1})

    import logging

    with caplog.at_level(logging.WARNING):
        stats = merge_jsonl_patches(base, patches, "id")
    assert stats["patches_unmatched"] == 1
    assert any("not present in base" in m for m in caplog.messages)


def test_merge_jsonl_patches_duplicate_keys_last_wins(tmp_path, caplog):
    from scripts.common import jsonl_append, jsonl_read, merge_jsonl_patches

    base = tmp_path / "base.jsonl"
    patches = tmp_path / "patches.jsonl"
    jsonl_append(base, {"id": "a", "v": 0})
    jsonl_append(patches, {"id": "a", "v": 1})
    jsonl_append(patches, {"id": "a", "v": 2})

    import logging

    with caplog.at_level(logging.WARNING):
        stats = merge_jsonl_patches(base, patches, "id")
    assert stats["duplicate_patches"] == 1
    merged = list(jsonl_read(base))
    assert merged[0]["v"] == 2
