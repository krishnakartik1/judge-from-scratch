---
license: apache-2.0
language:
  - en
size_categories:
  - 1K<n<10K
task_categories:
  - text-classification
tags:
  - social-bias
  - judge-training
  - bbq
  - sft
  - dpo
  - llm-as-judge
---

# gemma4-social-bias-judge-pairs

Training and evaluation data for the [judge-from-scratch
project](https://github.com/krishnakartik1/judge-from-scratch), which
fine-tuned Gemma 4 E4B into a specialist social-bias judge
([primary model](https://huggingface.co/krishnakartik/gemma4-social-bias-judge),
[SFT-only secondary](https://huggingface.co/krishnakartik/gemma4-social-bias-judge-sft)).

This dataset contains:

- **`sft.jsonl`** (3,844 rows) — the SFT training set, in TRL
  prompt-completion shape. 1,938 base pairs × position-swap doubling
  to teach the judge to mirror verdicts under A/B order swap.
- **`dpo.jsonl`** (2,200 rows) — the DPO preference set in TRL
  prompt/chosen/rejected shape. 70% of rejecteds synthesized by
  Sonnet 4.6 as plausible-but-wrong reasoning; 30% verdict-flipped
  versions of the chosen.
- **`pairs.jsonl`** (2,370 rows) — the raw labeled-pair pool
  (pre-formatting). Each row is `{pair_id, question, answer_choices,
  response_a, response_b, pair_category, human_verdict | claude_verdict,
  confidence, notes, eval_slice}`.
- **`pairs_to_label.jsonl`** (1,938 rows) — the labeling-pipeline
  input set (pairs.jsonl minus the 300-pair eval holdout).
- **`eval_set_unlabeled.jsonl`** (300 rows) — the human-labeled
  holdout used for Stage 8 eval. Despite the file name, all 300
  rows have a `human_verdict` field; the suffix is a
  legacy-naming-not-worth-renaming choice flagged in the project
  status doc.

---

## Data provenance

### Generation (Stage 1 of the project)

Underspecified questions from the Bias Benchmark for QA
([Parrish et al., 2022](https://github.com/nyu-mll/BBQ)) were used
to elicit candidate responses from a deliberately small generator
pool (7–8B base models — large enough to be coherent, small enough
to elicit bias). The pool is documented in the project repo's Stage
1 script. Both negative-question (where bias produces the wrong
answer) and non-negative variants were used.

### Pair construction (Stage 2)

Candidates were paired across 5 categorical buckets:

- **`clear_bias_vs_clean`** — one response openly relies on a
  stereotype, the other answers from context. Easy bucket.
- **`subtle_bias_vs_clean`** — both responses sound plausible, only
  one carries an unstated stereotype assumption. Hardest bucket the
  judge has any leverage on.
- **`tracked_bias_vs_alternate`** — both responses pick a stereotyped
  answer but for *different* stereotypes. Tests whether the judge
  follows the specific bias the question was built around vs. an
  alternate.
- **`both_clean_tie`** — both responses answer from context. Verdict
  should be TIE.
- **`adversarial`** — short-and-clean vs long-and-biased; tests
  verbosity-bias robustness.

The 300-pair eval holdout is stratified across the 5 buckets and
across an in-distribution / OOD-religion split (240/60).

### Labeling (Stage 4)

**Primary labeler**: Anthropic Sonnet 4.6 via the Batch API.
**Cross-check**: GPT-5.4 (OpenAI) and Qwen 3 235B (Together AI).
Three labelers from three different model lineages give stronger
triangulation than two. Cross-check ran on 500 pairs drawn from the
hardest buckets (subtle, tracked-vs-alternate, both-clean-tie,
adversarial).

**Cross-checker disagreement rate**: ~17.4% on hard-bucket pairs.
This is *not* noise — it is signal that the bucket genuinely is
hard, that frontier labelers disagree about what counts as a
stereotype, and that DPO needs hard negatives drawn from the
disagreement set rather than easy buckets where Sonnet's verdict is
unambiguous. The ~17% rate is itself an interesting datum for the
field.

The eval holdout (300 pairs) was hand-labeled by the project author
across multiple sessions, with bucket-stratified confidence ratings
(1–5) and free-form notes. These labels are the ground truth for
all Stage 8 κ values reported in the model cards.

### Synth pipeline for DPO rejecteds (Stage 5b)

70% of DPO rows have a Sonnet-synthesized rejected response: same
question/answer-choices, plausible-but-wrong reasoning that lands on
a stereotype-driven verdict. Drawing rejecteds from synth (rather
than from a weaker model's actual outputs) was the project's main
methodology bet — see decision #22 in the project status doc.

---

## Schema

### `sft.jsonl`

```json
{
  "prompt": "<chat-template-rendered system+user>",
  "completion": "<reasoning>...</reasoning><verdict>A|B|TIE</verdict>",
  "pair_id": "...",
  "swap": true|false,
  "pair_category": "clear_bias_vs_clean|...|adversarial",
  "bias_category": "..."
}
```

### `dpo.jsonl`

```json
{
  "prompt": "<chat-template-rendered system+user>",
  "chosen": "<correct reasoning>...<verdict>X</verdict>",
  "rejected": "<plausible-but-wrong reasoning>...<verdict>Y</verdict>",
  "pair_id": "...",
  "rejected_source": "synth|verdict_flip",
  "pair_category": "..."
}
```

### `eval_set_unlabeled.jsonl` (despite the name, fully labeled)

```json
{
  "pair_id": "...",
  "question_text": "...",
  "answer_choices": [{"letter": "A", "text": "..."}, ...],
  "bias_category": "...",
  "response_a": {"model": "...", "text": "..."},
  "response_b": {"model": "...", "text": "..."},
  "pair_category": "...",
  "human_verdict": "A|B|TIE",
  "confidence": 1-5,
  "notes": "..."|null,
  "eval_slice": "in_dist|ood_religion"
}
```

---

## Splits

| File | Rows | Use |
|---|---|---|
| `sft.jsonl` | 3,844 | Stage 6 SFT training |
| `dpo.jsonl` | 2,200 | Stage 7 DPO training |
| `pairs.jsonl` | 2,370 | Pre-format full pool (labeled) |
| `pairs_to_label.jsonl` | 1,938 | Stage 4 labeling input |
| `eval_set_unlabeled.jsonl` | 300 | Stage 8 eval holdout |

---

## Limitations and warnings

- **English only**, BBQ-derived. The trained patterns will not
  transfer to other languages or non-BBQ-shape questions.
- **The 17% cross-checker disagreement rate** means even frontier
  labelers don't agree on what counts as a stereotype on subtle
  cases. Treat individual pair labels as approximate; the κ-against-
  human eval set (Stage 8) is the ground truth that matters.
- **Tracked-vs-alternate is supply-bound** at 220 pairs. There is
  no scalable way to expand this bucket without changing question
  structure.
- **OOD generalization claim is religion-only** in this dataset.
  The v2 roadmap (in the project repo) calls for extending the OOD
  slice with UnQover-derived data to broaden the OOD claim from
  "held-out BBQ category" to "different dataset entirely".

---

## License

Apache 2.0 for the dataset structure and the synthesized DPO
rejecteds. The underlying BBQ questions are CC-BY-4.0
([Parrish et al., 2022](https://github.com/nyu-mll/BBQ)).

## Citation

```
@misc{darsipudi2026gemma4judgepairs,
  author = {Krishna Kartik Darsipudi},
  title = {gemma4-social-bias-judge-pairs: Training and eval data for a Gemma 4 social-bias judge},
  year = {2026},
  howpublished = {\url{https://huggingface.co/datasets/krishnakartik/gemma4-social-bias-judge-pairs}},
}
```
