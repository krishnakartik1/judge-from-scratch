# Claude Code Prompts — Staged Project Build

> **You are here:** the build manual for `judge-from-scratch`. For concepts, see [`fine-tuning-primer.md`](fine-tuning-primer.md). For current state, see [`project-status.md`](project-status.md). For an entry-point overview, see the [README](../README.md).

A sequenced set of prompts for building `judge-from-scratch` in Claude Code, one stage at a time.

## How to use this file

**Run one prompt per Claude Code session.** Don't chain them. After each stage:

1. Review what was produced. Read the code yourself.
2. Run it. Verify the artifact (JSONL file, model checkpoint, eval table) looks right.
3. Update the status checkboxes in `project-status.md`.
4. Then start the next stage.

The prompts assume `docs/fine-tuning-primer.md` and `project-status.md` are already in the repo. Each prompt explicitly tells Claude Code to read the primer first — don't skip this even if you think context is fresh.

The prompts are deliberately scoped narrow. "Build me the project" is a bad prompt; "build Stage 1 with these constraints" is a good one.

**Skip-ahead:** if you're already past some stages, jump to the relevant Stage section directly. Each stage prompt is self-contained.

## Project-wide invariants

**Model:** the fine-tuning target is **Gemma 4 E4B**. Use Unsloth's day-zero support. VRAM math, chat template, and quantization recommendations differ from older Gemmas.

**No native thinking mode.** The judge uses custom `<reasoning>...</reasoning><verdict>...</verdict>` tags. The system prompt does NOT contain `<|think|>` anywhere — not at training time, not at eval time, not in the published Modelfile, not in user-facing instructions. Train and infer with the same configuration.

**HF artifact names:**
- Model (fp16): `krishnak/gemma4-social-bias-judge`
- GGUF: `krishnak/gemma4-social-bias-judge-gguf`
- Dataset: `krishnak/gemma4-social-bias-judge-pairs`

---

## Stage 0: Repo bootstrap

Use this once, before everything else.

```
Set up the project skeleton for judge-from-scratch.

Read docs/fine-tuning-primer.md (especially Appendix B for the data
pipeline structure) and project-status.md before starting.

Create:
1. pyproject.toml using uv. Dependencies: unsloth (latest, must
   include Gemma 4 support), trl, peft, transformers (>=4.53.0 for
   Gemma 4), datasets, accelerate, bitsandbytes, openai, anthropic,
   together, python-dotenv, scikit-learn, pandas, tqdm.
   Python 3.11.
2. .env.example listing required keys: OPENROUTER_API_KEY,
   TOGETHER_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, HF_TOKEN.
   Comments next to each explaining what stage uses it:
   - OPENROUTER_API_KEY: Stage 1 candidate generation
   - TOGETHER_API_KEY: Stage 4 cross-check (DeepSeek V3.1) and
     Stage 5b weaker labeling (Qwen 2.5 7B for DPO rejected)
   - ANTHROPIC_API_KEY: Stage 4 primary labeling
   - OPENAI_API_KEY: Stage 4 cross-check (GPT-5.4)
   - HF_TOKEN: Stage 9 publishing
3. .gitignore covering .env, outputs/, *.jsonl in data/, model
   checkpoints, __pycache__, .venv.
4. Empty directory structure:
   data/{raw,pairs,labeled,formatted}/
   train/configs/
   outputs/
   eval/
   deployment/         # Stage 10 lives here
5. A scripts/common.py with:
   - load_env() helper
   - jsonl_read(path) and jsonl_append(path, record) helpers
   - already_processed(path, key_field) for resumable scripts

Do NOT install anything yet. Do NOT write any pipeline scripts. Just
the skeleton and helpers.

When done, summarize what you created and what's still empty.
```

---

## Stage 1: Candidate response generation

```
Build Stage 1 of the data pipeline: candidate response generation.

Read docs/fine-tuning-primer.md, especially Appendix B section
"The five-stage labeling pipeline" Stage 1. Also read project-status.md
for project conventions.

DESIGN INTENT (read before writing code):
The generator pool is deliberately small/less-RLHFed. BBQ candidate
generation needs a meaningful fraction of biased responses to
construct training pairs. Larger, more-aligned models hedge too much
on ambiguous questions, leaving nothing to pair against. Small models
in the 7-8B range fall back on training-data priors when context is
ambiguous — which is where stereotypes live. Stage 1's role is
*eliciting bias*, not *judging it*.

INPUT: BBQ sample at data/raw/bbq_sample.jsonl (3,000 records,
already produced by the sampling step).

PROVIDER: OpenRouter. Use the OpenAI-compatible API:
    base_url = "https://openrouter.ai/api/v1"
    api_key  = os.environ["OPENROUTER_API_KEY"]

GENERATION: One response per (question, model) pair from these 4
models:
- meta-llama/llama-3-8b-instruct
- meta-llama/llama-3.1-8b-instruct
- mistralai/mistral-7b-instruct-v0.1
- qwen/qwen-2.5-7b-instruct

Use temperature=0.7. Max 300 output tokens. Use a clear, neutral
prompt asking the model to read the BBQ context and answer briefly
(2-3 sentences). Format the prompt with all three answer choices
visible (A/B/C labels) so the model picks one explicitly.

OUTPUT: data/raw/candidates.jsonl. One record per (question, model):
{question_id, model, prompt, response, generation_params,
 chosen_letter (A/B/C parsed from response), chosen_idx (0/1/2)}

PREFLIGHT CHECK: As the very first thing the script does, hit each
model with a one-token "ping" request. If any returns 404 or auth
errors, abort with a clear message naming the failed model.

CONSTRAINTS:
- Resumable: skip (question_id, model) pairs already in output.
- Async with semaphore=20 concurrency.
- tqdm for progress.
- Log failures to data/raw/candidates_failed.jsonl, don't crash on
  one bad response.

After writing, do a 10-question dry run and show me:
1. The first 3 records produced.
2. The chosen_idx distribution across the 40 generations.
3. A rough count of how many candidates picked the stereotype-
   aligned answer (target_label match). If <10%, the pool is too
   aligned — stop and reconsider.

Don't run the full generation until I confirm the dry-run looks healthy.
```

---

## Stage 2: Pair construction with the pairing strategy

```
Build Stage 2 of the data pipeline: pair construction.

Read docs/fine-tuning-primer.md, especially Appendix B sections
"The pairing strategy" and "Generator diversity rules." Also read
project-status.md for the BBQ structural constraint discovery
(decision #6) — "tracked-bias vs alternate-bias" replaces the
originally-planned "bias vs bias" bucket.

Write data/02_construct_pairs.py:

INPUT: data/raw/candidates.jsonl from Stage 1 (with bias_classification
field added by Stage 1.5 enrichment).

PAIRING STRATEGY: Generate pairs targeting this distribution:
- Clear bias vs. clear neutral: 800 pairs
- Subtle bias vs. neutral: 550 pairs
- Tracked-bias vs. alternate-bias: up to 220 pairs (BBQ supply ceiling)
- Both-clean tie: 550 pairs
- Adversarial (length/confidence/style asymmetry): 250 pairs

Trust the bias_classification field from Stage 1.5; do NOT re-derive
bias status from raw chosen_idx vs target_label (this was the
classification leak caught in audit, decision #7).

Pair sampling rules:
- ALWAYS pair across different generator models for cross-model
  diversity, EXCEPT for "subtle vs. neutral" where same-model pairs
  are acceptable to control for style.
- For each pair, randomize which response goes in slot A vs. B.
- For adversarial pairs: deliberately construct length-asymmetric
  pairs (longer biased + shorter clean), confidence-asymmetric
  pairs, etc. Document the construction logic in code comments.

BUCKET ORDERING: Allocate adversarial buckets BEFORE clear/subtle
(decision #9). Strict subsets get their share before looser
categories consume the pool.

OUTPUT: data/pairs/pairs.jsonl. One record per pair:
{
  pair_id, question_id, question_text, bias_category,
  response_a: {model, text, suspected_bias_level},
  response_b: {model, text, suspected_bias_level},
  pair_category  # which of the 5 strategy buckets
}

After writing, run it and show me:
1. The actual distribution achieved.
2. 5 example pairs, one from each category.
3. Confirmation of cross-model diversity in the sample.
4. Distribution by bias_category — verify all 11 categories represented.

Don't proceed to Stage 3 until I review.
```

---

## Stage 3a: Hold out the human eval set

Reflects the religion-only OOD holdout decision.

```
Build the eval set holdout step.

Read docs/fine-tuning-primer.md Appendix B section "Stage 5: Hold
out a human-labeled eval set" and project-status.md decision #11
(religion-only holdout for OOD eval).

DESIGN INTENT:
For v1, the OOD eval slice comes from a held-out BBQ category, NOT
a different dataset. The judge is trained on 10 of the 11 BBQ
categories (religion held out entirely). Religion pairs in eval
test whether the judge generalizes its bias-detection skill to an
unseen bias category — same task, unseen content. v2b will add
UnQover-derived pairs for a stronger "different dataset entirely"
OOD claim, but that's deferred until v1 ships.

INPUT: data/pairs/pairs.jsonl from Stage 2.
The bias_category field is lowercase (e.g., "religion", not "Religion").

LOGIC:

1. Identify all pairs where bias_category == "religion". Call this
   set RELIGION_POOL. Use actual file counts at runtime, not estimates.

2. From RELIGION_POOL, sample 60 pairs for the OOD eval slice.
   Stratify across pair_category to mirror the in-dist proportions
   (28 clear / 12 subtle / 9 tracked_vs_alternate / 6 tie / 5 adversarial,
   derived from the in-dist 110/50/35/25/20 ratios with largest-
   residual rounding). Use seed=42.
   
   The remaining religion pairs (RELIGION_POOL minus 60) are unused
   in v1 — they do NOT go to training (would defeat the holdout)
   and they are not in eval. Log this clearly.
   
   Fail-loud check: before sampling, print the religion-bucket
   distribution and verify it can support the 28/12/9/6/5 split.
   If any cell is short, abort with a clear message rather than
   silently substituting from another bucket.

3. From the remaining pairs (all 10 non-religion bias_categories),
   sample 240 pairs for the in-distribution eval slice. Stratify
   by pair_category as follows:
     110 clear / 50 subtle / 35 tracked_vs_alternate / 25 tie / 20 adversarial

4. Mix: 240 in-dist + 60 OOD = 300 total holdout pairs.

OUTPUT:

- data/pairs/eval_set_unlabeled.jsonl — 300 pairs awaiting hand
  labels. Each record gets:
    * human_verdict: null (filled in by Stage 3b labeling tool)
    * confidence: null
    * notes: null
    * eval_slice: "in_dist" or "ood_religion"

- data/pairs/pairs_to_label.jsonl — the remaining pairs destined
  for Claude labeling in Stage 4. Religion pairs NOT selected for
  OOD eval are EXCLUDED from this file.

- data/pairs/pairs_unused_religion.jsonl — the religion pairs not
  selected for OOD eval. Saved for transparency.

DETERMINISM: Use seed=42 for all sampling.

After writing and running, confirm:
1. Counts: 300 unlabeled eval (240 in-dist + 60 OOD religion),
   pairs_to_label, unused religion. Total should equal the input
   pairs.jsonl count.
2. Category distribution in the eval set matches targets.
3. No religion pairs in pairs_to_label.jsonl.
4. No overlap: every pair_id in eval_set_unlabeled is absent from
   pairs_to_label and pairs_unused_religion.
5. Show the first 3 records of each output file, including one
   in-dist and one OOD religion eval record.
```

---

## Stage 3b: Hand-labeling tool

For *me* to use, not Claude Code. But Claude Code can build the tool.

```
Build a small CLI tool to help me hand-label the 300 eval pairs.

Read docs/fine-tuning-primer.md Appendix C for what the labels mean.

Write eval/label_tool.py:

A simple terminal-based labeling interface. For each unlabeled pair
in data/pairs/eval_set_unlabeled.jsonl:
1. Display the question, response A, response B clearly formatted.
   Show eval_slice ("in_dist" or "ood_religion") prominently.
2. Prompt me for: verdict (a/b/t for tie), confidence (1-5), and
   optional notes.
3. Save my label inline to the same file (mutating
   eval_set_unlabeled.jsonl in place by adding human_verdict,
   confidence, notes fields).
4. Allow quitting and resuming — skip pairs that already have
   human_verdict set.

Add a --random-order flag that shuffles the order of pairs.

Add a --review flag that re-displays already-labeled pairs and lets
me edit the verdict.

Add a --slice flag (in_dist | ood_religion) that filters which
slice of pairs to label. Useful for batching by slice across
multiple sessions.

Estimate: ~6-10 hours of careful work over multiple sessions.
Stopping and resuming should be easy.
```

---

## Stage 4: Claude labeling pipeline

```
Build Stage 4 of the data pipeline: Claude labeling.

Read docs/fine-tuning-primer.md, especially Appendix B sections on
Stage 3 labeling and "Hard cases and tiered labeling." Also read
docs/project-status.md decisions #16 (answer_choices schema) and
#17 (Sonnet 4.6 primary labeler).

NOTE: I will write the labeling prompt myself. Look for it at
data/labeling_prompt.md. Use it as-is. Do not generate or modify
it. The prompt MUST render the answer_choices field from each pair
record into its question framing — verify this is the case before
the dry-run. If labeling_prompt.md does not reference
answer_choices, abort and tell me to fix it first.

Write data/04_label_pairs.py:

INPUT: data/pairs/pairs_to_label.jsonl. Each pair record now has
an answer_choices field (per decision #16). The labeling code must
pass answer_choices into the prompt template alongside
question_text, response_a, and response_b.

PRIMARY LABELING:
- Anthropic API with model claude-sonnet-4-6 (verified API model ID).
- Batch API for 50% cost reduction.
- Prompt caching: system prompt and labeling instructions cached;
  only per-pair content is fresh.
- Output schema enforced: JSON with reasoning, verdict (A/B/TIE),
  confidence (1-5).
- Resumable: skip pairs already labeled.

CROSS-CHECK LABELING:
- After primary labeling, identify 500 pairs from the subtle/tie/
  tracked-vs-alternate/adversarial categories. Stratify across
  these four buckets proportionally so no single category dominates.
- Run those 500 through OpenAI API (gpt-5.4 — verify exact model
  name with `web_search` before running).
- ALSO run those same 500 through DeepSeek V3.1 via Together AI.
  Use Together's Batch API for 50% discount.
- Flag pairs where any of the three labelers disagree.
- Cost estimate: GPT-5.4 ~$3, DeepSeek V3.1 ~$1.

OUTPUT: data/labeled/labeled_pairs.jsonl. One record per pair:
{
  pair_id, ...original fields including answer_choices...,
  sonnet_verdict, sonnet_reasoning, sonnet_confidence,
  gpt_verdict (null if not cross-checked), gpt_reasoning,
  deepseek_verdict (null if not cross-checked), deepseek_reasoning,
  disagreement (bool, true if any pair of labelers disagreed
                on cross-checked rows)
}

Use field names sonnet_verdict / sonnet_reasoning / sonnet_confidence
(not claude_*) so the schema is unambiguous about which model
produced the primary label. Stage 5 formatting code will reference
these field names.

DRY RUN — COMPARISON RUN:

Before the full labeling run, validate that Sonnet 4.6 holds up
against Opus 4.6 on this task. The cost saving (~$25 vs ~$8) is only
worth taking if Sonnet's labels on the harder buckets are close to
Opus's.

1. Sample 50 pairs stratified to overweight hard cases:
   - 30 from {subtle, tracked-vs-alternate, adversarial} (10 each)
   - 20 from {clear, tie} (10 each)
   Use seed=42.

2. Run all 50 through BOTH:
   - claude-sonnet-4-6 (Batch API, prompt caching enabled)
   - claude-opus-4-6 (Batch API, prompt caching enabled)

3. Compute and show me:
   - Overall agreement rate (Sonnet verdict == Opus verdict).
   - Agreement rate broken down by pair_category.
   - Confidence distribution side-by-side per model.
   - Total cost actually charged for each model (from response
     headers).
   - The first 5 labeled pairs from each model side-by-side, so I
     can read the reasoning and judge quality directly.
   - A list of every pair where Sonnet and Opus disagreed,
     including both reasoning traces.

DECISION GATE:
- If overall agreement >= 90% AND hard-bucket agreement >= 75%:
  proceed with Sonnet 4.6 as primary on the full 1,938 pairs.
- If hard-bucket agreement falls below 70%: abort. Tell me, and
  I'll decide whether to switch back to Opus or revisit the
  labeling prompt.
- If 70-75%: stop and show me the disagreements. I'll judge.

Do not proceed past the dry run without my explicit confirmation.

BUDGET GUARDRAIL:
Hard-stop and prompt me if total spend across all labelers
(Sonnet + Opus dry-run + GPT-5.4 + DeepSeek) exceeds $20.

A note on the dry-run cost: 50 pairs × 2 models is roughly $1-2
total at these prices, so the validation is genuinely cheap insurance.

After the full primary labeling run completes, before kicking off
cross-check, show me:
- Total pairs labeled
- Confidence distribution
- Verdict distribution (A / B / TIE)
- Total spend so far

Wait for confirmation before launching cross-check.
```

---

## Stage 5: Format SFT and DPO datasets

```
Build Stage 5 of the data pipeline: format final training datasets.

Read docs/fine-tuning-primer.md Appendix B sections "The data row
structure: SFT vs DPO," "Stage 4a: Build the SFT dataset," and
"Stage 4b: Build the DPO dataset." Also read Step 5 sections
"Chat template" and "A specific design decision: custom tags vs.
native thinking mode."

Read docs/project-status.md decisions #16 (answer_choices schema),
#17 (Sonnet 4.6 primary, sonnet_* field names), #18 (cache_control
in system block), #19 (1 pair dropped, final count 1,937),
#20-21 (cross-check via Qwen 3 235B; deepseek_verdict field
contains Qwen output), and #22 (DPO sourcing: synthesis +
verdict-flip only, no cross-check supplement, due to judging-
rubric divergence between Sonnet and cross-checkers).

PRE-FLIGHT (do these before writing any code):
1. Verify the exact Unsloth Gemma 4 E4B model ID by searching
   their HF gemma-4 collection. The Stage 5 tokenizer must match
   the Stage 6 training tokenizer exactly.
2. Read data/labeled/labeled_pairs.jsonl head -3 to confirm field
   names: should be sonnet_verdict, sonnet_reasoning,
   sonnet_confidence (not claude_*). Abort if schema differs.
3. Confirm pair_id 96b558e0bf7cbd01 is NOT present (decision #19).
   Abort if it is.
4. Confirm data/judge_system_prompt.md exists. Abort if missing —
   Stage 5 cannot run without it.

CRITICAL CONSTRAINTS:
1. Use the Gemma 4 E4B tokenizer to apply the chat template via
   tokenizer.apply_chat_template(). Do NOT hand-roll the template.
2. The system prompt does NOT contain `<|think|>`. Native thinking
   mode is disabled. Custom tags only. Verify with a grep at the
   end.
3. Target format is `<reasoning>...</reasoning><verdict>X</verdict>`
   where X is A, B, or TIE. Match exactly — Stage 8 eval parses this.
4. The labeling prompt asked Sonnet for three tags including
   <confidence>N</confidence>. The trained judge should emit only
   <reasoning> and <verdict>. Stage 5 MUST strip <confidence> from
   sonnet_reasoning before writing targets. Verify with a grep at
   the end.

Write data/05_format_datasets.py:

INPUT: data/labeled/labeled_pairs.jsonl.

SFT FORMATTING:
- For each labeled pair, produce one row:
    {prompt: <full chat-template-wrapped prompt>,
     target: <reasoning>{cleaned_sonnet_reasoning}</reasoning><verdict>{sonnet_verdict}</verdict>}
- The prompt content uses the system role with text loaded from
  data/judge_system_prompt.md.
- The user role contains: question_text + answer_choices block
  (per decision #16, formatted as "A. ...\nB. ...\nC. ...") +
  Response A + Response B + the verdict question.
- Apply position-swap doubling: also produce the version with A
  and B swapped (and verdict flipped accordingly: A↔B, TIE→TIE).
- Filter out pairs where sonnet_confidence < 3.
- Save to data/formatted/sft.jsonl.

DPO FORMATTING (TWO-SOURCE — no cross-check supplement):

Filter to: sonnet_confidence >= 4 AND sonnet_verdict != "TIE".
Print pool size after filter. Should be ~1,058 unique pairs.
If < 800, stop and show me — pool is unexpectedly small.

Construct rejected response from one of two sources:
  * 70%: SYNTHESIZED FAILURE-MODE REJECTED.
    Call Claude (claude-sonnet-4-6, same as labeler for cost)
    separately to generate a "bad rationalization" of the WRONG
    verdict. Cache the synthesis prompt. The synthesis prompt
    should ask Claude to write reasoning that:
      - Lands on the wrong verdict (the opposite of sonnet_verdict)
      - Uses one of these failure modes (vary across rows):
        * Verbose hedging that obscures the actual answer
        * Surface-level engagement with the question without
          analyzing the reasoning chain
        * Stereotype-aligned reasoning presented confidently
        * Length asymmetry — overlong response that buries the
          flawed verdict
      - Stays in the <reasoning>...</reasoning><verdict>X</verdict>
        format
    Cost estimate: ~700 calls × ~300 output tokens ≈ $3-4 with
    caching on the synthesis prompt.

  * 30%: SIMPLE VERDICT-FLIP.
    Take sonnet_reasoning verbatim, flip the verdict letter
    (A↔B), keep the reasoning text unchanged. The result is a
    rejected response whose reasoning argues for one verdict
    but selects the other — a real failure mode worth
    discriminating against.

Apply position-swap doubling to all rows.

Save to data/formatted/dpo.jsonl with fields {prompt, chosen,
rejected}.

Both datasets follow TRL's expected format.

DRY-RUN PROTOCOL:
Before running synthesis on the full ~700-pair pool, do a
20-pair synthesis dry run. Show me:
- The first 5 synthesized rejecteds, full text.
- Cost actually charged for the 20 calls.
- Token-length distribution of synthesized rejecteds vs the
  corresponding chosens.

If synthesized rejecteds are systematically longer than chosens
by >20%, abort — verbosity bias risk in DPO. Adjust the synthesis
prompt to constrain length and re-test.

Wait for confirmation before running full synthesis.

AFTER FULL RUN, show me:
- Row counts: SFT (raw, post-confidence-filter, post-position-swap),
  DPO (raw, post-filter, post-position-swap).
- DPO source breakdown: how many rejecteds came from synthesis,
  how many from verdict-flip.
- 3 example rows from each (SFT, DPO), including the raw prompt
  string so I can verify chat-template wrapping.
- Confirmation: grep for `<|think|>` in every record (system,
  user, target, chosen, rejected). Must return zero hits.
- Confirmation: grep for `<confidence>` in any target/chosen/
  rejected. Must return zero hits.
- Length distribution: chosen vs rejected token counts (mean,
  median, p90). If chosen is >15% longer on average, flag and
  stop — verbosity bias risk.
- Length distribution: synthesis-rejecteds vs flip-rejecteds.
  These should be roughly comparable; large divergence means
  the synthesis prompt is producing oddly-shaped rejecteds.

Don't proceed to Stage 6 until I review.
```

---

## Stage 6: SFT training

The first GPU-spending stage. Targets Gemma 4 E4B.

```
Build the SFT training script.

Read docs/fine-tuning-primer.md Steps 2-5 (LoRA, QLoRA, Unsloth,
SFT) and "Hyperparameters that actually matter." Also read
project-status.md decisions #12 (Gemma 4 E4B) and #13 (thinking
mode disabled).

Before writing code, verify with `web_search`:
1. The exact Unsloth model ID for Gemma 4 E4B fine-tuning (check
   Unsloth's HF collection).
2. That Unsloth's current release supports Gemma 4 fine-tuning with
   FastLanguageModel and LoRA adapters.
3. Whether Gemma 4 E4B has any specific kwargs needed in
   FastLanguageModel.from_pretrained().

Write train/sft.py:

MODEL: the Gemma 4 E4B variant identified above, loaded with
FastLanguageModel, load_in_4bit=True. Apply LoRA with:
  r=16, lora_alpha=32, target_modules=all linear layers,
  lora_dropout=0.05, bias="none", random_state=3407

Memory expectation: ~4-5 GB for the 4-bit base + adapter overhead.
Should fit comfortably on a 12 GB+ GPU.

DATA: data/formatted/sft.jsonl. Use SFTTrainer from TRL.

CHAT TEMPLATE: rely on the tokenizer's apply_chat_template
mechanism. Do NOT hand-roll templates.

THINKING MODE: NOT enabled. The system prompt in the data does not
contain `<|think|>`. Verify this once at the start of the training
script — read the first SFT row, assert no `<|think|>` appears
anywhere. If it does, abort with a clear error message.

CONFIG (in train/configs/sft.yaml):
  learning_rate: 2e-4
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  warmup_ratio: 0.05
  lr_scheduler_type: cosine
  optim: adamw_8bit
  max_seq_length: 2048
  weight_decay: 0.01
  logging_steps: 10
  save_strategy: epoch

TRACKING: integrate Weights & Biases. Project name
"judge-from-scratch", run name from datetime + "sft".

OUTPUT: outputs/sft-checkpoints/, with the final adapter at
outputs/sft-final/.

DRY RUN PROTOCOL (mandatory — Gemma 4 was 2 weeks old at project
start and has reports of OOM and broken-quantization issues during
early fine-tuning attempts):

Before launching the full training run:
1. Run a 50-row, 1-epoch training to verify the pipeline works
   end-to-end. Log VRAM usage.
2. After the dry run, load the trained adapter and run inference
   on 5 held-out pairs (without thinking mode). Print the generated
   verdicts.

Don't run the full training yet. Show me the script, the config
file, the dry-run output, and the expected GPU memory + wall-clock
time, and wait for confirmation.
```

---

## Stage 7: DPO training

```
Build the DPO training script.

Read docs/fine-tuning-primer.md Step 7 (DPO) and Appendix C.

Write train/dpo.py:

STARTING POINT: load the SFT-trained adapter from outputs/sft-final/.
Continue training from there with DPO.

DATA: data/formatted/dpo.jsonl. Use DPOTrainer from TRL.

CHAT TEMPLATE: same as Stage 6.

THINKING MODE: NOT enabled. Same assertion check as Stage 6 at the
start of training.

CONFIG (train/configs/dpo.yaml):
  learning_rate: 5e-6
  num_train_epochs: 1
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  beta: 0.1
  max_length: 2048
  max_prompt_length: 1024
  warmup_ratio: 0.1
  lr_scheduler_type: cosine
  loss_type: sigmoid
  logging_steps: 5
  save_strategy: epoch

REFERENCE MODEL: rely on TRL's PEFT integration to use the SFT model
as reference by disabling adapters during the reference forward pass.
Don't load a separate copy of the model — that doubles VRAM.

TRACKING: same W&B project, run name with "dpo" suffix.

OUTPUT: outputs/dpo-final/ — final merged adapter ready for eval.

After training, also produce a merged-and-unloaded full-precision
checkpoint at outputs/merged-fp16/ for use with vLLM and GGUF
conversion.

DRY RUN: 50-row, 1-epoch DPO dry-run on top of the SFT adapter
before launching the full run.

Show the script, config, dry-run output, and resource estimate.
Don't run the full training yet.
```

---

## Stage 8: Eval harness

```
Build the evaluation harness.

Read docs/fine-tuning-primer.md Appendix C in full.

Write train/eval_harness.py:

INPUTS:
- A model checkpoint path (positional arg).
- The eval set at data/pairs/eval_set_unlabeled.jsonl (which now
  contains hand-labeled human_verdict fields after Stage 3b).

INFERENCE CONFIG (CRITICAL):
- temperature=0 for the headline metrics.
- System prompt does NOT include `<|think|>`. Run a startup
  assertion: load the data's system prompt, check no thinking
  token. Abort if one slips in.
- Use the same chat template as training.

METRICS:
1. Overall Cohen's κ vs. human verdicts (sklearn.metrics).
2. Per-category κ for each pair_category, computed separately for
   the in_dist slice and the ood_religion slice.
3. OOD κ — overall κ on just the ood_religion subset.
4. Position-bias rate. Run each pair twice with A and B swapped.
5. Verbosity-bias score. Average length-difference (chosen − rejected)
   in tokens, across non-TIE verdicts.
6. Self-consistency at temperature 0.3.

OUTPUT:
- A markdown table printed to stdout matching the format in the
  primer's Appendix C "The full eval suite" example.
- A JSON file at eval/results/{checkpoint_name}_{datetime}.json with
  all raw predictions and metrics.

BASELINE FLAG: support a --baseline flag that runs the same eval
on the base unmodified Gemma 4 E4B model. Run baseline inference
with the SAME no-thinking config — comparing fine-tuned-no-thinking
against base-with-thinking would muddy the numbers.

CACHING: store predictions in eval/cache/{checkpoint}_predictions.jsonl
so re-aggregating metrics doesn't require re-running inference.

Test on a small subset (50 pairs) first to verify metrics calculate
correctly. Show me the test output before running on the full 300.
```

---

## Stage 9: Publishing

```
Build the publishing pipeline.

Read docs/fine-tuning-primer.md Step 9 and project-status.md
decisions about thinking mode disabled (#13) and Gemma 4 E4B
target (#12).

Write three scripts under publish/:

1. publish/export_gguf.py:
   - Take outputs/merged-fp16/ as input.
   - Convert to GGUF using llama.cpp's converter. Verify
     llama.cpp is installed; if not, document the install steps.
   - For Gemma 4 small models, Unsloth recommends 8-bit GGUF as
     the Pareto starting point. Produce both Q8_0 and Q5_K_M.
   - Save to outputs/gguf/.

2. publish/build_modelfile.py:
   - Generate an Ollama Modelfile pointing to the Q8_0 GGUF file.
   - Set system prompt to the judge system prompt (loaded from
     data/judge_system_prompt.md).
   - CRITICAL: the system prompt must NOT contain `<|think|>`.
     Add a startup assertion that checks this before writing
     the Modelfile.
   - Set temperature 0, num_ctx 2048.
   - Output: outputs/Modelfile.

3. publish/upload_hf.py:
   - Upload three artifacts to Hugging Face:
     a. krishnak/gemma4-social-bias-judge — merged fp16 model
        with full model card.
     b. krishnak/gemma4-social-bias-judge-gguf — GGUF quantizations
        with Modelfile and Ollama instructions.
     c. krishnak/gemma4-social-bias-judge-pairs — SFT and DPO
        training datasets with a dataset card.
   - Use huggingface_hub. Read HF_TOKEN from .env.
   - The model card MUST include:
     * Tutorial framing: this model is the artifact produced by
       the judge-from-scratch tutorial. Link to the GitHub repo.
     * Training methodology summary
     * The eval results table from Stage 8
     * Intended use, limitations
     * Reproduction instructions
     * **Explicit warning** in a prominent section: "This model
       was fine-tuned with Gemma 4's native thinking mode DISABLED.
       Do NOT include `<|think|>` in the system prompt at inference
       time — doing so will produce degraded, untrained behavior."
     * Output format spec: `<reasoning>...</reasoning><verdict>A|B|TIE</verdict>`
     * A working Ollama one-liner so users can try it in 30 seconds:
       `ollama run hf.co/krishnak/gemma4-social-bias-judge-gguf:Q8_0`

For the model card, generate a draft from the eval results JSON
file. I will hand-edit it before uploading. Don't auto-publish —
leave the upload behind a --confirm flag.
```

---

## Stage 10: Deployment recipes

This is the educational-tutorial deliverable. Two deployment paths, written so readers can follow either one.

```
Build deployment recipes for judge-from-scratch.

Read docs/fine-tuning-primer.md Step 9 ("Putting the pipeline
together") and project-status.md decision #15 (three deployment
paths). Note: HF Space (Tier 1) is deferred to Stage 11; this
stage covers Ollama (Tier 2) and vLLM (Tier 3).

DESIGN INTENT:
This stage produces deployment artifacts that readers of the
tutorial can use to either (a) try the model locally in ~10
minutes via Ollama, or (b) stand up a production-pattern
OpenAI-compatible API server via vLLM. Both serve different
educational purposes:
- Ollama: showing how the GGUF artifact translates to a working
  local model with zero infrastructure.
- vLLM: showing the production deployment pattern, including
  Dockerfile, OpenAI-compatible serving, example clients.

Write the following under deployment/:

1. deployment/ollama/README.md:
   - Step-by-step instructions for pulling the published GGUF and
     running it locally via Ollama.
   - Working one-liner: `ollama run hf.co/krishnak/gemma4-social-bias-judge-gguf:Q8_0`
   - Example curl command and Python client (using openai SDK
     pointed at Ollama's OpenAI-compatible endpoint at
     http://localhost:11434/v1).
   - Example judge invocation: send a question and two responses,
     parse the `<reasoning>`/`<verdict>` output.
   - Troubleshooting: "if the model emits an empty thought block,
     check that you didn't include `<|think|>` in the system prompt."
   - System requirements: ~8 GB free disk, ~6 GB free RAM.

2. deployment/vllm/Dockerfile:
   - Base image: vllm/vllm-openai (latest tag).
   - Mount or download the merged fp16 model from HF.
   - ENTRYPOINT runs vLLM's OpenAI-compatible server with
     appropriate flags for Gemma 4 E4B (--model krishnak/gemma4-social-bias-judge,
     --max-model-len 2048, --dtype bfloat16, --port 8000).

3. deployment/vllm/docker-compose.yml:
   - Single-service compose file for local testing.
   - GPU passthrough configured.
   - Volume mount for HF cache so model isn't re-downloaded each restart.

4. deployment/vllm/README.md:
   - How to build and run the container locally.
   - Example client code in Python (openai SDK pointed at
     http://localhost:8000/v1) and curl.
   - One-paragraph note on what cloud hosting would cost (Modal,
     RunPod, HF Inference Endpoints) without prescribing a specific
     vendor. Approximate range only.
   - Troubleshooting: thinking-mode warning, OOM expectations on
     small GPUs.

5. deployment/example_client.py:
   - A standalone Python script that works against EITHER deployment.
   - Takes a backend flag (--backend ollama|vllm).
   - Sends a sample judge prompt, parses the output, prints verdict.
   - Demonstrates both happy path and error handling.

CRITICAL CONSTRAINTS:
- All system prompts in all deployment artifacts must NOT contain
  `<|think|>`. Add a comment block in each file noting why.
- The system prompt content must match data/judge_system_prompt.md
  exactly. If that file changes, these artifacts must be regenerated.
- Test the Ollama path locally before committing — it's the lowest-
  friction path for tutorial readers and must work reliably.

After writing, do a smoke test:
1. Build the Docker image (locally if possible, otherwise validate
   the Dockerfile syntactically).
2. Pull the GGUF via the Ollama one-liner and run a single test
   prompt end-to-end.
3. Run example_client.py against the local Ollama endpoint and
   confirm it parses output correctly.

Show me the smoke test results before considering this stage done.
```

---

## Stage 11: Tutorial layer (post-v1)

**Deliberately deferred.** Don't start this until v1 (Stages 0-10) is shipped and working end-to-end.

```
Build the tutorial layer for judge-from-scratch.

PREREQUISITES (HARD GATE):
- Stages 0-10 are complete.
- The model is published to Hugging Face.
- Both deployment paths (Ollama, vLLM) are tested and working.
- The eval table is populated with real numbers.

DO NOT start this stage until all of the above are true. The
temptation to write tutorial content while figuring out what works
is real and produces neither good code nor good tutorials. Build
first, document second.

Read docs/fine-tuning-primer.md (the entire file — this stage
turns it into the load-bearing pedagogical artifact) and
project-status.md (especially the Stage 11 description).

DESIGN INTENT:
The tutorial layer is a refactoring pass on top of the working
v1 pipeline. It does NOT add new functionality. It adds:
1. Notebook walkthroughs of each stage with working code, analysis,
   and explanations.
2. Polish to the README and docs based on what was actually
   load-bearing during the build.
3. A hosted Gradio demo on HF Spaces (the deferred Tier 1 from
   decision #15).

Write the following:

1. notebooks/01-data-exploration.ipynb:
   - Load BBQ, explore distributions, show what biased vs. unbiased
     candidate responses look like, plot the bias_classification
     distribution from Stage 1.5.

2. notebooks/02-pairing-strategy.ipynb:
   - Walk through how pairs are constructed. Show worked examples
     for each of the 5 buckets. Explain the BBQ structural
     constraint and the tracked-vs-alternate substitution.

3. notebooks/03-labeling-prompt-design.ipynb:
   - Show the iteration history of the labeling prompt. Run Claude
     on 10 calibration pairs with different prompt versions and
     compare against author labels. Demonstrate the value of
     iterative prompt design.

4. notebooks/04-sft-walkthrough.ipynb:
   - Annotated SFT training run. Show loss curves, gradient norms.
     Run inference at intermediate checkpoints and show how the
     model's behavior evolves.

5. notebooks/05-dpo-discrimination.ipynb:
   - The pedagogical centerpiece. Show pre-DPO vs. post-DPO logits
     for the same prompts. Demonstrate the confidence-gap effect
     described in primer Step 6 with concrete numbers.

6. notebooks/06-eval-methodology.ipynb:
   - Walk through computing Cohen's κ, position-bias rate,
     verbosity-bias score. Show what a "good" eval table looks
     like vs. a misleading one (raw accuracy alone, no robustness).

7. notebooks/07-deployment-comparison.ipynb:
   - Run the same prompt against both Ollama and vLLM backends.
     Compare latency, output consistency, and resource usage.

8. README polish:
   - Update with real eval numbers from the published model.
   - Add screenshots / output examples.
   - Clarify the three reading paths now that the doc structure
     is known.

9. HF Space (deferred Tier 1 demo):
   - Create a Gradio app at huggingface.co/spaces/krishnak/gemma4-social-bias-judge-demo.
   - Auto-sleep enabled to bound costs.
   - UI: two text boxes (Response A, Response B) plus a question
     field, a "Judge" button, and an output panel showing reasoning
     and verdict.
   - System prompt baked in (no `<|think|>` per project-wide rule).
   - Link from the README and the model card.
   - Document approximate monthly cost in the README so readers
     understand the tradeoff vs. the Ollama path.

10. troubleshooting.md (new doc):
    - Common issues encountered during the build, with solutions.
    - Gemma 4 OOM symptoms and fixes.
    - Chat template debugging.
    - "My eval κ is much lower than the table — what to check."
    - Thinking mode mistakes ("if your model produces empty leading
      tokens, you've enabled thinking somewhere").

The notebooks should be readable as standalone tutorials but also
designed to be run by readers with their own data. Include cell-by-
cell explanations, not just code.

After writing, do a self-review pass: pretend you're a reader who
just found this repo on GitHub. Can you go from "I know Python and
basic ML" to "I have a working judge model" by following the
README's first reading path?
```

---

## Workflow notes

**Don't skip the dry runs.** Every stage above has a "show me X before running on the full thing" step. These exist because the failure modes (wrong distribution in pairing, broken chat template in SFT, miscalculated κ in eval) are expensive to discover after the fact. Stage 6 dry-run is especially important — Gemma 4 was two weeks old at this project's start and may have rough edges in fine-tuning paths.

**Update `project-status.md` checkboxes after each stage.** This is your continuity-of-context across sessions.

**Commit after each stage to git.** One stage = one PR-shaped commit. If something breaks at stage 7, you can isolate the change.

**Resist scope creep within a stage.** If you find yourself thinking "while we're at it, let me also add X" — write it down for the next stage instead.

**Budget check at three points:** before Stage 4 (Claude labeling), before Stage 6 (first GPU spend), before Stage 9 (HF upload is free but irreversible).

**Thinking-mode discipline:** anywhere a system prompt is touched (Stages 5, 6, 7, 8, 9, 10), assert that `<|think|>` is absent. This is a project-wide invariant.

**Parallelism opportunities:** Stage 4 and Stage 3b can run concurrently — Stage 4 is wall-clock-bound on Anthropic's Batch API, Stage 3b is your hand-labeling time. The labeling prompt (Stage 4 input) and judge system prompt (Stage 5 input) can be drafted while earlier stages are running.

## After v1 ships

Once Stages 0-10 are done and the v1 model + deployment recipes are on Hugging Face:

- **v2b first**: build the UnQover OOD eval slice. See primer Appendix D's "V2b" subsection. ~1 day of work, mostly hand-labeling. Republish model card with a third κ column.

- **v2a second**: build the autoresearch-style automated dataset iteration loop. See primer Appendix D's main content.

- **Stage 11**: tutorial layer. See above.