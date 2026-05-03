# judge-from-scratch — Project Status

> **You are here:** state of the build. For concepts, see [`fine-tuning-primer.md`](fine-tuning-primer.md). For staged build prompts, see [`claude-code-prompts.md`](claude-code-prompts.md). For an entry-point overview, see the [README](../README.md).

Living document. Update after each stage completion or significant decision.

Last updated: end of Stage 2 (pair construction); pivot to educational tutorial framing; renamed from REVAL Judge to judge-from-scratch.

---

## What this project is

**`judge-from-scratch`** is an end-to-end educational tutorial that builds a specialized social-bias evaluation judge by fine-tuning **Gemma 4 E4B**. The artifact published to Hugging Face is `gemma4-social-bias-judge`. The repo and its docs walk readers through the entire process: data generation, labeling, SFT, DPO, eval, deployment.

**The judge's task:** given a question and two model responses, output reasoning + a verdict on which response shows more social bias (A / B / TIE) plus a confidence score.

**Methodology in one sentence:** Distillation through synthetic data — Claude Opus 4.7 as labeler, BBQ-derived pairs as training corpus, QLoRA + SFT + DPO as the training recipe.

**Audience:** readers who know Python and basic ML, learning fine-tuning from scratch. The conceptual primer covers gradient descent, LoRA, QLoRA, SFT, and DPO with the assumption that readers are comfortable with the math but haven't fine-tuned a transformer before.

**Multi-phase plan:**
- **v1** — ship a working judge with full eval (held-out BBQ category as OOD slice), end-to-end Stages 0-9, plus Stage 10 deployment recipes.
- **v2b** — add an UnQover-derived OOD slice for a stronger "different dataset entirely" generalization claim. Improves the eval rigor of the same judge.
- **v2a** — autoresearch-style automated dataset iteration loop (see primer Appendix D). Optimizes the training data through agent-driven experiments.
- **Stage 11** — the tutorial layer. Notebooks, walkthroughs, hosted demo. Done as a refactoring pass *after* v1 ships and works end-to-end. Building tutorial-quality content while figuring out what works produces neither.

REVAL — the factual-deference and rhetorical-parity evaluation project — is a separate future project, not a milestone of this repo.

---

## Pipeline status

| Stage | Description | Status |
|---|---|---|
| 0 | Repo bootstrap | ✅ Done |
| 1 | BBQ sampling + candidate generation | ✅ Done |
| 1.5 | Enrichment (bias classification) | ✅ Done |
| 2 | Pair construction | ✅ Done |
| 3a | Eval set holdout (BBQ in-dist + religion held-out OOD) | ⏳ Running |
| 3b | Hand-labeling tool + 300-pair manual labeling | ⏳ |
| 4 | Claude labeling (primary + cross-check) | ⏳ |
| 5 | SFT/DPO dataset formatting (custom tags, no thinking mode) | ⏳ |
| 6 | SFT training (Gemma 4 E4B QLoRA) | ⏳ |
| 7 | DPO training | ⏳ |
| 8 | Eval harness | ⏳ |
| 9 | Publishing (HF model + GGUF + dataset) | ⏳ |
| 10 | Deployment recipes (Ollama instructions + vLLM Dockerfile) | ⏳ |
| 11 | Tutorial layer (notebooks, walkthroughs, optional HF Space) | ⏳ Post-v1 |

---

## Current dataset shape

**BBQ sample:** 3,000 questions across 11 categories (including intersectional race_x_gender and race_x_SES). Stratified on context_condition (50/50 ambig/disambig) and question_polarity (50/50 neg/nonneg). Deterministic with seed=42.

**Stage 1 candidates:** 12,000 generations (3,000 questions × 4 generator models).

**Generator pool (via OpenRouter):**
- meta-llama/llama-3-8b-instruct
- meta-llama/llama-3.1-8b-instruct
- mistralai/mistral-7b-instruct-v0.1
- qwen/qwen-2.5-7b-instruct

**Stage 1.5 enrichment classification:**
- correct: ~78%
- biased (chose target_label, not also answer_label): ~9%
- incorrect_other: ~10%
- parse_failed: ~3%

**Stage 2 pair distribution:**

| Bucket | Count |
|---|---|
| Clear bias vs clean | 800 |
| Subtle bias vs clean | 550 |
| Tracked-bias vs alternate-bias | 220 (true supply ceiling) |
| Both-clean tie | 550 |
| Adversarial (length + confidence) | 250 |
| **Total** | **2,370 pairs** |

**Eval split (planned for Stage 3a, religion-only OOD holdout):** religion category (192 pairs in actual data, lowercase `bias_category` field) is held out from training entirely. From those, 60 pairs become the OOD eval slice via 28/12/9/6/5 stratification (mirroring the in-dist split's proportions). The remaining 132 pairs from religion are unused in v1 (preserves the holdout). 240 pairs stratified across the 10 trained-on categories form the in-distribution eval slice. Training pool: 2,370 − 192 − 240 = 1,938 pairs. After position-swap doubling: ~3,876 SFT rows. Below the primer's 5,000-row "comfort floor" but workable; eval will reveal whether the pool was too tight.

---

## Key methodological decisions (chronological)

1. **Switched provider from Together AI to OpenRouter** for Stage 1. Together's catalog deprecated multiple smaller models we needed; OpenRouter aggregates across providers and is more stable.

2. **Kept generator pool deliberately small (7-8B models).** Larger/more-RLHFed models hedge too much on ambiguous BBQ questions and produce too few biased candidates to construct training pairs. Stage 1's role is *eliciting bias*, not *judging it*.

3. **Included intersectional BBQ categories** (race_x_gender, race_x_SES). Originally planned to skip these; including them is methodologically stronger and produces a better tutorial story.

4. **Switched ambig/disambig from 60/40 to 50/50.** Disambig cases test whether models can override stereotypes with context — arguably the more important judge capability.

5. **Stratified on question_polarity** (50/50 neg/nonneg). Different polarities test different bias patterns (attributing negatives vs withholding positives). Natural BBQ distribution isn't necessarily balanced.

6. **Discovered BBQ structural constraint.** "Bias-vs-bias same-question" is impossible because each BBQ row has a single target_label. Replaced with "tracked-bias vs alternate-bias" pairs (biased candidate vs incorrect_other candidate from same question). Documented in primer Appendix B.

7. **Audit caught classification leak.** Stage 2's pair construction was filtering on raw `chosen_idx == target_label` instead of the enriched `bias_classification` field. Fix: trust the classifier from Stage 1.5; don't re-derive bias from raw labels in downstream stages. Reduced biased pool from 1,665 to 526 (the correct number).

8. **Doubled Stage 1 input** from 1,500 to 3,000 BBQ questions after the classification fix shrank the dataset. Kept seed=42 to preserve determinism on the first 1,500.

9. **Bucket ordering matters.** Adversarial buckets allocate before clear/subtle (their loose superset) so strict subsets get their share before the looser categories consume the pool.

10. **Replaced CrowS-Pairs with held-out BBQ category for v1 OOD eval.** CrowS-Pairs tests "which sentence reflects a recognized stereotype" — a different task from "which model response inappropriately leaned on a stereotype to answer." A judge picking the more stereotype-shaped sentence isn't doing bias detection; some "biased" sentences in CrowS may also be factually correct. Held-out BBQ category is genuine OOD for the same task with zero new pipeline work. UnQover via allenai/unqover deferred to v2b for a stronger "different dataset entirely" OOD claim.

11. **Holdout = religion only (single category).** Considered religion + disability_status (two-axis OOD), but two-category holdout dropped the SFT training pool by ~28% from baseline and DPO-eligible cases nearly in half. Religion-only holdout is a ~19% drop, keeps DPO closer to primer targets, and still gives a defensible "judge never trained on religion bias, here's its κ on it" story. Disability_status remains in training and in the in-distribution eval slice.

12. **Switched fine-tuning target from Gemma 3 4B to Gemma 4 E4B.** Gemma 4 dropped late April 2026; Unsloth has day-zero support and specifically recommends E4B QLoRA for the small-model path. Gains: native system prompt support, longer context (128K), fresher base model. Costs: VRAM math differs (~6.87B raw params load even though it operates at 4B effective), chat template differs, two-week-old model with possible kinks. Mitigation: rigorous Stage 6 dry-run on ~50 rows before committing to full training. Considered Gemma 4 26B-A4B; deferred — MoE architecture, Unsloth specifically warns against QLoRA on it.

13. **Custom output tags, native thinking mode disabled.** Gemma 4 has a native thinking mode (`<|think|>` token in system prompt). Considered using it but decided against: would require remapping all of Claude's labeled `<reasoning>...</reasoning><verdict>...</verdict>` outputs into thinking-channel format, locks judge format to Gemma 4, hurts portability. Custom tags map cleanly from Claude labels and survive a base-model swap. **Critical consistency rule:** train and infer with the same configuration. No `<|think|>` token in system prompt anywhere — Stage 5 formatting, Stage 6 training, Stage 8 eval, Stage 9 published Modelfile, model card user instructions. The model card must explicitly tell users not to enable thinking mode at inference time.

14. **Repivoted from "REVAL Judge" project to `judge-from-scratch` educational tutorial repo.** Examining the existing REVAL tool prompts revealed they target political bias and argumentative parity — different tasks from what the BBQ-trained judge does. Rather than retrofitting, the BBQ judge ships as its own focused artifact (`gemma4-social-bias-judge`) inside an educational repo (`judge-from-scratch`) that walks through the full fine-tuning pipeline. REVAL itself is a separate future project. Tutorial framing is honest about scope (3,800 SFT rows is "tutorial-appropriate," not "production-underpowered") and reaches a wider audience.

15. **Three deployment paths, not one.** Stage 9 publishes the model artifacts to HF; Stage 10 adds two deployment recipes for readers: Ollama (local, zero cost, ~10-min setup) and vLLM (production-pattern, OpenAI-compatible API, Docker). A third path — a hosted HF Space with Gradio UI for "click here to try it" — is deferred to Stage 11 to keep v1 scope bounded and avoid ongoing hosting costs while the pipeline is still being validated.

---

## Open threads / known constraints

- **Tracked-bias vs alternate-bias is supply-bound at 220 pairs.** True ceiling, not a heuristic problem. Cannot expand without different question structure.

- **Stage 4 labeling not yet started.** $20-30 estimated cost for Claude Opus 4.7 with Batch API + prompt caching. Plus optional ~$3 cross-check via GPT-5.4 + DeepSeek V3.1 on 500 ambiguous pairs.

- **Stage 5b weaker labeling for DPO rejected** uses Qwen 2.5 7B via Together AI ($5 credit allocated).

- **Hand-labeling 300 eval pairs** is the next big time sink. 6-10 hours of careful manual work. Don't rush — this is the foundation of every reported metric.

- **Gemma 4 E4B is two weeks old at time of writing.** Unsloth's day-zero support is real but at least one community report describes OOM and broken-quantization issues during early Gemma 4 fine-tuning attempts. Mitigation: Stage 6 dry-run on ~50 rows must succeed before scaling to full training.

- **Stage 6+ VRAM expectations need re-baselining.** The primer's "8 GB fp16, 2 GB 4-bit" math was for plain Gemma 3 4B. Gemma 4 E4B has ~6.87B raw params; expect ~14 GB fp16, ~4 GB 4-bit. Unsloth says 12 GB+ VRAM works for QLoRA, free Colab T4 fits.

- **Training pool below primer's "comfort floor."** Primer suggests ~5,000 SFT rows as a saturation-curve sweet spot; we're projecting ~3,876. Trainable but may underperform. The 5k number was a rough rule of thumb, not a measured threshold for this specific task. Eval results will tell us whether the pool size was the binding constraint or whether data quality compensates.

- **Labeling prompt and judge system prompt are unwritten.** Stage 4 needs the labeling prompt; Stage 5 needs the judge system prompt. Both are user-authored artifacts. Drafting and iterating these (with manual calibration on 10 pairs) is on the critical path before Stage 4 dry-run can run.

- **HF Space (Tier 1 demo) deferred to Stage 11.** Reasons: ongoing hosting cost, would distract from getting the pipeline working end-to-end, Ollama covers most of the "try it" need without ongoing cost. Revisit after v1 ships.

---

## Files in the project

- `README.md` — entry point, three reading paths, repo overview
- `docs/fine-tuning-primer.md` — conceptual reference (Steps 1-9 + Appendices A-D)
- `docs/claude-code-prompts.md` — staged build prompts for Claude Code
- `docs/project-status.md` — this file

These are the load-bearing documents. The chat conversations that produced them are not — once these files exist, fresh chat sessions or Claude Code sessions can pick up cleanly with just these as context.

---

## What's next

### Immediate

Stage 3a is currently running. Completes the 300-pair eval set holdout (240 in-dist + 60 OOD religion).

While 3a runs:
- Draft `data/labeling_prompt.md` (Stage 4 input). Iterate manually on 10 pairs to calibrate Claude's labels against your own judgments.
- Draft `data/judge_system_prompt.md` (Stage 5 input). Should be much shorter than the labeling prompt — the trained model has learned the task; this just reminds it.
- Verify the exact Unsloth model ID for Gemma 4 E4B fine-tuning by reading their gemma-4 collection on HF.

### v1 build (Stages 3b-10)

After 3a finishes:
- 3b: hand-label 300 pairs over multiple sessions (~6-10 hours).
- 4: Claude labeling pipeline (background, 12-24h Batch API turnaround).
- 5-9: format → SFT → DPO → eval → publish.
- 10: deployment recipes (Ollama instructions in model card, vLLM Dockerfile).

Stages 3b and 4 can run in parallel — 3b is your time, 4 is wall-clock background time.

### v2 milestones (deferred until v1 ships)

**v2b — UnQover-derived OOD eval slice.** Strengthens the OOD claim from "held-out BBQ category" to "different dataset entirely." Process: clone allenai/unqover, generate ~80 underspecified questions in religion + nationality categories, run through existing generator pool (mini Stage 1), construct ~60 pairs (mini Stage 2), hand-label, integrate as a third κ group in the eval harness. ~4-6 hours of work, mostly hand-labeling.

**v2a — autoresearch-style automated dataset iteration.** Described in primer Appendix D. Frozen training+eval harness, editable `data_pipeline.py`, agent-driven optimization over dataset design choices. Reward signal: κ on the human-labeled eval set.

**Order: v1 → v2b → v2a.** Reasoning: v2a's autoresearch loop optimizes against the eval set as reward signal, so strengthening the eval set (v2b) before running the optimization (v2a) means the agent is optimizing against a more meaningful number.

### Stage 11 — tutorial layer (post-v1)

After v1 ships and the deployment recipes work end-to-end:
- Notebooks for each pipeline stage with working code and analysis
- Hosted HF Space with Gradio UI for "click here to try it" demo
- README polish, examples, troubleshooting docs

Deliberately deferred. The temptation to write tutorial content while figuring out what works is real and produces neither good code nor good tutorials. Build first, document second.

### Future projects (separate repos)

- **REVAL** — factual-deference and rhetorical-parity evaluation. Different task, different training corpus, different judge. The two judge prompts referenced earlier in the conversation (political bias, argumentative parity) become starting points for that project. Will reuse the pipeline patterns from `judge-from-scratch` but the data is entirely different.

- **Gemma 4 26B-A4B path** — larger base model, MoE architecture, 16-bit LoRA (not QLoRA per Unsloth guidance), ~40+ GB VRAM. Treat as a separate experiment, not "the same project but bigger."