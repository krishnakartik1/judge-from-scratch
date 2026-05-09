# judge-from-scratch — agent context

## If you're adapting this repo

This repo is a working end-to-end fine-tuning pipeline for a social-bias
judge. If your user wants to build a different judge — toxicity, factual-
deference, refusal-quality, anything pairwise — or any other specialist
LLM, the pipeline structure and staged-prompt pattern transfer directly.
Here's what to substitute and what to keep.

### Reading order before you write code

1. **`docs/story-data.md` § "Before you start"** — the prerequisites
   checklist (base model, task definition, training dataset, eval
   methodology, output-format consistency rule, deployment targets).
   Resolve all six with your user *before* writing code or prompting
   downstream stages. Changing any of them mid-pipeline means redoing
   everything downstream.
2. **`docs/fine-tuning-primer.md`** — conceptual foundation. Steps 1-9
   are model-agnostic (LoRA → QLoRA → SFT → DPO → eval). Appendices A-D
   are project-specific but the *patterns* generalize (B = data
   pipeline shape, C = eval methodology, D = post-v1 milestones).
3. **`docs/claude-code-prompts.md`** — the prompt playbook. Read the
   Stage 0 prompt to understand the skeleton, then adapt each
   subsequent stage prompt by swapping the project-specific values
   below. Keep the dryrun gates and the "show me X before running on
   the full thing" pattern — those are the load-bearing safety
   features.
4. **`docs/project-status.md`** — the decision log. Read the
   "Key methodological decisions" section to see which forks came up
   and how they were resolved (33 numbered decisions). Your project
   will hit similar forks: Sonnet vs Opus for labeling, dropout=0 vs
   0.05, max_tokens for eval, holdout shape, etc.
5. **`docs/how-this-was-built.md`** — the AI-assisted workflow loop
   (brainstorm in chat → codify in project docs → scoped prompt to
   Claude Code → review → commit → repeat). Use the same loop for
   the adapted project; it's task-agnostic.

### Swap these (project-specific)

- **Base model.** Currently `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit`
  (Gemma 4 E4B, 4-bit pre-quantized). Replace with your target.
  Cascading changes: VRAM math, chat template, quantization
  recommendations, Unsloth model ID, `train/configs/sft.yaml`'s
  `model_id`. Re-run the Stage 6 smoke test before scaling.
- **Dataset.** Currently BBQ (social bias benchmark). Replace with
  whatever fits your task. The 5-bucket pairing strategy (clear /
  subtle / tracked-vs-alternate / tie / adversarial) is generalizable
   — the buckets map to whatever your task's difficulty spectrum looks
  like. The "tracked-vs-alternate" pattern (both wrong, only one
  carries the bad signal) is task-agnostic too; figure out its
  analog for your task.
- **Output format.** Currently `<reasoning>2-5 sentences</reasoning>
  <verdict>A|B|TIE</verdict>`. Define your own. **Lock it before
  Stage 1.** Everything downstream — labeling prompts, SFT targets,
  eval parsing, deployment recipes — depends on this being stable.
- **Labeling prompt.** `data/labeling_prompt.md` is task-specific.
  Write your own; iterate on 10 calibration examples before scaling
  to the full dataset (the Sonnet-vs-Opus dry-run pattern, decision
  #17).
- **Judge system prompt.** `data/judge_system_prompt.md`. Same — task-
  specific, must match training format exactly.
- **HF repo names.** `krishnakartik/*` throughout. Global find-replace
  to your namespace + judge name. Touched in `publish/`, model cards,
  `deployment/ollama/README.md`, `space/app.py`.
- **Bias categories / task-specific labels.** The 11 BBQ categories
  and the religion holdout. Replace with whatever categories your
  task has. The holdout pattern (one category excluded from training
  for OOD eval) generalizes — pick a category whose absence doesn't
  cripple training but produces a defensible "judge never trained
  on X, here's its κ on it" claim.
- **Generator pool.** The 4 small models via OpenRouter (Llama 3 8B,
  Llama 3.1 8B, Mistral 7B, Qwen 2.5 7B) — chosen for their tendency
  to produce biased outputs (decision #2). Replace with models
  appropriate for *eliciting* the behavior your judge needs to
  detect. For toxicity: less-RLHFed base models. For factual
  deference: smaller models that hallucinate. The principle is
  "elicit, don't judge" at this stage.

### Keep these (structural / task-agnostic)

- **The staged-prompt pattern in `docs/claude-code-prompts.md`.**
  One prompt per stage, scoped narrow, dryrun before full run, one
  PR-shaped commit per stage. Works regardless of task.
- **The eval methodology** (Cohen's κ vs human labels + position-
  bias rate + self-consistency at T=0.3 + verbosity-bias score +
  parse failure rate). Applies to any pairwise judge. The
  truncation-bias finding (decision #31 — eval `max_tokens` must
  fit the trained model's distribution, not the baseline's) is
  general too.
- **The train/infer consistency rule.** Whatever format or mode
  decisions you make, they must hold across training, eval, and
  deployment. This project's `<|think|>` invariant is one instance
  of the general principle.
- **Position-swap doubling in SFT data.** Free 2× pool size and
  measurable position-bias reduction; generalizes to any pairwise
  judgment task.
- **The DPO synthesis pipeline** (decision #22-23): generate
  plausible-but-wrong rejected responses with named failure modes
  (verbose hedging / surface engagement / stereotype-aligned
  reasoning / length asymmetry) rather than using weaker-model
  outputs. The verbosity-bias check (chosen/rejected length ratio
  must stay under 1.15×) is task-agnostic too.
- **Project docs structure.** Primer (concepts) + prompts (build
  manual) + status (living state) + the four core context files
  (`README.md`, primer, prompts, status). An agent starting a new
  project should create these four files first; everything else
  layers on top.
- **Hyperparameters in YAML configs**, not hardcoded. See
  `train/configs/README.md` for the per-parameter justification
  pattern.
- **Resumable scripts** with `scripts/common.py` `already_processed()`
  checks. Cheap insurance against partial runs.
- **The cost-ledger pattern** (`record_cost`, `total_spend`,
  `BudgetExceededError`, `BUDGET_OVERRIDE=1` env override). Per-stage
  budget caps prevent runaway API spend. See `data/04_label_pairs.py`
  and `train/modal/_cost_ledger.py` for examples.

---

## What this project is

End-to-end educational tutorial that builds a specialized social-bias
evaluation judge by fine-tuning **Gemma 4 E4B** (decision #12 — switched
from Gemma 3 4B at project pivot). The artifact published to Hugging
Face is `gemma4-social-bias-judge`. Distillation through synthetic data:
Claude Sonnet 4.6 as primary labeler (decision #17), GPT-5.4 + Qwen 3
235B for cross-check triangulation, BBQ-derived pairs as the training
corpus, QLoRA + SFT + DPO as the training recipe.

For the narrative walkthrough of how this was built and the decisions
that shaped it, see [`docs/story-data.md`](docs/story-data.md) and
[`docs/story-training.md`](docs/story-training.md).

## The plan

The full project plan, including dataset design, evaluation methodology,
and v2 autoresearch enhancement, is in `docs/fine-tuning-primer.md`.
Living state-of-the-build is in `docs/project-status.md`. Staged build
prompts are in `docs/claude-code-prompts.md`. Read those before doing
substantive work.

## Stack

- **Unsloth** — fast LoRA/QLoRA on Gemma 4 E4B (day-zero support).
- **TRL** — `SFTTrainer`, `DPOTrainer`.
- **PEFT** — LoRA via Unsloth.
- **OpenRouter** — candidate generation (decision #1; Together's
  catalog was deprecating models we needed).
- **Anthropic API** — labeling, via Batch API for cost.
- **OpenAI + OpenRouter** — cross-check labelers (GPT-5.4, Qwen 3 235B).
- **Modal** — serverless GPU. Stage 6 SFT and Stage 7 DPO run on Modal
  A100-40GB ($3.86 + $1.34); Stage 8 eval and Stage 10 benchmarks run
  on Modal A100-80GB. Per-second billing, layered image cache. Original
  plan was a single Lambda Labs A100-80GB; migrated to Modal per
  decision #25.
- **vLLM** — bf16 inference for Stage 8 eval (decision #30 pivoted off
  Unsloth for the precision-confound reason) and the production
  serving recipe in `deployment/vllm/`. Requires nightly vLLM + cu129
  + transformers ≥ 5.5 for Gemma 4 (decision #30 details).
- **llama.cpp** — Stage 9 GGUF export (Q8_0 + Q5_K_M).

## Conventions

- All data files are JSONL, one record per line.
- Each pipeline stage is a numbered script that reads its predecessor's
  output and writes its own artifact. Stages are resumable (skip records
  already in output). Exception: Stage 3a is sentinel-based, not
  per-record resumable — sampling depends on the full pool, so the
  whole stage is rebuilt via `--force` rather than appended to.
- Hyperparameters are in `train/configs/*.yaml`, not hardcoded.
- Use `uv` for dependency management.
- All inference runs at temperature=0 unless explicitly noted.

## Project-wide invariants

These rules must hold *everywhere* — every stage, every system prompt,
every script, every deployment artifact. Violating any of them at a
single stage silently corrupts downstream results.

- **No `<|think|>` in any system prompt.** Train, eval, deploy, model
  card, Modelfile. Decision #13. The single source of truth is
  `eval/eval_harness.py:assert_no_thinking_in_prompt`. Every stage
  that touches a system prompt asserts this at startup.
- **Output format is `<reasoning>...</reasoning><verdict>{A|B|TIE}</verdict>`.**
  Custom tags, not native thinking-mode tags. The trained judge emits
  exactly this format; eval parses exactly this format; deployment
  expects exactly this format.
- **Temperature = 0 for all eval and production inference.** T=0.3
  is allowed *only* for the self-consistency robustness metric. The
  headline κ is computed at T=0 (decision #32).
- **Position-swap doubling on every SFT row.** Each labeled pair
  produces two SFT rows: one with the original A/B order, one with
  A and B swapped (and verdict flipped: A↔B, TIE→TIE). Halves
  position bias at zero data-generation cost.
- **Eval holdout is never leaked into training.** Stage 4 reads
  `data/pairs/pairs_to_label.jsonl`, NOT `data/pairs/pairs.jsonl`.
  The 300-pair holdout (240 in-dist + 60 OOD religion) is excluded
  from labeling and from all downstream training files.

If you're adapting this repo, define your own invariants list. These
are the rules that, if violated at any single stage, silently corrupt
downstream results — and the cost of catching them late (re-run
training, re-run eval, re-publish) is much higher than the cost of
asserting them at every stage boundary.

## Status

All v1 stages (0-11) complete. Living state in
[`docs/project-status.md`](docs/project-status.md).

- [x] Stage 0: repo bootstrap
- [x] Stage 1: candidate generation (`data/01_generate_candidates.py`)
- [x] Stage 1.5: enrichment (`data/01b_enrich_candidates.py`)
- [x] Stage 2: pair construction (`data/02_construct_pairs.py`)
- [x] Stage 3a: holdout eval set (`data/03a_holdout_eval.py`)
- [x] Stage 3b: hand-label 300 eval pairs (`eval/label_tool.py`)
- [x] Stage 4: Claude labeling (`data/04_label_pairs.py`)
- [x] Stage 5: dataset formatting (`data/05_format_datasets.py`)
- [x] Stage 6: SFT training (Gemma 4 E4B QLoRA on Modal A100-40GB)
- [x] Stage 7: DPO training (Modal A100-40GB)
- [x] Stage 8: Eval harness (vLLM/bf16; in-dist κ 0.48→0.65→0.68; OOD 0.54→0.70→0.64)
- [x] Stage 9: HuggingFace publish + GGUF export (4 repos under `krishnakartik/`)
- [x] Stage 10: deployment recipes (Ollama + vLLM + benchmarks)
- [x] Stage 11: tutorial layer (`docs/story-{data,training}.md`,
  `docs/stage-refs/stage{0..10}.md`, `docs/troubleshooting.md`,
  `docs/how-this-was-built.md`, `space/` Gradio recipe)

## Pipeline wiring

Stage 4 (Claude labeling) reads `data/pairs/pairs_to_label.jsonl`,
NOT `data/pairs/pairs.jsonl` — the holdout in Stage 3a is the canonical
gate. Bypassing it leaks the human-eval pairs into training.

## Out of scope for v1

- Autoresearch v2 loop (see Appendix D of the primer; only after v1 ships)
- Quantization beyond standard GGUF Q4_K_M / Q5_K_M
- KV cache compression (TurboQuant etc.)
