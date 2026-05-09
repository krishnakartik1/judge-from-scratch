# judge-from-scratch — Project Status

> **You are here:** state of the build. For concepts, see [`fine-tuning-primer.md`](fine-tuning-primer.md). For staged build prompts, see [`claude-code-prompts.md`](claude-code-prompts.md). For an entry-point overview, see the [README](../README.md).

Living document. Update after each stage completion or significant decision.

Last updated: end of Stage 7 (DPO training). Stages 3a, 4, 5, 6, and 7 complete. DPO adapter at `/vol/checkpoints/dpo-final/`, merged fp16 at `/vol/checkpoints/merged-fp16/`.

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
| 3a | Eval set holdout (BBQ in-dist + religion held-out OOD) | ✅ Done |
| 3b | Hand-labeling tool + 300-pair manual labeling | ⏳ |
| 4 | Claude labeling (primary + cross-check) | ✅ Done |
| 5 | SFT/DPO dataset formatting (custom tags, no thinking mode) | ✅ Done |
| 6 | SFT training (Gemma 4 E4B QLoRA on Modal) | ✅ Done |
| 7 | DPO training | ✅ Done |
| 8 | Eval harness | ⏳ |
| 9 | Publishing (HF model + GGUF + dataset) | ⏳ |
| 10 | Deployment recipes (Ollama instructions + vLLM Dockerfile) | ✅ Done |
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

14. **Pivoted to `judge-from-scratch` as a focused educational tutorial repo.** The BBQ-trained judge ships as its own focused artifact (`gemma4-social-bias-judge`) inside an educational repo (`judge-from-scratch`) that walks through the full fine-tuning pipeline. The earlier idea of folding it into a broader REVAL umbrella (factual-deference + rhetorical-parity evaluation) was dropped because those tools target different tasks; REVAL itself is a separate future project (see "Future projects" below). Tutorial framing is honest about scope (3,800 SFT rows is "tutorial-appropriate," not "production-underpowered") and reaches a wider audience.

15. **Three deployment paths, not one.** Stage 9 publishes the model artifacts to HF; Stage 10 adds two deployment recipes for readers: Ollama (local, zero cost, ~10-min setup) and vLLM (production-pattern, OpenAI-compatible API, Docker). A third path — a hosted HF Space with Gradio UI for "click here to try it" — is deferred to Stage 11 to keep v1 scope bounded and avoid ongoing hosting costs while the pipeline is still being validated.

16. **answer_choices field added to pair records.** Pair schema was missing BBQ answer choices, so downstream couldn't decode <answer>B</answer>. Fixed by joining on question_id; patched Stage 2, re-ran 2 + 3a, updated label_tool.py. Labeling prompt, judge system prompt, model card spec still need updating to render the field.

17. **Switched primary labeler from Opus 4.7 to Sonnet 4.6.** Cost drops ~$25 → ~$8; Sonnet 4.6 is frontier-class for this task. 50-pair Opus-vs-Sonnet dry run gates the swap (≥90% overall, ≥75% hard-bucket agreement). Cross-check on 500 pairs unchanged. Update Stage 9 model card and primer Appendix B framing.

18. **Cache_control moved from user-content to system block**. v1 placed cache_control on a 1,575-token user-message content block; produced cache_read=0 across batch. v3 moved the rubric into a system block (canonical batch-caching pattern). Revalidated on same 50 pairs: v3 49/50 vs Opus (v1: 47/50), zero verdict flips, 6 ±1 confidence shifts. Primary labels produced under v3. Caching still cache_read=0 — environmental issue, deferred.

19. **One pair dropped from primary labeling**. Pair 96b558e0bf7cbd01 failed parsing twice with the same pattern: Sonnet emitted <thinking> tags (native mode) instead of <reasoning>, hit max_tokens=1024 before producing the verdict block. Deterministic, not transient. Dropped rather than patched to keep the 1,937 labels under one consistent configuration. Final labeled count: 1,937. SFT pool: ~3,874 rows pre-confidence-filter.

20. **DeepSeek V3.1 disabled on Together account**. switched cross-check to Qwen 3 235B Instruct (Qwen/Qwen3-235B-A22B-Instruct-2507-tput). Different lineage (Alibaba), preserves three-labeler triangulation framing in model card.

21. **Cross-check complete** Sonnet primary + GPT-5.4 + Qwen 3 235B (via OpenRouter, not Together — Together's batch hung, OpenRouter completed 500 calls in 3 min for $0.07). 17.4% disagreement rate on hard buckets. Three-lineage triangulation preserved despite DeepSeek V3.1 swap (#20). Total Stage 4 spend: $14.34 of $20 cap.

22. **DPO sourcing** synthesis (60-75%) + verdict-flip (25-40%), no cross-check supplement. Hand-review of 9 'both-cross-checkers-disagreed' pairs revealed a judging-philosophy gap, not Sonnet errors: Sonnet weights letter answers; GPT/Qwen weight reasoning chains. Cross-checker disagreements thus signal rubric difference, not weaker-model mistakes. Trained judge will inherit Sonnet's letter-aware rubric; documented as a deliberate choice in the model card. Stage 3b hand-labeling must apply the same rubric to avoid eval κ deflation.

23. **Stage 5 shipped at 70/30 synth/flip mid-range.** Final SFT 3,844 rows (1,922 pairs × position-swap, 15 dropped at conf<3); DPO 2,200 rows (1,100 pool × swap; 1,558 synth + 642 flip). Total spend $2.05 / $15 cap (Sonnet 4.6 Batch API at 50% discount). Synth prompt was re-tuned mid-run from "~120 reasoning tokens" to "200-300 tokens (soft target)" — v1 produced rejecteds whose bottom decile sat entirely below chosen's bottom decile, opening a "shorter = rejected" verbosity-bias shortcut. v2 closed it: 0/779 synth records below chosen p10. Three synth rejecteds leaked the wrong-verdict instruction ("we need to say the verdict is B"); re-synthesized via the stricter retry suffix and merged under phase=`leak-retry`. Final verify gates: 0 `<|think|>` hits, 0 verdict-flip violations across all 2,200 DPO rows, chosen/rejected median ratio 0.90× (under 1.15× block threshold).

24. **Stage 6 SFT shipped on Modal.** 3,844 rows × 3 epochs (720 optimizer steps at effective batch 16) trained in 88 min wall-clock on a single A100-40GB. Final `train_loss` 0.889, `mean_token_accuracy` 86.1% (vs random ~10%), peak VRAM 23.4 GB, `entropy` collapsed from 3.07 → 0.475. Loss curve monotonic, no spikes. Total Stage 6 spend $3.86 / $30 cap including all dryrun debugging. Adapter published privately at `krishnakartik/gemma4-judge-sft-checkpoint` and held on Modal volume at `/vol/checkpoints/sft-final/` for Stage 7 consumption. 5/5 probe outputs on the religion-OOD holdout were format-clean; 1 was truncated mid-reasoning at the probe's `max_new_tokens=256` — bump to 384 for Stage 8 eval.

25. **Migrated to Modal for serverless GPU; original Lambda Labs A100-80GB plan dropped.** Modal A100-40GB is sufficient (23 GB peak), per-second billing eliminates idle cost, layered image cache makes iteration cheap (~$0.10/dryrun after first build). Pattern: private `@app.function` for remote work, `@app.local_entrypoint` with budget gating for invocation. Cost ledger at `train/.cost_ledger.jsonl` (gitignored, per-developer) mirrors the Stage 4/5 pattern with `record_cost`, `total_spend`, and `BudgetExceededError` plus interactive `CONTINUE` prompt or `BUDGET_OVERRIDE=1` env override. `try`/`finally` recording so failed runs still log spend.

26. **Stage 6 required multiple Unsloth/TRL/HF-datasets stack workarounds.** Each failure mode was found by reading library source after a Modal run:
    a. Unsloth source-injects code into `SFTTrainer.__init__` (`unsloth/models/rl.py:1195`) that overrides `dataset_num_proc=None` to ~21 on fork-style Linux. HF datasets at `arrow_dataset.py:3318` then spawns a multiprocess Pool for any `num_proc >= 1`, which fails to pickle the Unsloth-patched processing class (references `torch._dynamo.config.ConfigModuleInstance`). Fix: pre-tokenize CPU-side single-process via a custom `_pretokenize` helper, then pass `dataset_kwargs={"skip_prepare_dataset": True}` to `SFTConfig` so TRL respects the prepared rows.
    b. `unsloth.chat_templates.train_on_responses_only` auto-computes the same large `num_proc` and triggers the same pickle error. Fix: invoke with `return_function=True` to get just the masking callable, then apply via single-process `dataset.map`.
    c. TRL 0.24's `SFTTrainer.compute_loss` subscripts `outputs.logits.shape` for entropy logging (`sft_trainer.py:1105`), which fails when Unsloth's cut-cross-entropy optimization returns a lazy callable. Fix: `os.environ["UNSLOTH_RETURN_LOGITS"] = "1"` *before* model load.
    d. TRL 0.24 renamed `tokenizer=` → `processing_class=` and moved `max_seq_length` → `max_length` (now on `SFTConfig`).
    e. Gemma 4 ships a multimodal `Gemma4Processor` not a plain text tokenizer; `tokenizer("text", ...)` mis-routes positional arg. Fix: always call with `text=` kwarg.

27. **`lora_dropout = 0` (not the primer-implied 0.05).** Researched after Unsloth's runtime warning that `dropout > 0` disables a fast-patch path. PEFT's `LoraConfig` defaults to 0.0; Unsloth's official LoRA hyperparameters guide explicitly recommends 0 unless overfitting is observed; Sebastian Raschka's "Hundreds of Experiments" article fixed dropout at 0.05 across every run but never ablated it and excluded it from the "hyperparameters that matter" list; Lin et al. (ICLR 2025) "LoRA Dropout as a Sparsity Regularizer" labels standard dropout-on-LoRA an "unreliable regularizer" for short fine-tuning. Empirical confirmation on our 50-row dryrun: switching 0.05 → 0 gave **68% steps/sec speedup** (0.028 → 0.047) with identical loss (8.852 → 8.856; within noise). Other regularizers (r=16, 3 epochs, weight_decay=0.01, distillation labels) cover the small-dataset overfit risk. Documented in `train/configs/README.md`.

28. **HF namespace correction: `krishnakartik` (not `krishnak` or `krishnakartik1`).** Original specs across docs had `krishnak/...`; the actual HF account is `krishnakartik`. Distinct from email `krishnakartik1@gmail.com` and GitHub `github.com/krishnakartik1/judge-from-scratch` (which legitimately retain the `1` on the username — separate identifiers). 15 references updated across `README.md`, `docs/claude-code-prompts.md`, `train/configs/sft.yaml`, `train/configs/README.md`, `tests/test_sft_config.py`. Stage 9 publish targets, Stage 10 Ollama one-liner, Stage 11 Gradio Space all now point at `krishnakartik/`.

29. **Stage 7 DPO shipped on Modal.** 2,200 rows × 1 epoch × effective batch 16 → 138 optimizer steps. Final `train_loss` 0.010, `rewards/accuracies` 1.00, `rewards/margins` oscillating 80–125 throughout, peak VRAM 27.8 GB, training wall-clock 34 min, total wall-clock 43 min (incl. probe), cost $1.34. DPO adapter at `/vol/checkpoints/dpo-final/`; merged fp16 at `/vol/checkpoints/merged-fp16/`. Cumulative spend $5.60 of $26.37 Stage 7 cap. Six surgical patches applied to dodge TRL 0.24 + transformers 5.5 + Gemma 4 multimodal compat issues (mergekit dep, llm_blender + weave module stubs, `warnings_issued` seed, `model_type` mask during DPOTrainer init, text-tokenizer pass-through). Merge decoupled from training as a standalone entrypoint — over-collapse verdict heuristic fired ABORT (false positive) on healthy training due to per-batch reward variance dominating the margin signal. 5/5 SFT-vs-DPO probe verdicts identical (probe set is sha1-deterministic, so it samples only easy pairs SFT already nailed).

30. **Stage 8 pivoted from Unsloth/4-bit to vLLM/bf16 inference.** The original Stage 8 ran inference via Unsloth's `model.generate()` with `load_in_4bit=True`. Two killers forced the rewrite: (a) Modal billing-cap preemption killed long detached runs (only baseline cache landed; SFT and DPO never finished), and (b) precision confound — comparing 4-bit baseline κ against bf16 SFT/DPO κ conflates training delta with precision delta, disqualifying for a project whose pedagogical point is rigorous eval. Fix: all three columns through vLLM in bf16 — same backend, same precision. Required nightly vllm + cu129 wheels + transformers ≥ 5.5.0 (Gemma 4 E4B uses `Gemma4ForConditionalGeneration` which stable vllm 0.8.x rejects on rope_scaling, and vllm 0.19 stable pins transformers ≤ 4.57.6 which doesn't recognize the gemma4 model class). Image base is `nvidia/cuda:12.9.0-devel-ubuntu22.04` so flashinfer's JIT-compiled sampling kernels can find `nvcc`; debian_slim is too thin. SFT-merged-fp16 produced via fresh `eval/modal/merge_adapter.py` (Stage 6 only saved adapter); DPO merged-fp16 reused as-is from Stage 7. All prompt rendering moved Modal-side because local transformers (4.57.x via the unsloth lockfile pin) can't load Gemma 4's `chat_template.jinja`; the on-container transformers-5.5+ stack inside `VllmRunner` does the rendering. Architecture: one `@app.cls VllmRunner` parameterized by `model_path` (Modal partitions container pool by parameter value), `scaledown_window=600` to keep the warm container alive across the 7 sequential `render_and_infer.remote()` calls per checkpoint (1 original + 1 swapped + 5 consistency at distinct seeds 0..4 for genuine self-consistency signal). 2100 rows/model written via a `write_cache` `@modal.method` that holds the volume mount. Legacy `eval/modal/run_eval.py` frozen as a reference; legacy 4-bit cache moved to `/vol/eval/cache/legacy_unsloth/` for the record but excluded from the final table. Budget gate scoped to `function.startswith("vllm_infer")` rows so prior failed legacy attempts (~$20 of `function="run_eval"` and `run_one[*]` entries) don't eat the new $10 STAGE8_BUDGET_CAP. SFT-merged-from-4bit caveat: the LoRA was trained against a 4-bit forward pass but the merge loads a non-quant base in bf16 — the merged bf16 weights inherit the quantization-aware training artifact, but vLLM sees a clean bf16 checkpoint at inference time. Legacy `eval_harness.SELF_CONSISTENCY_RUNS = 1` still applies to `run_eval.py`; the vLLM path uses 5 runs directly via `len(consistency_preds) == 5`, which routes to `aggregate_metrics`'s `≥2`-runs branch.

31. **Stage 8 `max_tokens` bumped 384 → 1024 for SFT/DPO inference.** First vLLM run-all under the post-pivot 384-token budget (sized off the Stage 6 dryrun per decision #24) hit 10.1% parse failures on SFT and 9.9% on DPO; baseline was 0%. Diagnosis: SFT-trained reasoning is materially longer than baseline's, and the truncated outputs always cut off mid-sentence before the closing `</reasoning><verdict>` tags. The truncation was systematically biased toward the harder cases — pairs where the model wrote longer reasoning are the ones where it was struggling most (subtle bias, tracked-vs-alternate, OOD religion). Reporting the parse-fail-excluded κ as the headline would have inflated the numbers by 0.06–0.18 depending on bucket: in-dist κ went from preview 0.731 → headline 0.647 for SFT (Δ=−0.084), DPO OOD κ went from 0.820 → 0.643 (Δ=−0.177). 1024 tokens leaves prompt-token ceiling at `4096 − 1024 − 32 = 3040`, well above the smoke test's max_prompt_tokens of 732. Re-run achieved 0.2–0.3% parse rate (5 SFT, 7 DPO out of 2100 each — these are the few cases where reasoning genuinely exceeds 1024 tokens). Both preview and headline result files retained under `eval/results/` for the methodology record. **Methodological lesson: token budget for eval must accommodate the trained model's output distribution, not just the baseline's** — the failure mode here is silent and selection-biased.

32. **Stage 8 evaluation complete.** Baseline / SFT / DPO evaluated on the 300-pair holdout (240 in-dist + 60 OOD religion). Headline numbers (in `eval/results/stage8_final_20260508T121153.md`): in-dist κ 0.481 → 0.647 → 0.682 (SFT lands at the 0.68 realistic target; DPO matches it). OOD religion κ 0.542 → 0.695 → 0.643 — DPO regresses on OOD by 0.052 vs SFT. Position-bias rate 21% → 8–9% in-dist (clean win). Self-consistency at T=0.3: 74% → 83%. Subtle-bias κ 0.63 → 0.74 → **0.89** (DPO's clearest win). Tie-cases κ −0.06 (SFT) → 0.36 (DPO) — SFT genuinely worse than chance on ties; DPO recovers. Tracked-vs-alternate stays at the supply-bound floor (0.20 SFT → 0.12 DPO; only 220 training pairs). Stretch κ target (0.76 in-dist) not reached. The DPO-OOD-regression is a real finding, not a hedge: synth-hard-negatives encoded patterns specific to the 10 in-dist categories rather than bias-in-general. SFT-only checkpoint should be published alongside DPO as the recommended artifact for OOD use cases. Total Stage 8 vLLM-pivot spend: ~$3 of the $10 STAGE8_BUDGET_CAP_USD (one full run-all + SFT/DPO re-run at max_tokens=1024).
---

## Open threads / known constraints

- **Tracked-bias vs alternate-bias is supply-bound at 220 pairs.** True ceiling, not a heuristic problem. Cannot expand without different question structure.

- ~~**Stage 4 labeling not yet started.**~~ Resolved: Stage 4 complete via Sonnet 4.6 primary + GPT-5.4 + Qwen 3 235B cross-check. Total spend $14.34 (decision #21).

- ~~**Stage 5b weaker labeling for DPO rejected** uses Qwen 2.5 7B via Together AI.~~ Resolved: dropped per decision #22 in favor of Sonnet-synthesized hard negatives (cross-checker disagreements signal rubric divergence, not weaker-model mistakes). See decisions #22-23.

- **Hand-labeling 300 eval pairs** is the next big time sink. 6-10 hours of careful manual work. Don't rush — this is the foundation of every reported metric.

- **Gemma 4 E4B is two weeks old at time of writing.** Unsloth's day-zero support is real but at least one community report describes OOM and broken-quantization issues during early Gemma 4 fine-tuning attempts. Mitigation: Stage 6 dry-run on ~50 rows must succeed before scaling to full training.

- **Stage 6+ VRAM expectations need re-baselining.** The primer's "8 GB fp16, 2 GB 4-bit" math was for plain Gemma 3 4B. Gemma 4 E4B has ~6.87B raw params; expect ~14 GB fp16, ~4 GB 4-bit. Unsloth says 12 GB+ VRAM works for QLoRA, free Colab T4 fits.

- **Training pool below primer's "comfort floor."** Primer suggested ~5,000 SFT rows as a saturation-curve sweet spot; final SFT pool landed at 3,844 rows (1,922 unique pairs × position-swap). Trainable but may underperform. Eval results will tell us whether the pool size was the binding constraint or whether data quality compensates.

- ~~**Labeling prompt and judge system prompt are unwritten.**~~ Resolved: `data/labeling_prompt.md` authored for Stage 4 dryrun (decision #17 calibration); `data/judge_system_prompt.md` authored for Stage 5 (decision #23). Both shipped.

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

Stages 3a / 3b / 4 / 5 / 6 / 7 / 8 complete. Next on the critical path:

- **Stage 9** — HuggingFace publish + GGUF export. Two model artifacts (DPO as primary, SFT-only as a secondary "use this if your bias categories are OOD"), Q8_0 + Q5_K_M GGUFs for both, dataset card with cross-checker disagreement statistics. See `docs/claude-code-prompts.md` Stage 9 prompt and the eval findings in `eval/results/stage8_final_20260508T121153.md` for the headline numbers and the OOD regression caveat that the model card should surface prominently.

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