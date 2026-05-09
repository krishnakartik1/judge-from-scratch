# Building a judge from scratch — Part 2: training, eval, deployment

> Part 2 of 2. Stages 6-10: SFT, DPO, eval, publishing, and deployment. [Part 1](story-data.md) covered the data pipeline through formatting; if you haven't read it, the SFT and DPO datasets it produces are this part's input.

By the end of Part 1 we had two training files (3,844 SFT rows, 2,200 DPO rows), a 300-pair hand-labeled eval set, and a documented rubric. The remaining stages are the part where things either work or don't: train the model, evaluate it honestly, and publish what's worth publishing.

## Stage 6 — SFT training

Supervised fine-tuning (SFT) on the 3,844-row dataset, using QLoRA on Gemma 4 E4B. Training runs on a single Modal A100-40GB. Total spend: $3.86.

**The smoke test came first.** Modal + Unsloth + Gemma 4 + 4-bit was a fresh combination — Gemma 4 was two weeks old when this project started, and Unsloth's day-zero support hadn't been stress-tested for fine-tuning. Before writing any training code, `train/modal/smoke_test.py` boots the image, loads the 4-bit model, and asserts peak VRAM lands in the expected ~4-5 GB range. ~$0.30 to catch environment failures here; multiple hours to catch them mid-SFT.

**The full hyperparameter set is in [`train/configs/README.md`](../train/configs/README.md)** with a per-parameter justification table. The two non-obvious ones are:

- `lora.r = 16` and `lora.lora_alpha = 32` (effective scaling = 2.0). Primer says 16-32; for 4B-effective on 3.8K rows, 16 is enough capacity at low overfit risk.
- `lora.lora_dropout = 0.0` — *not* the 0.05 the primer suggested. This was the one decision worth pushing back on. Sebastian Raschka's "Hundreds of Experiments" article fixed dropout at 0.05 across every run but never ablated it. Unsloth's official guide recommends 0 unless overfitting is observed, because dropout > 0 disables a fast-patch path. Lin et al. (ICLR 2025) labels standard dropout-on-LoRA an "unreliable regularizer" for short fine-tuning. The 50-row dryrun confirmed it: 0.05 → 0 gave a **68% steps/sec speedup with identical loss** (8.852 vs 8.856; within noise). Decision #27 documents the swap. Other regularizers (r=16, 3 epochs, weight_decay=0.01, distillation labels) cover the small-dataset overfit risk.

**Five Unsloth/TRL stack workarounds** were caught by the dryrun, each found by reading library source after a Modal run failed:

1. Unsloth source-injects `dataset_num_proc=21` into TRL's `SFTTrainer.__init__`, which then fails to pickle the Unsloth-patched processing class. Fix: pre-tokenize CPU-side single-process and pass `dataset_kwargs={"skip_prepare_dataset": True}`.
2. `unsloth.chat_templates.train_on_responses_only` triggers the same pickle error. Fix: invoke with `return_function=True` and apply via single-process `dataset.map`.
3. TRL 0.24's `compute_loss` subscripts `outputs.logits.shape` for entropy logging, which fails when Unsloth's cut-cross-entropy returns a lazy callable. Fix: `os.environ["UNSLOTH_RETURN_LOGITS"] = "1"` before model load.
4. TRL 0.24 renamed `tokenizer=` → `processing_class=` and moved `max_seq_length` → `max_length` (now on `SFTConfig`).
5. Gemma 4 ships a multimodal `Gemma4Processor`, not a plain text tokenizer. `tokenizer("text", ...)` mis-routes the positional arg. Fix: always call with `text=` kwarg.

Each one is the kind of failure that costs $0.30 on a dryrun but a couple hours of debugging if you skip the dryrun.

**Run results:** 720 optimizer steps at effective batch 16, 88 minutes wall-clock, peak VRAM 23.4 GB, final `train_loss` 0.889, final `mean_token_accuracy` 86.1% (vs random ~10%), `entropy` collapsed from 3.07 → 0.475. The loss curve was monotonic, no spikes. Adapter saved to `/vol/checkpoints/sft-final/` and pushed privately to HF as insurance against Modal volume issues.

**What we learned.** When the canonical recipe (primer says 0.05 dropout) conflicts with the runtime warning (Unsloth says 0 is faster) and recent research (ICLR 2025), the recipe loses. Run the ablation; trust the data. Also: stack-incompatibility bugs in fast-moving training stacks are most cheaply caught by a 50-row dryrun, not by reading docs.

→ Reference: [`docs/stage-refs/stage6.md`](stage-refs/stage6.md). Prompt: [`claude-code-prompts.md` § Stage 6](claude-code-prompts.md#stage-6-sft-training).

## Stage 7 — DPO training

DPO continues on top of the SFT adapter, using the 2,200-row DPO dataset. The reference model is the SFT model with adapters disabled (TRL's PEFT integration handles this — no separate model copy, no doubled VRAM). Total spend: $1.34.

DPO *feels* different from SFT in three ways:

1. **Shorter.** 138 optimizer steps for 1 epoch on 2,200 rows. SFT was 720. DPO converges in roughly half an hour of training time.
2. **More LR-sensitive.** SFT runs at `lr=2e-4`; DPO at `5e-6` — almost two orders of magnitude lower. DPO's loss is a log-sigmoid of reward differences, and reward magnitudes are large. A higher LR collapses the policy.
3. **Different metrics.** SFT reports token loss and accuracy. DPO reports `rewards/chosen`, `rewards/rejected`, `rewards/margins`, `rewards/accuracies`, `logps/chosen`, `logps/rejected`. The thing to watch is `margins` (should grow) and the *gap* between `logps/chosen` and `logps/rejected` (should diverge). If both `logps` decrease together, that's the over-collapse failure mode — abort.

The hyperparameters are routine: β=0.1, effective batch=16 (matches SFT), `loss_type=sigmoid`, 1 epoch.

Six surgical patches were needed for TRL 0.24 + transformers 5.5 + Gemma 4 multimodal compat — `mergekit` dependency, `llm_blender` and `weave` module stubs, `warnings_issued` seed, `model_type` mask during `DPOTrainer.__init__`, text-tokenizer pass-through. Detail in decision #29; the lesson is that bleeding-edge model + bleeding-edge trainer = expect six patches.

**Run results:** 34 minutes training, 43 minutes total (incl. probe), final `train_loss` 0.010, final `rewards/accuracies` 1.00, `rewards/margins` oscillating 80-125 throughout. Peak VRAM 27.8 GB.

The merge step deserves a note. The original plan ran a "verdict-collapse" heuristic that aborted the merge if probe outputs looked degraded. It fired ABORT on a healthy training run because per-batch reward variance dominates the margin signal at this dataset size. The fix was to decouple merge from training as a standalone entrypoint and trust the loss curve plus `accuracies=1.0` over the heuristic.

The probe set is sha1-deterministic. 5/5 SFT-vs-DPO probe verdicts came out identical because the deterministic sample lands on easy pairs SFT had already nailed. The interesting differences only show up on the 300-pair Stage 8 holdout — see Stage 8 below.

**What we learned.** Heuristic ABORT gates need to be calibrated against actual training-noise levels, not idealized signals. When the heuristic disagrees with the loss curve and the accuracy metric, the heuristic is wrong.

→ Reference: [`docs/stage-refs/stage7.md`](stage-refs/stage7.md). Prompt: [`claude-code-prompts.md` § Stage 7](claude-code-prompts.md#stage-7-dpo-training).

## Stage 8 — Eval

Stage 8 is where the project's claims are made. Three checkpoints (base Gemma 4 E4B, SFT, SFT+DPO), one 300-pair human-labeled holdout, six metrics. Total Stage 8 spend: ~$3 of the $10 cap.

Two pivots worth understanding before reading the table.

**Pivot 1: Unsloth → vLLM (decision #30).** The original Stage 8 ran inference via Unsloth's `model.generate()` with `load_in_4bit=True`. Two problems killed it. First, Modal billing-cap preemption killed long detached runs — only the baseline cache landed; SFT and DPO never finished. Second, the precision confound: comparing 4-bit baseline κ against bf16 SFT/DPO κ conflates training delta with arithmetic precision delta. For a project whose pedagogical point is rigorous eval, that's disqualifying. The fix was to re-run all three checkpoints through vLLM at bf16 — same backend, same precision, same sampling, only the training delta varies across columns. Required nightly vllm + cu129 wheels + transformers ≥ 5.5 (Gemma 4 E4B uses `Gemma4ForConditionalGeneration` which stable vllm rejects on `rope_scaling`).

**Pivot 2: `max_tokens` 384 → 1024 (decision #31).** This is the quiet methodology lesson of the project. The first vLLM run-all used `max_tokens=384` (sized off the Stage 6 dryrun where SFT outputs barely cleared 256 tokens). It hit 10.1% parse failures on SFT and 9.9% on DPO, compared to 0% on baseline. Diagnosis: SFT-trained reasoning is materially longer than baseline's, and truncated outputs always cut off mid-sentence before the closing tags. Worse, the truncation correlated with case difficulty — the harder the case, the longer the reasoning, the more likely it ran past 384. Reporting parse-fail-excluded κ as the headline would have inflated the numbers by 0.06-0.18. DPO-OOD κ shifted from 0.820 (preview, 9.9% truncation) to 0.643 (headline, 0.3% truncation): −0.177 from a methodology fix alone. Both result files are kept under `eval/results/` for the methodology record.

The lesson: token budget for eval must accommodate the *trained* model's output distribution, not just the baseline's. Truncation that correlates with case difficulty is silent selection bias on the metric you actually care about.

### The eval table

300 pairs (240 in-dist + 60 OOD religion), Cohen's κ vs human verdicts, vLLM bf16, `temperature=0` for headlines, `T=0.3` for self-consistency.

| Metric | Base Gemma 4 E4B | After SFT | After SFT+DPO |
|---|---|---|---|
| Overall κ (in-dist) | 0.481 | 0.647 | **0.682** |
| Overall κ (OOD religion) | 0.542 | **0.695** | 0.643 |
| Clear cases κ | 0.453 | 0.665 | 0.727 |
| Subtle cases κ | 0.632 | 0.743 | **0.890** |
| Tracked-vs-alternate κ | 0.145 | 0.197 | 0.119 |
| Tie cases κ | 0.202 | −0.056 | **0.359** |
| Position-bias rate (in-dist) | 21.2% | 8.4% | 9.2% |
| Position-bias rate (OOD) | 21.7% | 11.7% | 16.7% |
| Verbosity bias score | +17.6 | +19.9 | +20.1 |
| Self-consistency (T=0.3) | 73.7% | 83.2% | 82.7% |
| Parse failure rate | 0.0% | 0.2% | 0.3% |

The headline: **in-dist κ 0.481 → 0.647 → 0.682**. SFT is the bigger jump (+0.166); DPO adds a modest +0.035. **DPO lands at the realistic 0.68 in-dist target** from primer Appendix C; SFT reaches 0.647. The stretch target (0.76) is not reached.

### Three findings worth highlighting

**1. DPO regresses on OOD (this is the project's most interesting finding).** SFT generalizes to held-out religion (κ=0.695) better than DPO (κ=0.643). DPO is the opposite of the primer's prediction. The likely explanation: the synthesized hard negatives (decision #23 — the 70/30 synth/flip approach) encoded patterns specific to the 10 in-dist categories, and DPO learned to discriminate those patterns rather than bias-in-general. This isn't a failure — it tells you that DPO preference data needs OOD diversity, not just in-dist difficulty. The same pattern shows up on position-bias-OOD (SFT 11.7% vs DPO 16.7%): DPO is more confident in-dist, and that confidence doesn't transfer.

This is also why two checkpoints get published (decision #32). The DPO model wins in-dist; the SFT-only model wins OOD. The model card surfaces this as a recommendation, not a hedge.

**2. Subtle bias is DPO's clearest win.** Subtle κ goes from 0.632 → 0.743 → **0.890**. This is the bucket DPO was designed for — discriminating between two responses where the bias signal is faint and the reasoning quality matters more than the verdict letter. The synthesized rejecteds were specifically crafted to have plausible-looking reasoning that lands wrong; DPO learned to spot exactly that pattern.

**3. Tie cases tell a strange story.** Baseline gets ties partially right (κ=0.202). SFT goes worse than chance (κ=−0.056). DPO recovers to κ=0.359. The SFT regression is real: training on hard non-tie pairs makes the model overconfident on cases where "neither is biased" is the correct verdict. DPO's preference structure (chosen vs rejected) actively penalizes overconfident verdict picking on ties.

**4. Tracked-vs-alternate is at the structural floor.** κ stays at 0.12-0.20 across all three models. This is the bucket where both responses are wrong but only one is biased — the hardest discrimination — and only 220 training pairs exist (BBQ's structural ceiling). The synth negatives weren't designed for this bucket specifically. DPO actually regresses here (0.197 → 0.119), confirming the synth-negatives-overfit story.

**What we learned.** SFT → DPO is not a strict improvement. DPO can sharpen in-distribution signal at a cost in OOD generalization. The right answer is to publish both checkpoints and let users pick based on whether their use case is in or out of training distribution.

→ Reference: [`docs/stage-refs/stage8.md`](stage-refs/stage8.md). Full eval write-up: [`eval/results/stage8_final_20260508T121153.md`](../eval/results/stage8_final_20260508T121153.md). Prompt: [`claude-code-prompts.md` § Stage 8](claude-code-prompts.md#stage-8-eval-harness).

## Stage 9 — Publishing

Four HF repositories under namespace `krishnakartik`:

| Repo | Contents |
|---|---|
| [`gemma4-social-bias-judge`](https://huggingface.co/krishnakartik/gemma4-social-bias-judge) | DPO model (merged fp16). The primary artifact. |
| [`gemma4-social-bias-judge-sft`](https://huggingface.co/krishnakartik/gemma4-social-bias-judge-sft) | SFT-only model. Recommended when bias categories are out-of-distribution. |
| [`gemma4-social-bias-judge-gguf`](https://huggingface.co/krishnakartik/gemma4-social-bias-judge-gguf) | Q8_0 + Q5_K_M GGUFs of both models, with sibling Modelfiles. |
| [`gemma4-social-bias-judge-pairs`](https://huggingface.co/datasets/krishnakartik/gemma4-social-bias-judge-pairs) | Dataset card with `sft.jsonl`, `dpo.jsonl`, raw labeled pairs, cross-checker disagreement statistics. |

Two model artifacts is the load-bearing decision. The DPO model is the primary; the SFT-only model is positioned explicitly as "use this if your bias categories are OOD relative to BBQ's training set." The model card's OOD-regression section surfaces a real finding instead of hiding behind aggregate κ.

The thinking-mode warning appears in three places per model card: a dedicated section near the top, a code-comment in the quick-start snippet, and a comment in the Ollama Modelfile. People skim. Say it three times.

The HF→Ollama bridge gap was a small surprise. When Ollama pulls a GGUF from HF (`ollama run hf.co/...`), it downloads the GGUF and the chat template but does *not* apply the sibling Modelfile's `SYSTEM` directive. So `ollama show judge --modelfile` shows a bare `FROM ... + TEMPLATE + stop tokens` Modelfile with no `SYSTEM` line. This is fine for API clients that send the system prompt explicitly on every request (the recommended pattern), but bare CLI usage like `ollama run judge "test"` would skip the system prompt. The Ollama deploy README documents the fallback path: `huggingface-cli download Modelfile.Q8_0 → ollama create judge -f Modelfile.Q8_0`.

**What we learned.** Two checkpoints with distinct positioning serves users better than one checkpoint with a hedged recommendation. The OOD regression is exactly the kind of finding that gets buried in "future work"; surfacing it in the model card with the SFT-only artifact alongside is what makes the project honest rather than just optimistic.

→ Reference: [`docs/stage-refs/stage9.md`](stage-refs/stage9.md). Prompt: [`claude-code-prompts.md` § Stage 9](claude-code-prompts.md#stage-9-publishing).

## Stage 10 — Deployment + benchmarks

Two recipes shipped: Ollama (local CPU, ~10-min setup) and vLLM (production-pattern, OpenAI-compatible API, Docker). They serve different educational purposes — the Ollama path shows how the GGUF artifact translates to a working local model with zero infrastructure; the vLLM path shows the production deployment pattern.

The Ollama one-liner:

```bash
ollama run hf.co/krishnakartik/gemma4-social-bias-judge-gguf:Q8_0
```

That downloads the 8.03 GB Q8_0 GGUF, starts an OpenAI-compatible API on `localhost:11434/v1`, and accepts requests in seconds. Tens of seconds per judgment on a modern laptop CPU; no GPU required.

The vLLM path is `docker compose up --build` from `deployment/vllm/`. ~16 GB VRAM minimum, bf16 inference, `--max-model-len 4096`, CUDA graphs **on**. The `example_client.py --backend vllm|ollama` script works against either backend with one flag.

### Benchmarks (Modal A100-80GB, full numbers in `deployment/benchmark_results.json`)

Offline batch (Stage 8 cache replay, 2,100 rows × 7 passes per pair):

| Model | Wall-clock | prompts/min | output tok/s |
|---|---|---|---|
| baseline | 330 s | 381.34 | 638.0 |
| sft | 554 s | 227.33 | 878.1 |
| dpo | 394 s | 319.95 | 1,236.2 |

Live serving with vLLM (CUDA graphs on):

| Mode | Concurrency | p50 latency | p95 latency | tok/s |
|---|---|---|---|---|
| Sequential | 1 | 2.16 s | 4.94 s | 73.0 |
| Concurrent | 16 | 14.0 s | 19.9 s | 170.7 |

The interesting Stage 10 finding: **CUDA graphs deliver ~3× sequential and ~5× concurrent throughput vs eager mode**. Stage 8 used `--enforce-eager` for run-to-run determinism across baseline/SFT/DPO eval (24.94 sequential and 36.6 concurrent tok/s). Stage 10 turned it off for production serving and the throughput jumped to 73 / 170. Eager mode is the right call for eval (determinism); CUDA graphs are the right call for production (throughput). Both numbers live in `deployment/benchmark_results.json` under `live_serving` and `legacy_enforce_eager` so the speedup is reproducible.

### Cost comparison

The resume-line number: **32.47× cheaper to self-host than to call Sonnet 4.6 per judgment**. Apples-to-apples per-call:

| Backend | USD per call |
|---|---|
| Sonnet 4.6 labeling (Batch API) | $0.004229 |
| Self-hosted DPO judge (Modal A100-80GB) | $0.000130 |
| **Ratio** | **32.47×** |

Per-pair-pipeline ratio (full Stage 4 pipeline including cross-checkers vs Modal eval) is 8.12× — meaningful context but not the headline.

**What we learned.** Eager mode is for determinism, CUDA graphs are for production throughput; pick per use case, don't try to use the same flag for both. The 32× cost ratio is the cleanest "what self-hosting buys you" number; everything else (latency, ergonomics) is workload-specific.

→ Reference: [`docs/stage-refs/stage10.md`](stage-refs/stage10.md). Prompt: [`claude-code-prompts.md` § Stage 10](claude-code-prompts.md#stage-10-deployment-recipes).

---

## What we'd do differently

Things that worked better than expected:

- **Position-swap doubling.** Half the effort for a doubled pool that also halved position bias. Cheapest data-side win in the project.
- **The vLLM pivot.** Switching all three checkpoints to bf16 vLLM was painful (nightly wheels + transformers 5.5 + cu129) but it eliminated the precision confound and gave us trustworthy numbers. Worth it.
- **Hand-labeling discipline.** 6-10 hours felt long but the eval κ depends on it. The label tool's `--slice` and `--review` flags kept the rubric consistent across sessions.
- **`lora_dropout = 0`.** Pushing back on the primer's 0.05 default saved ~30% wall-clock on training without sacrificing quality. Trust the runtime warnings.
- **Two model artifacts.** Publishing SFT-only alongside DPO let us be honest about the OOD regression instead of hiding it.

Things that were harder than expected:

- **Gemma 4 + Unsloth + TRL.** Five workarounds for SFT, six for DPO. Bleeding-edge model + bleeding-edge trainer = expect a lot of stack patches. Plan for it.
- **Truncation bias in eval.** The preview-vs-headline κ shift (up to −0.177 on OOD religion) was the project's quietest methodology near-miss. Right token budget = trained model's output distribution, not baseline's.
- **DPO over-collapse heuristics.** Per-batch reward variance can dominate the margin signal at this dataset size; trust the loss curve and `accuracies=1.0` instead of an "ABORT if margins collapse" rule.

Where the data points next:

- **v2b — UnQover OOD eval slice.** The DPO OOD regression on held-out religion is suggestive but small (n=60). Strengthening the OOD claim from "held-out BBQ category" to "different dataset entirely" via UnQover-derived pairs (~80 questions, ~60 pairs after pair construction) would let us say more confidently whether the regression is structural or sample-noise. ~4-6 hours of work, mostly hand-labeling.
- **v2a — autoresearch dataset iteration.** Frozen training + eval harness, editable `data_pipeline.py`, agent-driven optimization over dataset design choices. Reward signal: κ on the human-labeled eval set. The order matters — v2b first (better eval signal) before v2a (optimizes against eval signal).
- **Tracked-vs-alternate is supply-bound.** 220 pairs is the BBQ ceiling. The only way to move that bucket's κ is a different question structure entirely. Out of v1 scope but worth flagging.

The artifact at the end of all this is the [`gemma4-social-bias-judge`](https://huggingface.co/krishnakartik/gemma4-social-bias-judge) model card and an Ollama one-liner. Everything in between — the data pipeline, the labeling rubric, the eval methodology, the deployment recipes — is the actual product.
