# Training config justification

This directory holds YAML configs consumed by `train/modal/*.py`. Each
parameter is justified below — sources are the project's
`docs/fine-tuning-primer.md`, Unsloth notebook defaults, decisions
in `docs/project-status.md`, and Stage 5's actual data shape.

If you change a value, update the matching row here. If you can't
defend the change in one sentence, don't make it.

---

## `sft.yaml` — Stage 6 SFT

### Model & runtime

| Param | Value | Source | Why this value |
|---|---|---|---|
| `model_id` | `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit` | Decision #12 + smoke test | Pre-quantized 4-bit mirror of Gemma 4 E4B-it. Same weights as the 16-bit `unsloth/gemma-4-E4B-it` mirror but stored already-quantized → faster cold start, predictable VRAM (10.85 GB measured), no on-the-fly quantization step |
| `max_seq_length` | 2048 | Stage 5 data shape | Stage 5 SFT rows: max prompt ~520 tokens, max target ~256 tokens → ~780 worst case. 2048 = 2.5× headroom + power of 2 (good GPU utilization). Doubling to 4096 doubles VRAM with zero benefit |
| `load_in_4bit` | true | Decision #12 | Required for QLoRA. Smoke test confirms 263 `Linear4bit` modules engaged at this setting |

### LoRA adapter shape

| Param | Value | Source | Why this value |
|---|---|---|---|
| `lora.r` | 16 | Primer Step 5 | Rank of the low-rank decomposition `ΔW = B @ A` where `A` is `r × in` and `B` is `out × r`. Primer says 16–32; "don't go above 64." For 4B-effective on 3.8K rows, 16 → ~50–100M trainable params (<1% of base). Enough capacity, low overfit risk |
| `lora.lora_alpha` | 32 | Convention | Scaling factor: effective LoRA contribution is `(alpha/r) × ΔW = 2 × ΔW`. Rule of thumb is `alpha = 2 × r`. Going higher amplifies adapter influence (more overfitting risk); lower mutes it |
| `lora.target_modules` | `"all-linear"` | Unsloth shortcut | Apply LoRA to every linear layer (q/k/v/o/gate/up/down). Maximum coverage. Restricting to only attention projections saves ~30% adapter params at meaningful quality cost on judge-format tasks |
| `lora.lora_dropout` | 0.0 | PEFT default + Unsloth recommendation | Set to 0 after empirical research (see "What dropout does" below). PEFT's `LoraConfig` default is `0.0`. Unsloth's official LoRA hyperparameters guide explicitly recommends 0 unless overfitting is observed; their warning during the first dryrun confirmed dropout=0 engages a fast-patch path (~5–15% faster). Sebastian Raschka's "Hundreds of Experiments" article fixed it at 0.05 across all runs but never ablated it and excluded it from the "hyperparameters that matter" list. Recent research (LoRA Dropout, ICLR 2025) labels standard dropout an "unreliable regularizer" for short fine-tuning runs. Other regularizers (r=16, 3 epochs, weight_decay=0.01, distillation labels) cover the small-dataset overfit risk |
| `lora.bias` | `"none"` | LoRA convention | Only train A/B matrices, leave bias terms frozen. Training biases adds parameter count without measurable benefit on most LoRA setups |
| `lora.random_state` | 3407 | Unsloth notebook convention | Reproducibility seed. Number is from Sebastian Raschka's "all you need is 16×16 patches" paper joke. Reproducible across runs at this seed |

### Training loop

| Param | Value | Source | Why this value |
|---|---|---|---|
| `training.learning_rate` | 2e-4 | Primer Step 5 | QLoRA's tiny adapter matrices need a relatively high LR (vs full SFT's 5e-5). Too low → no learning; too high → adapters explode. **The most sensitive knob — first to tune if results disappoint** |
| `training.num_train_epochs` | 3 | Primer Step 5 | "More than 3 epochs usually overfits." With 3,844 rows, by epoch 3 the model has seen each row 3× — sweet spot before memorization of labeler quirks |
| `training.per_device_train_batch_size` | 4 | A100 VRAM budget | At 4-bit base (~10 GB) + activations + gradients + AdamW-8bit state, batch size 4 fits comfortably under A100-40GB peak (~25 GB measured target) |
| `training.gradient_accumulation_steps` | 4 | Effective batch math | Effective batch = `per_device × accum = 16`. Primer says 16–32; we picked the small end because (a) gradient noise helps generalization on small datasets, (b) larger effective batch → fewer optimizer steps → less room for cosine LR schedule to do work |
| `training.warmup_ratio` | 0.05 | Unsloth default | 5% of total steps spent ramping LR from 0 → 2e-4. Avoids early-step instability when LR is high. Smaller datasets benefit less from longer warmup |
| `training.lr_scheduler_type` | `cosine` | Primer + Unsloth | Linear warmup → cosine decay to ~0 over remaining steps. Smoother than step decay, no manual milestone tuning |
| `training.optim` | `adamw_8bit` | Unsloth default for QLoRA | Stores Adam's momentum/variance state in 8-bit instead of 32-bit → ~75% optimizer-state memory savings. Quality cost is empirically near zero on QLoRA workloads |
| `training.weight_decay` | 0.01 | AdamW default | Mild L2 regularization on adapter weights. Standard across HF Trainer recipes |
| `training.logging_steps` | 10 | Logging cadence | W&B reports every 10 optimizer steps. At ~720 total steps for full run, that's 72 datapoints — enough to see the loss curve, not so many that W&B chokes |
| `training.save_strategy` | `epoch` | Checkpoint cadence | Save adapter at end of each epoch (3 checkpoints + final). `VolumeCommitCallback` persists each save to Modal volume so a mid-epoch crash recovers from the prior epoch's checkpoint |
| `bf16`/`fp16` | auto-detect | Plan-reviewer loop 1 finding | Set at runtime via `unsloth.is_bfloat16_supported()`. A100 → bf16 (wider exponent range, more numerical stability), T4/L4 fallback → fp16. Avoids dtype mismatch with bnb's 4-bit compute_dtype |

### Output paths

| Param | Value | Why |
|---|---|---|
| `output.checkpoints_dir` | `/vol/checkpoints/sft/` | Intermediate epoch checkpoints. Lives on Modal volume `judge-from-scratch` |
| `output.final_dir` | `/vol/checkpoints/sft-final/` | Final adapter after `trainer.save_model`. This is what gets pushed to HF Hub and consumed by Stage 7 (DPO) |

### Observability

| Param | Value | Why |
|---|---|---|
| `wandb.project` | `judge-from-scratch` | Wandb project name. Set as `WANDB_PROJECT` env var before trainer init (HF `TrainingArguments` has no native `wandb_project` arg) |

### Probe (dry-run + full-run inference probe)

| Param | Value | Why |
|---|---|---|
| `probe.source` | `/vol/data/eval_set_unlabeled.jsonl` | The 300-row religion-only OOD holdout from Stage 3a. Clean held-out source — neither dryrun nor full-train have ever seen these rows. See plan-reviewer loop 1 finding #4 |
| `probe.count` | 5 | 5 deterministically-selected rows (sha1-sorted by `pair_id`) for the side-by-side comparison in Step 4. Same 5 rows in both modes so the comparison is meaningful |

### HF push (insurance)

| Param | Value | Why |
|---|---|---|
| `hf_push.repo_id` | `krishnakartik/gemma4-judge-sft-checkpoint` | Per user spec at `docs/claude-code-prompts.md:800`. Distinct from Stage 9 publish target `krishnakartik/gemma4-social-bias-judge` — no collision |
| `hf_push.private` | true | Private until Stage 9. Cheap (~30 s upload) insurance against Modal volume loss |

---

## What dropout does

Dropout is a regularization technique. During training (not inference),
at each forward pass it randomly zeroes out a fraction of activations in
a given layer. The fraction is the dropout rate.

Why it works:
- Prevents over-reliance on any single neuron — forces redundant pathways
- Acts like training many sub-models simultaneously, ensembled at
  inference time when no neurons are dropped
- Reduces overfitting, especially on small datasets

For **LoRA specifically**, `lora_dropout=0.05` zeros 5% of activations
*along the LoRA path* (the A→B low-rank matrices). The base model's
frozen weights are unaffected — base forward is deterministic. Only the
trained adapter sees the stochasticity.

Trade-off table:

| `lora_dropout` | Speed | Regularization | When to use |
|---|---|---|---|
| 0 | Fastest (Unsloth fast-patch path engages) | None | Large datasets, low overfit risk, perf-bound runs |
| 0.05 | ~5–15% slower than 0 | Mild | Small-to-medium datasets like ours (3.8K rows). Default |
| 0.1–0.2 | Same as 0.05 | Heavy | Very small datasets, large models, strong overfit signal |

If Stage 8 eval shows the model has memorized labels (high train acc,
low eval acc), bump to 0.1. If the full run takes longer than the 4–6 h
budget and we're not overfitting, drop to 0.

---

## What we'd tune if results disappoint

Per the primer's "what actually moves the needle," tune in this order:

1. **`learning_rate`** — try 1e-4 (slower, more stable) or 5e-4 (faster, riskier). LR is the most sensitive knob on QLoRA SFT
2. **`num_train_epochs`** — 5 if still underfitting, 2 if overfitting
3. **`r`** — 32 if the model can't capture judge-format nuance at 16. Keep `lora_alpha = 2 × r` so effective scaling stays at 2.0
4. **`lora_dropout`** — bump to 0.1 if we see strong overfitting signal in W&B (train loss drops while eval stalls)
5. Everything else — leave alone unless the primer changes

Most of these are not tuned-from-our-data. They're educated defaults from
people who've fine-tuned a lot of models. The dry-run is the first signal
that any of them might be wrong.

---

## See also

- `docs/fine-tuning-primer.md` — primer, especially Step 5 ("Hyperparameters that actually matter")
- `docs/project-status.md` — decisions #12 (Gemma 4 E4B target), #13 (no `<|think|>`)
- `docs/claude-code-prompts.md` — user spec for each stage
- `train/modal/sft.py` — the script that consumes this config
- `tests/test_sft_config.py` — asserts the YAML matches the primer recipe
