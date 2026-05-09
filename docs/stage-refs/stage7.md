# Stage 7 — DPO training

**Prompt:** [`docs/claude-code-prompts.md` § Stage 7: DPO training](../claude-code-prompts.md#stage-7-dpo-training)

**Scripts:**
- `train/modal/dpo.py` — DPO via TRL's `DPOTrainer` on top of the SFT adapter. Reference model is the SFT model with adapters disabled (TRL's PEFT integration; no separate model copy).
- `train/configs/dpo.yaml` — DPO hyperparameters.

**Inputs:**
- `/vol/data/dpo.jsonl` on Modal volume — 2,200 rows (Stage 5 output).
- `/vol/checkpoints/sft-final/` — SFT adapter from Stage 6.

**Outputs:**
- `/vol/checkpoints/dpo-final/` — DPO adapter on Modal volume.
- `/vol/checkpoints/merged-fp16/` — merged-and-unloaded fp16 checkpoint for vLLM and GGUF conversion.

**Hyperparameters:**

| Param | Value |
|---|---|
| `learning_rate` | 5e-6 |
| `num_train_epochs` | 1 |
| `per_device_train_batch_size` | 2 |
| `gradient_accumulation_steps` | 8 |
| `beta` | 0.1 |
| `max_length` | 2048 |
| `max_prompt_length` | 1024 |
| `loss_type` | `sigmoid` |

Effective batch = 16 (matches SFT).

**Run results:**

| Metric | Value |
|---|---|
| Wall-clock training | 34 min |
| Wall-clock total (incl. probe) | 43 min |
| Optimizer steps | 138 |
| Final `train_loss` | 0.010 |
| Final `rewards/accuracies` | 1.00 |
| `rewards/margins` | 80–125 (oscillating throughout — high per-batch variance) |
| Peak VRAM | 27.8 GB |
| Cost | $1.34; cumulative Stage 7 $5.60 of $26.37 cap |

**Decisions made:**
- [#29](../project-status.md#key-methodological-decisions-chronological) — Stage 7 DPO shipped on Modal. Six surgical patches required for TRL 0.24 + transformers 5.5 + Gemma 4 multimodal compat (mergekit dep, llm_blender + weave module stubs, `warnings_issued` seed, `model_type` mask during DPOTrainer init, text-tokenizer pass-through). Merge decoupled from training as a standalone entrypoint — over-collapse heuristic fired ABORT (false positive) on healthy training due to per-batch reward variance dominating the margin signal.

**Key outputs:**
- Probe set is sha1-deterministic. 5/5 SFT-vs-DPO probe verdicts identical because the deterministic sample lands on easy pairs SFT already nailed. The interesting DPO-vs-SFT differences only show up on the 300-pair Stage 8 holdout — see [`stage8.md`](stage8.md).
- The high `rewards/margins` oscillation (80-125) is what tripped the ABORT heuristic. Per-batch reward variance dominates the margin signal at this dataset size; trust the loss curve and `accuracies=1.0` instead.
