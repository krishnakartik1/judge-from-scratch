# Stage 6 — SFT training

**Prompt:** [`docs/claude-code-prompts.md` § Stage 6: SFT training](../claude-code-prompts.md#stage-6-sft-training)

**Scripts:**
- `train/modal/smoke_test.py` — Modal + Unsloth + Gemma 4 + 4-bit env check on A100 before any real training.
- `train/modal/upload_data.py` — stage SFT data onto the `judge-from-scratch` Modal volume.
- `train/modal/sft.py` — QLoRA SFT via Unsloth + TRL. Has dryrun (50 rows × 1 epoch) and full-run (3,844 rows × 3 epochs) entrypoints.
- `train/configs/sft.yaml` — hyperparameters (consumed by `sft.py`).
- `train/configs/README.md` — per-parameter justification table.

**Inputs:**
- `/vol/data/sft.jsonl` on Modal volume — 3,844 rows (Stage 5 output).
- Base model: `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit` (pre-quantized 4-bit mirror).

**Outputs:**
- `/vol/checkpoints/sft-final/` — final adapter on Modal volume (consumed by Stage 7).
- HF Hub: `krishnakartik/gemma4-judge-sft-checkpoint` (private, insurance push).

**Hyperparameters (key ones; full table in [`train/configs/README.md`](../../train/configs/README.md)):**

| Param | Value | Why |
|---|---|---|
| `lora.r` | 16 | Low-rank decomposition rank; primer says 16-32, capacity adequate for 3.8K rows |
| `lora.lora_alpha` | 32 | `2 × r` convention; effective scaling = 2.0 |
| `lora.target_modules` | `"all-linear"` | Apply LoRA to every linear layer (q/k/v/o/gate/up/down) |
| `lora.lora_dropout` | 0.0 | Decision #27 — Unsloth fast-patch path engages at 0 |
| `learning_rate` | 2e-4 | QLoRA's small adapter matrices need a relatively high LR |
| `num_train_epochs` | 3 | Sweet spot for 3,844 rows; >3 typically overfits |
| `per_device_train_batch_size` | 4 | Fits A100-40GB at peak ~24 GB |
| `gradient_accumulation_steps` | 4 | Effective batch = 16 |
| `optim` | `adamw_8bit` | Optimizer state stored in 8-bit, ~75% memory savings |
| `lr_scheduler_type` | `cosine` | Linear warmup → cosine decay |

**Run results (full SFT, single A100-40GB on Modal):**

| Metric | Value |
|---|---|
| Wall-clock | 88 min |
| Optimizer steps | 720 |
| Final `train_loss` | 0.889 |
| Final `mean_token_accuracy` | 86.1% (vs random ~10%) |
| Peak VRAM | 23.4 GB |
| `entropy` | 3.07 → 0.475 (collapsed as expected) |
| Cost | $3.86 / $30 cap (incl. all dryrun debugging) |

**Decisions made:**
- [#12](../project-status.md#key-methodological-decisions-chronological) — Switched fine-tuning target from Gemma 3 4B to Gemma 4 E4B. Day-zero Unsloth support; 128K context; native system prompt.
- [#13](../project-status.md#key-methodological-decisions-chronological) — Native thinking mode disabled. Custom tags only.
- [#24](../project-status.md#key-methodological-decisions-chronological) — Stage 6 SFT shipped on Modal. 5/5 OOD probe outputs format-clean; 1 truncated mid-reasoning at probe `max_new_tokens=256` → bumped to 384 for Stage 8.
- [#25](../project-status.md#key-methodological-decisions-chronological) — Migrated to Modal for serverless GPU. Original Lambda Labs A100-80GB plan dropped; A100-40GB on Modal is sufficient (23 GB peak).
- [#26](../project-status.md#key-methodological-decisions-chronological) — Five Unsloth/TRL/HF-datasets stack workarounds for SFT on Gemma 4 (`skip_prepare_dataset`, `train_on_responses_only` with `return_function=True`, `UNSLOTH_RETURN_LOGITS=1`, `processing_class=` rename, Gemma4Processor `text=` kwarg).
- [#27](../project-status.md#key-methodological-decisions-chronological) — `lora_dropout = 0` (not 0.05). Empirical: 68% steps/sec speedup at identical loss on the 50-row dryrun.

**Key outputs:**
- Loss curve was monotonic with no spikes. Stage 6 is "the training mostly worked" — the interesting findings come in Stage 8 eval, not training metrics. `train_loss=0.889` at epoch 3 is consistent with the model having memorized the labeler's verdict patterns to roughly the level the dataset supports.
