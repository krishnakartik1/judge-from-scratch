# Training results

Per-stage results log for `judge-from-scratch`. Each section captures the
artifacts produced, headline metrics, cost, and any non-obvious behavior
that future-me would want to find. Configs and their justifications live
in `train/configs/README.md`; methodological decisions live in
`docs/project-status.md`.

---

## Stage 6 — SFT (Gemma 4 E4B QLoRA on Modal)

**Date:** 2026-05-04
**Branch:** `feat/stage-6-sft-training`
**Modal app:** `ap-HEh7UBa6oT0DUegncEo0CC` ([Modal dashboard](https://modal.com/apps/krishnakartik1/main/ap-HEh7UBa6oT0DUegncEo0CC))
**Run name:** `20260504-033151-sft-full`
**Adapter (private):** [`krishnakartik/gemma4-judge-sft-checkpoint`](https://huggingface.co/krishnakartik/gemma4-judge-sft-checkpoint)
**Volume path:** `/vol/checkpoints/sft-final/` on Modal volume `judge-from-scratch`

### Inputs

| | |
|---|---|
| Base model | `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit` (~6.87B raw params, 4-bit Unsloth Dynamic) |
| Training data | `data/formatted/sft.jsonl` (3,844 rows = 1,922 labeled pairs × 2 position-swap copies) |
| Probe data | `data/pairs/eval_set_unlabeled.jsonl` (5 deterministic sha1-sorted rows from the religion-OOD holdout) |

### Hyperparameters (see `train/configs/README.md` for justifications)

| | |
|---|---|
| LoRA rank `r` | 16 |
| `lora_alpha` | 32 |
| `target_modules` | `all-linear` (q, k, v, o, gate, up, down) |
| `lora_dropout` | **0.0** (deviation from primer's 0.05; see decision #27) |
| `lora.bias` | `none` |
| Learning rate | 2e-4 |
| Schedule | cosine, 5% warmup |
| Optimizer | AdamW-8bit |
| Weight decay | 0.01 |
| Effective batch size | 16 (4 per-device × 4 grad-accum) |
| Epochs | 3 |
| Max sequence length | 2048 |
| Precision | bf16 (auto-detected via `is_bfloat16_supported()`) |
| Random state | 3407 |

### Final-step metrics

| Metric | Value |
|---|---|
| `train_runtime` | 5,275 s (87.9 min) |
| `train_samples_per_second` | 2.186 |
| `train_steps_per_second` | 0.137 |
| `train_loss` (epoch-3 average) | **0.889** |
| `mean_token_accuracy` | **86.1%** (vs ~10% chance level) |
| `entropy` (averaged) | 0.475 (collapsed from 3.07 baseline) |
| `num_tokens` (total trained) | 8.234 M |
| Optimizer steps completed | 720 (= 3,844 × 3 / 16) |
| Peak VRAM | **23.4 GB** (under 30 GB gate, well under A100-40GB ceiling) |

### Loss curve (selected steps)

| Step | Epoch | Loss | Token accuracy |
|---|---|---|---|
| 1 | 0.04 | 9.32 | 9.7% |
| 6 | 0.25 | 1.30 | 67.2% |
| 12 | 0.50 | 0.96 | 73.5% |
| 24 | 1.00 | 0.82 | 76.1% |
| 25 | 1.04 | 0.70 | 78.4% |
| 720 | 3.00 | 0.89 | 86.1% (avg) |

Curve was monotonic with no spikes. Format compliance was learned within
the first ~6 optimizer steps (epoch 0.25); subsequent training compressed
entropy and refined verdict choice.

### Probe results (5 religion-OOD held-out pairs, same `pair_id`s in dryrun and full)

| `pair_id` | Dryrun verdict (50 rows × 1 epoch) | Full verdict (3,844 × 3) | Format |
|---|---|---|---|
| `c78b5769d2397d62` | TIE | (truncated mid-reasoning at 256 tokens) | Reasoning OK; no verdict tag |
| `ff06d34c694a8518` | B | **B** | clean |
| `8d2242064d47609d` | A | **A** | clean |
| `276e453aa5b6dd74` | TIE | **A** (genuine judgment shift) | clean |
| `3e61526e52561d21` | B | **B** | clean |

Zero `<|think|>` / `<thinking>` tokens across any of the 10 generated outputs (5 dryrun + 5 full). The single truncation is a probe-side `max_new_tokens=256` cap, not a training problem; the reasoning content was high-quality. **Bump probe `max_new_tokens` to 384 in Stage 8 eval.**

### Cost ledger (per `train/.cost_ledger.jsonl`)

| Timestamp (UTC) | Function | GPU | Wallclock | Cost | Status |
|---|---|---|---|---|---|
| 2026-05-04 02:41:10 | `train_sft_dryrun` | A100 | 145.8 s | $0.075 | ❌ failed: SFTConfig kwarg rename |
| 2026-05-04 02:51:19 | `train_sft_dryrun` | A100 | 166.2 s | $0.086 | ❌ failed: pickling under multiproc map |
| 2026-05-04 03:08:56 | `train_sft_dryrun` | A100 | 541.5 s | $0.280 | ✅ ok (`lora_dropout=0.05`) |
| 2026-05-04 03:24:40 | `train_sft_dryrun` | A100 | 356.3 s | $0.184 | ✅ ok (`lora_dropout=0`, **34% faster wallclock**) |
| 2026-05-04 05:07:54 | **`train_sft_full`** | A100 | 5,808.8 s | **$3.001** | ✅ ok |
| 2026-05-04 12:26:43 | `push_sft_to_hf` | CPU | 5.4 s | $0.0002 | ❌ failed: 403 (wrong HF namespace) |
| 2026-05-04 12:29:54 | `push_sft_to_hf` | CPU | 27.5 s | $0.0010 | ✅ ok |
| | | | | **$3.628** | |

Plus three smoke-test runs (~$0.10 each) at the very start that pre-dated the cost ledger; total Stage 6 spend ≈ **$3.86**.

Budget cap was $30 (`STAGE6_BUDGET_CAP_USD` in `train/modal/_cost_ledger.py`); we used 13% of it.

### Issues debugged during Stage 6 (full details in `docs/project-status.md` decisions #24–#28)

1. **`SFTTrainer.__init__()` got an unexpected keyword `tokenizer`** — TRL 0.24 renamed to `processing_class`; `max_seq_length` → `max_length` (now on `SFTConfig`).
2. **`TypeError: cannot pickle 'ConfigModuleInstance' object`** during dataset prep — Unsloth source-injects an override of `dataset_num_proc=None` to ~21 on fork-style Linux (`unsloth/models/rl.py:1195`). HF datasets' `arrow_dataset.py:3318` then spawns a multiprocess Pool for any `num_proc>=1`. Workaround: pre-tokenize CPU-side single-process, pass `dataset_kwargs={"skip_prepare_dataset": True}`.
3. **Same pickle error inside `train_on_responses_only`** — wrapper auto-computes large `num_proc`. Workaround: `return_function=True` to get just the masking callable, apply via single-process map manually.
4. **`TypeError: 'function' object is not subscriptable`** during training step — TRL 0.24's `compute_loss` subscripts `outputs.logits.shape` for entropy logging; Unsloth's cut-cross-entropy returns a lazy callable. Fix: `os.environ["UNSLOTH_RETURN_LOGITS"] = "1"` *before* model load.
5. **`TypeError: 'NoneType' object is not subscriptable`** in tokenizer call — Gemma 4 ships a multimodal `Gemma4Processor`, not a plain text tokenizer; positional arg routes incorrectly. Fix: always call with `text=` kwarg.
6. **VRAM bound assertion fired at 10.85 GB** — original spec assumed 4-5 GB for a non-Matryoshka 6.87B-param model in pure 4-bit. Gemma 4 E4B's MatFormer design + Unsloth Dynamic keeps embeddings/lm_head/scaffolding in fp16, pushing to ~10–11 GB. Widened smoke-test bound to (3.5, 16.0) GB and added a positive `Linear4bit`-count check that doesn't depend on VRAM math.
7. **HF push 403 Forbidden** — original config had wrong namespace. Fixed via decision #28; 15 references updated across docs.

### Steps to reproduce

From project root:

```bash
modal secret create huggingface HF_TOKEN=<token-with-write-scope>
modal secret create wandb WANDB_API_KEY=<wandb-key>

modal run train/modal/smoke_test.py
modal run train/modal/upload_data.py
modal run train/modal/sft.py::train_sft_dryrun
modal run train/modal/sft.py::train_sft_full
modal run train/modal/sft.py::push_sft_to_hf
```

The local entrypoints handle budget gating and ledger updates automatically. Set `BUDGET_OVERRIDE=1` in env to skip the interactive `CONTINUE` prompt.

### Verdict

Adapter is shippable. Stage 7 (DPO) consumes it from either the Modal volume or the HF repo. Stage 8 eval will produce the headline κ numbers vs Krishna's 300-pair human labels.
