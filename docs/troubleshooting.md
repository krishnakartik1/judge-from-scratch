# Troubleshooting

Real failure modes encountered during the `judge-from-scratch` build, with concrete fixes. Each entry is **Symptom → Cause → Fix**.

For deep context on any decision-numbered fix, see [`docs/project-status.md`](project-status.md).

---

## CUDA out of memory during Stage 6 SFT or Stage 7 DPO

**Symptom.** Training crashes with `CUDA out of memory` or VRAM utilization sits at 100% before training begins.

**Cause.** Gemma 4 E4B has ~6.87B raw parameters via MatFormer; VRAM math is different from plain Gemma 3 4B. Expect roughly:
- 4-bit quantized weights: ~4 GB
- bf16 weights: ~14 GB
- Peak VRAM during training (weights + activations + gradients + AdamW-8bit state): 24-28 GB

**Fix.** Use the QLoRA path (`load_in_4bit=True`); a 40 GB A100 fits at peak ~24 GB. If you're on a smaller card, drop `per_device_train_batch_size` to 2 and `gradient_accumulation_steps` to 8 (preserves effective batch=16). Free Colab T4 (16 GB) works for the training runs per Unsloth's QLoRA recommendation.

---

## Model outputs garbage / wrong template tokens

**Symptom.** Generated text contains visible chat-template literals like `<start_of_turn>` or `<|turn>`, or starts with whitespace and never produces `<reasoning>`.

**Cause.** Tokenizer version mismatch between Stage 5 formatting and Stage 6/7 training, or hand-rolled chat templates that don't match Gemma 4's native format.

**Fix.** Always render prompts via `tokenizer.apply_chat_template()` — never hand-roll the wrapper. Verify the same `model_id` is loaded in Stage 5 (`data/05_format_datasets.py`) and Stage 6 (`train/modal/sft.py`). Gemma 4 ships a multimodal `Gemma4Processor`, not a plain text tokenizer; always pass user text via `text=` kwarg, not as a positional arg.

---

## Empty leading tokens or `<thinking>` tags in output

**Symptom.** Trained model emits an empty leading token, or produces `<thinking>...</thinking>` tags, or generates nothing parseable before hitting `max_tokens`.

**Cause.** Native Gemma 4 thinking mode is enabled somewhere — typically `<|think|>` slipped into a system prompt at training time, eval time, or in the published Modelfile. The judge was trained without thinking mode (decision #13); enabling it at inference routes generation into a path the model never saw.

**Fix.** Grep every system prompt source for `<|think|>` — must be zero. The single source of truth for the invariant is `eval/eval_harness.py:125-137` (`assert_no_thinking_in_prompt`). For published artifacts, check `publish/build_modelfile.py` (Ollama Modelfile generator), the model card's quick-start snippets, and any client code. The thinking-mode warning is reproduced in three places per model card for a reason — people skim.

---

## Eval κ much lower than the table reports

**Symptom.** You re-run Stage 8 eval and your numbers are 0.05-0.20 κ lower than what the model card or `eval/results/stage8_final_20260508T121153.md` reports.

**Cause.** One of four:
1. Eval pairs leaked into training (the religion holdout was bypassed).
2. Inference temperature ≠ 0 (use `temperature=0` for the headline κ; T=0.3 only for self-consistency).
3. System prompt drift — yours doesn't match `data/judge_system_prompt.md` exactly.
4. `max_tokens` too low. The most insidious failure: SFT/DPO outputs are materially longer than baseline's, so a `max_tokens=384` budget systematically truncates harder cases (decision #31). Truncated outputs parse-fail; if you exclude them from the denominator, κ inflates by 0.06-0.18.

**Fix.** Verify holdout integrity (`grep -c '"eval_slice":"in_dist"' data/pairs/eval_set_unlabeled.jsonl` should be 240; OOD should be 60). Set `temperature=0`. Diff your system prompt against `data/judge_system_prompt.md`. Use `max_tokens=1024` for SFT/DPO inference — the methodology lesson of Stage 8.

---

## Modal billing-cap kills long detached runs

**Symptom.** Training or eval kicked off via `modal run --detach`, but the function gets preempted partway through and only some checkpoints land on the volume.

**Cause.** Modal workspaces have a default billing cap. When the workspace exceeds it during a detached run, container-level preemption inflates per-call cost (e.g., Stage 8's first attempt went from ~10 s/entry to ~68 s/entry) and the run hits its time budget before completing. Decisions #25 and #30 both hit this.

**Fix.** Check the workspace billing cap before launching multi-hour detached runs and raise it if needed. For risky long jobs, prefer foreground `modal run` (not `--detach`) so you see preemption immediately. The cost-ledger pattern (see `train/modal/_cost_ledger.py`) records spend in a `try`/`finally` so failed runs still log spend, making post-mortem easier.

---

## Ollama doesn't apply the system prompt from the published Modelfile

**Symptom.** `ollama run hf.co/krishnakartik/gemma4-social-bias-judge-gguf:Q8_0` runs, but bare CLI usage like `ollama run judge "test"` produces unparseable output. `ollama show judge --modelfile` shows no `SYSTEM` line.

**Cause.** The HF→Ollama bridge downloads the GGUF and chat template but does *not* apply the sibling Modelfile's `SYSTEM` directive (verified on `ollama 0.20.5`).

**Fix.** Two paths:
1. **Recommended (API clients).** Send the system message explicitly on every request. Both `deployment/example_client.py` and the curl/Python examples in `deployment/ollama/README.md` do this — bare CLI usage is the only case that needs the SYSTEM block baked in.
2. **Bake the SYSTEM block in.** Use the fallback path documented in `deployment/ollama/README.md` § Troubleshooting:
   ```bash
   huggingface-cli download krishnakartik/gemma4-social-bias-judge-gguf \
       Modelfile.Q8_0 Q8_0.gguf --local-dir ./gguf-cache
   cd ./gguf-cache
   ollama create judge -f Modelfile.Q8_0
   ```

---

## Parse failures from `max_tokens` too low

**Symptom.** Eval reports a non-trivial parse failure rate (>1%), and looking at the failed cases shows the output cuts off mid-sentence inside a `<reasoning>` block — never reaching `</reasoning><verdict>`.

**Cause.** `max_tokens` budget is sized for the baseline model's output distribution but the trained model writes longer reasoning. Truncation correlates with case difficulty (the harder the case, the longer the reasoning, the more likely it overruns).

**Fix.** Set `max_tokens=1024` for SFT/DPO inference (vs 384 from Stage 6 dryrun). Confirm: parse failure rate should drop to 0.2-0.3% (genuine cases where reasoning actually exceeds 1024 tokens). The methodological principle — token budget for eval must accommodate the trained model's distribution, not just the baseline's — is the load-bearing lesson of Stage 8 (decision #31).

---

## Unsloth + TRL pickle errors during SFT

**Symptom.** SFT training fails at startup with `_pickle.PicklingError` referencing `torch._dynamo.config.ConfigModuleInstance`, or with `TypeError` on `processing_class`.

**Cause.** Multiple stack incompatibilities between Unsloth's source-injection patches, TRL 0.24's API changes, and Gemma 4's multimodal processor. Decision #26 documents five distinct manifestations.

**Fix.** Apply each workaround:
1. **Pickle error during dataset prep.** Pre-tokenize CPU-side single-process via a custom `_pretokenize` helper, then pass `dataset_kwargs={"skip_prepare_dataset": True}` to `SFTConfig`.
2. **Pickle error from `train_on_responses_only`.** Invoke with `return_function=True` to get just the masking callable; apply via single-process `dataset.map`.
3. **Lazy-logits subscript error in `compute_loss`.** Set `os.environ["UNSLOTH_RETURN_LOGITS"] = "1"` *before* model load.
4. **`tokenizer=` is not accepted.** Rename to `processing_class=` and move `max_seq_length` → `max_length` (now on `SFTConfig`).
5. **Gemma 4 multimodal processor mis-routes.** Always call with `text=` kwarg, never positional.

The patterns are stable across Unsloth/TRL releases as of mid-2026, but verify against your specific lockfile if either dependency moved.

---

## vLLM rejects Gemma 4 with `rope_scaling` error

**Symptom.** `vllm serve` or `LLM(...)` fails to load `krishnakartik/gemma4-social-bias-judge` with an error mentioning `rope_scaling`, or transformers complains it doesn't recognize `gemma4` as a model class.

**Cause.** Gemma 4 E4B uses `Gemma4ForConditionalGeneration`. Stable vllm 0.8.x rejects it on `rope_scaling` validation; vllm 0.19 stable pins `transformers ≤ 4.57.6`, which doesn't recognize the `gemma4` model class at all.

**Fix.** Use the Stage 8 image stack (`eval/modal/vllm_infer.py:478-492` for the constructor):
- Base image: `nvidia/cuda:12.9.0-devel-ubuntu22.04` (FlashInfer's JIT kernels need `nvcc`; `debian_slim` is too thin).
- Wheels: nightly vllm + cu129 + `transformers ≥ 5.5.0`.
- For local Docker users, `deployment/vllm/Dockerfile` builds against `vllm/vllm-openai:latest`, which lags the Stage 8 image but works once the upstream stable image lands the Gemma 4 fix. Throughput will differ from the Modal benchmark numbers; the `_image_divergence_note` in `deployment/benchmark_results.json` flags this explicitly.

---

## Reference

- [`docs/project-status.md`](project-status.md) — full decision log with reasoning for every numbered fix.
- [`docs/fine-tuning-primer.md`](fine-tuning-primer.md) — conceptual reference (LoRA, QLoRA, SFT, DPO, eval methodology).
- [`eval/eval_harness.py`](../eval/eval_harness.py) — single source of truth for output parsing (`parse_output`) and the thinking-mode invariant (`assert_no_thinking_in_prompt`).
