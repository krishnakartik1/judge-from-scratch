# Stage 8 — Eval harness

**Prompt:** [`docs/claude-code-prompts.md` § Stage 8: Eval harness](../claude-code-prompts.md#stage-8-eval-harness)

**Scripts:**
- `eval/eval_harness.py` — metric computation: Cohen's κ overall + per-bucket, position-bias rate, verbosity-bias score, self-consistency at T=0.3, parse failure rate. `parse_output` is the canonical regex for the trained judge's output format.
- `eval/modal/vllm_infer.py` — vLLM bf16 inference runner. One `@app.cls VllmRunner` parameterized by `model_path` (Modal partitions container pool by parameter value).
- `eval/modal/merge_adapter.py` — produce SFT-merged-fp16 from the Stage 6 adapter (Stage 6 only saved the adapter, not a merged fp16 checkpoint).

**Inputs:**
- `data/pairs/eval_set_unlabeled.jsonl` — 300 hand-labeled holdout pairs (Stage 3a + 3b output). The filename reflects its origin; Stage 3b's label tool writes `human_verdict` / `confidence` / `notes` in place into the same file.
- Three checkpoints: base Gemma 4 E4B, SFT-merged-fp16, DPO-merged-fp16.

**Outputs:**
- `eval/results/stage8_final_20260508T121153.md` — headline metrics + findings (canonical eval table).
- `eval/results/stage8_final_20260508T121153.json` — raw predictions + per-pair metrics.
- `eval/results/stage8_final_20260508T115420.md` — preview run at `max_tokens=384` (kept as the truncation-bias methodology reference).

**Headline eval table (300 pairs: 240 in-dist + 60 OOD religion):**

| Metric | Base Gemma 4 E4B | After SFT | After SFT+DPO |
|---|---|---|---|
| Overall κ (in-dist) | 0.481 | 0.647 | 0.682 |
| Overall κ (OOD religion) | 0.542 | 0.695 | 0.643 |
| Clear cases κ | 0.453 | 0.665 | 0.727 |
| Subtle cases κ | 0.632 | 0.743 | 0.890 |
| Tracked-vs-alternate κ | 0.145 | 0.197 | 0.119 |
| Tie cases κ | 0.202 | −0.056 | 0.359 |
| Position-bias rate (in-dist) | 21.2% | 8.4% | 9.2% |
| Position-bias rate (OOD) | 21.7% | 11.7% | 16.7% |
| Verbosity bias score | +17.6 | +19.9 | +20.1 |
| Self-consistency (T=0.3) | 73.7% | 83.2% | 82.7% |
| Parse failure rate | 0.0% | 0.2% | 0.3% |

**Decisions made:**
- [#30](../project-status.md#key-methodological-decisions-chronological) — Stage 8 pivoted from Unsloth/4-bit to vLLM/bf16 inference. Two killers: Modal billing-cap preemption killed long detached Unsloth runs; the precision confound (4-bit baseline κ vs bf16 trained-checkpoint κ) conflated training delta with precision delta.
- [#31](../project-status.md#key-methodological-decisions-chronological) — `max_tokens` bumped 384 → 1024. SFT-trained reasoning is materially longer than baseline's; truncation systematically dropped the harder cases. Preview-vs-headline κ shifted by up to −0.177 on OOD religion.
- [#32](../project-status.md#key-methodological-decisions-chronological) — Stage 8 evaluation complete. SFT-only checkpoint should be published alongside DPO as the recommended artifact for OOD use cases.

**Cost ledger:** ~$3 of $10 STAGE8_BUDGET_CAP_USD (one full run-all + SFT/DPO re-run at `max_tokens=1024`).

**Key outputs:**

The full findings discussion is in `eval/results/stage8_final_20260508T121153.md`; key takeaways:
- **SFT is the bigger jump (+0.166), DPO adds +0.035** to in-dist κ.
- **DPO regresses on OOD by 0.052** (0.695 → 0.643). Likely cause: synthesized hard negatives encoded patterns specific to the 10 in-dist categories rather than bias-in-general.
- **Subtle-bias κ 0.632 → 0.743 → 0.890** is DPO's clearest win.
- **Tie cases** go from worse-than-chance under SFT (κ = −0.056) to recovered under DPO (κ = 0.359).
- **Tracked-vs-alternate stays at the structural floor** (κ = 0.12-0.20) — the bucket is supply-bound at 220 training pairs.

**Methodological lesson (decision #31):** token budget for eval must accommodate the trained model's output distribution, not just the baseline's. Truncation that correlates with case difficulty is silent selection bias on the metric you actually care about.
