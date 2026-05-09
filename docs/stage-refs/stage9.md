# Stage 9 — Publishing

**Prompt:** [`docs/claude-code-prompts.md` § Stage 9: Publishing](../claude-code-prompts.md#stage-9-publishing)

**Scripts:**
- `publish/export_gguf.py` — convert merged fp16 → Q8_0 + Q5_K_M GGUFs via llama.cpp.
- `publish/build_modelfile.py` — generate Ollama Modelfiles with the judge system prompt baked in (asserts no `<|think|>` before writing).
- `publish/upload_hf.py` — upload to four HF repos behind a `--confirm` flag.

**Inputs:**
- `/vol/checkpoints/merged-fp16/` — DPO-merged fp16.
- `/vol/checkpoints/sft-merged-fp16/` — SFT-merged fp16 (produced by `eval/modal/merge_adapter.py` in Stage 8).
- `eval/results/stage8_final_20260508T121153.md` — eval table for the model card.

**Outputs (4 HF repos under namespace `krishnakartik`):**

| Repo | Contents |
|---|---|
| `krishnakartik/gemma4-social-bias-judge` | DPO model (merged fp16). Primary artifact. |
| `krishnakartik/gemma4-social-bias-judge-sft` | SFT-only model (merged fp16). Secondary; recommended for OOD bias categories. |
| `krishnakartik/gemma4-social-bias-judge-gguf` | Both models as Q8_0 + Q5_K_M GGUFs + sibling Modelfiles. |
| `krishnakartik/gemma4-social-bias-judge-pairs` | Dataset card with `sft.jsonl`, `dpo.jsonl`, raw labeled pairs, cross-checker disagreement statistics. |

**Local model card sources (committed in repo):**
- `publish/model_cards/gemma4-social-bias-judge.md`
- `publish/model_cards/gemma4-social-bias-judge-sft.md`
- `publish/model_cards/gemma4-social-bias-judge-gguf.md`
- `publish/model_cards/gemma4-social-bias-judge-pairs.md`

**Generated Modelfiles:** `publish/modelfiles/Modelfile.{Q8_0,Q5_K_M}` (DPO), `Modelfile.Q8_0-sft`, `Modelfile.Q5_K_M-sft`.

**Decisions made:**
- [#13](../project-status.md#key-methodological-decisions-chronological) — Thinking-mode warning in three places per model card: dedicated section near top, code-comment in quick-start snippet, comment in Ollama Modelfile.
- [#15](../project-status.md#key-methodological-decisions-chronological) — Three deployment paths (Ollama + vLLM in Stage 10, HF Space deferred to Stage 11).
- [#28](../project-status.md#key-methodological-decisions-chronological) — HF namespace correction: `krishnakartik` (not `krishnak` or `krishnakartik1`). 15 references updated across docs and configs.
- [#32](../project-status.md#key-methodological-decisions-chronological) — Two model artifacts published, not one. SFT-only positioned explicitly as "use this if your bias categories are out-of-distribution."

**Key outputs:**
- The model card's OOD-regression section (decision #32) is the load-bearing piece — it surfaces a real finding instead of hiding behind aggregate κ. Future readers picking between checkpoints have the data they need.
- The HF→Ollama bridge gap (sibling Modelfile `SYSTEM` block not auto-applied) is documented in `deployment/ollama/README.md` § "About the SYSTEM block" with the explicit `huggingface-cli download → ollama create` fallback path.
