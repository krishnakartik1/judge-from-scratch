# judge-from-scratch — Project Context for Claude Code

## What this project is
End-to-end educational tutorial that builds a specialized social-bias
evaluation judge by fine-tuning **Gemma 4 E4B** (decision #12 — switched
from Gemma 3 4B at project pivot). The artifact published to Hugging
Face is `gemma4-social-bias-judge`. Distillation through synthetic data:
Claude Sonnet 4.6 as primary labeler (decision #17), GPT-5.4 + Qwen 3
235B for cross-check triangulation, BBQ-derived pairs as the training
corpus, QLoRA + SFT + DPO as the training recipe.

## The plan
The full project plan, including dataset design, evaluation methodology,
and v2 autoresearch enhancement, is in `docs/fine-tuning-primer.md`.
Living state-of-the-build is in `docs/project-status.md`. Staged build
prompts are in `docs/claude-code-prompts.md`. Read those before doing
substantive work.

## Stack
- Unsloth (fast LoRA/QLoRA on Gemma 4 E4B)
- TRL (SFTTrainer, DPOTrainer)
- PEFT (LoRA via Unsloth)
- OpenRouter (candidate generation — decision #1; Together's catalog
  was deprecating models we needed)
- Anthropic API (labeling, via Batch API for cost)
- OpenAI + OpenRouter (cross-check labelers)
- A single Lambda Labs A100 80GB for training

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

## Status
[Update this as you go]
- [x] Stage 1: candidate generation (data/01_generate_candidates.py)
- [x] Stage 1.5: enrichment (data/01b_enrich_candidates.py)
- [x] Stage 2: pair construction (data/02_construct_pairs.py)
- [x] Stage 3a: holdout eval set (data/03a_holdout_eval.py)
- [ ] Stage 3b: hand-label 300 eval pairs (eval/label_tool.py)
- [x] Stage 4: Claude labeling (data/04_label_pairs.py)
- [x] Stage 5: dataset formatting (data/05_format_datasets.py)
- [x] Stage 6: SFT training (Gemma 4 E4B QLoRA on Modal)
- [ ] Stage 7: DPO training
- [ ] Stage 8: Eval harness + model card
- [ ] Stage 9: HuggingFace publish + GGUF export
- [ ] Stage 10: deployment recipes (Ollama + vLLM)

## Pipeline wiring
Stage 4 (Claude labeling) reads `data/pairs/pairs_to_label.jsonl`,
NOT `data/pairs/pairs.jsonl` — the holdout in Stage 3a is the canonical
gate. Bypassing it leaks the human-eval pairs into training.

## Out of scope for v1
- Autoresearch v2 loop (see Appendix D of the primer; only after v1 ships)
- Quantization beyond standard GGUF Q4_K_M / Q5_K_M
- KV cache compression (TurboQuant etc.)