# REVAL Judge — Project Context for Claude Code

## What this project is
Fine-tuning Gemma 3 4B into a specialized bias evaluation judge for the
REVAL framework. Distillation through synthetic data: Claude Opus 4.7 as
teacher, Gemma 3 4B as student.

## The plan
The full project plan, including dataset design, evaluation methodology,
and v2 autoresearch enhancement, is in `docs/fine-tuning-primer.md`.
Read that before doing substantive work.

## Stack
- Unsloth (fast LoRA/QLoRA on Gemma 3 4B)
- TRL (SFTTrainer, DPOTrainer)
- PEFT (LoRA via Unsloth)
- Together AI (candidate generation)
- Anthropic API (labeling)
- A single Lambda Labs A100 80GB for training

## Conventions
- All data files are JSONL, one record per line.
- Each pipeline stage is a numbered script that reads its predecessor's
  output and writes its own artifact. Stages are resumable (skip records
  already in output).
- Hyperparameters are in `train/configs/*.yaml`, not hardcoded.
- Use `uv` for dependency management.
- All inference runs at temperature=0 unless explicitly noted.

## Status
[Update this as you go]
- [ ] Stage 1: candidate generation
- [ ] Stage 2: pair construction
- [ ] Stage 3: Claude labeling
- [ ] Stage 4: dataset formatting
- [ ] Stage 5: hand-labeling 300 eval pairs
- [ ] SFT training
- [ ] DPO training
- [ ] Eval harness + model card
- [ ] HuggingFace publish
- [ ] GGUF export + Ollama Modelfile

## Out of scope for v1
- Autoresearch v2 loop (see Appendix D of the primer; only after v1 ships)
- Quantization beyond standard GGUF Q4_K_M / Q5_K_M
- KV cache compression (TurboQuant etc.)