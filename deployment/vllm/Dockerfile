# Stage 10: vLLM Dockerfile for the social-bias judge.
#
# Constraint: the system prompt MUST NOT contain `<|think|>` (Gemma 4
# thinking-mode token; decision #13 — train/infer parity). This image
# only serves weights — it does NOT bake in a system prompt the way an
# Ollama Modelfile does. Clients must include the contents of
# `data/judge_system_prompt.md` as the `system` message on every
# request. See `deployment/vllm/README.md` for the client contract and
# `eval/eval_harness.py:125-137` for the canonical assertion.
#
# Flags mirror the Stage 8 LLM() constructor at
# `eval/modal/vllm_infer.py:478-492` EXCEPT we omit `--enforce-eager`
# so vLLM can capture CUDA graphs and AOT-compile. Stage 8 used eager
# mode for run-to-run determinism across baseline/SFT/DPO eval; serving
# doesn't need that and the speedup is large (see
# `deployment/benchmark_results.json` — sequential tok/s 25→73,
# concurrent agg tok/s 37→171 on A100-80GB). Cold start with graphs is
# ~3 min longer because of torch.compile; subsequent restarts hit the
# cache. Add `--enforce-eager` back if you need bit-exact reproducibility.
# Drop or lower `--gpu-memory-utilization` on small consumer cards.

FROM vllm/vllm-openai:latest

ENV HF_HOME=/root/.cache/huggingface

EXPOSE 8000

ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--model", "krishnakartik/gemma4-social-bias-judge", \
     "--max-model-len", "4096", \
     "--dtype", "bfloat16", \
     "--gpu-memory-utilization", "0.85", \
     "--port", "8000"]
