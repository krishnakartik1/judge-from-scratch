---
title: Gemma4 Social Bias Judge
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: gemma
hardware: cpu-basic
---

# `gemma4-social-bias-judge` demo

A Gradio UI for the [`krishnakartik/gemma4-social-bias-judge`](https://huggingface.co/krishnakartik/gemma4-social-bias-judge) model. Paste a question and two candidate responses; the judge returns reasoning plus a verdict (`A`, `B`, or `TIE`) on which response shows more inappropriate reliance on social or demographic stereotypes.

## Status

This directory holds the Space recipe — `app.py`, `requirements.txt`, frontmatter — committed to the [`judge-from-scratch`](https://github.com/krishnakartik1/judge-from-scratch) repo. **The Space is not yet deployed**; pushing it to `huggingface.co/spaces/krishnakartik/gemma4-social-bias-judge-demo` is a follow-up step you (the maintainer) trigger when ready:

```bash
huggingface-cli upload krishnakartik/gemma4-social-bias-judge-demo space/ --repo-type space
```

(Create the empty Space first via `huggingface-cli repo create gemma4-social-bias-judge-demo --type space --space_sdk gradio` if it doesn't already exist, or via the HF web UI.)

## Backend

`app.py` loads the **Q5_K_M GGUF** of the DPO model directly via `llama-cpp-python`, downloading it from the published [`gemma4-social-bias-judge-gguf`](https://huggingface.co/krishnakartik/gemma4-social-bias-judge-gguf) repo on first startup. This runs on the free `cpu-basic` Spaces tier (16 GB RAM, 2 vCPU). No GPU required, no Inference API dependency.

## Performance expectations

- **Cold start: 30-60 seconds.** First request downloads the ~4.5 GB Q5_K_M GGUF and loads it into RAM.
- **Per-inference latency: 10-30 seconds** on `cpu-basic` for typical judgments. CPU-only inference at this model size is slow; a single judgment is several seconds of token generation.
- **Auto-sleep is on by default.** HF Spaces free tier sleeps idle Spaces after a few minutes of no traffic; the next request pays the cold-start tax again.

If you want a faster local try-it path, the [Ollama recipe](../deployment/ollama/README.md) gets you to first judgment in ~10 minutes with the same GGUF on your own hardware.

## Cost

The free `cpu-basic` tier costs $0/month with auto-sleep. Spaces moves to paid tiers if you upgrade hardware (GPU tiers run $0.40-$4/hr depending on accelerator). The recipe in this directory is designed to stay on the free tier — it deliberately does not use the HF Inference API (which would require a paid endpoint for fine-tuned models) or a GPU Space.

## Constraints (load-bearing)

- **The system prompt MUST NOT contain `<|think|>`.** The trained judge was fine-tuned with native Gemma 4 thinking mode disabled (decision #13). Including the thinking token would route generation into a path the model never saw — degraded, unparseable output. `app.py` asserts this at startup.
- **The system prompt is inlined in `app.py`**, mirroring [`data/judge_system_prompt.md`](../data/judge_system_prompt.md) verbatim. If the canonical prompt changes, update `app.py`'s `SYSTEM_PROMPT` constant.
- **Output schema:** `<reasoning>...</reasoning><verdict>{A|B|TIE}</verdict>`. Same parser regex as [`eval/eval_harness.py`](../eval/eval_harness.py) `parse_output`.
