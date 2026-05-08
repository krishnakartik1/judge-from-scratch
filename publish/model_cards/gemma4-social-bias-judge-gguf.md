---
license: gemma
language:
  - en
base_model: google/gemma-4-E4B-it
tags:
  - gguf
  - llama.cpp
  - ollama
  - judge
  - social-bias
  - gemma
---

# gemma4-social-bias-judge — GGUF quantizations

GGUF quantizations of the [judge-from-scratch](https://github.com/krishnakartik1/judge-from-scratch)
social-bias judge. Both the **DPO** primary release and the
**SFT-only** secondary release are bundled in this single repo with
suffixed Ollama tags so users can pick the right checkpoint for
their use case from one discovery point.

Full model description, eval results, and the OOD-regression caveat
that decides DPO vs SFT live in the model cards, not here:

- **DPO (default)** — [`krishnakartik/gemma4-social-bias-judge`](https://huggingface.co/krishnakartik/gemma4-social-bias-judge)
- **SFT-only** — [`krishnakartik/gemma4-social-bias-judge-sft`](https://huggingface.co/krishnakartik/gemma4-social-bias-judge-sft)

---

## ⚠️ Important: Thinking Mode

These models were fine-tuned with **Gemma 4's native thinking mode
DISABLED**. The bundled Modelfiles' SYSTEM blocks omit `<|think|>`
deliberately. Do not modify them to enable thinking mode — see the
[primary model card](https://huggingface.co/krishnakartik/gemma4-social-bias-judge#%E2%9A%A0%EF%B8%8F-important-thinking-mode)
for the full explanation.

---

## Available tags

| Tag | Checkpoint | Quant | Approx size | Use case |
|---|---|---|---|---|
| `:Q8_0` | DPO (default) | Q8_0 | ~6 GB | Best quality; recommended starting point |
| `:Q5_K_M` | DPO | Q5_K_M | ~3.5 GB | Smaller; minor quality trade-off |
| `:Q8_0-sft` | SFT-only | Q8_0 | ~6 GB | Use for OOD bias categories |
| `:Q5_K_M-sft` | SFT-only | Q5_K_M | ~3.5 GB | Smaller SFT-only |

**If your bias categories are outside BBQ's training set** (politics,
ideology, novel demographic axes), prefer the `-sft` tags — see the
[OOD-regression discussion](https://huggingface.co/krishnakartik/gemma4-social-bias-judge#%EF%B8%8F-the-ood-regression---read-this-before-deploying)
on the primary model card.

---

## Quick start

### Ollama (default DPO Q8_0)

```bash
ollama run hf.co/krishnakartik/gemma4-social-bias-judge-gguf:Q8_0
```

### Ollama (SFT-only Q8_0)

```bash
ollama run hf.co/krishnakartik/gemma4-social-bias-judge-gguf:Q8_0-sft
```

### llama.cpp

The judge expects a specific system prompt; pull both the GGUF and
the system-prompt text into a working directory before invoking
`llama-cli`.

```bash
# 1. Pull the GGUF.
huggingface-cli download \
  krishnakartik/gemma4-social-bias-judge-gguf Q8_0.gguf \
  --local-dir ./gemma4-judge

# 2. Pull the system prompt (Apache-2.0; canonical source in the
#    judge-from-scratch repo).
curl -fsSL -o ./gemma4-judge/system_prompt.md \
  https://raw.githubusercontent.com/krishnakartik1/judge-from-scratch/main/data/judge_system_prompt.md

# 3. Run.
llama-cli -m ./gemma4-judge/Q8_0.gguf \
  --temp 0 --ctx-size 4096 \
  --system-prompt-file ./gemma4-judge/system_prompt.md
```

Ollama users can skip the system-prompt download — the Modelfiles
in this repo (`Modelfile.Q8_0` etc.) embed the judge system prompt
and the `temperature` / `num_ctx` settings already, so
`ollama run hf.co/...:Q8_0` is fully self-contained.

---

## License & citation

Same as the [primary model
card](https://huggingface.co/krishnakartik/gemma4-social-bias-judge#citation).
