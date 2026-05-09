# judge-from-scratch

> Generate data, train, evaluate, and deploy a specialized LLM judge — explained step by step.

This repo walks through fine-tuning a small open-weights model into a specialized social-bias evaluation judge, end to end. You'll see how to construct training pairs from a benchmark dataset, label them with a frontier model, run SFT + DPO with QLoRA, evaluate with rigor, and deploy.

The model produced is `gemma4-social-bias-judge`: a fine-tuned Gemma 4 E4B that takes a question and two model responses and outputs a verdict on which response shows more social bias. The methodology generalizes to other judge-style fine-tuning tasks.

## Where to start

**New to fine-tuning?** Start with [`docs/fine-tuning-primer.md`](docs/fine-tuning-primer.md). The primer walks through gradient descent → LoRA → QLoRA → SFT → DPO, building intuition before any code.

**Comfortable with LoRA, want to see a real pipeline?** Jump to [`docs/fine-tuning-primer.md`](docs/fine-tuning-primer.md) Appendix B. It covers how to construct training pairs from BBQ, label them with Claude, and format for SFT/DPO. From there, the build itself lives in [`docs/claude-code-prompts.md`](docs/claude-code-prompts.md) — staged prompts you can run through Claude Code one stage at a time.

**Just want the deployment recipe and the model?** See [`docs/fine-tuning-primer.md`](docs/fine-tuning-primer.md) Step 9 for the deployment overview, and [the model card on HF](https://huggingface.co/krishnakartik/gemma4-social-bias-judge) for results and a working Ollama one-liner.

**Reading a file directly without this README?** Each doc file has a breadcrumb at the top showing where it fits in the sequence. Follow those if you want the full path.

## What you'll need

- Python 3.11
- A GPU with ≥12 GB VRAM (free Colab T4 works for the training runs)
- API keys: Anthropic (labeling), OpenRouter (candidate generation), Together (cross-checking), Hugging Face (publishing)
- ~$30-50 in API + compute costs for the full pipeline
- ~6-10 hours of hand-labeling time for the eval set

## Repo structure

```
docs/                  conceptual primer, design decisions, project status
  fine-tuning-primer.md
  claude-code-prompts.md
  project-status.md
src/                   pipeline code (Stages 0-9)
deployment/            vLLM Dockerfile, Ollama Modelfile (added in Stage 10)
eval/                  eval harness, results
notebooks/             step-by-step walkthroughs (added in Stage 11, post-v1)
```

## Status

This repo is being built incrementally. Current state, completed stages, and pending work are tracked in [`docs/project-status.md`](docs/project-status.md).