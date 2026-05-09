---
license: gemma
language:
  - en
library_name: transformers
base_model: google/gemma-4-E4B-it
base_model_relation: finetune
pipeline_tag: text-generation
tags:
  - judge
  - llm-as-judge
  - evaluation
  - social-bias
  - bbq
  - gemma
  - lora
  - sft
  - fine-tuned
datasets:
  - krishnakartik/gemma4-social-bias-judge-pairs
model-index:
  - name: gemma4-social-bias-judge-sft
    results:
      - task:
          type: text-classification
          name: Social-bias judge (A / B / TIE verdict)
        dataset:
          type: krishnakartik/gemma4-social-bias-judge-pairs
          name: Gemma 4 Social Bias Judge Pairs (eval holdout)
          config: eval_holdout
          split: train
        metrics:
          - type: cohen_kappa
            name: "Cohen's κ (in-distribution, 240 pairs)"
            value: 0.647
          - type: cohen_kappa
            name: "Cohen's κ (OOD religion, 60 pairs)"
            value: 0.695
          - type: cohen_kappa
            name: "Cohen's κ (tracked-vs-alternate)"
            value: 0.197
          - type: cohen_kappa
            name: "Cohen's κ (subtle-bias bucket)"
            value: 0.743
          - type: position_bias_rate
            name: "Position-bias rate (in-distribution; lower is better)"
            value: 0.084
          - type: self_consistency
            name: "Self-consistency rate (T=0.3)"
            value: 0.832
---

# Gemma 4 E4B — Social-Bias Judge (SFT only)

This is the **SFT-only checkpoint** from the [judge-from-scratch
project](https://github.com/krishnakartik1/judge-from-scratch). It is the
intermediate artifact before the DPO refinement pass that produced
[`krishnakartik/gemma4-social-bias-judge`](https://huggingface.co/krishnakartik/gemma4-social-bias-judge)
(the primary release).

**Use this checkpoint instead of the DPO version if your bias
categories are out-of-distribution relative to BBQ's training set.**
The DPO refinement narrows generalization by overfitting to the 10
in-distribution bias categories' specific patterns — fine when your
inputs match the training distribution, harmful when they don't.

For the full project narrative, eval methodology, training pipeline,
and limitations, **read the [primary model
card](https://huggingface.co/krishnakartik/gemma4-social-bias-judge)**.
This card focuses on what differs between the SFT-only and DPO
checkpoints.

---

## ⚠️ Important: Thinking Mode

This model was fine-tuned with **Gemma 4's native thinking mode
DISABLED**. Do **NOT** include `<|think|>` in the system prompt at
inference time — the model never saw that token during training and
will generate degraded, unparseable output. See the [primary model
card's thinking-mode
section](https://huggingface.co/krishnakartik/gemma4-social-bias-judge#%E2%9A%A0%EF%B8%8F-important-thinking-mode)
for the full explanation.

---

## Quick start

### Ollama

```bash
# IMPORTANT: thinking mode is disabled — do NOT add <|think|> to /system.
ollama run hf.co/krishnakartik/gemma4-social-bias-judge-gguf:Q8_0-sft
```

### Python (transformers)

```python
# Identical usage to the DPO checkpoint — only the model_id changes.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "krishnakartik/gemma4-social-bias-judge-sft"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="cuda"
)
# ... see primary model card for the full inference snippet.
```

---

## When to choose this over the DPO checkpoint

| Use case | Recommended |
|---|---|
| Bias categories in BBQ's 10 trained set (age, disability, gender identity, nationality, physical appearance, race/ethnicity inc. intersectional, religion, sexual orientation, SES) | DPO (primary) |
| Bias categories outside the trained set (politics, ideology, novel demographic axes, intersectional categories not in training) | **This checkpoint (SFT)** |
| Tie-case detection (both responses clean) is critical | DPO — tie-κ jumps from −0.06 (SFT) to 0.36 (DPO) |
| Subtle bias discrimination on in-dist data | DPO — subtle-κ jumps from 0.74 (SFT) to 0.89 (DPO) |
| Tracked-vs-alternate (which specific stereotype is invoked) | This checkpoint (SFT-κ 0.20 vs DPO-κ 0.12) |
| Position-bias robustness on OOD | This checkpoint (SFT 11.7% vs DPO 16.7%) |

---

## Eval results (selected)

Same 300-pair holdout, same vLLM/bf16 backend as the [primary model
card's eval
table](https://huggingface.co/krishnakartik/gemma4-social-bias-judge#eval-results).

| Metric | Base | **SFT (this)** | DPO |
|---|---|---|---|
| Overall κ (in-dist) | 0.481 | 0.647 | 0.682 |
| **Overall κ (OOD religion)** | 0.542 | **0.695** | 0.643 |
| Tracked-vs-alternate κ | 0.145 | **0.197** | 0.119 |
| Subtle cases κ | 0.632 | 0.743 | 0.890 |
| Tie cases κ | 0.202 | −0.056 | 0.359 |
| Position-bias rate (OOD) | 21.7% | **11.7%** | 16.7% |
| Self-consistency (T=0.3) | 73.7% | 83.2% | 82.7% |

This checkpoint **wins on OOD κ, tracked-vs-alternate κ, and
OOD position-bias**. The DPO checkpoint wins on in-dist κ, subtle
cases, and tie cases — the metrics where the synth-hard-negatives
training shape was specifically designed to help.

The OOD-κ delta (+0.052 in this checkpoint's favor) is the load-bearing
reason this artifact exists. See the [primary model card's
OOD-regression
discussion](https://huggingface.co/krishnakartik/gemma4-social-bias-judge#%EF%B8%8F-the-ood-regression---read-this-before-deploying)
for the full analysis.

---

## Training summary

QLoRA SFT: 3,844 rows (1,938 base pairs × position-swap doubling), 3
epochs, 720 optimizer steps, r=16, α=32, dropout=0, all-linear LoRA
targets, lr=2e-4 cosine, peak VRAM 23.4 GB on A100-40GB. Final
`train_loss` 0.889, `mean_token_accuracy` 86.1%. Total Stage 6
spend: ~$4. Adapter merged to bf16 for Stage 8 eval and this
release.

The DPO step was applied to a copy of this checkpoint (not gated by
this checkpoint's existence), so the SFT artifact is the same one
that fed into DPO — it's a checkpoint snapshot of the pipeline,
unmodified.

---

## License & citation

Same as the [primary model
card](https://huggingface.co/krishnakartik/gemma4-social-bias-judge#citation).
