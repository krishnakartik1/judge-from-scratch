# Building a judge from scratch — Part 1: data

> Part 1 of 2. This walks through Stages 0-5: setting up the repo, generating candidate responses, building training pairs, holding out an eval set, labeling with Claude, and formatting the final SFT and DPO datasets. [Part 2](story-training.md) covers training, eval, and deployment.

This is a tour of how `judge-from-scratch` actually got built. It's not a tutorial that re-runs code in a notebook — the scripts already exist and you can read them. It's a narrative that explains *why* each stage looks the way it does and which decisions matter.

If LoRA, QLoRA, SFT, and DPO are new to you, [`docs/fine-tuning-primer.md`](fine-tuning-primer.md) is the conceptual foundation — start there. If you want to see the exact prompts that produced each script, [`docs/claude-code-prompts.md`](claude-code-prompts.md) is the build manual.

## How this project was built

Two AI tools, each doing what it's good at:

- **Claude in chat (via a Claude Project).** The conceptual work — the [primer](fine-tuning-primer.md), the data design, the eval methodology — was all developed in long-form conversation. The Project's "knowledge" was the primer, the [build prompts](claude-code-prompts.md), and the [project status](project-status.md) — three files attached as project knowledge so any new chat could pick up context without re-explaining the project. When a stage finished, the status file got updated and the next chat session inherited that context automatically.
- **Claude Code for staged implementation.** Each pipeline stage was built from a single scoped prompt in [`docs/claude-code-prompts.md`](claude-code-prompts.md). One prompt per stage, one PR per stage. Inside each stage, Claude Code wrote the script, ran the dryrun, surfaced the dryrun output, waited for review, and then ran the full thing.

The split worked because the two tools have different strengths. Long chat is good at "should we hold out religion or religion+disability?" and "how do we close the verbosity-bias hole in the synthesis prompt?" Claude Code is good at "given this spec, write a 200-line resumable async script with proper error handling." Mixing them up — using long chat to write code, or using Claude Code to argue about methodology — wastes both.

What stayed yours, what to delegate:

| You own | The assistant owns |
|---|---|
| Data design (which buckets, which categories, why) | Script implementation, async/retry plumbing |
| Eval methodology (κ, position bias, OOD definition) | API client glue, JSON parsing, file I/O |
| The labeling rubric | Boilerplate (chat-template wrapping, Modal images, configs) |
| Deciding when the AI's plan is wrong | Translating a clear plan into working code |
| What "good enough" looks like for the dataset | Resumability, retries, log formatting |

The skill in using a coding assistant well is partly knowing when *not* to delegate — pushing back when its plan is plausible-looking but wrong. The next sections include several examples of that.

## Stage 0 — Repo bootstrap

Standard Python project skeleton: `pyproject.toml` (uv-managed), `.env.example`, `.gitignore`, the `data/{raw,pairs,labeled,formatted}/` tree, and a small `scripts/common.py` with the helpers every later stage relies on — especially `already_processed`, which makes resumable scripts a one-line check instead of a custom dedup pass per stage.

The dependency list is the load-bearing part. Pinning `transformers >= 4.53.0` was for Gemma 4 support; later stages pulled in `unsloth`, `trl`, `peft`, `accelerate`, `bitsandbytes` for training, plus `openai`, `anthropic`, `together`, `python-dotenv` for the API plumbing.

**What we learned.** The repo skeleton is the cheapest part of the project, but skipping it costs you later. Without resumable JSONL helpers established here, every stage downstream invents its own dedup mechanism and they don't agree.

→ Reference: [`docs/stage-refs/stage0.md`](stage-refs/stage0.md). Prompt: [`claude-code-prompts.md` § Stage 0](claude-code-prompts.md#stage-0-repo-bootstrap).

## Stage 1 — Candidate generation

The judge's training data is pairs of model responses. Stage 1's job is generating the responses themselves — 12,000 of them (3,000 BBQ questions × 4 generator models) — with enough biased outputs in the pool that pair construction has something to work with.

The generator pool was deliberately small: four 7-8B-parameter models (Llama 3 8B, Llama 3.1 8B, Mistral 7B, Qwen 2.5 7B), all reached via OpenRouter. The instinct is to pick "good" generator models, but for this stage that's wrong. Larger and more RLHF-aligned models hedge on ambiguous BBQ questions and produce too few biased candidates to construct training pairs against. Stage 1's role is *eliciting bias*, not judging it. Smaller, less-RLHFed models fall back on training-data priors when context is ambiguous — which is where stereotypes live.

After generation, **Stage 1.5** classifies each candidate as `correct` / `biased` / `incorrect_other` / `parse_failed`. The distribution landed at roughly 78 / 9 / 10 / 3 percent. The 9% biased pool is what makes everything downstream possible. If the generator pool had been more aligned, that number would collapse below 2% and Stage 2 would have nothing to pair against.

The audit that caught the *classification leak* was the most important moment of Stage 1. Stage 2 was originally filtering on raw `chosen_idx == target_label` to identify biased candidates — re-deriving bias status downstream from the BBQ structural metadata. The Stage 1.5 classifier was already doing this work better, looking at the candidate's actual reasoning. The fix was a one-liner — trust the enriched `bias_classification` field — but it shrunk the biased pool from 1,665 to 526 (the correct number). That's why the input was doubled from 1,500 to 3,000 BBQ rows: with the corrected bias detection, the original 1,500 didn't yield enough pairs.

**What we learned.** When a downstream stage starts re-deriving a fact an upstream stage already established, that's a bug, not an optimization. The fix is upstream-trust, not downstream-cleverness.

→ Reference: [`docs/stage-refs/stage1.md`](stage-refs/stage1.md). Prompt: [`claude-code-prompts.md` § Stage 1](claude-code-prompts.md#stage-1-candidate-response-generation).

## Stage 2 — Pair construction

Pair construction turns 12,000 candidate responses into 2,370 training pairs across five buckets:

| Bucket | Count | What it tests |
|---|---|---|
| Clear bias vs clean | 800 | Headline cases the judge must get right |
| Subtle bias vs clean | 550 | Stereotype-aligned reasoning that needs careful reading |
| Tracked-bias vs alternate-bias | 220 | Both wrong; only one is biased — the hardest discrimination |
| Both-clean tie | 550 | Teaches the judge that "neither is biased" is a real verdict |
| Adversarial (length / confidence asymmetry) | 250 | Resistance to surface signals |

The original plan had a "bias vs bias" bucket — two biased responses to the same question, judge picks the more biased one. That turned out to be impossible. Each BBQ question has a single `target_label` (the stereotype-aligned answer), so "biased" candidates for a given question all converge on the same letter. There's nothing to discriminate. The fix — captured as decision #6 — was to replace it with **tracked-bias vs alternate-bias**: a biased candidate paired against an `incorrect_other` candidate from the same question. Both are wrong; only one carries a bias signal. This is the hardest bucket and the most pedagogically interesting one, and it's structurally capped at 220 pairs (the BBQ supply ceiling).

Bucket *ordering* matters too. Adversarial pairs are a strict subset of clear/subtle (an adversarial pair is also a clear-or-subtle pair). If clear and subtle allocate first, they consume the pool and adversarial gets the leftovers. So the script allocates strict subsets first and looser categories second.

**What we learned.** When a planned bucket is structurally impossible, document why and substitute something equivalent — don't paper over it. The "tracked-vs-alternate" substitution turned out to be the most interesting bucket in the dataset.

→ Reference: [`docs/stage-refs/stage2.md`](stage-refs/stage2.md). Prompt: [`claude-code-prompts.md` § Stage 2](claude-code-prompts.md#stage-2-pair-construction-with-the-pairing-strategy).

## Stage 3 — Eval holdout (3a) and hand-labeling (3b)

Stage 3 is two scripts but one decision: 300 pairs are held out from training and labeled by hand. Without this, every reported metric is suspect — the trained judge would be evaluated against the same labels it was trained on.

**Stage 3a** stratifies the 300 holdout pairs into a 240-pair in-distribution slice (across the 10 BBQ categories the judge will train on) and a 60-pair OOD slice (religion category, held out from training entirely). The OOD design is deliberate: an unseen bias category tests whether the judge generalizes its skill, not whether it memorized category-specific patterns. Religion-only (vs religion + disability_status) was chosen because two-axis OOD would have dropped the SFT pool by 28% and the DPO-eligible pool by nearly half. Religion-only is a 19% drop and still gives a defensible "judge never trained on religion bias, here's its κ on it" claim.

**Stage 3b** is the hand-labeling. `eval/label_tool.py` is a small CLI: it shows you a pair, asks for verdict (a/b/t for tie), confidence (1-5), and optional notes; saves the labels in place; lets you skip pairs you've already done; and supports `--slice in_dist|ood_religion` for batching by slice across multiple sessions. The whole job took 6-10 hours over multiple sessions. There's no shortcut — this is the foundation of every reported metric, including the eval κ that decides whether the trained judge actually got better.

The confidence score earned its keep. Looking at where you marked confidence 1 or 2 is how you find pairs whose label is genuinely ambiguous (where reasonable judges would split) vs pairs where a clearer answer exists but you missed it. That signal also matters in Stage 5, which filters DPO-eligible pairs by `sonnet_confidence ≥ 4`.

**What we learned.** Hand-labeling 300 pairs is decision-fatiguing in a way that's hard to predict before doing it. Batching by slice, allowing review/edit, and recording confidence are not optional features — they're how the work stays consistent across sessions.

→ Reference: [`docs/stage-refs/stage3.md`](stage-refs/stage3.md). Prompts: [Stage 3a](claude-code-prompts.md#stage-3a-hold-out-the-human-eval-set), [Stage 3b](claude-code-prompts.md#stage-3b-hand-labeling-tool).

## Stage 4 — Claude labeling

Stage 4 labels the remaining 1,938 pairs (everything outside the holdout) using Claude Sonnet 4.6 as the primary labeler, with GPT-5.4 and Qwen 3 235B cross-checking 500 hard-bucket pairs. Total spend: $14.34, well under the $20 cap.

The interesting decision wasn't *which* labeler to use — it was Sonnet vs Opus. Opus 4.7 is the higher-tier model; the instinct is to use it. But on a 50-pair stratified dry-run, Opus and Sonnet agreed on 49/50 verdicts, and Sonnet's reasoning quality was indistinguishable on the cases they both got right. Cost dropped from ~$25 to ~$8 by switching. The dry-run was the gate: ≥90% overall agreement and ≥75% hard-bucket agreement to proceed; the actual numbers blew past both. Decision #17 captured the swap.

The cross-check pipeline produced the most interesting *finding* of Stage 4. On 500 hard-bucket pairs, Sonnet disagreed with at least one of GPT-5.4 / Qwen 3 235B on 17.4% of them. The first reading was "Sonnet might be missing subtle cases." But hand-review of 9 representative disagreements told a different story: Sonnet weights letter answers, GPT and Qwen weight reasoning chains. On a pair where response A's reasoning is biased but lands on the correct letter, Sonnet says A is the better response (correct letter wins) and GPT says A is the worse response (biased reasoning wins). Both rubrics are defensible. The disagreement signals rubric divergence, not weaker-model mistakes.

That reframing — captured as decision #22 — is what justified *not* mixing cross-checker labels into training. Mixing them would encode rubric inconsistency, not signal. The trained judge inherits Sonnet's letter-aware rubric; the model card documents this as a deliberate choice.

One pair was dropped (decision #19). `pair_id` 96b558e0bf7cbd01 failed parsing twice with the same pattern — Sonnet emitted `<thinking>` tags (Claude's native mode) instead of the requested `<reasoning>` tag, and ran out of `max_tokens` before producing the verdict block. Deterministic, not transient. Drop one row over patching the prompt mid-run; final labeled count is 1,937.

**What we learned.** The Sonnet-vs-Opus dry-run paid for itself in real money saved (~$17). When two labelers disagree, the right question is "which rubric is each one applying?" — not "which one is right?". Cross-checker disagreements are a signal about your own rubric, not a verdict on the labelers.

→ Reference: [`docs/stage-refs/stage4.md`](stage-refs/stage4.md). Prompt: [`claude-code-prompts.md` § Stage 4](claude-code-prompts.md#stage-4-claude-labeling-pipeline).

## Stage 5 — Dataset formatting

Stage 5 turns 1,937 labeled pairs into two training files:

- `data/formatted/sft.jsonl` — 3,844 SFT rows. The pipeline is 1,937 labeled pairs → 1,922 retained after the `sonnet_confidence ≥ 3` filter (drops 15 low-confidence pairs) → 3,844 rows after position-swap doubling (each pair becomes one `(prompt, target)` row plus a second row with A/B swapped and verdict flipped: A↔B, TIE→TIE). The target is `<reasoning>...</reasoning><verdict>{A|B|TIE}</verdict>` — Sonnet's `<confidence>` tag is stripped because the trained judge doesn't emit it.
- `data/formatted/dpo.jsonl` — 2,200 DPO rows. Filter to `sonnet_confidence ≥ 4` and `verdict ≠ TIE` (∼1,100 unique pairs). For each, construct a `rejected` response in one of two ways:
  - **70% synthesized** — call Sonnet 4.6 separately to write a "bad rationalization" of the *wrong* verdict, using one of four failure modes: verbose hedging, surface-level engagement without analyzing the reasoning chain, stereotype-aligned reasoning presented confidently, length asymmetry. The cost is small (~$3-4 with caching).
  - **30% verdict-flip** — take Sonnet's reasoning verbatim, flip the verdict letter. This produces a rejected whose reasoning argues for one verdict but selects the other. A real failure mode worth discriminating against.

Then position-swap doubling everywhere → 2,200 rows.

The most useful thing about Stage 5 was a course-correction caught mid-run. The first synthesis pass produced rejecteds whose token-length distribution was systematically *shorter* than the chosens — opening a "shorter = rejected" verbosity-bias shortcut for DPO to exploit. The synthesis prompt was re-tuned (from "~120 reasoning tokens" to "200-300 tokens, soft target") and the affected rows re-synthesized. Final length ratio: chosen/rejected median 0.90×, well under the 1.15× block threshold. Decision #23 documents this. Three rows leaked the synthesis instruction into the rejected text ("we need to say the verdict is B" — a direct prompt artifact); a stricter retry suffix fixed those.

The whole stage cost $2.05 against a $15 cap (Batch API plus the 50% off promotion).

**What we learned.** Length distribution of synthesized rejecteds is the silent killer of DPO. If chosens are systematically longer than rejecteds (or vice versa), the model learns "length signal" instead of "quality signal," and you find out about it only at eval. Verify the chosen/rejected length ratio before kicking off DPO training.

→ Reference: [`docs/stage-refs/stage5.md`](stage-refs/stage5.md). Prompt: [`claude-code-prompts.md` § Stage 5](claude-code-prompts.md#stage-5-format-sft-and-dpo-datasets).

---

That's the data side. The training data is now in `data/formatted/sft.jsonl` and `data/formatted/dpo.jsonl`, the eval set is hand-labeled, and the cross-checked rubric is documented. Part 2, [`story-training.md`](story-training.md), picks up at Stage 6 (SFT training on Modal) and runs through eval, publishing, and deployment.
