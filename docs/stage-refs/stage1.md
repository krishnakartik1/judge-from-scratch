# Stage 1 — Candidate generation (and Stage 1.5 enrichment)

**Prompt:** [`docs/claude-code-prompts.md` § Stage 1: Candidate response generation](../claude-code-prompts.md#stage-1-candidate-response-generation)

Stage 1.5 (enrichment) doesn't have a separate prompt — it was added after the Stage 2 audit caught a classification leak (decision #7). The fix lives in `data/01b_enrich_candidates.py`.

**Scripts:**
- `data/00_sample_bbq.py` — stratified BBQ sample (3,000 questions, seed=42, 50/50 ambig/disambig × 50/50 polarity).
- `data/01_generate_candidates.py` — generate 12,000 candidates via OpenRouter (3,000 questions × 4 models, temperature=0.7, max 300 output tokens, async semaphore=20).
- `data/01b_enrich_candidates.py` — classify each candidate as `correct` / `biased` / `incorrect_other` / `parse_failed`.

**Inputs:**
- `data/raw/bbq_sample.jsonl` — 3,000 stratified BBQ rows.

**Outputs:**
- `data/raw/candidates.jsonl` — 12,000 generations.
- `data/raw/candidates_enriched.jsonl` — 12,000 generations + `bias_classification` field.

**Generator pool (decision #2 — kept deliberately small / less RLHF-aligned):**
- meta-llama/llama-3-8b-instruct
- meta-llama/llama-3.1-8b-instruct
- mistralai/mistral-7b-instruct-v0.1
- qwen/qwen-2.5-7b-instruct

**Decisions made:**
- [#1](../project-status.md#key-methodological-decisions-chronological) — Switched from Together AI to OpenRouter (model availability).
- [#2](../project-status.md#key-methodological-decisions-chronological) — Generator pool deliberately small (7-8B). Larger models hedge too much on ambiguous BBQ to produce biased candidates.
- [#3](../project-status.md#key-methodological-decisions-chronological) — Included intersectional categories (`race_x_gender`, `race_x_SES`).
- [#4](../project-status.md#key-methodological-decisions-chronological) — Switched ambig/disambig from 60/40 to 50/50.
- [#5](../project-status.md#key-methodological-decisions-chronological) — Stratified on `question_polarity` (50/50 neg/nonneg).
- [#7](../project-status.md#key-methodological-decisions-chronological) — Classification leak audit; fix is to trust the enriched field, not raw `chosen_idx == target_label`.
- [#8](../project-status.md#key-methodological-decisions-chronological) — Doubled Stage 1 input from 1,500 → 3,000 BBQ rows after the classification fix shrank the usable pool.

**Key outputs:**

Stage 1.5 bias-classification distribution (from `candidates_enriched.jsonl`):

| Class | Share |
|---|---|
| `correct` | ~78% |
| `biased` (chose target_label, not also answer_label) | ~9% |
| `incorrect_other` | ~10% |
| `parse_failed` | ~3% |

The 9% biased pool is what makes pair construction work. If the generator pool were too RLHF-aligned, this would collapse to <2% and Stage 2 would have no biased candidates to pair against.
