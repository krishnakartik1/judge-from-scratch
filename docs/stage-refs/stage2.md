# Stage 2 — Pair construction

**Prompt:** [`docs/claude-code-prompts.md` § Stage 2: Pair construction with the pairing strategy](../claude-code-prompts.md#stage-2-pair-construction-with-the-pairing-strategy)

**Scripts:**
- `data/02_construct_pairs.py` — build training pairs across 5 strategy buckets, randomize A/B slot, prefer cross-model pairings.

**Inputs:**
- `data/raw/candidates_enriched.jsonl` — 12,000 enriched candidates (Stage 1.5 output).

**Outputs:**
- `data/pairs/pairs.jsonl` — 2,370 pairs.

**Bucket distribution (planned vs achieved):**

| Bucket | Target | Achieved | Notes |
|---|---|---|---|
| Clear bias vs clean | 800 | 800 | |
| Subtle bias vs clean | 550 | 550 | |
| Tracked-bias vs alternate-bias | 220 | 220 | BBQ structural ceiling — not a heuristic problem (decision #6) |
| Both-clean tie | 550 | 550 | |
| Adversarial (length / confidence asymmetry) | 250 | 250 | Allocated *before* clear/subtle so strict subsets get their share (decision #9) |
| **Total** | **2,370** | **2,370** | |

Pair record schema: `{pair_id, question_id, question_text, bias_category, answer_choices, response_a: {model, text, suspected_bias_level}, response_b: {...}, pair_category}`.

**Decisions made:**
- [#6](../project-status.md#key-methodological-decisions-chronological) — BBQ structural constraint discovery. "Bias-vs-bias same-question" is impossible because each BBQ row has a single `target_label`. Replaced with "tracked-bias vs alternate-bias" (biased candidate vs `incorrect_other` candidate from same question).
- [#9](../project-status.md#key-methodological-decisions-chronological) — Bucket ordering: adversarial allocates before clear/subtle to keep strict subsets fed.
- [#16](../project-status.md#key-methodological-decisions-chronological) — `answer_choices` field added to pair records so Stage 4 labeler and Stage 8 judge can render the BBQ multiple choice.

**Key outputs:**

The "tracked-bias vs alternate-bias" bucket is the most pedagogically interesting one — it pairs a candidate that took the stereotype-aligned path against another candidate that landed on a different wrong answer. Both are "wrong"; only one carries a bias signal. This is exactly the discrimination the trained judge needs to learn but is supply-bound at 220 pairs (cannot expand without a different question structure than BBQ provides).
