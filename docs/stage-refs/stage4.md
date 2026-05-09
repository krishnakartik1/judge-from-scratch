# Stage 4 — Claude labeling

**Prompt:** [`docs/claude-code-prompts.md` § Stage 4: Claude labeling pipeline](../claude-code-prompts.md#stage-4-claude-labeling-pipeline)

**Scripts:**
- `data/04_label_pairs.py` — primary labeling (Sonnet 4.6, Anthropic Batch API, prompt caching) + cross-check on 500 hard-bucket pairs (GPT-5.4 + Qwen 3 235B).
- `data/labeling_prompt.md` — the rubric (Krishna-authored, decision #16-aware: renders `answer_choices` into the question framing).

**Inputs:**
- `data/pairs/pairs_to_label.jsonl` — 1,938 pairs (Stage 3a output).

**Outputs:**
- `data/labeled/labeled_pairs.jsonl` — 1,937 labeled pairs (1 dropped, decision #19). Fields: `sonnet_verdict`, `sonnet_reasoning`, `sonnet_confidence` (primary, from Sonnet 4.6); plus `crosscheck_verdict_b` / `crosscheck_reasoning_b` (cross-checker B = GPT-5.4) and `crosscheck_verdict_c` / `crosscheck_reasoning_c` (cross-checker C = Qwen 3 235B per decision #20) on the 500 hard-bucket pairs.
- `data/labeled/dropped_pairs.jsonl` — 1 row (`pair_id` 96b558e0bf7cbd01, deterministic parse failure).

**Cost ledger:**

| Phase | Spend |
|---|---|
| Sonnet-vs-Opus dry-run (50 pairs × 2 models) | ~$1 |
| Sonnet primary (1,937 pairs, Batch API) | $8.20 |
| GPT-5.4 cross-check (500 pairs, Batch API) | ~$3 |
| Qwen 3 235B cross-check (500 pairs, OpenRouter) | $0.07 |
| **Stage 4 total** | **$14.34** of $20 cap |

**Decisions made:**
- [#17](../project-status.md#key-methodological-decisions-chronological) — Switched primary labeler from Opus 4.7 to Sonnet 4.6 after a 50-pair dry run showed equivalent labels at ~30% the cost (~$8 vs ~$25).
- [#18](../project-status.md#key-methodological-decisions-chronological) — Moved `cache_control` from user-content to system block (canonical batch-caching pattern). Revalidated on the same 50 pairs.
- [#19](../project-status.md#key-methodological-decisions-chronological) — One pair dropped after deterministic Sonnet parse failure (emitted `<thinking>` instead of `<reasoning>`, hit `max_tokens=1024`).
- [#20](../project-status.md#key-methodological-decisions-chronological) — DeepSeek V3.1 disabled on Together account; cross-check switched to Qwen 3 235B Instruct via OpenRouter.
- [#21](../project-status.md#key-methodological-decisions-chronological) — Cross-check complete; 17.4% disagreement rate on hard buckets.
- [#22](../project-status.md#key-methodological-decisions-chronological) — Hand-review of 9 cross-checker disagreements revealed a *judging-philosophy gap*, not Sonnet errors: Sonnet weights letter answers; GPT/Qwen weight reasoning chains. The trained judge inherits Sonnet's letter-aware rubric.

**Key outputs:**

The 17.4% cross-check disagreement on hard buckets is the headline finding. Initially read as "Sonnet might be missing things"; hand-review of disagreements reframed it as "different valid rubrics." That reframing is what justifies *not* mixing cross-checker labels into training (decision #22) — the labels would encode rubric inconsistency, not signal.
