# Stage 3 — Eval holdout (3a) + hand-labeling tool (3b)

**Prompts:**
- [`docs/claude-code-prompts.md` § Stage 3a: Hold out the human eval set](../claude-code-prompts.md#stage-3a-hold-out-the-human-eval-set)
- [`docs/claude-code-prompts.md` § Stage 3b: Hand-labeling tool](../claude-code-prompts.md#stage-3b-hand-labeling-tool)

**Scripts:**
- `data/03a_holdout_eval.py` — stratified holdout sampler.
- `eval/label_tool.py` — interactive CLI for labeling 300 pairs (verdict A/B/T, confidence 1-5, optional notes; supports `--slice`, `--review`, `--random-order` flags; resumable).

**Inputs:**
- `data/pairs/pairs.jsonl` — 2,370 pairs (Stage 2 output).

**Outputs:**
- `data/pairs/eval_set_unlabeled.jsonl` — 300 holdout pairs (240 in-dist + 60 OOD religion). After Stage 3b runs, the same file holds the human labels in place.
- `data/pairs/pairs_to_label.jsonl` — 1,938 pairs destined for Claude labeling in Stage 4. Religion pairs not selected for OOD eval are excluded (preserves the holdout).
- `data/pairs/pairs_unused_religion.jsonl` — 132 religion pairs not selected for OOD eval. Saved for transparency; not used in v1.

**Eval set stratification (300 pairs):**

| Slice | Bucket | Count |
|---|---|---|
| in_dist (10 categories) | clear | 110 |
| in_dist | subtle | 50 |
| in_dist | tracked_vs_alternate | 35 |
| in_dist | tie | 25 |
| in_dist | adversarial | 20 |
| ood_religion | clear | 28 |
| ood_religion | subtle | 12 |
| ood_religion | tracked_vs_alternate | 9 |
| ood_religion | tie | 6 |
| ood_religion | adversarial | 5 |
| **Total** | | **300** |

OOD bucket counts (28/12/9/6/5) mirror the in-dist proportions (110/50/35/25/20) via largest-residual rounding.

**Decisions made:**
- [#10](../project-status.md#key-methodological-decisions-chronological) — Replaced CrowS-Pairs with held-out BBQ category for v1 OOD. CrowS tests "which sentence reflects a stereotype" — different task; some "biased" sentences are factually correct.
- [#11](../project-status.md#key-methodological-decisions-chronological) — Holdout = religion only (single category). Two-axis (religion + disability_status) was considered but dropped 28% of the SFT pool. Religion-only is a 19% drop, keeps DPO closer to primer targets, still gives a defensible "judge never trained on religion bias" story.

**Key outputs:**
- The label tool's `--slice` flag is what made hand-labeling 300 pairs across multiple sessions tractable — batch by `in_dist` or `ood_religion` to avoid context-switching the rubric.
- The 6-10 hours of hand-labeling time is non-negotiable: this is the foundation of every reported metric, including the eval κ that decides whether the trained judge actually got better.
