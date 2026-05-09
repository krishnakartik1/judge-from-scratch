# Stage 5 — Dataset formatting (SFT + DPO)

**Prompt:** [`docs/claude-code-prompts.md` § Stage 5: Format SFT and DPO datasets](../claude-code-prompts.md#stage-5-format-sft-and-dpo-datasets)

**Scripts:**
- `data/05_format_datasets.py` — chat-template-wraps prompts, applies position-swap doubling, filters by confidence, writes both training files.
- `data/synth_rejected.py` — synthesizes "bad rationalization" rejecteds for DPO (Sonnet 4.6 with caching). 70% of DPO rejecteds; the other 30% are simple verdict flips of `sonnet_reasoning`.
- `data/_format_helpers.py` — `build_user_message`, target rendering, `<confidence>` stripping (Sonnet emits 3 tags; the trained judge emits only `<reasoning>` + `<verdict>`).
- `data/judge_system_prompt.md` — system prompt (no `<|think|>` per project invariant).

**Inputs:**
- `data/labeled/labeled_pairs.jsonl` — 1,937 rows (Stage 4 output).

**Outputs:**
- `data/formatted/sft.jsonl` — 3,844 rows (1,922 unique pairs after `confidence ≥ 3` filter, × position-swap = 2).
- `data/formatted/dpo.jsonl` — 2,200 rows (1,100 pool with `confidence ≥ 4` AND `verdict ≠ TIE`, × position-swap; 1,558 synth + 642 verdict-flip rejecteds).
- `data/formatted/synthesis_results.jsonl` — 779 synthesis audits.

**Decisions made:**
- [#13](../project-status.md#key-methodological-decisions-chronological) — Custom `<reasoning>...</reasoning><verdict>...</verdict>` tags. Native Gemma 4 thinking mode is disabled. **Train and infer with the same configuration** — no `<|think|>` token in any system prompt anywhere.
- [#22](../project-status.md#key-methodological-decisions-chronological) — DPO sourcing = synthesis (60-75%) + verdict-flip (25-40%). No cross-check supplement (cross-checker disagreements signal rubric divergence, not weaker-model mistakes).
- [#23](../project-status.md#key-methodological-decisions-chronological) — Final 70/30 synth/flip split. Synth prompt re-tuned mid-run from "~120 reasoning tokens" to "200-300 tokens" to close a verbosity-bias shortcut (v1 produced rejecteds whose bottom decile sat entirely below chosen's bottom decile).

**Cost ledger:** $2.05 / $15 cap (Sonnet 4.6 Batch API at 50% discount).

**Key outputs (verify gates passed):**
- 0 `<|think|>` hits across all 3,844 SFT + 2,200 DPO records.
- 0 `<confidence>` hits in any target/chosen/rejected (stripped per #13).
- 0 verdict-flip violations across all 2,200 DPO rows.
- chosen/rejected median length ratio: 0.90× (under the 1.15× block threshold — verbosity bias closed).

**SFT row schema:**
```
{prompt: <chat-template-wrapped string>, target: "<reasoning>...</reasoning><verdict>{A|B|TIE}</verdict>"}
```

**DPO row schema:**
```
{prompt, chosen, rejected}
```
where `chosen` is the cleaned Sonnet output and `rejected` is either a synthesized failure-mode rationalization (verbose hedging / surface engagement / stereotype-aligned reasoning / length asymmetry) or a verdict flip of `chosen`.
