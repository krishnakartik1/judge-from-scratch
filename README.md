# REVAL Judge

Fine-tuning Gemma 3 4B into a specialized bias evaluation judge for the
REVAL framework. See `docs/fine-tuning-primer.md` for the full plan.


Initial candidate generation produced 526 biased candidates (8.8% of 6,000), aligning with documented BBQ bias rates for similar-scale open models. Pair distribution was rebalanced from the planned 45% clear-bias share to 30%, with the freed allocation moved to bias-vs-bias pairs that teach relative-severity judgment.

Pair construction was constrained by BBQ's structure: each question has a single tracked stereotype target, making 'two biased responses on the same question' structurally identical. The pair distribution was redesigned to substitute 'tracked-bias vs alternate-bias' (biased vs. partially-biased-along-different-axis) for the impossible 'bias-vs-bias same-target' category. Final dataset: 2,400 pairs, with bias categories validated against cross-model supply.