"""Stage 2 pair construction — assemble judge-training pairs from enriched candidates.

Reads ``data/raw/candidates_enriched.jsonl`` (Stage 1.5 output) and
``data/raw/bbq_sample.jsonl`` (Stage 0 output). Emits one record per pair
to ``data/pairs/pairs.jsonl`` and a metadata sidecar to ``data/pairs/pairs.meta.json``.

Five pair categories are populated, with targets matching cross-model supply
observed in the enriched candidate file:

    clear_bias_vs_clean       (~530)  unhedged-biased x neutral, cross-model
    subtle_bias_vs_clean      (~520)  hedged-biased x neutral, same-model OK
    tracked_bias_vs_alternate (~600)  biased x incorrect_other, cross-model
    both_clean_tie            (~400)  neutral x neutral, cross-model
    adversarial               (~350)  length-asym + confidence-asym, cross-model

Pre-pass design notes — why these targets, and not the doc's 4,500:

  Only 369 of 1,500 questions yielded both a biased and a neutral candidate.
  Strict cross-model pair supply caps the biased-side buckets at ~537/521 each.
  The literal "bias-vs-bias same-target different-severity" bucket is
  structurally impossible because all four candidates of one question share
  the same context_condition, hence the same bias_severity tag, and any two
  biased candidates share chosen_idx == target_label. We replace it with
  "tracked_bias_vs_alternate" (biased + incorrect_other, same question) which
  matches the worked example in docs/fine-tuning-primer.md (Llama-1
  "grandfather" vs Mistral-1 "grandson"). Per-pair chosen/rejected is NOT
  decided here — Stage 3 (Claude labeling) makes that call.

Resumable: a re-run skips ``pair_id`` values already in the output. Determinism
is governed by ``--seed`` (default 42); given a fixed input file, fixed targets,
and fixed seed, two fresh runs produce identical pairs.

Usage:
    uv run python data/02_construct_pairs.py [--seed N] [--max-per-q-per-bucket K]
                                              [--dry-run] [--force]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import random
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from scripts.common import already_processed, jsonl_append, jsonl_read

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PAIRS_DIR = REPO_ROOT / "data" / "pairs"
BBQ_PATH = RAW_DIR / "bbq_sample.jsonl"
ENRICHED_PATH = RAW_DIR / "candidates_enriched.jsonl"
PAIRS_PATH = PAIRS_DIR / "pairs.jsonl"
META_PATH = PAIRS_DIR / "pairs.meta.json"

# Hedge vocabulary used to identify "subtle bias" — biased responses that
# wrap a stereotype-aligned answer in tentative language. Used both for the
# subtle-vs-clean bucket and for the biased-side check in confidence-asym
# adversarial pairs (a hedged biased response disqualifies a pair from being
# "confident biased + uncertain neutral").
HEDGE_TOKENS: tuple[str, ...] = (
    "probably", "might", "tends to", "often", "usually",
    "more likely", "perhaps", "maybe", "could be", "seems",
    "appears", "suggests", "i think", "i believe", "i'd guess",
    "i would assume", "may have", "might have", "could have",
    "would tend", "somewhat", "rather", "fairly", "quite possibly",
)

# Explicit uncertainty markers — broader than HEDGE_TOKENS. A response with
# any of these signals "the speaker explicitly declines to commit," which is
# the signature we want for the neutral side of confidence-asym pairs.
UNCERTAINTY_TOKENS: tuple[str, ...] = (
    "not sure", "hard to say", "depends", "without more info",
    "could be either", "difficult to tell",
)


def _build_token_regex(tokens: tuple[str, ...]) -> re.Pattern[str]:
    """Compile a case-insensitive whole-token regex from a phrase list.

    Internal whitespace is matched permissively (\\s+) so "i  think" or
    "tends    to" are still hits.
    """
    parts = [t.replace(" ", r"\s+") for t in tokens]
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.IGNORECASE)


HEDGE_RE: re.Pattern[str] = _build_token_regex(HEDGE_TOKENS)
UNCERTAINTY_RE: re.Pattern[str] = _build_token_regex(UNCERTAINTY_TOKENS)

# Order matters: cross-bucket dedup is first-wins, so the strict-subset
# buckets must run before their supersets. adversarial confidence-asym is
# a strict subset of clear (biased-unhedged × neutral), and adversarial
# length-asym overlaps with both clear and subtle. Putting adversarial
# first keeps it from being starved by upstream buckets eating its supply.
PAIR_CATEGORIES: tuple[str, ...] = (
    "adversarial",
    "clear_bias_vs_clean",
    "subtle_bias_vs_clean",
    "tracked_bias_vs_alternate",
    "both_clean_tie",
)

# Target ceilings. Two of these (subtle, adversarial confidence_asym) are
# heuristic-constrained rather than supply-constrained — even with a broader
# hedge regex the pool fluctuates with vocab choice. Those are dynamically
# capped at min(ceiling, round(0.7 * pool)) at runtime so the dataset always
# leaves headroom rather than draining the pool. The other three are
# genuinely supply-constrained and the ceilings reflect what the data yields.
PAIR_TARGETS: dict[str, int] = {
    # High-value buckets: pools doubled at 3,000-question expansion, scale up.
    "clear_bias_vs_clean": 800,
    "subtle_bias_vs_clean": 550,
    "adversarial": 250,
    # tracked stays at the true supply ceiling (~220 cross-model at this size).
    "tracked_bias_vs_alternate": 220,
    # Tie pairs hit diminishing pedagogical value; capped scaling.
    "both_clean_tie": 550,
}

# Adversarial sub-targets within the 250-pair bucket. confidence_asym is the
# tighter heuristic, length_asym the looser one; both are independent (a
# shortfall in one is NOT compensated by the other).
ADVERSARIAL_SPLIT: dict[str, int] = {"length_asym": 120, "confidence_asym": 130}

# Heuristic-constrained buckets get pool-aware capping at this fraction. Keeps
# us from emptying a heuristic pool just because it happens to be exactly
# target-sized — leaves slack for vocabulary refinement without re-running.
HEURISTIC_POOL_FILL_RATE: float = 0.7

# A bucket that achieves <(1 - SHORTFALL_THRESHOLD) of its target is flagged
# as a hard failure; the script aborts before writing unless --force.
SHORTFALL_THRESHOLD: float = 0.10

# Length-asym predicate floor: avoids degenerate "B." style replies in the
# shorter slot. Tuned empirically against the candidate file.
LENGTH_ASYM_RATIO: int = 2
LENGTH_ASYM_FLOOR_WC: int = 5


# -----------------------------------------------------------------------------
# Pure helpers
# -----------------------------------------------------------------------------


def classify_level(bias_classification: str | None) -> str:
    """Map Stage 1.5's ``bias_classification`` to a pair-construction level.

    The 1.5 classifier already handles the edge cases (e.g.
    ``answer_label == target_label`` where "correct" wins over "biased").
    Stage 2 must NOT re-derive bias from raw labels — it trusts the field
    written by 1.5. Mapping:

        bias_classification == "biased"         -> "biased"
        bias_classification == "correct"        -> "neutral"
        bias_classification == "incorrect_other"-> "unclear"
        bias_classification == "parse_failed"   -> "excluded"
        anything else (None, unknown)           -> "excluded"

    Candidates with level "excluded" never enter any pair pool.
    """
    if bias_classification == "biased":
        return "biased"
    if bias_classification == "correct":
        return "neutral"
    if bias_classification == "incorrect_other":
        return "unclear"
    return "excluded"


def is_hedged(text: str | None) -> bool:
    """True iff ``text`` contains any hedging marker from ``HEDGE_RE``."""
    if not text:
        return False
    return HEDGE_RE.search(text) is not None


def is_uncertain(text: str | None) -> bool:
    """True iff ``text`` contains a hedge OR an explicit uncertainty marker.

    Wider than ``is_hedged`` — used for the neutral side of confidence-asym
    pairs where "the speaker declines to commit" is the signal we want.
    """
    if not text:
        return False
    return HEDGE_RE.search(text) is not None or UNCERTAINTY_RE.search(text) is not None


def word_count(text: str | None) -> int:
    """Whitespace-delimited word count. Returns 0 on None/empty."""
    if not text:
        return 0
    return len(text.split())


def build_question_text(bbq_record: dict[str, Any]) -> str:
    """Concatenate ``context`` and ``question`` for the judge-prompt input."""
    context = (bbq_record.get("context") or "").strip()
    question = (bbq_record.get("question") or "").strip()
    if context and question:
        return f"{context} {question}"
    return context or question


def make_pair_id(question_id: str, model_a: str, model_b: str) -> str:
    """Stable swap-symmetric 16-hex-char id for (question, model_a, model_b).

    Sorted on the (qid, model) pair so the id does not depend on which slot
    a candidate lands in. Ensures that resume after a crash sees the same id
    regardless of A/B orientation.
    """
    members = sorted([f"{question_id}::{model_a}", f"{question_id}::{model_b}"])
    digest = hashlib.sha1("|".join(members).encode("utf-8")).hexdigest()
    return digest[:16]


# -----------------------------------------------------------------------------
# Annotation
# -----------------------------------------------------------------------------


def load_bbq_index(path: Path | str) -> dict[str, dict[str, Any]]:
    """Build ``{question_id: bbq_record}`` from a BBQ JSONL file."""
    index: dict[str, dict[str, Any]] = {}
    for record in jsonl_read(path):
        qid = record["question_id"]
        if qid in index:
            raise ValueError(
                f"Duplicate question_id in BBQ sample: {qid!r}. "
                "Stage 0 invariant violated; refusing to silently overwrite."
            )
        index[qid] = record
    return index


def annotate_candidates(
    enriched: list[dict[str, Any]],
    bbq_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return candidates augmented with derived fields used by the sampler.

    Adds:
        _level          str  — biased / neutral / unclear
        _hedged         bool — HEDGE_RE hit on response
        _uncertain      bool — HEDGE_RE OR UNCERTAINTY_RE hit on response
        _wc             int  — whitespace word count of response
        _question_text  str  — context + " " + question
        _bbq_category   str  — BBQ category (age, gender, ...) for the output

    Candidates whose ``question_id`` is not in ``bbq_index`` are dropped with
    a logged warning; this should never happen if Stage 1.5 ran cleanly.
    """
    out: list[dict[str, Any]] = []
    for cand in enriched:
        qid = cand["question_id"]
        bbq = bbq_index.get(qid)
        if bbq is None:
            logger.warning("Candidate references unknown question_id %r; dropping.", qid)
            continue
        annotated = dict(cand)
        annotated["_level"] = classify_level(cand.get("bias_classification"))
        if annotated["_level"] == "excluded":
            # Parse fails / unknown classifications never participate in pairs.
            continue
        annotated["_hedged"] = is_hedged(cand.get("response"))
        annotated["_uncertain"] = is_uncertain(cand.get("response"))
        annotated["_wc"] = word_count(cand.get("response"))
        annotated["_question_text"] = build_question_text(bbq)
        annotated["_bbq_category"] = bbq.get("category", "")
        out.append(annotated)
    return out


# -----------------------------------------------------------------------------
# Per-bucket pool generators
# -----------------------------------------------------------------------------
#
# Each generator takes the four annotated candidates for ONE question and
# returns a list of unordered (rec_a, rec_b) tuples that satisfy the bucket's
# predicate. The sampler shuffles, dedupes, and caps these per-question pools
# to fill global targets.
#
# Cross-model rule: every generator except `candidate_pairs_subtle` requires
# `rec_a["model"] != rec_b["model"]`. Subtle permits same-model so that style
# is held constant while the bias signal varies, per the user's spec.


def candidate_pairs_clear(cands: list[dict[str, Any]]) -> list[tuple[dict, dict]]:
    """Unhedged biased x neutral, cross-model only."""
    biased = [c for c in cands if c["_level"] == "biased" and not c["_hedged"]]
    neutral = [c for c in cands if c["_level"] == "neutral"]
    pairs: list[tuple[dict, dict]] = []
    for b in biased:
        for n in neutral:
            if b["model"] != n["model"]:
                pairs.append((b, n))
    return pairs


def candidate_pairs_subtle(cands: list[dict[str, Any]]) -> list[tuple[dict, dict]]:
    """Hedged biased x neutral. Same-model permitted (style control)."""
    biased = [c for c in cands if c["_level"] == "biased" and c["_hedged"]]
    neutral = [c for c in cands if c["_level"] == "neutral"]
    pairs: list[tuple[dict, dict]] = []
    for b in biased:
        for n in neutral:
            pairs.append((b, n))
    return pairs


def candidate_pairs_tracked_vs_alternate(
    cands: list[dict[str, Any]],
) -> list[tuple[dict, dict]]:
    """Biased (chose stereotype) x unclear-with-non-null-answer, cross-model.

    Replaces the structurally-impossible "two biased candidates with different
    target answers" formulation. Both responses are wrong; one is wrong in the
    stereotype-aligned direction and the other is wrong in some other direction.
    Stage 3 decides which is "more biased."

    Candidates at level "unclear" are bias_classification == "incorrect_other",
    which by 1.5's definition already excludes parse failures.
    """
    biased = [c for c in cands if c["_level"] == "biased"]
    other = [c for c in cands if c["_level"] == "unclear"]
    pairs: list[tuple[dict, dict]] = []
    for b in biased:
        for o in other:
            if b["model"] != o["model"]:
                pairs.append((b, o))
    return pairs


def candidate_pairs_tie(cands: list[dict[str, Any]]) -> list[tuple[dict, dict]]:
    """Both-clean: neutral x neutral, cross-model.

    Restricted to neutral pairs per the user's spec — biased x biased pairs
    on the same question are degenerate (same chosen_idx) and unclear x unclear
    is too noisy a "tie" signal.
    """
    neutral = [c for c in cands if c["_level"] == "neutral"]
    pairs: list[tuple[dict, dict]] = []
    for i in range(len(neutral)):
        for j in range(i + 1, len(neutral)):
            a, b = neutral[i], neutral[j]
            if a["model"] != b["model"]:
                pairs.append((a, b))
    return pairs


def candidate_pairs_adversarial_length(
    cands: list[dict[str, Any]],
) -> list[tuple[dict, dict]]:
    """Cross-model biased (longer) x neutral (shorter), wc(biased) >= 2*wc(neutral)
    AND wc(neutral) >= 5.

    Adversarial intent: a verbose stereotype-aligned response paired with a
    terse correct response. A judge with verbosity bias preferentially picks
    the longer answer regardless of substance — this pair surfaces that.
    The biased side is always the longer one to make the trap explicit.
    """
    biased = [c for c in cands if c["_level"] == "biased"]
    neutral = [c for c in cands if c["_level"] == "neutral"]
    pairs: list[tuple[dict, dict]] = []
    for b in biased:
        for n in neutral:
            if b["model"] == n["model"]:
                continue
            if n["_wc"] < LENGTH_ASYM_FLOOR_WC:
                continue
            if b["_wc"] < LENGTH_ASYM_RATIO * n["_wc"]:
                continue
            pairs.append((b, n))
    return pairs


def candidate_pairs_adversarial_confidence(
    cands: list[dict[str, Any]],
) -> list[tuple[dict, dict]]:
    """Cross-model biased (unhedged, confident) x neutral (hedged).

    Adversarial intent: the stereotype-aligned response sounds certain ("X is
    Y, full stop"); the correct deferral response sounds tentative ("it's
    probably impossible to say"). A judge that conflates confidence with
    correctness will pick the biased side.
    """
    biased = [c for c in cands if c["_level"] == "biased" and not c["_hedged"]]
    neutral = [c for c in cands if c["_level"] == "neutral" and c["_uncertain"]]
    pairs: list[tuple[dict, dict]] = []
    for b in biased:
        for n in neutral:
            if b["model"] != n["model"]:
                pairs.append((b, n))
    return pairs


# Bucket name -> generator function. Adversarial is special: it composes two
# generators with sub-budgets; see _fill_adversarial.
BUCKET_GENERATORS: dict[
    str, Callable[[list[dict[str, Any]]], list[tuple[dict, dict]]]
] = {
    "clear_bias_vs_clean": candidate_pairs_clear,
    "subtle_bias_vs_clean": candidate_pairs_subtle,
    "tracked_bias_vs_alternate": candidate_pairs_tracked_vs_alternate,
    "both_clean_tie": candidate_pairs_tie,
}


# -----------------------------------------------------------------------------
# Sampling driver
# -----------------------------------------------------------------------------


def _fill_simple_bucket(
    cands_by_question: dict[str, list[dict[str, Any]]],
    generator: Callable[[list[dict[str, Any]]], list[tuple[dict, dict]]],
    target: int,
    rng: random.Random,
    max_per_q: int,
    seen_pair_ids: set[str],
    picked_pair_ids: set[str],
) -> tuple[list[tuple[dict, dict]], int]:
    """Fill one bucket. Returns (picked, pool_total).

    Iterates questions in shuffled order; within each question, generates all
    candidate pairs, shuffles them, and contributes up to ``max_per_q`` to the
    bucket. ``picked_pair_ids`` is the cross-bucket dedup set: a pair_id taken
    by an earlier bucket is not reused.
    """
    picked: list[tuple[dict, dict]] = []
    if target <= 0:
        return picked, 0
    qids = list(cands_by_question.keys())
    rng.shuffle(qids)
    pool_total = 0
    for qid in qids:
        per_q = generator(cands_by_question[qid])
        pool_total += len(per_q)
        if not per_q:
            continue
        rng.shuffle(per_q)
        contributed = 0
        for a, b in per_q:
            if contributed >= max_per_q:
                break
            pid = make_pair_id(qid, a["model"], b["model"])
            if pid in seen_pair_ids or pid in picked_pair_ids:
                continue
            picked.append((a, b))
            picked_pair_ids.add(pid)
            contributed += 1
            if len(picked) >= target:
                return picked, pool_total
    return picked, pool_total


def _fill_adversarial(
    cands_by_question: dict[str, list[dict[str, Any]]],
    target: int,
    split: dict[str, int],
    rng: random.Random,
    max_per_q: int,
    seen_pair_ids: set[str],
    picked_pair_ids: set[str],
) -> tuple[list[tuple[dict, dict]], dict[str, int], dict[str, int]]:
    """Fill the adversarial bucket as length-asym + confidence-asym.

    The two sub-buckets are independent — they verify different judge
    failure modes (verbosity bias vs confidence bias), so a shortfall in
    one is NOT compensated by overflow from the other. Each runs to its
    own sub-target from ``split``.

    The ``target`` argument acts as a sanity check: if it's <= 0, both
    sub-buckets are skipped. Otherwise the sub-targets in ``split`` drive
    the actual fill (caller's responsibility to keep them coherent).

    Returns (picked, sub_achieved, sub_pool_totals) where sub_achieved /
    sub_pool_totals are keyed by "length_asym" / "confidence_asym".
    """
    if target <= 0:
        return (
            [],
            {"length_asym": 0, "confidence_asym": 0},
            {"length_asym": 0, "confidence_asym": 0},
        )

    conf_picked, conf_pool = _fill_simple_bucket(
        cands_by_question,
        candidate_pairs_adversarial_confidence,
        split["confidence_asym"],
        rng,
        max_per_q,
        seen_pair_ids,
        picked_pair_ids,
    )
    length_picked, length_pool = _fill_simple_bucket(
        cands_by_question,
        candidate_pairs_adversarial_length,
        split["length_asym"],
        rng,
        max_per_q,
        seen_pair_ids,
        picked_pair_ids,
    )
    sub_achieved = {
        "length_asym": len(length_picked),
        "confidence_asym": len(conf_picked),
    }
    sub_pool = {"length_asym": length_pool, "confidence_asym": conf_pool}
    return conf_picked + length_picked, sub_achieved, sub_pool


def measure_pools(
    cands_by_question: dict[str, list[dict[str, Any]]],
) -> dict[str, int]:
    """Count maximum candidate pairs per bucket / sub-bucket without sampling.

    Returns counts keyed by:
        clear_bias_vs_clean, subtle_bias_vs_clean, tracked_bias_vs_alternate,
        both_clean_tie, adversarial_length, adversarial_confidence

    The numbers are upper bounds — sampling caps per-question contribution
    via ``max_per_q_per_bucket`` and dedups across buckets, so the actual
    achievable count is smaller. But the upper bound is enough to drive the
    dynamic 0.7-of-pool target cap for heuristic-constrained buckets.
    """
    pools: dict[str, int] = {
        "clear_bias_vs_clean": 0,
        "subtle_bias_vs_clean": 0,
        "tracked_bias_vs_alternate": 0,
        "both_clean_tie": 0,
        "adversarial_length": 0,
        "adversarial_confidence": 0,
    }
    for cands in cands_by_question.values():
        pools["clear_bias_vs_clean"] += len(candidate_pairs_clear(cands))
        pools["subtle_bias_vs_clean"] += len(candidate_pairs_subtle(cands))
        pools["tracked_bias_vs_alternate"] += len(
            candidate_pairs_tracked_vs_alternate(cands)
        )
        pools["both_clean_tie"] += len(candidate_pairs_tie(cands))
        pools["adversarial_length"] += len(candidate_pairs_adversarial_length(cands))
        pools["adversarial_confidence"] += len(
            candidate_pairs_adversarial_confidence(cands)
        )
    return pools


def apply_dynamic_caps(
    base_targets: dict[str, int],
    base_split: dict[str, int],
    pools: dict[str, int],
    fill_rate: float,
) -> tuple[dict[str, int], dict[str, int]]:
    """Cap heuristic-constrained targets at min(base, round(fill_rate * pool)).

    Specifically caps:
        subtle_bias_vs_clean (capped against pools["subtle_bias_vs_clean"])
        adversarial confidence_asym sub-target (capped against
        pools["adversarial_confidence"])
        adversarial_length sub-target (capped against
        pools["adversarial_length"]) — supply-constrained, but capping by
        the same rule keeps the overall adversarial target consistent.

    Returns (effective_targets, effective_split). The adversarial bucket
    target is recomputed as the sum of the (possibly capped) sub-targets.
    """
    effective = dict(base_targets)
    effective_split = dict(base_split)

    subtle_cap = round(fill_rate * pools.get("subtle_bias_vs_clean", 0))
    effective["subtle_bias_vs_clean"] = min(base_targets["subtle_bias_vs_clean"], subtle_cap)

    conf_cap = round(fill_rate * pools.get("adversarial_confidence", 0))
    effective_split["confidence_asym"] = min(base_split["confidence_asym"], conf_cap)

    length_cap = round(fill_rate * pools.get("adversarial_length", 0))
    effective_split["length_asym"] = min(base_split["length_asym"], length_cap)

    # Adversarial bucket total = sum of (capped) sub-targets, BUT bounded by
    # the base ceiling. A caller that sets base["adversarial"] = 0 disables
    # the bucket regardless of pool sizes.
    sub_sum = effective_split["length_asym"] + effective_split["confidence_asym"]
    effective["adversarial"] = min(base_targets.get("adversarial", sub_sum), sub_sum)
    if effective["adversarial"] < sub_sum:
        # Base ceiling is tighter than the split caps. Scale split down so
        # the sub-targets sum exactly to the bucket's effective ceiling.
        scale = effective["adversarial"] / sub_sum if sub_sum > 0 else 0.0
        effective_split["length_asym"] = round(effective_split["length_asym"] * scale)
        effective_split["confidence_asym"] = (
            effective["adversarial"] - effective_split["length_asym"]
        )
    return effective, effective_split


def sample_pairs(
    cands_by_question: dict[str, list[dict[str, Any]]],
    targets: dict[str, int],
    rng: random.Random,
    max_per_q_per_bucket: int = 4,
    seen_pair_ids: set[str] | None = None,
    adversarial_split: dict[str, int] | None = None,
) -> tuple[
    dict[str, list[tuple[dict, dict]]],
    dict[str, dict[str, int]],
    dict[str, int],
]:
    """Run the full sampling loop.

    Returns:
        picked_by_bucket: bucket name -> list of (rec_a, rec_b) pairs.
        diagnostics: bucket name -> dict with keys
            target, achieved, pool_total, cross_model, same_model.
        adversarial_sub_achieved: {"length_asym": N, "confidence_asym": N}.
    """
    if seen_pair_ids is None:
        seen_pair_ids = set()
    if adversarial_split is None:
        adversarial_split = ADVERSARIAL_SPLIT
    picked_pair_ids: set[str] = set()
    picked_by_bucket: dict[str, list[tuple[dict, dict]]] = {}
    diagnostics: dict[str, dict[str, int]] = {}
    adversarial_sub: dict[str, int] = {"length_asym": 0, "confidence_asym": 0}
    adversarial_pool: dict[str, int] = {"length_asym": 0, "confidence_asym": 0}

    for bucket in PAIR_CATEGORIES:
        target = targets[bucket]
        if bucket == "adversarial":
            picked, sub_achieved, sub_pool = _fill_adversarial(
                cands_by_question,
                target,
                adversarial_split,
                rng,
                max_per_q_per_bucket,
                seen_pair_ids,
                picked_pair_ids,
            )
            adversarial_sub = sub_achieved
            adversarial_pool = sub_pool
            pool_total = sub_pool["length_asym"] + sub_pool["confidence_asym"]
        else:
            picked, pool_total = _fill_simple_bucket(
                cands_by_question,
                BUCKET_GENERATORS[bucket],
                target,
                rng,
                max_per_q_per_bucket,
                seen_pair_ids,
                picked_pair_ids,
            )
        cross = sum(1 for a, b in picked if a["model"] != b["model"])
        same = len(picked) - cross
        picked_by_bucket[bucket] = picked
        diagnostics[bucket] = {
            "target": target,
            "achieved": len(picked),
            "pool_total": pool_total,
            "cross_model": cross,
            "same_model": same,
        }

    # Stash adversarial sub-pool sizes inside the diagnostics for the report.
    diagnostics["adversarial"]["length_pool"] = adversarial_pool["length_asym"]
    diagnostics["adversarial"]["confidence_pool"] = adversarial_pool["confidence_asym"]
    return picked_by_bucket, diagnostics, adversarial_sub


# -----------------------------------------------------------------------------
# Output record + shortfall guard
# -----------------------------------------------------------------------------


def build_output_record(
    rec_a: dict[str, Any],
    rec_b: dict[str, Any],
    pair_category: str,
    rng: random.Random,
) -> dict[str, Any]:
    """Build the JSONL record for one pair, with random A/B slot assignment.

    Output schema is locked. Ground-truth fields (chosen_idx, target_label,
    answer_label, bias_severity, bias_classification, context_condition) are
    deliberately excluded — Stage 3 (Claude labeling) must not see them.
    """
    if rng.random() < 0.5:
        a, b = rec_a, rec_b
    else:
        a, b = rec_b, rec_a
    pair_id = make_pair_id(a["question_id"], a["model"], b["model"])
    return {
        "pair_id": pair_id,
        "question_id": a["question_id"],
        "question_text": a["_question_text"],
        "bias_category": a["_bbq_category"],
        "response_a": {
            "model": a["model"],
            "text": a.get("response", ""),
            "suspected_bias_level": a["_level"],
        },
        "response_b": {
            "model": b["model"],
            "text": b.get("response", ""),
            "suspected_bias_level": b["_level"],
        },
        "pair_category": pair_category,
    }


def check_shortfalls(
    achieved: dict[str, int],
    targets: dict[str, int],
    threshold: float,
) -> dict[str, float]:
    """Return ``{bucket: shortfall_fraction}`` for buckets whose shortfall
    exceeds ``threshold``. Empty dict means every bucket cleared the bar.
    """
    out: dict[str, float] = {}
    for bucket, target in targets.items():
        if target <= 0:
            continue
        got = achieved.get(bucket, 0)
        shortfall = (target - got) / target
        if shortfall > threshold:
            out[bucket] = shortfall
    return out


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def _truncate(text: str | None, n: int = 200) -> str:
    if text is None:
        return ""
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."


def print_report(
    diagnostics: dict[str, dict[str, int]],
    adversarial_sub: dict[str, int],
    examples: dict[str, dict[str, Any] | None],
    shortfalls: dict[str, float],
    effective_split: dict[str, int] | None = None,
) -> None:
    """Print the five-section post-sampling report."""
    bar = "=" * 72
    print()
    print(bar)
    print("STAGE 2 PAIR-CONSTRUCTION REPORT")
    print(bar)

    # [1] bucket distribution
    print()
    print("[1] Bucket distribution:")
    header = (
        f"    {'bucket':<28s}{'target':>8s}{'achieved':>10s}{'pct':>8s}{'pool':>10s}"
    )
    print(header)
    print("    " + "-" * (len(header) - 4))
    total_target = 0
    total_achieved = 0
    for bucket in PAIR_CATEGORIES:
        d = diagnostics[bucket]
        pct = (d["achieved"] / d["target"] * 100) if d["target"] else 0.0
        pool = d.get("pool_total", 0)
        print(
            f"    {bucket:<28s}{d['target']:>8d}{d['achieved']:>10d}"
            f"{pct:>7.1f}%{pool:>10d}"
        )
        total_target += d["target"]
        total_achieved += d["achieved"]
    total_pct = (total_achieved / total_target * 100) if total_target else 0.0
    print("    " + "-" * (len(header) - 4))
    print(
        f"    {'TOTAL':<28s}{total_target:>8d}{total_achieved:>10d}{total_pct:>7.1f}%"
    )

    # [2] adversarial split
    print()
    print("[2] Adversarial sub-split achieved:")
    adv = diagnostics.get("adversarial", {})
    split_to_show = effective_split if effective_split is not None else ADVERSARIAL_SPLIT
    for sub, sub_target in split_to_show.items():
        got = adversarial_sub.get(sub, 0)
        pool = adv.get(f"{sub.split('_')[0]}_pool", 0)
        print(f"    {sub:<28s}{sub_target:>8d}{got:>10d}{pool:>10d} (pool)")

    # [3] cross-model diversity
    print()
    print("[3] Cross-model diversity:")
    print(
        f"    {'bucket':<28s}{'cross':>8s}{'same':>8s}{'same%':>8s}"
    )
    print("    " + "-" * 52)
    for bucket in PAIR_CATEGORIES:
        d = diagnostics[bucket]
        cross = d["cross_model"]
        same = d["same_model"]
        total = cross + same
        same_pct = (same / total * 100) if total else 0.0
        print(f"    {bucket:<28s}{cross:>8d}{same:>8d}{same_pct:>7.1f}%")

    # [4] sample pairs
    print()
    print("[4] Sample pairs (one per category):")
    for bucket in PAIR_CATEGORIES:
        ex = examples.get(bucket)
        print(f"    ----- {bucket} -----")
        if ex is None:
            print("    (none)")
            continue
        print(f"    pair_id:       {ex['pair_id']}")
        print(f"    question_id:   {ex['question_id']}")
        print(f"    bias_category: {ex['bias_category']}")
        print(f"    question:      {_truncate(ex['question_text'], 200)}")
        ra = ex["response_a"]
        rb = ex["response_b"]
        print(
            f"    response_a:    [{ra['model']} | {ra['suspected_bias_level']}] "
            f"{_truncate(ra['text'], 200)}"
        )
        print(
            f"    response_b:    [{rb['model']} | {rb['suspected_bias_level']}] "
            f"{_truncate(rb['text'], 200)}"
        )

    # [5] shortfall check
    print()
    print("[5] Shortfall check (>10% short flagged):")
    if not shortfalls:
        print("    OK — every bucket within 10% of target.")
    else:
        for bucket, frac in shortfalls.items():
            d = diagnostics[bucket]
            print(
                f"    WARNING {bucket}: target={d['target']} "
                f"achieved={d['achieved']} short={frac * 100:.1f}%"
            )
    print(bar)


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Tmp + rename for the metadata sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    tmp.replace(path)


def existing_pair_categories(path: Path) -> dict[str, int]:
    """Count pairs already in ``pairs.jsonl`` grouped by ``pair_category``.

    Used to make the shortfall guard resume-aware: a bucket whose target was
    met on a prior run should not re-trip the guard just because no NEW pairs
    were produced on the current run.
    """
    counts: dict[str, int] = dict.fromkeys(PAIR_CATEGORIES, 0)
    if not path.exists():
        return counts
    for record in jsonl_read(path):
        cat = record.get("pair_category")
        if cat in counts:
            counts[cat] += 1
    return counts


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's contents, hex-encoded."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2 — construct judge-training pairs from enriched candidates. "
            "Resumable; deterministic given --seed."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="RNG seed for question/pair shuffles and A/B swap (default 42).",
    )
    parser.add_argument(
        "--max-per-q-per-bucket",
        type=int,
        default=4,
        metavar="K",
        help=(
            "Cap on pairs contributed per question per bucket "
            "(prevents one rich question from dominating; default 4)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run sampling and print the report without writing pairs.jsonl.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Bypass the 10%% shortfall guard and write whatever was achieved. "
            "Recorded in pairs.meta.json."
        ),
    )
    parser.add_argument(
        "--qc-subtle",
        type=int,
        default=0,
        metavar="N",
        nargs="?",
        const=20,
        help=(
            "Quality-check mode: print N random subtle-bucket pairs (default 20 "
            "when --qc-subtle is supplied without a value) and exit without "
            "writing pairs.jsonl. Lets you eyeball the hedge-vocabulary heuristic "
            "before committing to the dataset."
        ),
    )
    return parser.parse_args(argv)


def print_qc_subtle(
    picks: list[tuple[dict[str, Any], dict[str, Any]]],
    n: int,
    rng: random.Random,
) -> None:
    """Print N random pairs from the subtle bucket for human review."""
    bar = "=" * 72
    print()
    print(bar)
    print(f"QC: {min(n, len(picks))} random subtle_bias_vs_clean pairs")
    print(bar)
    sample_idx = rng.sample(range(len(picks)), min(n, len(picks)))
    for i, idx in enumerate(sample_idx, start=1):
        a, b = picks[idx]
        # Identify which side carries the biased label so the eyeball test
        # focuses on the hedged-bias text.
        biased = a if a["_level"] == "biased" else b
        clean = b if biased is a else a
        print(f"\n----- pair {i}/{len(sample_idx)} -----")
        print(f"question_id:   {biased['question_id']}")
        print(f"category:      {biased['_bbq_category']}")
        print(f"question:      {_truncate(biased['_question_text'], 240)}")
        print(
            f"BIASED ({biased['model']}): {_truncate(biased.get('response', ''), 320)}"
        )
        print(
            f"CLEAN  ({clean['model']}): {_truncate(clean.get('response', ''), 320)}"
        )
    print(bar)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not BBQ_PATH.exists():
        logger.error("%s not found. Run data/00_sample_bbq.py first.", BBQ_PATH)
        return 2
    if not ENRICHED_PATH.exists():
        logger.error(
            "%s not found. Run data/01b_enrich_candidates.py first.", ENRICHED_PATH
        )
        return 2

    bbq_index = load_bbq_index(BBQ_PATH)
    logger.info("Loaded %d BBQ rows.", len(bbq_index))

    enriched = list(jsonl_read(ENRICHED_PATH))
    logger.info("Loaded %d enriched candidates.", len(enriched))

    annotated = annotate_candidates(enriched, bbq_index)
    cands_by_question: dict[str, list[dict[str, Any]]] = {}
    for cand in annotated:
        cands_by_question.setdefault(cand["question_id"], []).append(cand)
    logger.info("Grouped into %d questions.", len(cands_by_question))

    is_qc = args.qc_subtle > 0
    if args.dry_run or is_qc:
        seen: set[str] = set()
        existing_counts = dict.fromkeys(PAIR_CATEGORIES, 0)
    else:
        seen = already_processed(PAIRS_PATH, "pair_id")
        existing_counts = existing_pair_categories(PAIRS_PATH)
        if seen:
            logger.info(
                "Resume: %d pair_ids already written; counts per bucket=%s",
                len(seen),
                existing_counts,
            )

    # Heuristic-constrained buckets get capped at 0.7 * pool to leave headroom
    # against vocabulary tweaks. Supply-constrained ones use base targets.
    pools = measure_pools(cands_by_question)
    logger.info("Measured pool sizes: %s", pools)
    effective_targets, effective_split = apply_dynamic_caps(
        PAIR_TARGETS, ADVERSARIAL_SPLIT, pools, HEURISTIC_POOL_FILL_RATE
    )
    logger.info("Effective targets after caps: %s", effective_targets)
    logger.info("Effective adversarial split: %s", effective_split)

    # Remaining target per bucket on this run; the shortfall guard then sees
    # the lifetime achieved (= existing + newly-picked) so a met-on-prior-run
    # bucket does not falsely trip the guard.
    remaining_targets = {
        b: max(0, effective_targets[b] - existing_counts.get(b, 0))
        for b in PAIR_CATEGORIES
    }

    rng = random.Random(args.seed)
    picked_by_bucket, diagnostics, adversarial_sub = sample_pairs(
        cands_by_question,
        remaining_targets,
        rng,
        max_per_q_per_bucket=args.max_per_q_per_bucket,
        seen_pair_ids=seen,
        adversarial_split=effective_split,
    )

    # QC mode: print 20 subtle pairs and exit, no shortfall check, no write.
    if is_qc:
        subtle_picks = picked_by_bucket.get("subtle_bias_vs_clean", [])
        if not subtle_picks:
            logger.error("No subtle pairs were picked — nothing to QC.")
            return 7
        qc_rng = random.Random(args.seed + 999)
        print_qc_subtle(subtle_picks, args.qc_subtle, qc_rng)
        logger.info(
            "--qc-subtle: printed %d subtle pairs; exiting without writing.",
            min(args.qc_subtle, len(subtle_picks)),
        )
        return 0

    # Lifetime achieved = existing + newly-picked. Diagnostics' "target" field
    # is reset to the effective (post-cap) target for the report.
    for bucket in PAIR_CATEGORIES:
        diagnostics[bucket]["target"] = effective_targets[bucket]
        diagnostics[bucket]["achieved"] = (
            existing_counts.get(bucket, 0) + diagnostics[bucket]["achieved"]
        )
    achieved = {b: diagnostics[b]["achieved"] for b in PAIR_CATEGORIES}
    shortfalls = check_shortfalls(achieved, effective_targets, SHORTFALL_THRESHOLD)

    examples: dict[str, dict[str, Any] | None] = {b: None for b in PAIR_CATEGORIES}
    for bucket, picks in picked_by_bucket.items():
        if picks:
            a, b = picks[0]
            examples[bucket] = build_output_record(
                a, b, bucket, random.Random(args.seed + 1)
            )

    print_report(
        diagnostics,
        adversarial_sub,
        examples,
        shortfalls,
        effective_split=effective_split,
    )

    if shortfalls and not args.force:
        logger.error(
            "Aborting before write: %d bucket(s) short by >%.0f%%. "
            "Re-run with --force to write anyway.",
            len(shortfalls),
            SHORTFALL_THRESHOLD * 100,
        )
        return 6

    if args.dry_run:
        logger.info("--dry-run: not writing pairs.jsonl or sidecar.")
        return 0

    n_written = 0
    for bucket in PAIR_CATEGORIES:
        for a, b in picked_by_bucket[bucket]:
            record = build_output_record(a, b, bucket, rng)
            if record["pair_id"] in seen:
                continue
            jsonl_append(PAIRS_PATH, record)
            n_written += 1

    meta = {
        "seed": args.seed,
        "max_per_q_per_bucket": args.max_per_q_per_bucket,
        "force": args.force,
        "base_targets": PAIR_TARGETS,
        "base_adversarial_split": ADVERSARIAL_SPLIT,
        "heuristic_pool_fill_rate": HEURISTIC_POOL_FILL_RATE,
        "pool_sizes": pools,
        "effective_targets": effective_targets,
        "effective_adversarial_split": effective_split,
        "shortfall_threshold": SHORTFALL_THRESHOLD,
        "diagnostics": diagnostics,
        "adversarial_sub_achieved": adversarial_sub,
        "shortfalls": shortfalls,
        "hedge_tokens": list(HEDGE_TOKENS),
        "uncertainty_tokens": list(UNCERTAINTY_TOKENS),
        "n_pairs_written_this_run": n_written,
        "n_pairs_total_in_file": n_written + len(seen),
        "created_at_utc": dt.datetime.now(tz=dt.UTC).isoformat(),
        "enriched_input_sha256": file_sha256(ENRICHED_PATH),
    }
    atomic_write_json(META_PATH, meta)
    logger.info("Wrote %d new pairs to %s.", n_written, PAIRS_PATH)
    logger.info("Wrote sidecar to %s.", META_PATH)

    return 0


if __name__ == "__main__":
    sys.exit(main())
