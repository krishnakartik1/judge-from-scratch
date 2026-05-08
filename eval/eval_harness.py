"""Stage 8 evaluation-harness logic — pure Python.

No ``modal`` or ``torch`` imports at module load time; ``torch`` is
imported lazily inside :func:`run_inference_pass` so the metric and
parsing helpers are exercisable on CPU with a stub tokenizer.

The harness produces the four-column table from primer Appendix C —
Cohen's κ vs. human verdicts, position-bias, verbosity-bias,
self-consistency at T=0.3, parse-failure rate — for any combination of
the base Gemma 4 E4B and fine-tuned (SFT, SFT+DPO) checkpoints.

Decision #13 invariant: train and infer with the same configuration.
The system prompt must NOT contain ``<|think|>`` —
:func:`assert_no_thinking_in_prompt` hard-fails if it does.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

# ``data/_format_helpers.py`` is the single source of truth for the
# train/infer chat-template parity (decision #13). It is imported via
# the ``data`` package locally; on Modal it sits beside this file in
# ``/root/eval_pkg/`` and the entrypoint patches ``sys.path`` so the
# bare-module form below resolves.
try:
    from data._format_helpers import (
        VALID_VERDICTS,
        apply_chat,
        build_user_message,
        flip_verdict,
    )
except ImportError:
    from _format_helpers import (  # type: ignore[no-redef]
        VALID_VERDICTS,
        apply_chat,
        build_user_message,
        flip_verdict,
    )

from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Legacy filename: the file holds 300/300 human labels post-Stage 3b
# despite the ``_unlabeled`` suffix. Renaming would silently break the
# Stage 3a/6/7 references that hardcode this name.
EVAL_SET_PATH = Path("data/pairs/eval_set_unlabeled.jsonl")
SYSTEM_PROMPT_PATH = Path("data/judge_system_prompt.md")

VERDICT_RE = re.compile(
    r"<reasoning>(.*?)</reasoning>\s*<verdict>(A|B|TIE)</verdict>",
    re.DOTALL,
)

# Bumped from the Stage 6 probe's 256 per status-doc decision #24:
# post-training reasoning routinely runs longer than 256 tokens.
MAX_NEW_TOKENS = 384

# Pinned label list keeps the ``cohen_kappa_score`` confusion-matrix
# shape stable when one column is constant (e.g. the ``both_clean_tie``
# slice where humans tend to all label TIE).
KAPPA_LABELS: list[str] = ["A", "B", "TIE"]

EVAL_SLICES: tuple[str, ...] = ("in_dist", "ood_religion")
IN_DIST_CATEGORIES: tuple[str, ...] = (
    "clear_bias_vs_clean",
    "subtle_bias_vs_clean",
    "tracked_bias_vs_alternate",
    "both_clean_tie",
)

PARSE_FAIL: Literal["PARSE_FAIL"] = "PARSE_FAIL"

Verdict = Literal["A", "B", "TIE", "PARSE_FAIL"]


@dataclass
class Prediction:
    """One inference call's parsed output, indexed in the cache."""

    pair_id: str
    verdict: Verdict
    reasoning: str | None
    raw_output: str
    prompt_hash: str
    temperature: float
    swapped: bool
    run_index: int


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def parse_output(text: str) -> tuple[Verdict, str | None]:
    """Match ``<reasoning>…</reasoning><verdict>X</verdict>`` in ``text``.

    Returns ``(verdict, reasoning)`` on a clean match where ``verdict``
    is one of ``"A"``, ``"B"``, ``"TIE"``. Returns ``("PARSE_FAIL", None)``
    on any other shape.
    """
    m = VERDICT_RE.search(text)
    if not m:
        return PARSE_FAIL, None
    verdict = m.group(2)
    if verdict not in VALID_VERDICTS:
        return PARSE_FAIL, None
    return verdict, m.group(1)  # type: ignore[return-value]


def assert_no_thinking_in_prompt(system: str) -> None:
    """Hard-fail if the system prompt contains ``<|think|>``.

    Decision #13 critical consistency rule: train and infer with the
    same config; ``<|think|>`` must not appear anywhere. Stage 5 has a
    block at format time; this catches drift between stages.
    """
    if "<|think|>" in system:
        raise AssertionError(
            "System prompt contains `<|think|>`. Decision #13 forbids "
            "Gemma 4 native thinking mode at train- or infer-time. "
            "Re-check data/judge_system_prompt.md."
        )


def load_eval_set(path: Path) -> list[dict[str, Any]]:
    """Read the labeled holdout JSONL. Aborts on null/empty verdicts.

    Stage 3b is the labeling gate; running Stage 8 against an unlabeled
    record is meaningless because human_verdict is the κ ground truth.
    """
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Eval set {path}: malformed JSON at line {i}: {exc}"
                ) from exc
            verdict = record.get("human_verdict")
            if verdict is None or verdict == "":
                raise ValueError(
                    f"Eval set {path}: record "
                    f"{record.get('pair_id', '?')!r} at line {i} has no "
                    "human_verdict — Stage 3b labeling incomplete."
                )
            records.append(record)
    if not records:
        raise ValueError(f"Eval set {path} is empty.")
    return records


def select_dryrun_subset(
    records: list[dict[str, Any]],
    n_in_dist: int = 25,
    n_ood: int = 25,
) -> list[dict[str, Any]]:
    """Stratified sha1-sorted dryrun subset.

    Filters to each ``eval_slice``, sha1-sorts by ``pair_id``, and takes
    the first N. Deterministic across runs and machines (the hash trick
    is the same one used by ``train/modal/sft.py:select_probe_pair_ids``).
    """

    def _take(slice_name: str, n: int) -> list[dict[str, Any]]:
        candidates = [r for r in records if r.get("eval_slice") == slice_name]
        if len(candidates) < n:
            raise ValueError(
                f"Eval slice {slice_name!r}: have {len(candidates)} "
                f"records, need {n} for dryrun stratification."
            )
        return sorted(
            candidates,
            key=lambda r: hashlib.sha1(r["pair_id"].encode()).hexdigest(),
        )[:n]

    return _take("in_dist", n_in_dist) + _take("ood_religion", n_ood)


def prompt_hash(system: str, user: str) -> str:
    """sha256 hex digest of the rendered prompt material."""
    h = hashlib.sha256()
    h.update(system.encode())
    h.update(b"\x00")
    h.update(user.encode())
    return h.hexdigest()


def seed_for(pair_id: str, run_index: int) -> int:
    """Deterministic 32-bit seed for ``do_sample=True`` generations.

    Distinct ``(pair_id, run_index)`` tuples yield distinct seeds, so
    the three self-consistency runs at T=0.3 see different RNG state
    and can disagree.
    """
    h = int(hashlib.sha1(pair_id.encode()).hexdigest()[:8], 16)
    return (h ^ run_index) & 0xFFFFFFFF


def slug(name: str) -> str:
    """Filesystem-safe slug for cache and result filenames."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower()
    return cleaned or "unknown"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class PredictionCache:
    """Append-only JSONL cache backed by an in-memory lookup map.

    Cache key = ``(pair_id, prompt_hash, temperature, swapped, run_index)``.
    Including ``pair_id`` makes a key collision structurally impossible
    (``prompt_hash`` alone would also suffice in practice).

    The constructor replays the file into the lookup map. Malformed
    lines (truncated tail from a prior container kill) are skipped with
    a warning so partial caches still recover everything before the
    bad line.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._store: dict[str, Prediction] = {}
        if path.exists():
            self._replay()

    @staticmethod
    def key(
        pair_id: str,
        prompt_hash: str,
        temperature: float,
        swapped: bool,
        run_index: int,
    ) -> str:
        return f"{pair_id}|{prompt_hash}|{temperature:.3f}|{int(swapped)}|{run_index}"

    def _replay(self) -> None:
        with open(self.path) as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    pred = Prediction(**payload)
                except (json.JSONDecodeError, TypeError) as exc:
                    logger.warning(
                        "PredictionCache %s: skipping malformed line %d (%s).",
                        self.path,
                        i,
                        exc,
                    )
                    continue
                k = self.key(
                    pred.pair_id,
                    pred.prompt_hash,
                    pred.temperature,
                    pred.swapped,
                    pred.run_index,
                )
                self._store[k] = pred

    def get(
        self,
        pair_id: str,
        prompt_hash: str,
        temperature: float,
        swapped: bool,
        run_index: int,
    ) -> Prediction | None:
        return self._store.get(
            self.key(pair_id, prompt_hash, temperature, swapped, run_index)
        )

    def put(self, prediction: Prediction) -> None:
        k = self.key(
            prediction.pair_id,
            prediction.prompt_hash,
            prediction.temperature,
            prediction.swapped,
            prediction.run_index,
        )
        self._store[k] = prediction
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(prediction)) + "\n")

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Inference loop (model-agnostic)
# ---------------------------------------------------------------------------


def run_inference_pass(
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    records: list[dict[str, Any]],
    *,
    temperature: float,
    swapped: bool,
    run_index: int,
    cache: PredictionCache,
) -> list[Prediction]:
    """One inference pass over ``records`` for a fixed (T, swap, run).

    Hits the cache first; only invokes ``model.generate`` for misses.
    Each new prediction is persisted before the next is generated so a
    crash mid-pass loses at most one inference.
    """
    import torch

    predictions: list[Prediction] = []
    for record in records:
        pair_id = record["pair_id"]
        user = build_user_message(record, swap=swapped)
        rendered = apply_chat(tokenizer, system_prompt, user)
        phash = prompt_hash(system_prompt, user)

        cached = cache.get(pair_id, phash, temperature, swapped, run_index)
        if cached is not None:
            predictions.append(cached)
            continue

        # Gemma 4 multimodal processor: pass ``text=`` as keyword,
        # else the positional arg routes to images/videos
        # (mirrors train/modal/sft.py:338).
        inputs = tokenizer(text=rendered, return_tensors="pt").to("cuda")
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if temperature == 0.0:
            gen_kwargs["do_sample"] = False
        else:
            torch.manual_seed(seed_for(pair_id, run_index))
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        suffix = tokenizer.decode(
            out[0][inputs.input_ids.shape[1] :], skip_special_tokens=False
        )
        verdict, reasoning = parse_output(suffix)

        pred = Prediction(
            pair_id=pair_id,
            verdict=verdict,
            reasoning=reasoning,
            raw_output=suffix,
            prompt_hash=phash,
            temperature=temperature,
            swapped=swapped,
            run_index=run_index,
        )
        cache.put(pred)
        predictions.append(pred)

    return predictions


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _index_predictions(predictions: Iterable[Prediction]) -> dict[str, Prediction]:
    return {p.pair_id: p for p in predictions}


def _kappa_pair(
    records: list[dict[str, Any]], pred_index: dict[str, Prediction]
) -> float:
    y_true: list[str] = []
    y_pred: list[str] = []
    for record in records:
        pred = pred_index.get(record["pair_id"])
        if pred is None or pred.verdict == PARSE_FAIL:
            continue
        y_true.append(record["human_verdict"])
        y_pred.append(pred.verdict)  # type: ignore[arg-type]
    if not y_true:
        return float("nan")
    return float(cohen_kappa_score(y_true, y_pred, labels=KAPPA_LABELS))


def compute_kappa(
    records: list[dict[str, Any]], predictions: list[Prediction]
) -> float:
    """Cohen's κ on parseable predictions, with PARSE_FAIL excluded."""
    return _kappa_pair(records, _index_predictions(predictions))


def kappa_by_slice(
    records: list[dict[str, Any]], predictions: list[Prediction]
) -> dict[str, float]:
    """κ split by ``eval_slice`` (in_dist, ood_religion)."""
    pred_index = _index_predictions(predictions)
    return {
        slice_name: _kappa_pair(
            [r for r in records if r.get("eval_slice") == slice_name],
            pred_index,
        )
        for slice_name in EVAL_SLICES
    }


def kappa_by_category_in_dist(
    records: list[dict[str, Any]], predictions: list[Prediction]
) -> dict[str, float]:
    """κ split by ``pair_category`` over the in_dist slice only.

    OOD per-category numbers are excluded by spec — the OOD slice
    splits 28/12/9/6/5 across buckets and κ on a 5-pair slice is
    statistically meaningless.
    """
    pred_index = _index_predictions(predictions)
    in_dist = [r for r in records if r.get("eval_slice") == "in_dist"]
    return {
        category: _kappa_pair(
            [r for r in in_dist if r.get("pair_category") == category],
            pred_index,
        )
        for category in IN_DIST_CATEGORIES
    }


def position_bias_rate(
    records: list[dict[str, Any]],
    normal_preds: list[Prediction],
    swapped_preds: list[Prediction],
) -> dict[str, float]:
    """Fraction of pairs where the verdict failed to mirror under swap.

    Mirror-preserving outcomes: A→B, B→A, TIE→TIE. Anything else
    (including A→A, A→TIE, TIE→A) is a flip. Pairs where either side
    PARSE_FAILs are excluded from the denominator.
    """
    n_index = _index_predictions(normal_preds)
    s_index = _index_predictions(swapped_preds)
    out: dict[str, float] = {}
    for slice_name in EVAL_SLICES:
        flips = 0
        compared = 0
        for record in records:
            if record.get("eval_slice") != slice_name:
                continue
            n_pred = n_index.get(record["pair_id"])
            s_pred = s_index.get(record["pair_id"])
            if (
                n_pred is None
                or s_pred is None
                or n_pred.verdict == PARSE_FAIL
                or s_pred.verdict == PARSE_FAIL
            ):
                continue
            compared += 1
            if flip_verdict(n_pred.verdict) != s_pred.verdict:  # type: ignore[arg-type]
                flips += 1
        out[slice_name] = float("nan") if compared == 0 else flips / compared
    return out


def verbosity_bias_score(
    records: list[dict[str, Any]],
    normal_preds: list[Prediction],
    tokenizer: Any,
) -> float:
    """Average ``len(chosen) − len(rejected)`` in tokens, on non-TIE pairs.

    Positive = the judge favors longer responses. Tokenizes via
    ``tokenizer.encode`` (the input goes through the same vocabulary
    the model uses, not character counts).
    """
    pred_index = _index_predictions(normal_preds)
    diffs: list[int] = []
    for record in records:
        pred = pred_index.get(record["pair_id"])
        if pred is None or pred.verdict in (PARSE_FAIL, "TIE"):
            continue
        if pred.verdict == "A":
            chosen = record["response_a"]["text"]
            rejected = record["response_b"]["text"]
        else:
            chosen = record["response_b"]["text"]
            rejected = record["response_a"]["text"]
        diffs.append(len(tokenizer.encode(chosen)) - len(tokenizer.encode(rejected)))
    if not diffs:
        return float("nan")
    return float(sum(diffs) / len(diffs))


# Number of T=0.3 sampling passes per checkpoint. With 1 run we compare
# the sampled verdict to the T=0 baseline (see ``aggregate_metrics``);
# with ≥2 runs we use the classical all-runs-agree definition. Sized
# down from the original 3 to fit the project-wide budget cap.
SELF_CONSISTENCY_RUNS = 1


def self_consistency_rate(
    records: list[dict[str, Any]],
    runs: list[list[Prediction]],
) -> float:
    """Fraction of pairs where all ``runs`` produce the same verdict.

    Pairs where any run PARSE_FAILs are excluded from the denominator.
    Caller passes ≥2 runs; for the single-T=0.3-run case, prepend the
    T=0 normal pass as the reference (see ``aggregate_metrics``).
    """
    if len(runs) < 2:
        raise ValueError(f"Need at least 2 runs for self-consistency, got {len(runs)}.")
    indexes = [_index_predictions(run) for run in runs]
    consistent = 0
    compared = 0
    for record in records:
        preds = [idx.get(record["pair_id"]) for idx in indexes]
        if any(p is None or p.verdict == PARSE_FAIL for p in preds):
            continue
        compared += 1
        verdicts = {p.verdict for p in preds}  # type: ignore[union-attr]
        if len(verdicts) == 1:
            consistent += 1
    if compared == 0:
        return float("nan")
    return consistent / compared


def parse_failure_rate(predictions: Iterable[Prediction]) -> float:
    """Fraction of predictions whose verdict is PARSE_FAIL."""
    total = 0
    fails = 0
    for pred in predictions:
        total += 1
        if pred.verdict == PARSE_FAIL:
            fails += 1
    if total == 0:
        return float("nan")
    return fails / total


# ---------------------------------------------------------------------------
# Aggregation and table rendering
# ---------------------------------------------------------------------------


def aggregate_metrics(
    records: list[dict[str, Any]],
    normal_preds: list[Prediction],
    swapped_preds: list[Prediction],
    consistency_preds: list[list[Prediction]],
    tokenizer: Any,
) -> dict[str, float]:
    """The eleven scalar metrics for one checkpoint."""
    slice_kappa = kappa_by_slice(records, normal_preds)
    cat_kappa = kappa_by_category_in_dist(records, normal_preds)
    pos_bias = position_bias_rate(records, normal_preds, swapped_preds)
    all_preds = (
        list(normal_preds)
        + list(swapped_preds)
        + [p for run in consistency_preds for p in run]
    )
    # With a single T=0.3 sampling run, ``self_consistency_rate`` needs
    # a reference run; we pass the T=0 baseline so the metric still
    # measures verdict stability under sampling perturbation. With ≥2
    # T=0.3 runs, the classical all-sampled-runs-agree definition applies.
    sc_runs = (
        [normal_preds, consistency_preds[0]]
        if len(consistency_preds) == 1
        else list(consistency_preds)
    )
    return {
        "kappa_in_dist": slice_kappa["in_dist"],
        "kappa_ood_religion": slice_kappa["ood_religion"],
        "kappa_clear": cat_kappa["clear_bias_vs_clean"],
        "kappa_subtle": cat_kappa["subtle_bias_vs_clean"],
        "kappa_tracked": cat_kappa["tracked_bias_vs_alternate"],
        "kappa_tie": cat_kappa["both_clean_tie"],
        "position_bias_in_dist": pos_bias["in_dist"],
        "position_bias_ood": pos_bias["ood_religion"],
        "verbosity_bias": verbosity_bias_score(records, normal_preds, tokenizer),
        "self_consistency_t03": self_consistency_rate(records, sc_runs),
        "parse_failure_rate": parse_failure_rate(all_preds),
    }


_TABLE_ROWS: list[tuple[str, str, str]] = [
    ("Overall κ (in-dist)", "kappa_in_dist", "kappa"),
    ("Overall κ (OOD religion)", "kappa_ood_religion", "kappa"),
    ("Clear cases κ", "kappa_clear", "kappa"),
    ("Subtle cases κ", "kappa_subtle", "kappa"),
    ("Tracked-vs-alternate κ", "kappa_tracked", "kappa"),
    ("Tie cases κ", "kappa_tie", "kappa"),
    ("Position-bias rate (in-dist)", "position_bias_in_dist", "pct"),
    ("Position-bias rate (OOD)", "position_bias_ood", "pct"),
    ("Verbosity bias score", "verbosity_bias", "tokens"),
    ("Self-consistency (T=0.3)", "self_consistency_t03", "pct"),
    ("Parse failure rate", "parse_failure_rate", "pct"),
]


def _fmt_value(value: float, kind: str) -> str:
    if value != value:  # NaN sentinel for empty/undefined slices
        return "—"
    if kind == "kappa":
        return f"{value:.3f}"
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "tokens":
        sign = "+" if value > 0 else ""
        return f"{sign}{value:.1f}"
    return str(value)


def render_markdown_table(
    per_checkpoint_metrics: dict[str, dict[str, float]],
) -> str:
    """Eleven-row results table in the user's spec order, one column per checkpoint."""
    if not per_checkpoint_metrics:
        return "(no results)"
    names = list(per_checkpoint_metrics.keys())
    header = "| Metric | " + " | ".join(names) + " |"
    sep = "|" + "|".join(["---"] * (len(names) + 1)) + "|"
    lines = [header, sep]
    for label, key, kind in _TABLE_ROWS:
        values = [_fmt_value(per_checkpoint_metrics[name][key], kind) for name in names]
        lines.append("| " + label + " | " + " | ".join(values) + " |")
    return "\n".join(lines)
