"""Unit tests for Stage 3a eval-set holdout (data/03a_holdout_eval.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import stage3a_holdout

allocate_ood_targets = stage3a_holdout.allocate_ood_targets
partition_pairs = stage3a_holdout.partition_pairs
annotate_eval_record = stage3a_holdout.annotate_eval_record
sample_holdout = stage3a_holdout.sample_holdout

PAIR_CATEGORIES = (
    "clear_bias_vs_clean",
    "subtle_bias_vs_clean",
    "tracked_bias_vs_alternate",
    "both_clean_tie",
    "adversarial",
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _mk_pair(
    pair_id: str,
    bias_category: str,
    pair_category: str,
    bias_a: str = "biased",
    bias_b: str = "neutral",
) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "question_id": f"q_{pair_id}",
        "question_text": f"text for {pair_id}",
        "bias_category": bias_category,
        "response_a": {
            "model": "model-a",
            "text": "answer A",
            "suspected_bias_level": bias_a,
        },
        "response_b": {
            "model": "model-b",
            "text": "answer B",
            "suspected_bias_level": bias_b,
        },
        "pair_category": pair_category,
    }


def _build_pool(
    bias_category: str, per_bucket: dict[str, int], id_prefix: str
) -> list[dict[str, Any]]:
    out = []
    for bucket, n in per_bucket.items():
        for i in range(n):
            out.append(_mk_pair(f"{id_prefix}_{bucket}_{i:04d}", bias_category, bucket))
    return out


# -----------------------------------------------------------------------------
# allocate_ood_targets
# -----------------------------------------------------------------------------


def test_allocate_ood_targets_production_case():
    """Locks the production allocation against drift (B1)."""
    in_dist = {
        "clear_bias_vs_clean": 110,
        "subtle_bias_vs_clean": 50,
        "tracked_bias_vs_alternate": 35,
        "both_clean_tie": 25,
        "adversarial": 20,
    }
    out = allocate_ood_targets(in_dist, 60)
    assert out == {
        "clear_bias_vs_clean": 28,
        "subtle_bias_vs_clean": 12,
        "tracked_bias_vs_alternate": 9,
        "both_clean_tie": 6,
        "adversarial": 5,
    }
    assert sum(out.values()) == 60


def test_allocate_ood_targets_tiebreak_path():
    """Exercises secondary (in-dist desc) and tertiary (name asc) tiebreaks.

    Fixture: in_dist={"big":4, "small":1, "med":1}, total=4.
    Sum=6 → shares (2.667, 0.667, 0.667), floors (2,0,0) sum to 2,
    need 2 bumps. All three residuals are 0.667 — three-way tie.
    Ranking by (residual desc, in-dist desc, name asc):
      1. big   (0.667, 4, "big")   — wins on in-dist=4
      2. med   (0.667, 1, "med")   — beats "small" on name asc
      3. small (0.667, 1, "small")
    Result: big=3, med=1, small=0.
    """
    out = allocate_ood_targets({"big": 4, "small": 1, "med": 1}, 4)
    assert out == {"big": 3, "med": 1, "small": 0}
    assert sum(out.values()) == 4


def test_allocate_ood_targets_no_residual_ties():
    """Sanity check on the simple path (no bumps required)."""
    out = allocate_ood_targets({"a": 80, "b": 20}, 5)
    assert out == {"a": 4, "b": 1}


def test_allocate_ood_targets_zero_residuals():
    """Exact-share case — no bumps."""
    out = allocate_ood_targets({"a": 2, "b": 2}, 4)
    assert out == {"a": 2, "b": 2}


def test_allocate_ood_targets_zero_total():
    """Total of 0 → all zeros."""
    out = allocate_ood_targets({"a": 5, "b": 5}, 0)
    assert out == {"a": 0, "b": 0}


def test_allocate_ood_targets_empty_input_raises():
    with pytest.raises(ValueError):
        allocate_ood_targets({}, 5)


# -----------------------------------------------------------------------------
# partition_pairs
# -----------------------------------------------------------------------------


def test_partition_pairs():
    pairs = [
        _mk_pair("a", "religion", "clear_bias_vs_clean"),
        _mk_pair("b", "age", "clear_bias_vs_clean"),
        _mk_pair("c", "religion", "both_clean_tie"),
        _mk_pair("d", "ses", "adversarial"),
    ]
    religion, nonreligion = partition_pairs(pairs)
    assert [r["pair_id"] for r in religion] == ["a", "c"]
    assert [r["pair_id"] for r in nonreligion] == ["b", "d"]


# -----------------------------------------------------------------------------
# annotate_eval_record
# -----------------------------------------------------------------------------


def test_annotate_eval_record_strips_suspected_bias_level():
    rec = _mk_pair("x", "religion", "clear_bias_vs_clean")
    out = annotate_eval_record(rec, "ood_religion")
    assert "suspected_bias_level" not in out["response_a"]
    assert "suspected_bias_level" not in out["response_b"]
    # Original is not mutated.
    assert "suspected_bias_level" in rec["response_a"]


def test_annotate_eval_record_keeps_model_field():
    """Explicit decision: model identity stays visible to the labeler."""
    rec = _mk_pair("x", "religion", "clear_bias_vs_clean")
    out = annotate_eval_record(rec, "ood_religion")
    assert out["response_a"]["model"] == "model-a"
    assert out["response_b"]["model"] == "model-b"


def test_annotate_eval_record_adds_stub_fields():
    rec = _mk_pair("x", "age", "subtle_bias_vs_clean")
    out = annotate_eval_record(rec, "in_dist")
    assert out["human_verdict"] is None
    assert out["confidence"] is None
    assert out["notes"] is None
    assert out["eval_slice"] == "in_dist"


# -----------------------------------------------------------------------------
# sample_holdout
# -----------------------------------------------------------------------------


def _scaled_fixture() -> tuple[list[dict[str, Any]], dict[str, int], int]:
    """100 religion (20/bucket) + 200 non-religion (40/bucket).

    With in_dist=8 per bucket and total_ood=20, the OOD allocation
    under :func:`allocate_ood_targets` is (4,4,4,4,4) — exact shares.
    Per-bucket religion supply (20) ≥ OOD target (4) and per-bucket
    non-religion supply (40) ≥ in-dist target (8), both with margin.
    """
    religion = _build_pool("religion", {b: 20 for b in PAIR_CATEGORIES}, "rel")
    nonreligion = _build_pool("age", {b: 40 for b in PAIR_CATEGORIES}, "age")
    pairs = religion + nonreligion
    in_dist_targets = {b: 8 for b in PAIR_CATEGORIES}
    return pairs, in_dist_targets, 20


def test_sample_holdout_counts():
    pairs, in_dist_targets, total_ood = _scaled_fixture()

    # Sufficiency precondition (test as documented in plan §Tests #5).
    rel_buckets = {b: 0 for b in PAIR_CATEGORIES}
    for p in pairs:
        if p["bias_category"] == "religion":
            rel_buckets[p["pair_category"]] += 1
    ood_targets = allocate_ood_targets(in_dist_targets, total_ood)
    for b in PAIR_CATEGORIES:
        assert rel_buckets[b] >= ood_targets[b]

    splits = sample_holdout(pairs, in_dist_targets, total_ood, seed=42)

    in_dist_counts = {b: 0 for b in PAIR_CATEGORIES}
    for r in splits["eval_in_dist"]:
        in_dist_counts[r["pair_category"]] += 1
    ood_counts = {b: 0 for b in PAIR_CATEGORIES}
    for r in splits["eval_ood"]:
        ood_counts[r["pair_category"]] += 1

    for b in PAIR_CATEGORIES:
        assert in_dist_counts[b] == 8, (b, in_dist_counts[b])
        assert ood_counts[b] == 4, (b, ood_counts[b])

    eval_total = len(splits["eval_in_dist"]) + len(splits["eval_ood"])
    assert eval_total == 60


def test_sample_holdout_no_religion_in_to_label():
    pairs, in_dist_targets, total_ood = _scaled_fixture()
    splits = sample_holdout(pairs, in_dist_targets, total_ood, seed=42)
    assert not any(r["bias_category"] == "religion" for r in splits["to_label"])


def test_sample_holdout_no_id_overlap():
    pairs, in_dist_targets, total_ood = _scaled_fixture()
    splits = sample_holdout(pairs, in_dist_targets, total_ood, seed=42)
    eval_ids = {r["pair_id"] for r in splits["eval_in_dist"]} | {
        r["pair_id"] for r in splits["eval_ood"]
    }
    to_label_ids = {r["pair_id"] for r in splits["to_label"]}
    unused_ids = {r["pair_id"] for r in splits["unused_religion"]}
    assert eval_ids.isdisjoint(to_label_ids)
    assert eval_ids.isdisjoint(unused_ids)
    assert to_label_ids.isdisjoint(unused_ids)


def test_sample_holdout_total_conservation():
    pairs, in_dist_targets, total_ood = _scaled_fixture()
    splits = sample_holdout(pairs, in_dist_targets, total_ood, seed=42)
    total = (
        len(splits["eval_in_dist"])
        + len(splits["eval_ood"])
        + len(splits["to_label"])
        + len(splits["unused_religion"])
    )
    assert total == len(pairs)


def test_sample_holdout_deterministic():
    pairs, in_dist_targets, total_ood = _scaled_fixture()
    a = sample_holdout(pairs, in_dist_targets, total_ood, seed=42)
    b = sample_holdout(pairs, in_dist_targets, total_ood, seed=42)

    def fingerprint(splits):
        return {k: [r["pair_id"] for r in v] for k, v in splits.items()}

    assert fingerprint(a) == fingerprint(b)


def test_sample_holdout_insufficient_supply_raises(tmp_path, monkeypatch):
    """Religion missing the adversarial bucket → AssertionError; no files written."""
    religion = _build_pool(
        "religion",
        {
            "clear_bias_vs_clean": 20,
            "subtle_bias_vs_clean": 20,
            "tracked_bias_vs_alternate": 20,
            "both_clean_tie": 20,
            "adversarial": 0,  # <-- no supply
        },
        "rel",
    )
    nonreligion = _build_pool("age", {b: 40 for b in PAIR_CATEGORIES}, "age")
    pairs = religion + nonreligion
    in_dist_targets = {b: 8 for b in PAIR_CATEGORIES}

    # Re-point output paths to tmp_path so we can assert nothing was written.
    eval_path = tmp_path / "eval.jsonl"
    monkeypatch.setattr(stage3a_holdout, "EVAL_PATH", eval_path)
    monkeypatch.setattr(stage3a_holdout, "TO_LABEL_PATH", tmp_path / "tl.jsonl")
    monkeypatch.setattr(stage3a_holdout, "UNUSED_RELIGION_PATH", tmp_path / "ur.jsonl")
    monkeypatch.setattr(stage3a_holdout, "META_PATH", tmp_path / "meta.json")

    with pytest.raises(AssertionError, match="adversarial"):
        sample_holdout(pairs, in_dist_targets, 20, seed=42)

    # No files should exist (sampler raises before any I/O).
    assert not eval_path.exists()
    assert not (tmp_path / "tl.jsonl").exists()
    assert not (tmp_path / "ur.jsonl").exists()
    assert not (tmp_path / "meta.json").exists()


# -----------------------------------------------------------------------------
# main() — end-to-end through monkeypatched paths
# -----------------------------------------------------------------------------


def _setup_main_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Path]:
    """Repoint module-level paths into tmp_path and write a synthetic input."""
    input_path = tmp_path / "pairs.jsonl"
    eval_path = tmp_path / "eval_set_unlabeled.jsonl"
    to_label_path = tmp_path / "pairs_to_label.jsonl"
    unused_path = tmp_path / "pairs_unused_religion.jsonl"
    meta_path = tmp_path / "eval_holdout.meta.json"

    paths = {
        "input": input_path,
        "eval": eval_path,
        "to_label": to_label_path,
        "unused": unused_path,
        "meta": meta_path,
    }

    monkeypatch.setattr(stage3a_holdout, "INPUT_PATH", input_path)
    monkeypatch.setattr(stage3a_holdout, "EVAL_PATH", eval_path)
    monkeypatch.setattr(stage3a_holdout, "TO_LABEL_PATH", to_label_path)
    monkeypatch.setattr(stage3a_holdout, "UNUSED_RELIGION_PATH", unused_path)
    monkeypatch.setattr(stage3a_holdout, "META_PATH", meta_path)
    monkeypatch.setattr(
        stage3a_holdout,
        "_ALL_OUTPUT_PATHS",
        (
            eval_path,
            eval_path.with_name(eval_path.name + ".tmp"),
            to_label_path,
            to_label_path.with_name(to_label_path.name + ".tmp"),
            unused_path,
            unused_path.with_name(unused_path.name + ".tmp"),
            meta_path,
            meta_path.with_name(meta_path.name + ".tmp"),
        ),
    )

    # Synthetic input: scaled fixture (100 religion + 200 non-religion).
    pairs, _, _ = _scaled_fixture()
    with input_path.open("w", encoding="utf-8") as fh:
        for r in pairs:
            fh.write(json.dumps(r) + "\n")

    return paths


def _patched_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override IN_DIST_TARGETS / OOD_TOTAL / IN_DIST_TOTAL for fixture scale."""
    in_dist = {b: 8 for b in PAIR_CATEGORIES}
    monkeypatch.setattr(stage3a_holdout, "IN_DIST_TARGETS", in_dist)
    monkeypatch.setattr(stage3a_holdout, "IN_DIST_TOTAL", sum(in_dist.values()))
    monkeypatch.setattr(stage3a_holdout, "OOD_TOTAL", 20)


def test_force_unlinks_then_rewrites(tmp_path, monkeypatch):
    paths = _setup_main_fixture(tmp_path, monkeypatch)
    _patched_targets(monkeypatch)

    # Pre-populate outputs with junk to confirm --force overwrites.
    junk = b"junk\n"
    for p in (paths["eval"], paths["to_label"], paths["unused"], paths["meta"]):
        p.write_bytes(junk)

    rc = stage3a_holdout.main(["--seed", "42", "--force"])
    assert rc == 0

    # Each file now contains valid JSON(L), not junk.
    for key in ("eval", "to_label", "unused"):
        with paths[key].open() as fh:
            for line in fh:
                if line.strip():
                    json.loads(line)
    sidecar = json.loads(paths["meta"].read_text())
    assert sidecar["seed"] == 42

    # Counts reflect a single rewrite, not double.
    n_eval = sum(1 for line in paths["eval"].open() if line.strip())
    n_to_label = sum(1 for line in paths["to_label"].open() if line.strip())
    n_unused = sum(1 for line in paths["unused"].open() if line.strip())
    assert n_eval == 60, n_eval
    assert n_to_label == 200 - 40 == 160, n_to_label
    assert n_unused == 100 - 20 == 80, n_unused
    assert sidecar["n_eval_unlabeled"] == n_eval
    assert sidecar["n_pairs_to_label"] == n_to_label
    assert sidecar["n_unused_religion"] == n_unused


def test_subset_outputs_present_exits_one(tmp_path, monkeypatch):
    paths = _setup_main_fixture(tmp_path, monkeypatch)
    _patched_targets(monkeypatch)

    # Pre-populate ONE output (no sidecar) → "files present but no sidecar".
    pre_existing = b"do not overwrite me\n"
    paths["eval"].write_bytes(pre_existing)

    rc = stage3a_holdout.main([])
    assert rc == 1
    # File untouched.
    assert paths["eval"].read_bytes() == pre_existing
    assert not paths["to_label"].exists()
    assert not paths["unused"].exists()
    assert not paths["meta"].exists()


def test_sidecar_input_sha_matches(tmp_path, monkeypatch):
    paths = _setup_main_fixture(tmp_path, monkeypatch)
    _patched_targets(monkeypatch)

    rc = stage3a_holdout.main(["--seed", "42"])
    assert rc == 0

    sidecar = json.loads(paths["meta"].read_text())
    assert sidecar["input_sha256"] == stage3a_holdout.file_sha256(paths["input"])


def test_sidecar_record_counts_match_files(tmp_path, monkeypatch):
    paths = _setup_main_fixture(tmp_path, monkeypatch)
    _patched_targets(monkeypatch)

    rc = stage3a_holdout.main(["--seed", "42"])
    assert rc == 0

    sidecar = json.loads(paths["meta"].read_text())
    n_eval = sum(1 for line in paths["eval"].open() if line.strip())
    n_to_label = sum(1 for line in paths["to_label"].open() if line.strip())
    n_unused = sum(1 for line in paths["unused"].open() if line.strip())
    assert sidecar["n_eval_unlabeled"] == n_eval
    assert sidecar["n_pairs_to_label"] == n_to_label
    assert sidecar["n_unused_religion"] == n_unused


def test_input_sha_mismatch_exits_one(tmp_path, monkeypatch, caplog):
    paths = _setup_main_fixture(tmp_path, monkeypatch)
    _patched_targets(monkeypatch)

    rc1 = stage3a_holdout.main(["--seed", "42"])
    assert rc1 == 0

    # Mutate the input (append a no-op extra record).
    extra = _mk_pair("extra_pair_999", "age", "clear_bias_vs_clean")
    with paths["input"].open("a") as fh:
        fh.write(json.dumps(extra) + "\n")

    with caplog.at_level("ERROR"):
        rc2 = stage3a_holdout.main([])
    assert rc2 == 1
    diag = " ".join(rec.getMessage() for rec in caplog.records)
    assert "input SHA mismatch" in diag
    assert "--force" in diag


def test_clean_state_is_idempotent(tmp_path, monkeypatch):
    """Re-running with no changes returns 0 without touching files."""
    paths = _setup_main_fixture(tmp_path, monkeypatch)
    _patched_targets(monkeypatch)

    rc1 = stage3a_holdout.main(["--seed", "42"])
    assert rc1 == 0
    sidecar_before = paths["meta"].read_bytes()
    eval_before = paths["eval"].read_bytes()

    rc2 = stage3a_holdout.main([])
    assert rc2 == 0
    assert paths["meta"].read_bytes() == sidecar_before
    assert paths["eval"].read_bytes() == eval_before


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    paths = _setup_main_fixture(tmp_path, monkeypatch)
    _patched_targets(monkeypatch)

    rc = stage3a_holdout.main(["--dry-run"])
    assert rc == 0
    assert not paths["eval"].exists()
    assert not paths["to_label"].exists()
    assert not paths["unused"].exists()
    assert not paths["meta"].exists()


def test_missing_input_exits_two(tmp_path, monkeypatch):
    monkeypatch.setattr(
        stage3a_holdout, "INPUT_PATH", tmp_path / "does-not-exist.jsonl"
    )
    rc = stage3a_holdout.main([])
    assert rc == 2
