"""Unit tests for ``train/modal/dpo.py`` YAML config loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from train.modal.dpo import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
DPO_YAML = REPO_ROOT / "train" / "configs" / "dpo.yaml"


def test_load_config_reads_real_dpo_yaml() -> None:
    cfg = load_config(DPO_YAML)
    assert cfg["model_id"] == "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
    assert cfg["sft_adapter_dir"] == "/vol/checkpoints/sft-final/"
    assert cfg["max_seq_length"] == 2048
    assert cfg["load_in_4bit"] is True


def test_load_config_training_block_matches_primer_recipe() -> None:
    """Primer Step 6 line 300: 'beta=0.1, ~1 epoch, low LR (5e-6 ish)'.
    Effective batch 16-32. The shipped YAML must encode these values."""
    cfg = load_config(DPO_YAML)
    t = cfg["training"]
    assert t["learning_rate"] == pytest.approx(5.0e-6)
    assert t["num_train_epochs"] == 1
    assert t["beta"] == pytest.approx(0.1)
    assert t["loss_type"] == "sigmoid"
    effective_batch = (
        t["per_device_train_batch_size"] * t["gradient_accumulation_steps"]
    )
    assert (
        16 <= effective_batch <= 32
    ), f"Effective batch {effective_batch} outside primer-recommended 16-32."
    assert t["lr_scheduler_type"] == "cosine"
    assert t["dataset_num_proc"] == 1, (
        "dataset_num_proc must be explicit 1 (not None) to survive Unsloth's "
        "source-injected override at unsloth/models/rl.py:1194-1208."
    )


def test_load_config_output_paths() -> None:
    cfg = load_config(DPO_YAML)
    out = cfg["output"]
    assert out["final_dir"] == "/vol/checkpoints/dpo-final/"
    assert out["merged_fp16_dir"] == "/vol/checkpoints/merged-fp16/"


def test_load_config_probe_block() -> None:
    cfg = load_config(DPO_YAML)
    probe = cfg["probe"]
    assert probe["count"] == 5
    assert probe["max_new_tokens"] == 384
    assert probe["sft_adapter_for_compare"] == "/vol/checkpoints/sft-final/"


def test_load_config_min_train_rows_safeguard() -> None:
    """Stage 5 decision #22 floor: <800 rows means too many were dropped."""
    cfg = load_config(DPO_YAML)
    assert cfg["min_train_rows"] >= 800


def test_load_config_missing_required_key(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("model_id: foo\n")
    with pytest.raises(ValueError, match="missing required key"):
        load_config(bad)


def test_load_config_non_dict(tmp_path: Path) -> None:
    bad = tmp_path / "list.yaml"
    bad.write_text("- foo\n- bar\n")
    with pytest.raises(ValueError, match="did not parse to a dict"):
        load_config(bad)
