"""Unit tests for ``train/modal/sft.py`` YAML config loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from train.modal.sft import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
SFT_YAML = REPO_ROOT / "train" / "configs" / "sft.yaml"


def test_load_config_reads_real_sft_yaml() -> None:
    """The shipped config must parse and contain the keys the
    training routine reads from it."""
    cfg = load_config(SFT_YAML)
    assert cfg["model_id"] == "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
    assert cfg["max_seq_length"] == 2048
    assert cfg["load_in_4bit"] is True


def test_load_config_lora_block() -> None:
    cfg = load_config(SFT_YAML)
    lora = cfg["lora"]
    assert lora["r"] == 16
    assert lora["lora_alpha"] == 32
    assert lora["target_modules"] == "all-linear"
    assert lora["bias"] == "none"
    assert lora["random_state"] == 3407
    assert lora["lora_dropout"] == 0.0  # PEFT/Unsloth default; see configs/README.md


def test_load_config_training_block_matches_primer_recipe() -> None:
    """Hyperparameters that actually matter (per primer Step 5):
    LR 2e-4, 3 epochs, effective batch 16-32. The shipped YAML must
    encode these values so the training routine reads them, not
    hardcoded defaults."""
    cfg = load_config(SFT_YAML)
    train = cfg["training"]
    assert train["learning_rate"] == pytest.approx(2.0e-4)
    assert train["num_train_epochs"] == 3
    effective_batch = (
        train["per_device_train_batch_size"] * train["gradient_accumulation_steps"]
    )
    assert (
        16 <= effective_batch <= 32
    ), f"Effective batch {effective_batch} outside primer-recommended 16-32."
    assert train["lr_scheduler_type"] == "cosine"
    assert train["optim"] == "adamw_8bit"


def test_load_config_probe_and_hf_blocks() -> None:
    cfg = load_config(SFT_YAML)
    assert cfg["probe"]["count"] == 5
    assert cfg["hf_push"]["repo_id"] == "krishnakartik/gemma4-judge-sft-checkpoint"
    assert cfg["hf_push"]["private"] is True


def test_load_config_missing_required_key(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("model_id: foo\n")  # missing max_seq_length, lora, etc.
    with pytest.raises(ValueError, match="missing required key"):
        load_config(bad)


def test_load_config_non_dict(tmp_path: Path) -> None:
    bad = tmp_path / "list.yaml"
    bad.write_text("- foo\n- bar\n")
    with pytest.raises(ValueError, match="did not parse to a dict"):
        load_config(bad)
