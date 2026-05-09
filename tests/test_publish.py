"""Unit tests for the Stage 9 publish helpers.

Pure-Python only — no Modal calls, no HF uploads, no llama.cpp.
Tests cover the helper functions and config tables exposed by:

- ``publish/export_gguf.py``
- ``publish/build_modelfile.py``
- ``publish/upload_hf.py``

This is the gate that catches structural drift between the model
cards (which embed file names and tag schemes) and the scripts that
build / upload them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from publish.build_modelfile import (
    MODEL_VARIANTS,
    NUM_CTX,
    TEMPERATURE,
    THINKING_MODE_WARNING,
    render_modelfile,
)
from publish.export_gguf import (
    MIN_GGUF_BYTES,
    MODEL_REGISTRY,
    QUANT_TYPES,
    _verify_quant,
    gguf_paths,
    validate_model_name,
)
from publish.upload_hf import (
    GGUF_LAYOUT,
    PAIRS_FILES,
    REPOS,
    UPLOAD_ORDER,
    validate_target,
)

# ---------------------------------------------------------------------------
# export_gguf.py
# ---------------------------------------------------------------------------


def test_export_gguf_model_registry_shape() -> None:
    """Both model variants must be registered with the volume paths."""
    assert set(MODEL_REGISTRY) == {"dpo", "sft"}
    for name, info in MODEL_REGISTRY.items():
        assert info["source_dir"].startswith("/vol/checkpoints/")
        assert "display_name" in info
        # display_name should match what gets published to HF
        assert "gemma4-social-bias-judge" in info["display_name"]


def test_export_gguf_quant_types() -> None:
    """The two quant levels (Q8_0 + Q5_K_M) must be registered."""
    assert QUANT_TYPES == ("Q8_0", "Q5_K_M")


def test_export_gguf_validate_model_name_known() -> None:
    for name in ("dpo", "sft"):
        validate_model_name(name)


def test_export_gguf_validate_model_name_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        validate_model_name("rlhf")
    with pytest.raises(ValueError):
        validate_model_name("")


@pytest.mark.parametrize("name", ["dpo", "sft"])
def test_export_gguf_paths_layout(name: str) -> None:
    """Paths land under /vol/gguf/<model>/{f16,Q8_0,Q5_K_M}.gguf."""
    paths = gguf_paths(name)
    assert set(paths) == {"f16", *QUANT_TYPES}
    for key, p in paths.items():
        assert isinstance(p, Path)
        assert str(p).startswith(f"/vol/gguf/{name}/")
        assert p.name.endswith(".gguf")
    # The f16 intermediate sits in the same dir as the quants.
    assert paths["f16"].parent == paths["Q8_0"].parent


def test_export_gguf_verify_quant_missing(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="not produced"):
        _verify_quant(tmp_path / "missing.gguf")


def test_export_gguf_verify_quant_too_small(tmp_path: Path) -> None:
    p = tmp_path / "tiny.gguf"
    p.write_bytes(b"x" * 100)
    with pytest.raises(AssertionError, match="suspiciously small"):
        _verify_quant(p)


def test_export_gguf_verify_quant_passes(tmp_path: Path) -> None:
    p = tmp_path / "ok.gguf"
    p.write_bytes(b"x" * (MIN_GGUF_BYTES + 1))
    out = _verify_quant(p)
    assert out["bytes"] >= MIN_GGUF_BYTES
    assert out["path"] == str(p)


# ---------------------------------------------------------------------------
# build_modelfile.py
# ---------------------------------------------------------------------------


def test_build_modelfile_variant_table() -> None:
    """The 4 variants are (dpo Q8_0/Q5_K_M, sft Q8_0/Q5_K_M) with
    flat tag/filename schemes that the GGUF repo and model cards
    expect."""
    assert len(MODEL_VARIANTS) == 4
    seen_tags = {tag for _, _, tag, _ in MODEL_VARIANTS}
    assert seen_tags == {"Q8_0", "Q5_K_M", "Q8_0-sft", "Q5_K_M-sft"}
    # DPO tags are the bare quants, SFT tags get the -sft suffix.
    for model, _quant, tag, fname in MODEL_VARIANTS:
        if model == "dpo":
            assert "-sft" not in tag
        else:
            assert tag.endswith("-sft")
        assert fname == f"{tag}.gguf"


def test_build_modelfile_render_includes_required_blocks() -> None:
    """Modelfile body must include FROM, both PARAMETERs, the warning,
    and the SYSTEM block — all four are load-bearing for Ollama."""
    body = render_modelfile("Q8_0.gguf", "you are a judge")
    assert "FROM ./Q8_0.gguf" in body
    assert f"PARAMETER temperature {TEMPERATURE}" in body
    assert f"PARAMETER num_ctx {NUM_CTX}" in body
    assert THINKING_MODE_WARNING in body
    assert 'SYSTEM """you are a judge"""' in body


def test_build_modelfile_thinking_mode_warning_mentions_think_token() -> None:
    """The warning must say what NOT to add (the actual token name)."""
    assert "<|think|>" in THINKING_MODE_WARNING
    assert "DISABLED" in THINKING_MODE_WARNING


def test_build_modelfile_render_strips_trailing_whitespace() -> None:
    """Modelfile SYSTEM block doesn't leak trailing newlines that would
    desync between the model card text and the system prompt the
    deployed model sees."""
    body = render_modelfile("Q8_0.gguf", "you are a judge\n\n")
    assert 'SYSTEM """you are a judge"""' in body


# ---------------------------------------------------------------------------
# upload_hf.py
# ---------------------------------------------------------------------------


def test_upload_hf_repos_shape() -> None:
    """All four targets present with their HF repo IDs and types."""
    assert set(REPOS) == {"model-dpo", "model-sft", "gguf", "pairs"}
    for target, info in REPOS.items():
        assert info["repo_id"].startswith("krishnakartik/")
        assert info["repo_type"] in ("model", "dataset")
        assert "card_path" in info
    assert REPOS["pairs"]["repo_type"] == "dataset"
    assert REPOS["model-dpo"]["repo_type"] == "model"
    assert REPOS["model-sft"]["repo_type"] == "model"
    assert REPOS["gguf"]["repo_type"] == "model"


def test_upload_hf_validate_target() -> None:
    for t in REPOS:
        validate_target(t)
    with pytest.raises(ValueError, match="Unknown target"):
        validate_target("unknown")


def test_upload_hf_upload_order_starts_with_gguf() -> None:
    """The Ollama one-liner in the model cards points at the GGUF repo,
    so GGUF must land first to keep the model-card link non-dead."""
    assert UPLOAD_ORDER[0] == "gguf"
    assert UPLOAD_ORDER[-1] == "model-dpo"
    assert set(UPLOAD_ORDER) == set(REPOS)


def test_upload_hf_gguf_layout_renames() -> None:
    """The GGUF layout maps /vol/gguf/<model>/<quant>.gguf to flat
    repo paths matching the build_modelfile.py tags."""
    expected_dest = {fname for _, _, _, fname in MODEL_VARIANTS}
    actual_dest = {dest for _src, dest in GGUF_LAYOUT}
    assert actual_dest == expected_dest


def test_upload_hf_pairs_files_resolve() -> None:
    """The dataset-target file list must point at real files on disk
    (running from the project root). Catches drift if the pipeline
    output paths change."""
    for fp in PAIRS_FILES:
        assert Path(fp).exists(), f"Missing pairs file: {fp}"


def test_upload_hf_card_paths_exist() -> None:
    """Each target's card_path must point at an existing markdown file."""
    for target, info in REPOS.items():
        p = Path(info["card_path"])
        assert p.exists(), f"Card missing for {target}: {p}"
        assert p.suffix == ".md"
