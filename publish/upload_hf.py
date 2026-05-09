"""Stage 9 — Publish all four artifacts to Hugging Face.

Four targets, in the order :func:`main` runs them when given
``--target all`` (matches the dependency the model cards declare):

1. ``gguf``       — krishnakartik/gemma4-social-bias-judge-gguf
                    (single repo with all 4 GGUFs + Modelfiles + a
                    dual-purpose README). Uploaded first so the
                    Ollama one-liner referenced in the model cards
                    is live before the model cards land.
2. ``pairs``      — krishnakartik/gemma4-social-bias-judge-pairs
                    (Dataset repo: SFT/DPO formatted JSONL +
                    pairs.jsonl + eval set + dataset card).
3. ``model-sft``  — krishnakartik/gemma4-social-bias-judge-sft
                    (the SFT-only secondary model + card).
4. ``model-dpo``  — krishnakartik/gemma4-social-bias-judge
                    (the DPO primary model + card; the headline
                    artifact, uploaded last so its README links to
                    the already-live SFT and GGUF repos).

Volume-resident artifacts (the two model checkpoints and the
GGUFs) upload from a Modal CPU container so we don't ship 50+ GB
through a local network. The dataset (~30 MB) uploads locally.

Behind ``--confirm`` for each invocation; without it, every target
prints what it WOULD upload and exits.

Run from the project root::

    # Dry run — print plan, upload nothing
    uv run python publish/upload_hf.py --target gguf

    # Real upload (requires HF_TOKEN env or huggingface-cli login,
    # plus ``modal`` set up for volume-resident targets)
    uv run python publish/upload_hf.py --target gguf --confirm
    modal run publish/upload_hf.py::upload_remote --target gguf --confirm

The volume-resident targets dispatch through a Modal entrypoint
(``modal run publish/upload_hf.py::upload_remote``) because the
huggingface_hub library streams uploads from disk and we want it
to read from /vol directly.
"""

import argparse
import logging
import os
from pathlib import Path

import modal

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
HF_ORG = "krishnakartik"

REPOS: dict[str, dict[str, str]] = {
    "model-dpo": {
        "repo_id": f"{HF_ORG}/gemma4-social-bias-judge",
        "repo_type": "model",
        "vol_source": "/vol/checkpoints/merged-fp16",
        "card_path": "publish/model_cards/gemma4-social-bias-judge.md",
    },
    "model-sft": {
        "repo_id": f"{HF_ORG}/gemma4-social-bias-judge-sft",
        "repo_type": "model",
        "vol_source": "/vol/checkpoints/sft-merged-fp16",
        "card_path": "publish/model_cards/gemma4-social-bias-judge-sft.md",
    },
    "gguf": {
        "repo_id": f"{HF_ORG}/gemma4-social-bias-judge-gguf",
        "repo_type": "model",
        "vol_source": "/vol/gguf",
        "card_path": "publish/model_cards/gemma4-social-bias-judge-gguf.md",
    },
    "pairs": {
        "repo_id": f"{HF_ORG}/gemma4-social-bias-judge-pairs",
        "repo_type": "dataset",
        "card_path": "publish/model_cards/gemma4-social-bias-judge-pairs.md",
    },
}

# GGUF repo flat layout (renames during upload).
GGUF_LAYOUT: list[tuple[str, str]] = [
    ("/vol/gguf/dpo/Q8_0.gguf", "Q8_0.gguf"),
    ("/vol/gguf/dpo/Q5_K_M.gguf", "Q5_K_M.gguf"),
    ("/vol/gguf/sft/Q8_0.gguf", "Q8_0-sft.gguf"),
    ("/vol/gguf/sft/Q5_K_M.gguf", "Q5_K_M-sft.gguf"),
]

UPLOAD_ORDER: tuple[str, ...] = ("gguf", "pairs", "model-sft", "model-dpo")

# Local-only target (small files; no Modal needed).
PAIRS_FILES: tuple[str, ...] = (
    "data/formatted/sft.jsonl",
    "data/formatted/dpo.jsonl",
    "data/pairs/pairs.jsonl",
    "data/pairs/pairs_to_label.jsonl",
    "data/pairs/eval_set_unlabeled.jsonl",
)

# ----------------------------------------------------------------------------
# Modal app (volume-resident targets only)
# ----------------------------------------------------------------------------
VOLUME_NAME = "judge-from-scratch"

app = modal.App("judge-upload-hf")

upload_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub>=1.0.0"
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]


# ----------------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------------


def validate_target(name: str) -> None:
    if name not in REPOS:
        raise ValueError(f"Unknown target {name!r}; valid: {sorted(REPOS)} or 'all'")


def _read_token() -> str:
    """Get the HF token from env or ``HF_TOKEN`` Modal secret.

    huggingface_hub also picks up cached credentials from
    ``~/.huggingface/token`` so a prior ``huggingface-cli login``
    works without any env var. This helper just sanity-checks one
    of the standard sources is set, returning the env value or the
    empty string (in which case the hub library falls back to its
    cache).
    """
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN", "")


# ----------------------------------------------------------------------------
# Modal-side helpers (volume-resident targets)
# ----------------------------------------------------------------------------


@app.function(
    image=upload_image,
    cpu=4.0,
    timeout=7200,
    volumes={"/vol": volume},
    secrets=secrets,
)
def upload_model_remote(target: str, card_text: str, dry_run: bool) -> dict:
    """Upload a fp16 model checkpoint from /vol to HF.

    The model card text is passed in from the local side because
    the model-card markdown lives outside the Modal volume.
    """

    from huggingface_hub import HfApi

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    info = REPOS[target]
    source = Path(info["vol_source"])
    if not (source / "config.json").exists():
        raise FileNotFoundError(
            f"Model checkpoint missing at {source} — run merge step first."
        )

    # Stage the README.md inside the source dir so HF's upload_folder
    # picks it up alongside the weights. We copy (not symlink) so a
    # later run with a different card body cleanly replaces it.
    readme_dst = source / "README.md"
    if dry_run:
        files = sorted(p.name for p in source.iterdir() if p.is_file())
        return {
            "target": target,
            "repo_id": info["repo_id"],
            "would_upload": files + ["README.md (synthesized from card)"],
            "source": str(source),
            "card_bytes": len(card_text),
        }

    readme_dst.write_text(card_text)
    api = HfApi(token=_read_token() or None)
    api.create_repo(info["repo_id"], repo_type=info["repo_type"], exist_ok=True)
    # ``upload_large_folder`` is HF's purpose-built path for multi-GB
    # uploads — more efficient chunking + resume semantics than
    # ``upload_folder``. The merged-fp16 checkpoints are ~16 GB each
    # so this is the right call here. ``upload_folder`` remains in
    # use for the small pairs upload (~30 MB).
    api.upload_large_folder(
        folder_path=str(source),
        repo_id=info["repo_id"],
        repo_type=info["repo_type"],
    )
    # Don't leave a stale README on the volume — the volume is the
    # source of truth for model weights, not for HF-rendered cards.
    readme_dst.unlink(missing_ok=True)
    return {"target": target, "repo_id": info["repo_id"], "status": "uploaded"}


@app.function(
    image=upload_image,
    cpu=4.0,
    timeout=7200,
    volumes={"/vol": volume},
    secrets=secrets,
)
def upload_gguf_remote(
    card_text: str, modelfiles: dict[str, str], dry_run: bool
) -> dict:
    """Upload all 4 GGUFs + their Modelfiles + the combined card.

    The 4 Modelfile bodies are passed in (they live locally in
    publish/modelfiles/, generated by build_modelfile.py).
    """
    import shutil

    from huggingface_hub import HfApi

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    info = REPOS["gguf"]
    staging = Path("/vol/gguf/_hf_staging")

    # Verify the GGUFs are present regardless of dry-run vs real,
    # so a missing-source bug surfaces in the dry-run preview.
    planned_ggufs: list[str] = []
    for src, dest_name in GGUF_LAYOUT:
        src_p = Path(src)
        if not src_p.exists():
            raise FileNotFoundError(
                f"GGUF missing at {src_p} — run export_gguf.py first."
            )
        planned_ggufs.append(dest_name)

    planned = planned_ggufs + [f"Modelfile.{tag}" for tag in modelfiles] + ["README.md"]

    if dry_run:
        return {
            "target": "gguf",
            "repo_id": info["repo_id"],
            "would_upload": planned,
            "staging_dir_when_real": str(staging),
            "card_bytes": len(card_text),
        }

    # Real upload path: stage the GGUFs under their published flat
    # filenames in a sibling dir on the volume, alongside the
    # Modelfiles + README. ``shutil.copy2`` (not ``os.link``) because
    # Modal volumes reject cross-path hard links — the snapshot
    # system can't track them.
    staging.mkdir(parents=True, exist_ok=True)
    for src, dest_name in GGUF_LAYOUT:
        dest = staging / dest_name
        dest.unlink(missing_ok=True)
        shutil.copy2(src, str(dest))
    for tag, body in modelfiles.items():
        (staging / f"Modelfile.{tag}").write_text(body)
    (staging / "README.md").write_text(card_text)

    api = HfApi(token=_read_token() or None)
    api.create_repo(info["repo_id"], repo_type=info["repo_type"], exist_ok=True)
    # See note on the model upload path above — same large-folder
    # rationale applies (4 GGUFs totalling ~19 GB).
    api.upload_large_folder(
        folder_path=str(staging),
        repo_id=info["repo_id"],
        repo_type=info["repo_type"],
    )
    return {
        "target": "gguf",
        "repo_id": info["repo_id"],
        "uploaded": planned,
    }


# ----------------------------------------------------------------------------
# Local-side helpers (pairs target)
# ----------------------------------------------------------------------------


def upload_pairs_local(card_text: str, dry_run: bool) -> dict:
    """Upload the dataset (small files, local disk → HF Dataset repo)."""
    info = REPOS["pairs"]
    files = list(PAIRS_FILES)
    missing = [p for p in files if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"pairs files missing: {missing}")

    if dry_run:
        return {
            "target": "pairs",
            "repo_id": info["repo_id"],
            "would_upload": files + ["README.md (from card)"],
            "card_bytes": len(card_text),
        }

    from huggingface_hub import HfApi

    api = HfApi(token=_read_token() or None)
    api.create_repo(info["repo_id"], repo_type=info["repo_type"], exist_ok=True)
    # Stage the dataset card next to the data files.
    staging = Path("publish/_pairs_staging")
    staging.mkdir(parents=True, exist_ok=True)
    for src in files:
        dest = staging / Path(src).name
        dest.write_bytes(Path(src).read_bytes())
    (staging / "README.md").write_text(card_text)
    api.upload_folder(
        folder_path=str(staging),
        repo_id=info["repo_id"],
        repo_type=info["repo_type"],
        commit_message="Stage 9 publish: pairs",
    )
    return {
        "target": "pairs",
        "repo_id": info["repo_id"],
        "uploaded": [Path(s).name for s in files] + ["README.md"],
    }


# ----------------------------------------------------------------------------
# Local entrypoint (CLI)
# ----------------------------------------------------------------------------


def _read_card(path_str: str) -> str:
    """Load a model card from disk; abort with a clear error if missing."""
    p = Path(path_str)
    if not p.exists():
        raise SystemExit(f"Model card missing at {p}. Author it before running upload.")
    return p.read_text()


def _read_modelfiles() -> dict[str, str]:
    """Read all 4 Modelfiles produced by build_modelfile.py."""
    mf_dir = Path("publish/modelfiles")
    if not mf_dir.exists():
        raise SystemExit(
            f"{mf_dir} not found — run "
            "`uv run python publish/build_modelfile.py` first."
        )
    out: dict[str, str] = {}
    for p in sorted(mf_dir.glob("Modelfile.*")):
        tag = p.name.removeprefix("Modelfile.")
        out[tag] = p.read_text()
    expected = {"Q8_0", "Q5_K_M", "Q8_0-sft", "Q5_K_M-sft"}
    if set(out) != expected:
        raise SystemExit(
            f"Modelfiles incomplete: have {sorted(out)}, expected {sorted(expected)}"
        )
    return out


@app.local_entrypoint()
def upload_remote(target: str = "gguf", confirm: bool = False) -> None:
    """Modal-backed upload entrypoint for volume-resident targets.

    Usage::

        modal run publish/upload_hf.py::upload_remote \\
            --target gguf --confirm

    Targets handled here: ``gguf``, ``model-sft``, ``model-dpo``.
    For ``pairs`` use the plain ``python publish/upload_hf.py`` CLI.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    validate_target(target)
    if target == "pairs":
        raise SystemExit(
            "Use `python publish/upload_hf.py --target pairs` for the "
            "dataset upload — it's small and runs locally."
        )

    info = REPOS[target]
    card_text = _read_card(info["card_path"])

    if target == "gguf":
        modelfiles = _read_modelfiles()
        result = upload_gguf_remote.remote(
            card_text=card_text, modelfiles=modelfiles, dry_run=not confirm
        )
    else:
        result = upload_model_remote.remote(
            target=target, card_text=card_text, dry_run=not confirm
        )

    import json

    print(json.dumps(result, indent=2, default=str))
    if not confirm:
        print("[upload_hf] DRY RUN — pass --confirm to actually upload.")


def _cli() -> None:
    """Plain-Python CLI for the local-only ``pairs`` target."""
    parser = argparse.ArgumentParser(
        description="Upload Stage 9 artifacts to Hugging Face."
    )
    parser.add_argument(
        "--target",
        choices=["pairs"],
        required=True,
        help=(
            "Local target. Use `modal run publish/upload_hf.py::upload_remote` "
            "for the volume-resident targets (gguf, model-sft, model-dpo)."
        ),
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required for actual upload. Without it, prints plan and exits.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.target == "pairs":
        card_text = _read_card(REPOS["pairs"]["card_path"])
        result = upload_pairs_local(card_text=card_text, dry_run=not args.confirm)
        import json

        print(json.dumps(result, indent=2, default=str))
        if not args.confirm:
            print("[upload_hf] DRY RUN — pass --confirm to actually upload.")


if __name__ == "__main__":
    _cli()


# Re-export for tests
__all__ = [
    "REPOS",
    "GGUF_LAYOUT",
    "UPLOAD_ORDER",
    "PAIRS_FILES",
    "validate_target",
]
