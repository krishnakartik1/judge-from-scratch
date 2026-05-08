"""Stage 9 — Generate Ollama Modelfiles for all four GGUF variants.

Reads ``data/judge_system_prompt.md`` and emits four Modelfiles, one
per Ollama tag in the published GGUF repo:

- ``publish/modelfiles/Modelfile.Q8_0``       (DPO Q8_0, default)
- ``publish/modelfiles/Modelfile.Q5_K_M``     (DPO Q5_K_M)
- ``publish/modelfiles/Modelfile.Q8_0-sft``   (SFT-only Q8_0)
- ``publish/modelfiles/Modelfile.Q5_K_M-sft`` (SFT-only Q5_K_M)

GGUF repo layout these Modelfiles assume (produced by
:mod:`publish.upload_hf`)::

    krishnakartik/gemma4-social-bias-judge-gguf/
    ├── README.md
    ├── Q8_0.gguf       Q5_K_M.gguf       (DPO; default tags)
    ├── Q8_0-sft.gguf   Q5_K_M-sft.gguf   (SFT-only)
    └── Modelfile.<tag> for each tag above

Each Modelfile embeds the judge system prompt with a thinking-mode
warning comment — this is the third of three placements of that
warning per decision #13 (model card §"Important: Thinking Mode" +
quick-start code comment + here).

Run from the project root::

    uv run python publish/build_modelfile.py

Local-only; no Modal, no GPU. The script asserts no ``<|think|>``
in the system prompt at startup so a future edit to
``judge_system_prompt.md`` that accidentally enables thinking mode
fails fast.
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Reuse the eval harness's assertion so the gate is exactly the
# train/infer-parity check that decision #13 mandates.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.eval_harness import assert_no_thinking_in_prompt  # noqa: E402

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
SYSTEM_PROMPT_PATH = Path("data/judge_system_prompt.md")
OUTPUT_DIR = Path("publish/modelfiles")

# (model_name, quant, ollama_tag, gguf_filename) for each variant.
# DPO is the default — its tags are the bare quant names. SFT tags are
# suffixed with ``-sft`` so a single GGUF repo serves both checkpoints
# while keeping the DPO one-liner short.
MODEL_VARIANTS: tuple[tuple[str, str, str, str], ...] = (
    ("dpo", "Q8_0", "Q8_0", "Q8_0.gguf"),
    ("dpo", "Q5_K_M", "Q5_K_M", "Q5_K_M.gguf"),
    ("sft", "Q8_0", "Q8_0-sft", "Q8_0-sft.gguf"),
    ("sft", "Q5_K_M", "Q5_K_M-sft", "Q5_K_M-sft.gguf"),
)

# Per Stage 9 prompt: temperature 0 and num_ctx 2048 for production
# judge use. Bumping num_ctx to 4096 to match the eval harness's
# MAX_MODEL_LEN; the eval verified prompts fit comfortably and a
# 4K context is the floor for any non-trivial use case.
TEMPERATURE = 0
NUM_CTX = 4096

THINKING_MODE_WARNING = """\
# IMPORTANT: this judge was fine-tuned with Gemma 4's native thinking
# mode DISABLED. Do NOT modify the SYSTEM block below to enable
# thinking mode — specifically, do NOT add `<|think|>` to the system
# text. Doing so routes the model into a generation path it never
# saw during training and produces degraded, unparseable output.
# The eval-harness assert_no_thinking_in_prompt() check is the
# single source of truth for this invariant; see decision #13 in
# the project status doc."""


def render_modelfile(gguf_filename: str, system_prompt: str) -> str:
    """Compose the Modelfile body for one variant.

    ``gguf_filename`` is the relative path to the sibling GGUF file
    inside the GGUF repo (flat layout — see module docstring), e.g.
    ``"Q8_0.gguf"`` or ``"Q5_K_M-sft.gguf"``.
    """
    return (
        f"FROM ./{gguf_filename}\n"
        f"PARAMETER temperature {TEMPERATURE}\n"
        f"PARAMETER num_ctx {NUM_CTX}\n"
        f"\n"
        f"{THINKING_MODE_WARNING}\n"
        f'SYSTEM """{system_prompt.rstrip()}"""\n'
    )


def main() -> None:
    """Read system prompt, assert thinking-mode invariant, write 4 Modelfiles."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not SYSTEM_PROMPT_PATH.exists():
        raise SystemExit(f"System prompt not found at {SYSTEM_PROMPT_PATH}")
    system_prompt = SYSTEM_PROMPT_PATH.read_text()
    assert_no_thinking_in_prompt(system_prompt)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for _model_name, _quant, tag, gguf_filename in MODEL_VARIANTS:
        body = render_modelfile(gguf_filename, system_prompt)
        out_path = OUTPUT_DIR / f"Modelfile.{tag}"
        out_path.write_text(body)
        logger.info("Wrote %s (%d bytes)", out_path, len(body))

    print(
        f"[build_modelfile] wrote {len(MODEL_VARIANTS)} Modelfiles to " f"{OUTPUT_DIR}/"
    )


if __name__ == "__main__":
    main()


# Re-export for tests
__all__ = [
    "MODEL_VARIANTS",
    "TEMPERATURE",
    "NUM_CTX",
    "THINKING_MODE_WARNING",
    "render_modelfile",
]
