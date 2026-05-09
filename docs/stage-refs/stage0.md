# Stage 0 — Repo bootstrap

**Prompt:** [`docs/claude-code-prompts.md` § Stage 0: Repo bootstrap](../claude-code-prompts.md#stage-0-repo-bootstrap)

**Scripts:**
- `pyproject.toml` — uv-managed dependencies (Unsloth, TRL, PEFT, transformers, openai, anthropic, etc.). Python 3.11.
- `scripts/common.py` — shared helpers: `load_env`, `jsonl_read`, `jsonl_append`, `already_processed`.

**Inputs:**
- None. This is project skeleton creation.

**Outputs:**
- Directory tree: `data/{raw,pairs,labeled,formatted}/`, `train/configs/`, `outputs/`, `eval/`, `deployment/`.
- `.env.example` — placeholder for `OPENROUTER_API_KEY`, `TOGETHER_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `HF_TOKEN`.
- `.gitignore` — covers `.env`, `outputs/`, `*.jsonl` in `data/`, model checkpoints.

**Decisions made:**
- None at this stage. The first methodological decisions (#1 OpenRouter switch, #2 small generator pool) land in Stage 1.

**Key outputs:**
- A reusable repo skeleton. The `scripts/common.py` helpers (`already_processed`) are what make every later stage resumable — skip records already in the output JSONL on rerun.
