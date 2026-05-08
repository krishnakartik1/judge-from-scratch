"""Stage 8 vLLM-based eval pipeline.

Replaces ``eval/modal/run_eval.py`` for the precision-parity rerun.
The legacy file stays as a frozen reference; this module does not
share code paths with it. The metric layer (``eval/eval_harness.py``)
is reused unchanged — only the inference backend differs.

Architecture overview:

- :class:`VllmRunner` is a Modal class with ``min_containers=1``,
  ``max_containers=1`` and a ``model_path`` parameter, so each unique
  checkpoint pins one warm container that loads vLLM exactly once.
  Seven sequential ``infer.remote()`` calls per checkpoint share that
  load (300 original + 300 swapped + 5 × 300 consistency = 2100
  prompts written into the cache).
- Local entrypoints :func:`run_model`, :func:`run_all` orchestrate
  the per-checkpoint runs; :func:`smoke_test` validates parse,
  tokenizer parity, prompt-length headroom, and VRAM before any
  full run; :func:`collect_results` reads the per-model cache files
  and renders the 11-row Appendix C table.

Cache schema (one JSONL row per inference, written from inside the
container so the volume mount is live)::

    {model, pair_id, run_type, verdict, reasoning, raw_output,
     prompt_hash, temperature, swapped, run_index}

``run_type`` ↔ ``(temperature, swapped, run_index)`` is a bijection
implemented in :func:`run_type_to_partition` /
:func:`partition_to_run_type`; the metric layer never sees
``run_type``.

Run from project root::

    modal run eval/modal/vllm_infer.py::smoke_test
    modal run eval/modal/vllm_infer.py::run_all
    modal run eval/modal/vllm_infer.py::collect_results

Note: this module deliberately does NOT use ``from __future__ import
annotations``. Modal's class-parameter validation evaluates the
``model_path: str = modal.parameter()`` annotation at class-definition
time and would see ``"str"`` (a string) rather than ``str`` (the type)
under future-annotations, raising InvalidError. All other type hints
in the file are valid Python 3.11 syntax (``list[str]``, ``X | None``).
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import modal

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Paths inside the Modal container
# ----------------------------------------------------------------------------
PKG_DIR = "/root/eval_pkg"
SYSTEM_PROMPT_REMOTE = f"{PKG_DIR}/judge_system_prompt.md"
EVAL_SET_REMOTE = f"{PKG_DIR}/eval_set.jsonl"
CACHE_DIR_VOL = "/vol/eval/cache"
LEGACY_CACHE_DIR_VOL = f"{CACHE_DIR_VOL}/legacy_unsloth"

# ----------------------------------------------------------------------------
# Local paths (resolved relative to project root)
# ----------------------------------------------------------------------------
LOCAL_EVAL_SET = Path("data/pairs/eval_set_unlabeled.jsonl")
LOCAL_SYSTEM_PROMPT = Path("data/judge_system_prompt.md")
RESULTS_DIR = Path("eval/results")

# ----------------------------------------------------------------------------
# Model registry (display name → volume path)
# ----------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "baseline": {
        "path": "/vol/checkpoints/baseline-bf16",
        "display_name": "Base Gemma 4 E4B",
    },
    "sft": {
        "path": "/vol/checkpoints/sft-merged-fp16",
        "display_name": "After SFT",
    },
    "dpo": {
        "path": "/vol/checkpoints/merged-fp16",
        "display_name": "After SFT+DPO",
    },
}

# Tokenizer used both for local prompt rendering (Step 2) and for
# verbosity-bias scoring (collect_results). Stage 5 used the same
# mirror for training-set construction; the smoke-test parity check
# proves this matches the on-disk merged-checkpoint tokenizers
# byte-for-byte.
HF_TOKENIZER_ID = "unsloth/gemma-4-E4B-it"

# ----------------------------------------------------------------------------
# Sampling and budget constants
# ----------------------------------------------------------------------------
# Bumped from 384 (the original eval-harness default per status doc
# decision #24) after the v1 vLLM run-all hit ~10% truncation on SFT
# and DPO — post-training reasoning regularly exceeds 384 generated
# tokens on the harder (subtle / tracked / adversarial) cases. Dropping
# those rows from the κ denominator would systematically inflate κ
# because the truncated cases are exactly the ones the model struggles
# with. 1024 leaves PROMPT_TOKEN_CEILING = 4096 - 1024 - 32 = 3040,
# well above the smoke-test max_prompt_tokens of 732.
MAX_NEW_TOKENS = 1024
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.85
PROMPT_TOKEN_SAFETY_MARGIN = 32  # vLLM may reserve a few special tokens
PROMPT_TOKEN_CEILING = MAX_MODEL_LEN - MAX_NEW_TOKENS - PROMPT_TOKEN_SAFETY_MARGIN

CONSISTENCY_RUN_INDEXES: tuple[int, ...] = (0, 1, 2, 3, 4)
CONSISTENCY_TEMPERATURE = 0.3
INFER_TIMEOUT_S = 1800  # per-method-call timeout
CLS_TIMEOUT_S = 3600

# ----------------------------------------------------------------------------
# Modal app
# ----------------------------------------------------------------------------
MODAL_GPU = os.environ.get("MODAL_GPU", "A100-80GB")
VOLUME_NAME = "judge-from-scratch"

app = modal.App("judge-vllm-eval")

# vLLM image: nightly + cu129 + transformers >= 5.5.0 — the only
# combination that loads Gemma 4 E4B (released April 2026, arch
# `Gemma4ForConditionalGeneration`). Stable vllm 0.8.x errors on
# rope_scaling at config-load time; vllm 0.19 stable pins
# transformers <= 4.57.6 which doesn't recognize the gemma4 model
# class. The vllm-project recipes
# (https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html)
# explicitly prescribe the nightly path with cu129 wheels.
#
# Base image MUST be a CUDA-devel image (with ``nvcc`` on PATH),
# not ``debian_slim``: vllm-nightly uses flashinfer for top-k/top-p
# sampling, and flashinfer JIT-compiles its sampling kernels at
# first invocation via the host nvcc. On a thin debian_slim base
# this fails with ``Could not find nvcc and default
# cuda_home='/usr/local/cuda' doesn't exist`` during the engine's
# VRAM-profile dummy run.
#
# Modal hosts ship NVIDIA driver 580+ which supports CUDA 13.0
# runtime, so cu129 wheels load cleanly on the A100-80GB.
#
# We use ``run_commands`` rather than ``pip_install`` because the
# recipe needs two ``--extra-index-url`` flags plus
# ``--index-strategy unsafe-best-match`` (uv-only); ``pip_install``'s
# kwargs only expose one extra index.
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.11")
    .run_commands(
        [
            "pip install --no-cache-dir uv",
            (
                "uv pip install --system --no-cache -U vllm --pre "
                "--extra-index-url https://wheels.vllm.ai/nightly/cu129 "
                "--extra-index-url https://download.pytorch.org/whl/cu129 "
                "--index-strategy unsafe-best-match"
            ),
            (
                "uv pip install --system --no-cache -U "
                "'transformers>=5.5.0' 'huggingface_hub>=1.0.0' "
                "scikit-learn"
            ),
        ]
    )
    .add_local_file(
        "data/_format_helpers.py", f"{PKG_DIR}/_format_helpers.py", copy=True
    )
    .add_local_file("data/judge_system_prompt.md", SYSTEM_PROMPT_REMOTE, copy=True)
    .add_local_file("data/pairs/eval_set_unlabeled.jsonl", EVAL_SET_REMOTE, copy=True)
    .add_local_file("eval/eval_harness.py", f"{PKG_DIR}/eval_harness.py", copy=True)
    .add_local_python_source("data")
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]


# ============================================================================
# Pure helpers (importable for unit tests; no Modal/vLLM imports)
# ============================================================================


VALID_RUN_TYPES: tuple[str, ...] = (
    "original",
    "swapped",
    "consistency_0",
    "consistency_1",
    "consistency_2",
    "consistency_3",
    "consistency_4",
)


def run_type_to_partition(run_type: str) -> tuple[float, bool, int]:
    """Translate ``run_type`` to ``(temperature, swapped, run_index)``.

    The metric layer in ``eval_harness.py`` partitions cached
    predictions by this triple; ``run_type`` is the human-readable
    label written into the JSONL cache for debugging.
    """
    if run_type == "original":
        return (0.0, False, 0)
    if run_type == "swapped":
        return (0.0, True, 0)
    if run_type.startswith("consistency_"):
        idx_part = run_type[len("consistency_") :]
        if not idx_part.isdigit():
            raise ValueError(f"Unknown run_type: {run_type!r}")
        idx = int(idx_part)
        if idx not in CONSISTENCY_RUN_INDEXES:
            raise ValueError(
                f"consistency run index {idx} out of range "
                f"{CONSISTENCY_RUN_INDEXES}"
            )
        return (CONSISTENCY_TEMPERATURE, False, idx)
    raise ValueError(f"Unknown run_type: {run_type!r}")


def partition_to_run_type(temperature: float, swapped: bool, run_index: int) -> str:
    """Inverse of :func:`run_type_to_partition`. Strict — raises on unknown."""
    if temperature == 0.0 and not swapped and run_index == 0:
        return "original"
    if temperature == 0.0 and swapped and run_index == 0:
        return "swapped"
    if (
        temperature == CONSISTENCY_TEMPERATURE
        and not swapped
        and run_index in CONSISTENCY_RUN_INDEXES
    ):
        return f"consistency_{run_index}"
    raise ValueError(
        f"No run_type for partition (T={temperature!r}, "
        f"swapped={swapped!r}, run_index={run_index!r})"
    )


PREDICTION_KEYS: tuple[str, ...] = (
    "pair_id",
    "verdict",
    "reasoning",
    "raw_output",
    "prompt_hash",
    "temperature",
    "swapped",
    "run_index",
)
EXTRA_CACHE_KEYS: tuple[str, ...] = ("model", "run_type")


def to_prediction(row: dict[str, Any]) -> Any:
    """Reconstruct an ``eval_harness.Prediction`` from one cache row.

    Strips the cache-only keys (``model``, ``run_type``) so the dict
    matches the dataclass's 8 fields exactly. Importing here keeps
    the helper usable from contexts that don't have ``Prediction``
    in scope at module load time.
    """
    from eval.eval_harness import Prediction

    rest = {k: v for k, v in row.items() if k not in EXTRA_CACHE_KEYS}
    missing = set(PREDICTION_KEYS) - rest.keys()
    if missing:
        raise ValueError(f"Cache row missing fields: {sorted(missing)}")
    return Prediction(**rest)


def build_cache_row(
    model_name: str,
    run_type: str,
    pair_id: str,
    raw_output: str,
    prompt_hash_str: str,
) -> dict[str, Any]:
    """Build one JSONL cache row from a generation result.

    Imports ``parse_output`` lazily so this helper is importable on
    a CPU-only test runner. The (temperature, swapped, run_index)
    tuple is derived from ``run_type`` for the schema's machine-key
    half.
    """
    from eval.eval_harness import parse_output

    temperature, swapped, run_index = run_type_to_partition(run_type)
    verdict, reasoning = parse_output(raw_output)
    return {
        "model": model_name,
        "pair_id": pair_id,
        "run_type": run_type,
        "verdict": verdict,
        "reasoning": reasoning,
        "raw_output": raw_output,
        "prompt_hash": prompt_hash_str,
        "temperature": temperature,
        "swapped": swapped,
        "run_index": run_index,
    }


def select_consistency_pair_ids(records: list[dict[str, Any]]) -> list[str]:
    """All pair_ids in deterministic sorted order — used to drive every
    eval pass identically across checkpoints."""
    return sorted(r["pair_id"] for r in records)


def render_prompts_for_pass(
    records: list[dict[str, Any]],
    tokenizer: Any,
    system_prompt: str,
    *,
    swap: bool,
) -> tuple[list[str], list[dict[str, str]]]:
    """Render (apply_chat) the full eval set for one pass.

    Returns ``(prompts, metadata)`` where ``metadata[i]`` carries the
    ``pair_id`` and pre-computed ``prompt_hash`` for the i-th prompt.
    Pure: no Modal calls, no GPU.
    """
    from data._format_helpers import apply_chat, build_user_message
    from eval.eval_harness import (
        assert_no_thinking_in_prompt,
        prompt_hash,
    )

    assert_no_thinking_in_prompt(system_prompt)
    prompts: list[str] = []
    metadata: list[dict[str, str]] = []
    for record in records:
        user = build_user_message(record, swap=swap)
        rendered = apply_chat(tokenizer, system_prompt, user)
        prompts.append(rendered)
        metadata.append(
            {
                "pair_id": record["pair_id"],
                "prompt_hash": prompt_hash(system_prompt, user),
            }
        )
    return prompts, metadata


def validate_model_name(name: str) -> None:
    """Reject unknown model names with a clear list of valid options."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}; valid: {sorted(MODEL_REGISTRY)}")


# ============================================================================
# Modal-side helpers
# ============================================================================


@app.function(
    image=vllm_image,
    volumes={"/vol": volume},
    timeout=300,
)
def assert_paths_exist(paths: list[str]) -> dict[str, bool]:
    """Confirm each volume path has a ``tokenizer.json``.

    Used as the smoke-test pre-flight: fails fast with a clear
    message rather than letting a downstream tokenizer-load crash
    obscure the missing-checkpoint root cause.
    """
    out: dict[str, bool] = {}
    for path in paths:
        out[path] = (Path(path) / "tokenizer.json").exists()
    missing = [p for p, ok in out.items() if not ok]
    if missing:
        raise FileNotFoundError(
            f"Step 0 must populate checkpoint paths: {missing}. "
            "Re-run download_baseline.py / merge_adapter.py."
        )
    return out


def _do_relocate_legacy_cache(cache_dir: Path, legacy_dir: Path) -> dict[str, Any]:
    """Pure helper: move top-level ``*_predictions.jsonl`` into ``legacy_dir``.

    Extracted from the Modal function so the file-system semantics
    (auto-create dest dir, per-file dest path, no-op when empty)
    are unit-testable on CPU without a Modal volume.
    """
    import shutil

    cache_dir.mkdir(parents=True, exist_ok=True)
    legacy_dir.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for src in sorted(cache_dir.glob("*_predictions.jsonl")):
        dest = legacy_dir / src.name
        shutil.move(str(src), str(dest))
        moved.append(src.name)
    return {"moved": moved, "count": len(moved)}


@app.function(
    image=vllm_image,
    volumes={"/vol": volume},
    timeout=300,
)
def relocate_legacy_cache() -> dict[str, Any]:
    """Move any top-level ``*_predictions.jsonl`` into ``legacy_unsloth/``.

    Idempotent on the file-presence trigger: a partial-failure resume
    finds nothing left to move and returns ``count=0``.
    """
    result = _do_relocate_legacy_cache(Path(CACHE_DIR_VOL), Path(LEGACY_CACHE_DIR_VOL))
    if result["count"] > 0:
        volume.commit()
    return result


@app.function(
    image=vllm_image,
    volumes={"/vol": volume},
    timeout=600,
    secrets=secrets,
)
def render_via_volume_tokenizer(
    model_path: str, system_prompt: str, user_message: str
) -> str:
    """Load ``AutoTokenizer.from_pretrained(model_path)`` and render once.

    Used by the smoke test's tokenizer-parity check: the same
    (system, user) pair is rendered through each on-disk tokenizer
    and the local HF-mirror tokenizer; the local side asserts all
    four return identical strings.
    """
    sys.path.insert(0, PKG_DIR)
    from transformers import AutoTokenizer

    from data._format_helpers import apply_chat

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return apply_chat(tokenizer, system_prompt, user_message)


# ============================================================================
# VllmRunner — one warm container per model_path
# ============================================================================


@app.cls(
    image=vllm_image,
    gpu=MODAL_GPU,
    timeout=CLS_TIMEOUT_S,
    volumes={"/vol": volume},
    secrets=secrets,
    scaledown_window=600,
)
class VllmRunner:
    """vLLM bf16 inference, one model load amortized over all calls.

    Modal partitions the container pool by parameter value, so each
    unique ``model_path`` gets its own warm container. Sequential
    ``infer.remote()`` calls on the same instance route to that
    container while it stays warm; ``scaledown_window=600`` (10 min)
    keeps the container alive between back-to-back calls inside one
    ``run_model`` invocation, so the seven calls per checkpoint share
    a single vLLM load.

    Modal forbids ``min_containers > 0`` on parameterized classes
    (the parameter-pool can't be pre-warmed without knowing the
    parameter values), so the long ``scaledown_window`` is the
    closest equivalent to manual pinning.
    """

    model_path: str = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        sys.path.insert(0, PKG_DIR)
        import torch
        from vllm import LLM

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        logger.info("Loading vLLM with model=%s", self.model_path)
        self.llm = LLM(
            model=self.model_path,
            dtype="bfloat16",
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            enforce_eager=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.vram_after_load_gb = torch.cuda.memory_allocated() / 1e9
        logger.info("vLLM ready: vram_after_load=%.2f GB", self.vram_after_load_gb)

    @modal.method()
    def render_and_infer(
        self,
        swap: bool,
        temperature: float,
        seed: int,
        pair_ids: list[str] | None = None,
        max_tokens: int = MAX_NEW_TOKENS,
    ) -> dict[str, Any]:
        """Render prompts on-container, then run vLLM batched inference.

        Loads eval set + system prompt from the image-mounted files,
        renders via ``self.tokenizer`` (vLLM's bundled tokenizer is
        transformers 5.5+ compatible — required for Gemma 4's
        ``chat_template.jinja``-based template). Optionally subsets
        records to ``pair_ids`` (smoke-test path).

        Returns texts, per-prompt metadata (pair_id + prompt_hash),
        and probe stats (max_prompt_tokens, wallclock, VRAM).
        ``seed`` is required — defaulting it would collapse 5 distinct
        consistency runs to identical samples.
        """
        sys.path.insert(0, PKG_DIR)
        from vllm import SamplingParams

        # Modal mounts ``data/`` via add_local_python_source (importable
        # as ``data._format_helpers``) and ``eval/eval_harness.py`` as a
        # bare file at PKG_DIR (so it imports as ``eval_harness`` once
        # PKG_DIR is on sys.path). The try/except mirrors the pattern in
        # eval/eval_harness.py for the same reason.
        try:
            from data._format_helpers import apply_chat, build_user_message
        except ImportError:
            from _format_helpers import (  # type: ignore[no-redef]
                apply_chat,
                build_user_message,
            )
        try:
            from eval.eval_harness import (
                assert_no_thinking_in_prompt,
                load_eval_set,
                prompt_hash,
            )
        except ImportError:
            from eval_harness import (  # type: ignore[no-redef]
                assert_no_thinking_in_prompt,
                load_eval_set,
                prompt_hash,
            )

        system_prompt = Path(SYSTEM_PROMPT_REMOTE).read_text()
        assert_no_thinking_in_prompt(system_prompt)
        records = load_eval_set(Path(EVAL_SET_REMOTE))
        if pair_ids is not None:
            wanted = set(pair_ids)
            records = [r for r in records if r["pair_id"] in wanted]
            missing = wanted - {r["pair_id"] for r in records}
            if missing:
                raise ValueError(f"pair_ids not in eval set: {sorted(missing)}")

        prompts: list[str] = []
        metadata: list[dict[str, str]] = []
        for record in records:
            user = build_user_message(record, swap=swap)
            rendered = apply_chat(self.tokenizer, system_prompt, user)
            prompts.append(rendered)
            metadata.append(
                {
                    "pair_id": record["pair_id"],
                    "prompt_hash": prompt_hash(system_prompt, user),
                }
            )

        t0 = time.time()
        sp = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            stop=None,
        )
        outputs = self.llm.generate(prompts, sp)
        wallclock_s = time.time() - t0
        max_prompt_tokens = max(len(self.tokenizer.encode(p)) for p in prompts)
        return {
            "texts": [o.outputs[0].text for o in outputs],
            "metadata": metadata,
            "max_prompt_tokens": max_prompt_tokens,
            "wallclock_s": wallclock_s,
            "vram_after_load_gb": self.vram_after_load_gb,
        }

    @modal.method()
    def read_vram(self) -> float:
        """Smoke-test helper: peak VRAM after model load."""
        return self.vram_after_load_gb

    @modal.method()
    def write_cache(self, model_name: str, rows: list[dict[str, Any]]) -> int:
        """Write JSONL cache file and commit the volume.

        Runs on the same container that holds the LLM so the volume
        mount is live. The local entrypoint cannot write to ``/vol``,
        so the write must happen here.
        """
        path = Path(f"{CACHE_DIR_VOL}/{model_name}_predictions.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        volume.commit()
        logger.info("Wrote %d rows to %s", len(rows), path)
        return len(rows)


# ============================================================================
# Local entrypoints
# ============================================================================


def _load_local_records() -> list[dict[str, Any]]:
    """Load the eval set from the local working tree."""
    from eval.eval_harness import load_eval_set

    return load_eval_set(LOCAL_EVAL_SET)


def _load_local_system_prompt() -> str:
    """Read the system prompt from the local working tree."""
    return LOCAL_SYSTEM_PROMPT.read_text()


def _git_sha() -> str | None:
    """Best-effort git HEAD sha for the results metadata."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


@app.local_entrypoint()
def smoke_test() -> None:
    """Validate parse, tokenizer parity, prompt length, VRAM before run_all.

    Runs against the baseline checkpoint with 4 prompts: 2 sorted by
    ``pair_id`` (smallest), 2 with the longest combined response
    text (prompt-length stress proxy). Aborts cleanly on any of:

    - tokenizer drift across the 3 on-disk checkpoints
    - max_prompt_tokens > PROMPT_TOKEN_CEILING
    - parse_output miss on any of the 4 generated texts
    - presence of ``<think>`` in any output (Gemma 4 thinking-mode leak)

    Local transformers (4.57 via the unsloth lockfile pin) cannot
    render Gemma 4's ``chat_template.jinja``-based chat template, so
    all rendering happens Modal-side using the on-container
    transformers ≥5.5 stack inside :class:`VllmRunner`.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from eval.eval_harness import parse_output

    paths = [info["path"] for info in MODEL_REGISTRY.values()]
    print(f"[smoke] pre-flight: assert_paths_exist {paths}")
    assert_paths_exist.remote(paths)

    records = _load_local_records()
    sorted_records = sorted(records, key=lambda r: r["pair_id"])
    by_length = sorted(
        records,
        key=lambda r: -(len(r["response_a"]["text"]) + len(r["response_b"]["text"])),
    )
    sample_records = sorted_records[:2] + by_length[:2]
    sample_pair_ids = [r["pair_id"] for r in sample_records]
    print(f"[smoke] selected {len(sample_records)} sample pair_ids: {sample_pair_ids}")

    system_prompt = _load_local_system_prompt()
    from data._format_helpers import build_user_message

    sample_user = build_user_message(sample_records[0], swap=False)

    print("[smoke] tokenizer parity check across 3 on-disk checkpoints")
    parity_results: dict[str, str] = {}
    for path in paths:
        rendered = render_via_volume_tokenizer.remote(path, system_prompt, sample_user)
        parity_results[path] = rendered
    distinct = set(parity_results.values())
    if len(distinct) != 1:
        for label, text in parity_results.items():
            print(f"[smoke] {label}: first 200 chars = {text[:200]!r}")
        raise AssertionError(
            "Tokenizer drift detected across "
            f"{list(parity_results)} — precision-parity comparison "
            "would be invalid."
        )
    print("[smoke] tokenizer parity OK (3 on-disk byte-identical)")

    baseline_path = MODEL_REGISTRY["baseline"]["path"]
    runner = VllmRunner(model_path=baseline_path)
    print(
        f"[smoke] runner.render_and_infer on baseline "
        f"({len(sample_pair_ids)} prompts)"
    )
    result = runner.render_and_infer.remote(
        swap=False,
        temperature=0.0,
        seed=0,
        pair_ids=sample_pair_ids,
    )

    print(f"[smoke] vram_after_load_gb = {result['vram_after_load_gb']:.2f}")
    print(f"[smoke] wallclock_s        = {result['wallclock_s']:.2f}")
    print(f"[smoke] max_prompt_tokens  = {result['max_prompt_tokens']}")
    if result["max_prompt_tokens"] > PROMPT_TOKEN_CEILING:
        raise AssertionError(
            f"max_prompt_tokens={result['max_prompt_tokens']} > "
            f"ceiling {PROMPT_TOKEN_CEILING}; raise MAX_MODEL_LEN."
        )

    failures: list[str] = []
    for i, text in enumerate(result["texts"]):
        pair_id = result["metadata"][i]["pair_id"]
        verdict, _ = parse_output(text)
        print(f"[smoke] --- output {i} (pair_id={pair_id}) ---")
        print(text)
        print(f"[smoke] parsed verdict: {verdict}")
        if verdict == "PARSE_FAIL":
            failures.append(f"prompt {i} (pair_id={pair_id})")
        if "<think>" in text or "<thinking>" in text:
            failures.append(
                f"prompt {i} emitted thinking-mode block (decision #13 violation)"
            )
    if failures:
        raise AssertionError("Smoke-test parse failures: " + "; ".join(failures))

    print("[smoke] OK — proceed to run_all")


def _run_model_logic(model_name: str) -> None:
    """Body of :func:`run_model`, factored so :func:`run_all` can call it.

    ``@app.local_entrypoint`` decorators don't expose a callable
    ``.local`` shim the way ``@app.function`` does, so the only safe
    way to invoke run_model from another local entrypoint is to call
    a plain Python helper that both wrap.
    """
    validate_model_name(model_name)
    info = MODEL_REGISTRY[model_name]
    model_path = info["path"]
    print(f"[run_model] {model_name} → {model_path}")

    runner = VllmRunner(model_path=model_path)

    rows: list[dict[str, Any]] = []

    def _run_pass(run_type: str) -> int:
        temperature, swapped, run_index = run_type_to_partition(run_type)
        seed = run_index
        t0 = time.time()
        result = runner.render_and_infer.remote(
            swap=swapped,
            temperature=temperature,
            seed=seed,
        )
        elapsed = time.time() - t0
        if result["max_prompt_tokens"] > PROMPT_TOKEN_CEILING:
            raise AssertionError(
                f"{run_type}: max_prompt_tokens={result['max_prompt_tokens']} "
                f"> ceiling {PROMPT_TOKEN_CEILING}"
            )
        n_failures = 0
        for text, meta in zip(result["texts"], result["metadata"], strict=True):
            row = build_cache_row(
                model_name=model_name,
                run_type=run_type,
                pair_id=meta["pair_id"],
                raw_output=text,
                prompt_hash_str=meta["prompt_hash"],
            )
            rows.append(row)
            if row["verdict"] == "PARSE_FAIL":
                n_failures += 1
        print(
            f"[run_model] {run_type}: {len(result['texts'])} prompts in "
            f"{elapsed:.1f}s, {n_failures} parse failures"
        )
        return n_failures

    total_t0 = time.time()
    n_fail_total = 0
    n_fail_total += _run_pass("original")
    n_fail_total += _run_pass("swapped")
    for run_index in CONSISTENCY_RUN_INDEXES:
        n_fail_total += _run_pass(f"consistency_{run_index}")

    print(f"[run_model] writing {len(rows)} rows to volume cache")
    n_written = runner.write_cache.remote(model_name, rows)
    elapsed_total = time.time() - total_t0
    fail_pair_ids = sorted(
        {row["pair_id"] for row in rows if row["verdict"] == "PARSE_FAIL"}
    )
    print(
        f"[run_model] {model_name} done: {n_written} rows, "
        f"{n_fail_total} parse failures across {len(fail_pair_ids)} unique "
        f"pair_ids, total wall-clock {elapsed_total:.1f}s"
    )
    if fail_pair_ids:
        print(f"[run_model] parse-fail pair_ids: {fail_pair_ids}")

    _record_run_cost(
        model_name=model_name,
        wallclock_s=elapsed_total,
        n_rows=n_written,
        n_parse_fails=n_fail_total,
    )


@app.local_entrypoint()
def run_model(model_name: str = "baseline") -> None:
    """Run all 7 inference passes for one checkpoint and persist to volume.

    Usage::

        modal run eval/modal/vllm_infer.py::run_model --model-name baseline
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    _run_model_logic(model_name)


def _record_run_cost(
    *, model_name: str, wallclock_s: float, n_rows: int, n_parse_fails: int
) -> None:
    """Append one Stage 8 entry to the cost ledger.

    Local-only call; the ledger lives at ``train/.cost_ledger.jsonl``
    and is read/written by ``train/modal/_cost_ledger.py``.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from train.modal._cost_ledger import record_cost  # type: ignore

    record_cost(
        stage="stage8",
        function=f"vllm_infer.{model_name}",
        gpu=MODAL_GPU,
        wallclock_s=wallclock_s,
        notes=(
            f"vLLM bf16 inference; {n_rows} rows; " f"{n_parse_fails} parse failures"
        ),
    )


@app.local_entrypoint()
def run_all() -> None:
    """Pre-flight, then sequentially run baseline, sft, dpo.

    Pre-flight steps:
      1. Confirm volume paths exist.
      2. Budget gate via :mod:`train.modal._cost_ledger`.
      3. Relocate any legacy 4-bit cache files to legacy_unsloth/.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from train.modal._cost_ledger import (  # type: ignore
        LEDGER_PATH,
        STAGE8_BUDGET_CAP_USD,
        check_budget,
        project_cost,
    )

    paths = [info["path"] for info in MODEL_REGISTRY.values()]
    print(f"[run_all] pre-flight: paths = {paths}")
    assert_paths_exist.remote(paths)

    # Worst-case under @app.cls amortization: 3 models × 30-min ceiling.
    # Compare against vllm-pivot spend only — not project-wide
    # ``total_spend()`` (which would pull in Stages 6/7), and not all
    # ``stage="stage8"`` rows either: legacy ``run_eval.py`` attempts
    # before this pivot burned ~$20 chasing the 4-bit Unsloth path
    # that failed for billing-cap and precision-confound reasons. Those
    # are sunk cost; the $10 cap envelopes the NEW backend's spend.
    pivot_spent = 0.0
    if LEDGER_PATH.exists():
        with open(LEDGER_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("stage") == "stage8" and str(
                    row.get("function", "")
                ).startswith("vllm_infer"):
                    pivot_spent += float(row.get("est_cost_usd", 0.0))
    worst_case = project_cost(MODAL_GPU, timeout_s=1800 * 3)
    check_budget(
        projected_usd=worst_case,
        cap_usd=STAGE8_BUDGET_CAP_USD,
        spent_usd=pivot_spent,
        label="stage8-vllm-pivot",
    )

    relocate_result = relocate_legacy_cache.remote()
    if relocate_result["count"] > 0:
        print(
            f"[run_all] moved {relocate_result['count']} legacy cache files "
            f"to {LEGACY_CACHE_DIR_VOL}: {relocate_result['moved']}"
        )

    for model_name in ("baseline", "sft", "dpo"):
        print(f"[run_all] === {model_name} ===")
        _run_model_logic(model_name)

    print("[run_all] all three models complete; run collect_results next")


def _load_cache_via_volume_get(model_name: str, dest_dir: Path) -> Path:
    """Download one cache file from /vol/eval/cache/ via ``modal volume get``."""
    remote_path = f"eval/cache/{model_name}_predictions.jsonl"
    local_path = dest_dir / f"{model_name}_predictions.jsonl"
    cmd = [
        "modal",
        "volume",
        "get",
        VOLUME_NAME,
        remote_path,
        str(local_path),
        "--force",
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return local_path


def _read_cache(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Cache {path}: malformed JSON at line {i}: {exc}"
                ) from exc
    return rows


def _partition_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Group cache rows into (normal, swapped, consistency_runs) lists.

    Returns dict with keys ``normal``, ``swapped``, ``consistency`` —
    the last is a list of 5 lists indexed by ``run_index``.
    """
    from eval.eval_harness import Prediction  # noqa: F401  (importable check)

    normal: list[Any] = []
    swapped: list[Any] = []
    consistency: dict[int, list[Any]] = {i: [] for i in CONSISTENCY_RUN_INDEXES}
    for row in rows:
        pred = to_prediction(row)
        t = row["temperature"]
        s = row["swapped"]
        idx = row["run_index"]
        if t == 0.0 and not s and idx == 0:
            normal.append(pred)
        elif t == 0.0 and s and idx == 0:
            swapped.append(pred)
        elif t == CONSISTENCY_TEMPERATURE and not s and idx in consistency:
            consistency[idx].append(pred)
        else:
            raise ValueError(
                f"Cache row has unexpected partition "
                f"(T={t}, swapped={s}, run_index={idx}) "
                f"for pair_id={row.get('pair_id')!r}"
            )
    return {
        "normal": normal,
        "swapped": swapped,
        "consistency": [consistency[i] for i in CONSISTENCY_RUN_INDEXES],
    }


@app.local_entrypoint()
def collect_results() -> None:
    """Download per-model caches, compute metrics, render the table."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from transformers import AutoTokenizer

    from eval.eval_harness import (
        aggregate_metrics,
        render_markdown_table,
    )

    records = _load_local_records()
    expected_rows = (
        len(records)  # original
        + len(records)  # swapped
        + len(records) * len(CONSISTENCY_RUN_INDEXES)  # consistency
    )
    print(f"[collect] expected rows per model: {expected_rows}")

    verb_tok = AutoTokenizer.from_pretrained(HF_TOKENIZER_ID)

    per_checkpoint_metrics: dict[str, dict[str, float]] = {}
    detailed: list[dict[str, Any]] = []
    n_records_per_checkpoint: dict[str, int] = {}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for model_name in ("baseline", "sft", "dpo"):
            print(f"[collect] downloading cache for {model_name}")
            local_path = _load_cache_via_volume_get(model_name, tmp_dir)
            rows = _read_cache(local_path)
            print(
                f"[collect] {model_name}: {len(rows)} rows "
                f"(expected {expected_rows})"
            )
            if len(rows) != expected_rows:
                logger.warning(
                    "%s cache has %d rows; expected %d — skipping column.",
                    model_name,
                    len(rows),
                    expected_rows,
                )
                continue
            partition = _partition_predictions(rows)
            metrics = aggregate_metrics(
                records,
                partition["normal"],
                partition["swapped"],
                partition["consistency"],
                verb_tok,
            )
            display = MODEL_REGISTRY[model_name]["display_name"]
            per_checkpoint_metrics[display] = metrics
            n_records_per_checkpoint[display] = len(rows)
            for row in rows:
                detailed.append(
                    {
                        "model": model_name,
                        "checkpoint_name": display,
                        "pair_id": row["pair_id"],
                        "run_type": row["run_type"],
                        "verdict": row["verdict"],
                    }
                )

    if not per_checkpoint_metrics:
        raise SystemExit("No checkpoints produced complete caches; nothing to render.")

    table = render_markdown_table(per_checkpoint_metrics)
    print("\n" + table + "\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    md_path = RESULTS_DIR / f"stage8_final_{stamp}.md"
    json_path = RESULTS_DIR / f"stage8_final_{stamp}.json"
    md_path.write_text(table + "\n")
    payload: dict[str, Any] = {
        "run_id": f"stage8-final-{stamp}",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_sha": _git_sha(),
        "modal_gpu": MODAL_GPU,
        "metrics": per_checkpoint_metrics,
        "n_records_per_checkpoint": n_records_per_checkpoint,
        "detailed": detailed,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[collect] wrote {md_path} and {json_path}")
