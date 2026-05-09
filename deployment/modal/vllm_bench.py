"""Stage 10 Task 6b — Modal-hosted vLLM server + benchmark driver.

Runs the live throughput/latency portion of the deployment benchmark on
Modal A100-80GB (parity with Stage 8 eval). Combines:

1. ``offline_batch_stats`` — tokenizes raw_output from the Stage 8
   prediction caches at ``/vol/eval/cache/{baseline,sft,dpo}_predictions.jsonl``
   and divides by Stage 8 wall-clocks read from
   ``train/.cost_ledger.jsonl``. No new inference; runs even when the
   live-serving leg fails.
2. ``VllmServer`` — boots vLLM's OpenAI-compatible API server pointing
   at ``/vol/checkpoints/merged-fp16/`` (the DPO merged fp16
   checkpoint). The in-container readiness probe blocks until
   ``/v1/models`` returns non-empty data so Modal's web-server proxy
   doesn't route traffic to a not-yet-ready server.
3. Local entrypoint that drives ``deployment/example_client.py
   --benchmark`` against the Modal proxy URL and merges all sections
   into ``deployment/benchmark_results.json`` (atomic write).

Module imports must succeed in BOTH the local Python env and the Modal
container env. The container has only the inlined ``vllm_image`` (no
project tree), so:

- Image, volume, GPU, and tokenizer constants are inlined verbatim
  from ``eval/modal/vllm_infer.py:155-184`` rather than imported.
- ROOT-relative paths (``train/.cost_ledger.jsonl``,
  ``data/judge_system_prompt.md``, etc.) are resolved lazily inside
  ``@app.local_entrypoint()``.
- ``assert_no_thinking_in_prompt`` is imported from the project tree
  inside the local entrypoint, never at module level.

Constraint: the system prompt MUST NOT contain ``<|think|>``. Asserted
locally before the Modal run starts.

Estimated cost ceiling: ~$0.50 (180 s startup + 90 s bench + 60 s
scaledown ≈ 5.5 min × $0.000694/s ≈ $0.23, doubled for safety).

Usage::

    modal run deployment/modal/vllm_bench.py::bench
"""

import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (inlined — must work in both local and Modal-container contexts)
# ---------------------------------------------------------------------------

# Mirrors eval/modal/vllm_infer.py:101,114-128.
HF_TOKENIZER_ID = "unsloth/gemma-4-E4B-it"
MAX_MODEL_LEN = 4096
MODAL_GPU = os.environ.get("MODAL_GPU", "A100-80GB")
VOLUME_NAME = "judge-from-scratch"

DPO_PATH = "/vol/checkpoints/merged-fp16/"
CACHE_DIR = "/vol/eval/cache"
CACHE_FILES: dict[str, str] = {
    "baseline": "baseline_predictions.jsonl",
    "sft": "sft_predictions.jsonl",
    "dpo": "dpo_predictions.jsonl",
}
SCALEDOWN_S = 60

MODAL_USD_PER_HOUR = 2.50
MODAL_USD_PER_S = MODAL_USD_PER_HOUR / 3600.0

# Stage 4/8 totals used in the cost-comparison ratios.
STAGE4_TOTAL_USD = 14.34
STAGE4_PAIRS = 1937
STAGE8_PAIRS = 300

# ---------------------------------------------------------------------------
# Image and volume (inlined from eval/modal/vllm_infer.py:155-184)
# ---------------------------------------------------------------------------
#
# We don't import vllm_image from eval.modal.vllm_infer because Modal
# containers re-import this script and would crash on the cross-package
# import (the project tree is not mounted inside the container). The
# image must build identically to Stage 8's; if Stage 8 changes its
# recipe we either propagate the change here or accept divergence.

vllm_image = modal.Image.from_registry(
    "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.11"
).run_commands(
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
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

app = modal.App("judge-vllm-bench")


# ---------------------------------------------------------------------------
# Modal: vLLM web server (live latency benchmark target)
# ---------------------------------------------------------------------------


@app.cls(
    image=vllm_image,
    gpu=MODAL_GPU,
    volumes={"/vol": volume},
    scaledown_window=SCALEDOWN_S,
    timeout=600,
)
class VllmServer:
    """Long-lived vLLM HTTP server proxied via Modal's web_server.

    No precedent for ``@modal.web_server`` exists in this codebase
    (Stage 8 used offline ``@app.cls`` + ``@modal.method()``). The
    in-container readiness probe is critical: vLLM binds the socket
    before weights finish loading, so without it Modal would proxy
    traffic to a not-yet-ready server.
    """

    @modal.web_server(port=8000, startup_timeout=600)
    def serve(self) -> None:
        """Spawn vLLM's OpenAI-compatible API server, then block until ready.

        ``--served-model-name judge`` is best-effort; if the pinned
        image silently ignores it the bench client discovers the live
        model id via ``GET /v1/models``.
        """
        # Mirrors eval.modal.vllm_infer.VllmRunner.load() flags EXCEPT
        # we drop --enforce-eager so vLLM can capture CUDA graphs.
        # Stage 8 used eager mode for run-to-run determinism across
        # baseline/SFT/DPO; serving doesn't need that and benefits
        # significantly from graph capture (1.5x → 5-8x at concurrency
        # 16 in vLLM's published benchmarks). If graph capture fails
        # on Gemma 4 E4B the server logs the error and the bench
        # surfaces a non-empty errors[] entry rather than falling
        # back silently.
        subprocess.Popen(
            [
                "python",
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                DPO_PATH,
                "--served-model-name",
                "judge",
                "--max-model-len",
                str(MAX_MODEL_LEN),
                "--dtype",
                "bfloat16",
                "--gpu-memory-utilization",
                "0.85",
                "--port",
                "8000",
            ]
        )
        # 480 s budget: model load (~30 s) + torch.compile AOT (~110 s)
        # + CUDA graph capture (~10 s) + safety. With --enforce-eager
        # the same wait completed in ~60-90 s; without it, AOT compile
        # is the dominant cost on the FIRST run. Subsequent runs may
        # hit a warm torch_compile_cache (not currently persisted).
        deadline = time.time() + 480
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    "http://127.0.0.1:8000/v1/models", timeout=2
                ) as r:
                    if r.status == 200 and b'"id"' in r.read():
                        logger.info("vLLM server ready")
                        return
            except (urllib.error.URLError, ConnectionResetError, OSError):
                pass
            time.sleep(2)
        raise RuntimeError("vLLM did not become ready within 480 s")


# ---------------------------------------------------------------------------
# Modal: offline batch stats (no new inference)
# ---------------------------------------------------------------------------


@app.function(
    image=vllm_image,
    volumes={"/vol": volume},
    timeout=600,
)
def offline_batch_stats(wall_clocks: dict[str, float]) -> dict[str, dict]:
    """Per-model offline-batch numbers from cached Stage 8 predictions.

    Tokenizes ``raw_output`` with the Gemma 4 tokenizer; row counts
    come from JSONL line counts (the ledger has no ``rows`` field).
    Wall-clocks come from the LOCAL ``train/.cost_ledger.jsonl`` and
    are passed in by the caller — Modal cannot mount the project tree.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF_TOKENIZER_ID)
    out: dict[str, dict] = {}
    for name, fname in CACHE_FILES.items():
        path = Path(CACHE_DIR) / fname
        rows = 0
        out_tokens = 0
        if not path.exists():
            out[name] = {"error": f"cache file missing: {path}"}
            continue
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows += 1
                rec = json.loads(line)
                raw = rec.get("raw_output") or ""
                if raw:
                    out_tokens += len(tok.encode(raw, add_special_tokens=False))
        wall = wall_clocks.get(name)
        if wall is None or wall <= 0:
            out[name] = {
                "rows": rows,
                "wallclock_s": None,
                "output_tokens": out_tokens,
                "prompts_per_min": None,
                "output_tok_s": None,
                "note": "no wall-clock in train/.cost_ledger.jsonl",
            }
            continue
        out[name] = {
            "rows": rows,
            "wallclock_s": round(wall, 2),
            "output_tokens": out_tokens,
            "prompts_per_min": round(rows / wall * 60.0, 2),
            "output_tok_s": round(out_tokens / wall, 2),
        }
    return out


# ---------------------------------------------------------------------------
# Local helpers (only invoked from @app.local_entrypoint())
# ---------------------------------------------------------------------------


def _local_root() -> Path:
    """Project root, computed lazily — fails inside Modal containers."""
    return Path(__file__).resolve().parents[2]


def _read_train_wallclocks(ledger_path: Path) -> dict[str, float]:
    """Final-by-timestamp wall-clock per ``vllm_infer.{name}`` function."""
    if not ledger_path.exists():
        return {}
    final: dict[str, dict] = {}
    for line in ledger_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        fn = rec.get("function") or ""
        if not fn.startswith("vllm_infer."):
            continue
        name = fn.split(".", 1)[1]
        prev = final.get(name)
        if prev is None or rec["timestamp"] > prev["timestamp"]:
            final[name] = rec
    return {name: float(rec["wallclock_s"]) for name, rec in final.items()}


def _read_sonnet_primary_cost(ledger_path: Path) -> tuple[float, int]:
    """Sum cost_usd and n_requests over Sonnet primary + retry_primary phases."""
    if not ledger_path.exists():
        return (0.0, 0)
    total = 0.0
    calls = 0
    for line in ledger_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("model") != "claude-sonnet-4-6":
            continue
        if rec.get("phase") not in {"primary", "retry_primary"}:
            continue
        total += float(rec.get("cost_usd", 0.0))
        calls += int(rec.get("n_requests", 0))
    return (total, calls)


def _git_sha(cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(cwd), text=True
        ).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("git rev-parse failed: %s", exc)
        return "unknown"


def _model_revision_sha() -> str:
    """HF revision SHA from huggingface_hub (NOT /v1/models, which lacks SHA)."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info("krishnakartik/gemma4-social-bias-judge")
        return getattr(info, "sha", None) or "unknown"
    except Exception as exc:  # noqa: BLE001
        logger.warning("HfApi model_info failed: %s", exc)
        return "unknown"


def _wait_until_ready(url: str, *, timeout_s: float = 300.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    body = json.loads(r.read())
                    data = body.get("data") or []
                    if data:
                        return
        except (urllib.error.URLError, ConnectionResetError, OSError, ValueError):
            pass
        time.sleep(5)
    raise TimeoutError(f"server at {url} did not become ready within {timeout_s} s")


def _discover_model_id(url: str) -> str:
    with urllib.request.urlopen(url, timeout=10) as r:
        body = json.loads(r.read())
    data = body.get("data") or []
    if not data:
        raise RuntimeError(f"{url} returned empty data list")
    return data[0]["id"]


def _compute_cost_comparison(
    offline: dict[str, dict],
    live: dict[str, dict],
    sonnet_cost: float,
    sonnet_calls: int,
) -> dict[str, dict]:
    # Per-call apples-to-apples uses the offline-batch DPO throughput
    # (prompts/min) per the spec: "Modal A100 hourly rate ÷ calls/hour
    # from the batch throughput numbers." This is the realistic-at-scale
    # cost — single-stream p50 latency would penalize Modal for not
    # being batched, which isn't how anyone actually runs a judge in
    # production. Live-serving p50 stays in the JSON as context (under
    # live_serving) but is not used for the headline ratio.
    sonnet_per_call: float | None = (
        sonnet_cost / sonnet_calls if sonnet_calls > 0 else None
    )
    dpo_offline = offline.get("dpo") or {}
    dpo_prompts_per_min = dpo_offline.get("prompts_per_min")
    if dpo_prompts_per_min and dpo_prompts_per_min > 0:
        modal_calls_per_hour = dpo_prompts_per_min * 60.0
        modal_per_call = MODAL_USD_PER_HOUR / modal_calls_per_hour
        ratio_per_call = (
            sonnet_per_call / modal_per_call
            if sonnet_per_call and modal_per_call > 0
            else None
        )
    else:
        modal_per_call = None
        ratio_per_call = None
    seq = live.get("sequential", {}) or {}
    p50_total_s = seq.get("p50_total_s")

    sonnet_pipeline_per_pair = STAGE4_TOTAL_USD / STAGE4_PAIRS
    dpo_wall = (offline.get("dpo") or {}).get("wallclock_s")
    if dpo_wall and dpo_wall > 0:
        modal_dpo_cost = MODAL_USD_PER_S * dpo_wall
        modal_pipeline_per_pair = modal_dpo_cost / STAGE8_PAIRS
        ratio_pipeline = sonnet_pipeline_per_pair / modal_pipeline_per_pair
    else:
        modal_pipeline_per_pair = None
        ratio_pipeline = None

    return {
        "per_call_apples_to_apples": {
            "sonnet_usd_per_call": (
                round(sonnet_per_call, 6) if sonnet_per_call is not None else None
            ),
            "sonnet_source": (
                f"data/labeled/.cost_ledger.jsonl phases primary+retry_primary, "
                f"${sonnet_cost:.4f} / {sonnet_calls} calls"
            ),
            "modal_usd_per_call": (
                round(modal_per_call, 6) if modal_per_call is not None else None
            ),
            "modal_source": (
                f"Modal A100-80GB ${MODAL_USD_PER_HOUR}/hr ÷ DPO offline "
                f"batch throughput ({dpo_prompts_per_min} prompts/min = "
                f"{round(dpo_prompts_per_min * 60.0, 0) if dpo_prompts_per_min else None} calls/hr)"
            ),
            "ratio_sonnet_over_modal_x": (
                round(ratio_per_call, 2) if ratio_per_call is not None else None
            ),
            "live_serving_p50_total_s_for_reference": p50_total_s,
            "notes": (
                "One Sonnet labeling call vs one self-hosted judge call, "
                "computed from realistic-at-scale batch throughput (matches "
                "the user's spec). Headline number for the resume line. "
                "Live-serving p50 latency is captured separately under "
                "live_serving for single-stream context."
            ),
        },
        "per_pair_pipeline": {
            "sonnet_usd_per_pair": round(sonnet_pipeline_per_pair, 6),
            "sonnet_source": (
                f"Stage 4 total ${STAGE4_TOTAL_USD} / {STAGE4_PAIRS} pairs "
                "(includes Sonnet primary + GPT-5.4 + Qwen 3 cross-checkers)"
            ),
            "modal_usd_per_pair": (
                round(modal_pipeline_per_pair, 6)
                if modal_pipeline_per_pair is not None
                else None
            ),
            "modal_source": (
                f"Stage 8 DPO Modal spend (${MODAL_USD_PER_HOUR}/hr × "
                f"{dpo_wall} s) / {STAGE8_PAIRS} eval pairs (7 passes per pair)"
            ),
            "ratio_sonnet_over_modal_x": (
                round(ratio_pipeline, 2) if ratio_pipeline is not None else None
            ),
            "notes": (
                "Full-pipeline labeling cost vs Modal eval-suite cost. "
                "Useful context but NOT apples-to-apples — the Sonnet "
                "pipeline includes cross-checkers the self-hosted judge "
                "doesn't run."
            ),
        },
    }


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def bench(results_path: str | None = None) -> None:
    """Run offline + live + cost comparison; write benchmark_results.json."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    root = _local_root()
    sys.path.insert(0, str(root))
    from eval.eval_harness import assert_no_thinking_in_prompt  # noqa: PLC0415

    train_ledger = root / "train" / ".cost_ledger.jsonl"
    labeled_ledger = root / "data" / "labeled" / ".cost_ledger.jsonl"
    system_prompt_path = root / "data" / "judge_system_prompt.md"
    example_client = root / "deployment" / "example_client.py"
    tmp_live_path = root / "deployment" / ".benchmark_live.tmp.json"
    out_path = (
        Path(results_path)
        if results_path
        else root / "deployment" / "benchmark_results.json"
    )

    # Sanity-check the system prompt before spending Modal time.
    assert_no_thinking_in_prompt(system_prompt_path.read_text())

    out: dict[str, Any] = {
        "git_sha": _git_sha(root),
        "captured_at": datetime.now(UTC).isoformat(),
        "modal_gpu": MODAL_GPU,
        "modal_usd_per_hour": MODAL_USD_PER_HOUR,
        "vllm_image_ref": (
            "modal nightly cu129 wheels on nvidia/cuda:12.9.0-devel-ubuntu22.04 "
            "(inlined from eval.modal.vllm_infer.vllm_image)"
        ),
        "local_docker_image_ref": (
            "vllm/vllm-openai:latest (stable; what users running "
            "deployment/vllm/Dockerfile get)"
        ),
        "_image_divergence_note": (
            "Modal benchmark numbers come from the nightly cu129 image used "
            "in Stage 8. Local Docker users running vllm/vllm-openai:latest "
            "will see different (typically slower) throughput. Treat "
            "live_serving as a Stage-8-image data point, not a guarantee for "
            "the local Dockerfile."
        ),
        "model_revision": _model_revision_sha(),
        "cuda_graphs": True,  # we no longer pass --enforce-eager
        "offline_batch": None,
        "live_serving": None,
        "cost_comparison": None,
        "errors": [],
        "_units_note": (
            "All *_usd_* fields are USD; *_s suffixes are seconds; ratio_* "
            "are dimensionless multipliers (use ratio_sonnet_over_modal_x as "
            "'Modal is X-fold cheaper')."
        ),
    }

    # 1) Offline batch first — runs even if live serving fails.
    try:
        wall_clocks = _read_train_wallclocks(train_ledger)
        if not wall_clocks:
            raise RuntimeError(f"could not read wall-clocks from {train_ledger}")
        out["offline_batch"] = offline_batch_stats.remote(wall_clocks)
        logger.info("offline_batch ok: %s", list(out["offline_batch"].keys()))
    except Exception as exc:  # noqa: BLE001
        logger.warning("offline_batch failed: %s", exc)
        out["errors"].append({"phase": "offline_batch", "error": str(exc)})

    # 2) Live serving — boot vLLM web server, drive example_client.py.
    try:
        server = VllmServer()
        # Modal v1.4 renamed the Function.web_url property to a
        # get_web_url() method. The cls is lazily provisioned on the
        # first call to get_web_url(); the actual cold start happens
        # when wait_until_ready hits the URL below.
        url_root = server.serve.get_web_url().rstrip("/")
        logger.info("VllmServer URL: %s", url_root)
        _wait_until_ready(f"{url_root}/v1/models", timeout_s=540.0)
        model_id = _discover_model_id(f"{url_root}/v1/models")
        logger.info("live model id: %s", model_id)
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        subprocess.run(
            [
                sys.executable,
                str(example_client),
                "--backend",
                "vllm",
                "--benchmark",
                "--base-url",
                f"{url_root}/v1",
                "--model",
                model_id,
                "--results-path",
                str(tmp_live_path),
            ],
            check=True,
            cwd=str(root),
            env=env,
        )
        out["live_serving"] = json.loads(tmp_live_path.read_text())
        tmp_live_path.unlink(missing_ok=True)
        logger.info("live_serving ok")
    except Exception as exc:  # noqa: BLE001
        logger.warning("live_serving failed: %s", exc)
        out["errors"].append({"phase": "live_serving", "error": str(exc)})

    # 3) Cost comparison only when both phases succeeded.
    if out["offline_batch"] and out["live_serving"]:
        sonnet_cost, sonnet_calls = _read_sonnet_primary_cost(labeled_ledger)
        out["cost_comparison"] = _compute_cost_comparison(
            out["offline_batch"], out["live_serving"], sonnet_cost, sonnet_calls
        )

    _atomic_write_json(out_path, out)
    logger.info("wrote %s", out_path)
    print(json.dumps(out, indent=2))
    if out["errors"]:
        sys.exit(1)
