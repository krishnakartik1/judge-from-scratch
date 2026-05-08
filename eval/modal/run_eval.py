"""Stage 8 — eval harness on Modal.

Runs the base Gemma 4 E4B and any number of fine-tuned checkpoints
through the 300-pair held-out evaluation set, computing Cohen's κ vs.
human verdicts plus the robustness metrics from primer Appendix C.

Run from project root::

    # Dryrun (50 pairs)
    modal run eval/modal/run_eval.py --baseline --dryrun

    # Full run (3 columns, parallel)
    modal run eval/modal/run_eval.py --baseline \\
        --checkpoints "/vol/checkpoints/sft-final/,/vol/checkpoints/dpo-final/"

Architecture: ``run_eval`` ``spawn``s one Modal function per checkpoint
in parallel, persists the FunctionCall IDs to ``IN_FLIGHT_PATH``, then
waits with retry. If the local CLI dies during the wait (network blip,
gRPC ``Deadline exceeded``), re-invoke the same command — it detects
the in-flight state file and resumes waiting on the same call IDs.

The CLI accepts comma-separated lists for ``--checkpoints`` and
``--names`` because Modal's local-entrypoint CLI does not natively
support repeat-flag list args.

Override the GPU variant with ``MODAL_GPU=A100-80GB modal run …``.
Override the budget cap interactively with ``BUDGET_OVERRIDE=1``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import modal

logger = logging.getLogger(__name__)

# --- Paths inside the Modal container ------------------------------------
PKG_DIR = "/root/eval_pkg"
SYSTEM_PROMPT_REMOTE = f"{PKG_DIR}/judge_system_prompt.md"
EVAL_SET_REMOTE = f"{PKG_DIR}/eval_set.jsonl"
CACHE_DIR_VOL = "/vol/eval/cache"

# --- Local paths (resolved relative to project root) --------------------
LOCAL_EVAL_SET = Path("data/pairs/eval_set_unlabeled.jsonl")
LOCAL_SYSTEM_PROMPT = Path("data/judge_system_prompt.md")
RESULTS_DIR = Path("eval/results")
# Persists the in-flight Modal FunctionCall IDs across local-side
# crashes / gRPC drops. Re-invoking ``run_eval`` while this exists
# resumes wait on the persisted IDs instead of re-spawning.
IN_FLIGHT_PATH = RESULTS_DIR / ".stage8_in_flight.json"

# Base model — must match cfg["model_id"] used by Stage 6 (sft.yaml).
BASELINE_MODEL_ID = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"

# --- Modal app -----------------------------------------------------------
MODAL_GPU = os.environ.get("MODAL_GPU", "A100")
VOLUME_NAME = "judge-from-scratch"
# Sized off the 2026-05-07 dryrun (~13.9 s/inference). With 3 passes per
# checkpoint (1 normal + 1 swapped + 1 self-consistency) × 300 records
# = 900 calls/ckpt × 13.9 s ≈ 3.5 h. 14000 s gives ~25 min headroom and
# keeps the worst-case ``check_budget`` projection (timeout × rate ×
# n_targets) under the project-wide $30 cap when combined with prior
# stage spend.
TIMEOUT_S = 14000

# Project-wide budget cap. Stages 6 and 7 also size against this $30
# ceiling (see train/modal/_cost_ledger.py).
PROJECT_BUDGET_CAP_USD = 30.0

app = modal.App("judge-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
    .add_local_file("uv.lock", "/root/uv.lock", copy=True)
    .uv_sync()
    .add_local_file(
        "data/_format_helpers.py", f"{PKG_DIR}/_format_helpers.py", copy=True
    )
    .add_local_file("data/judge_system_prompt.md", SYSTEM_PROMPT_REMOTE, copy=True)
    .add_local_file("data/pairs/eval_set_unlabeled.jsonl", EVAL_SET_REMOTE, copy=True)
    .add_local_file("eval/eval_harness.py", f"{PKG_DIR}/eval_harness.py", copy=True)
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]


# --- Local helpers (no torch / unsloth imports) -------------------------


def _parse_csv(value: str) -> list[str]:
    """Split a comma-separated CLI value into stripped non-empty parts."""
    return [v.strip() for v in value.split(",") if v.strip()]


def _resolve_targets(
    baseline: bool, checkpoints: str, names: str
) -> list[tuple[str, str | None]]:
    """Build the (display_name, checkpoint_path) list to evaluate.

    A ``checkpoint_path`` of ``None`` indicates the un-fine-tuned base
    model. Display names default to a heuristic on the path basename
    (``dpo`` → ``After SFT+DPO``; ``sft`` → ``After SFT``); supply
    ``--names`` for full control.
    """
    paths = _parse_csv(checkpoints)
    user_names = _parse_csv(names)
    targets: list[tuple[str, str | None]] = []

    if baseline:
        targets.append(("Base Gemma 4 E4B", None))

    if paths:
        if user_names:
            if len(user_names) != len(paths):
                raise ValueError(
                    f"--names must match --checkpoints in length: "
                    f"got {len(user_names)} names for {len(paths)} paths."
                )
            display_names = user_names
        else:
            display_names = [_default_display_name(p) for p in paths]
        targets.extend(zip(display_names, paths, strict=False))

    if not targets:
        raise ValueError(
            "No targets to evaluate. Pass --baseline and/or "
            '--checkpoints "path1,path2".'
        )
    return targets


def _default_display_name(path: str) -> str:
    stem = Path(path.rstrip("/")).name
    lowered = stem.lower()
    if "dpo" in lowered:
        return "After SFT+DPO"
    if "sft" in lowered:
        return "After SFT"
    return stem


def _git_sha() -> str | None:
    """Best-effort current git SHA. ``None`` when unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


# --- Remote inference function ------------------------------------------


@app.function(
    image=image,
    gpu=MODAL_GPU,
    timeout=TIMEOUT_S,
    volumes={"/vol": volume},
    secrets=secrets,
)
def _run_inference_remote(
    checkpoint_path: str | None, dryrun: bool, display_name: str
) -> dict[str, Any]:
    """Five-pass inference for one checkpoint (or the baseline).

    Passes (in order): normal headline (T=0), position-swapped (T=0),
    self-consistency runs 0/1/2 (T=0.3). Each pass is cache-aware; only
    misses invoke ``model.generate``. The volume is committed after
    every pass so a crash mid-run preserves prior-pass progress.
    """
    sys.path.insert(0, PKG_DIR)
    from eval_harness import (  # type: ignore[import-not-found]
        PredictionCache,
        assert_no_thinking_in_prompt,
        load_eval_set,
        run_inference_pass,
        select_dryrun_subset,
        slug,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    system_prompt = Path(SYSTEM_PROMPT_REMOTE).read_text()
    assert_no_thinking_in_prompt(system_prompt)

    records = load_eval_set(Path(EVAL_SET_REMOTE))
    if dryrun:
        records = select_dryrun_subset(records)
        logger.info("Dryrun: subset to %d records.", len(records))

    from unsloth import FastLanguageModel  # type: ignore[import-not-found]

    if checkpoint_path is not None and not Path(checkpoint_path).exists():
        available = sorted(p.name for p in Path("/vol/checkpoints").glob("*"))
        raise FileNotFoundError(
            f"Checkpoint {checkpoint_path!r} not found on volume. "
            f"Available under /vol/checkpoints/: {available}"
        )
    model_name = checkpoint_path if checkpoint_path is not None else BASELINE_MODEL_ID
    logger.info("Loading model: %s", model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    # The base ``unsloth/gemma-4-E4B-it-unsloth-bnb-4bit`` upstream
    # reports ``eos_token='<eos>'`` while the SFT/DPO checkpoints save
    # ``'<turn|>'`` from Stage 6's training-time state. Inference does
    # not depend on eos parity (chat template handles turn structure;
    # ``max_new_tokens`` caps generation regardless), so we only log.
    logger.info("Tokenizer eos_token=%r", tokenizer.eos_token)

    Path(CACHE_DIR_VOL).mkdir(parents=True, exist_ok=True)
    cache_path = Path(CACHE_DIR_VOL, f"{slug(display_name)}_predictions.jsonl")
    cache = PredictionCache(cache_path)
    logger.info("Cache loaded from %s (%d entries).", cache_path, len(cache))

    pass_specs: list[tuple[float, bool, int]] = [
        (0.0, False, 0),  # normal headline
        (0.0, True, 0),  # position-swapped
        (0.3, False, 0),  # single sampled run (compared to T=0 baseline)
    ]
    t0 = time.time()
    all_preds: list[dict[str, Any]] = []
    for temperature, swapped, run_index in pass_specs:
        logger.info("Pass: T=%.1f swapped=%s run=%d", temperature, swapped, run_index)
        preds = run_inference_pass(
            model,
            tokenizer,
            system_prompt,
            records,
            temperature=temperature,
            swapped=swapped,
            run_index=run_index,
            cache=cache,
        )
        all_preds.extend(asdict(p) for p in preds)
        volume.commit()
    wallclock_s = time.time() - t0

    logger.info(
        "Inference done. wallclock=%.1fs, predictions=%d, cache_size=%d",
        wallclock_s,
        len(all_preds),
        len(cache),
    )
    return {
        "display_name": display_name,
        "model_name": model_name,
        "checkpoint_path": checkpoint_path,
        "wallclock_s": wallclock_s,
        "predictions": all_preds,
        "n_records": len(records),
    }


# --- Local entrypoints ---------------------------------------------------


@app.local_entrypoint()
def run_one(
    checkpoint_path: str = "",
    display_name: str = "",
    dryrun: bool = False,
) -> None:
    """Run inference for a single checkpoint, then exit.

    One Modal function per invocation, which means this works correctly
    with ``modal run --detach`` (the "only last triggered function
    survives" caveat is satisfied because there is exactly one spawn).
    Run three of these in sequence (or in three terminals) plus a
    final ``collect_results`` to assemble the table::

        modal run --detach eval/modal/run_eval.py::run_one \\
            --display-name "Base Gemma 4 E4B"
        modal run --detach eval/modal/run_eval.py::run_one \\
            --checkpoint-path /vol/checkpoints/sft-final/ \\
            --display-name "After SFT"
        modal run --detach eval/modal/run_eval.py::run_one \\
            --checkpoint-path /vol/checkpoints/dpo-final/ \\
            --display-name "After SFT+DPO"
        # check progress at any time:
        modal run eval/modal/run_eval.py::cache_status
        # then aggregate metrics from the cached predictions:
        modal run eval/modal/run_eval.py::collect_results

    Args:
        checkpoint_path: volume path to a fine-tuned checkpoint, or ``""``
            for the base ``unsloth/gemma-4-E4B-it-unsloth-bnb-4bit`` model.
        display_name: column label used in the results table and as the
            cache filename slug. Required (no auto-derivation here so
            it stays consistent across re-runs).
        dryrun: subset to 25 in_dist + 25 OOD pairs.
    """
    if not display_name:
        raise ValueError("--display-name is required for run_one (e.g., 'After SFT').")
    ckpt_arg: str | None = checkpoint_path or None

    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from train.modal._cost_ledger import (
        check_budget,
        project_cost,
        record_cost,
        total_spend,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    print(
        f"[stage8] run_one: {display_name!r} "
        f"({ckpt_arg or BASELINE_MODEL_ID}); dryrun={dryrun}."
    )
    projected = project_cost(MODAL_GPU, TIMEOUT_S)
    spent = total_spend()
    check_budget(
        projected_usd=projected,
        cap_usd=PROJECT_BUDGET_CAP_USD,
        spent_usd=spent,
        label=f"stage8-eval/{display_name} on {MODAL_GPU}",
    )

    t0 = time.time()
    notes_status = "ok"
    result: dict[str, Any] | None = None
    try:
        result = _run_inference_remote.remote(ckpt_arg, dryrun, display_name)
    except Exception as exc:  # noqa: BLE001
        notes_status = f"failed:{type(exc).__name__}"
        raise
    finally:
        wallclock_s = time.time() - t0
        remote_s = result["wallclock_s"] if result else wallclock_s
        cost_row = record_cost(
            stage="stage8",
            function=f"run_one[{display_name}]" + ("-dryrun" if dryrun else ""),
            gpu=MODAL_GPU,
            wallclock_s=remote_s,
            notes=(
                f"status={notes_status} ckpt={ckpt_arg!r} "
                f"display_name={display_name!r} dryrun={dryrun}"
            ),
        )
        print(
            f"[cost] Recorded ${cost_row['est_cost_usd']:.3f} "
            f"({remote_s:.0f}s remote, {notes_status}). "
            f"Cumulative spend: ${total_spend():.2f}"
        )

    if result is not None:
        print(
            f"\n[stage8] {display_name} done. "
            f"{len(result['predictions'])} predictions cached on volume.\n"
            "Run `modal run eval/modal/run_eval.py::collect_results` "
            "after all checkpoints finish to produce the table."
        )


@app.local_entrypoint()
def collect_results(
    names: str = "Base Gemma 4 E4B,After SFT,After SFT+DPO",
    checkpoint_paths: str = ",/vol/checkpoints/sft-final/,/vol/checkpoints/dpo-final/",
    dryrun: bool = False,
) -> None:
    """Read cached predictions from /vol and produce the results table.

    No GPU work — pulls each ``/vol/eval/cache/{slug}_predictions.jsonl``
    via ``modal volume get``, computes metrics locally, prints the
    markdown table, writes ``eval/results/{run_id}.json``. Skips any
    checkpoint whose cache is incomplete (warns and excludes from the
    table). Run after all desired ``run_one`` invocations finish.

    Args:
        names: comma-separated display names; column order in the table.
        checkpoint_paths: comma-separated checkpoint paths matching
            ``names`` length. Empty entries (``""``) mark the baseline
            (no fine-tuning); the leading empty in the default value
            corresponds to "Base Gemma 4 E4B".
        dryrun: aggregate using the dryrun 50-pair subset rather than
            the full 300-pair holdout.
    """
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from eval.eval_harness import (
        SELF_CONSISTENCY_RUNS,
        Prediction,
        aggregate_metrics,
        flip_verdict,
        load_eval_set,
        render_markdown_table,
        select_dryrun_subset,
        slug,
    )

    name_list = _parse_csv(names)
    path_parts = [p.strip() for p in checkpoint_paths.split(",")]
    if len(path_parts) != len(name_list):
        raise ValueError(
            f"--names ({len(name_list)}) and --checkpoint-paths "
            f"({len(path_parts)}) must have the same length."
        )

    records = load_eval_set(LOCAL_EVAL_SET)
    if dryrun:
        records = select_dryrun_subset(records)

    needed_passes = [(0.0, False, 0), (0.0, True, 0), (0.3, False, 0)]
    target_n = len(records)

    per_checkpoint_metrics: dict[str, dict[str, float]] = {}
    detailed: list[dict[str, Any]] = []
    completed: list[tuple[str, str]] = []

    print("\n[stage8] Loading tokenizer locally for verbosity-bias scoring…")
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-4-E4B-it")

    for display_name, ckpt_path in zip(name_list, path_parts, strict=True):
        cache_path = Path(f"/vol/eval/cache/{slug(display_name)}_predictions.jsonl")
        counts = _count_cache_entries(cache_path)
        if counts is None:
            print(f"[stage8] {display_name}: no cache file — skipping.")
            continue
        useful = sum(min(counts.get(k, 0), target_n) for k in needed_passes)
        if useful < target_n * len(needed_passes):
            print(
                f"[stage8] {display_name}: only {useful}/{target_n * len(needed_passes)} "
                "cached entries — skipping (run `run_one` to finish)."
            )
            continue
        # Re-download to populate predictions; _count_cache_entries deletes its tempfile.
        all_preds = list(_load_cache_predictions(cache_path))
        ids = {r["pair_id"] for r in records}
        preds_for_records = [Prediction(**p) for p in all_preds if p["pair_id"] in ids]
        normal = [
            p for p in preds_for_records if not p.swapped and p.temperature == 0.0
        ]
        swapped = [p for p in preds_for_records if p.swapped and p.temperature == 0.0]
        consistency = [
            [p for p in preds_for_records if p.temperature == 0.3 and p.run_index == i]
            for i in range(SELF_CONSISTENCY_RUNS)
        ]
        per_checkpoint_metrics[display_name] = aggregate_metrics(
            records, normal, swapped, consistency, tokenizer
        )
        completed.append((display_name, ckpt_path))

        n_index = {p.pair_id: p for p in normal}
        s_index = {p.pair_id: p for p in swapped}
        c_indexes = [{p.pair_id: p for p in run} for run in consistency]
        for record in records:
            pid = record["pair_id"]
            n = n_index.get(pid)
            s = s_index.get(pid)
            cs = [c.get(pid) for c in c_indexes]
            detailed.append(
                {
                    "pair_id": pid,
                    "checkpoint_name": display_name,
                    "human_verdict": record["human_verdict"],
                    "model_verdict": n.verdict if n else None,
                    "swapped_verdict": s.verdict if s else None,
                    "consistency_verdicts": [c.verdict if c else None for c in cs],
                    "raw_reasoning": n.reasoning if n else None,
                    "eval_slice": record.get("eval_slice"),
                    "pair_category": record.get("pair_category"),
                    "confidence": record.get("confidence"),
                    "notes": record.get("notes"),
                }
            )

    if not per_checkpoint_metrics:
        print("[stage8] No checkpoint had a complete cache. Nothing to aggregate.")
        return

    table = render_markdown_table(per_checkpoint_metrics)
    print("\n" + ("=" * 60))
    print(f"Stage 8 results ({len(records)} pairs{', dryrun' if dryrun else ''})")
    print("=" * 60)
    print(table)
    print()

    run_id = (
        f"stage8-eval-{'dryrun-' if dryrun else ''}"
        f"{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{run_id}.json"
    payload = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_sha": _git_sha(),
        "modal_gpu": MODAL_GPU,
        "checkpoints": [
            {"display_name": dn, "checkpoint_path": cp} for dn, cp in completed
        ],
        "metrics": per_checkpoint_metrics,
        "predictions": detailed,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[stage8] Results written to {out_path}")
    # ``flip_verdict`` is loaded for parity with the metric module's
    # public surface; flag-checking it here avoids an unused-import
    # warning if a future contributor wires sample-flip diagnostics
    # into ``collect_results``.
    _ = flip_verdict


def _load_cache_predictions(cache_path: Path) -> list[dict[str, Any]]:
    """Re-download a cache file via ``modal volume get`` and parse it."""
    import tempfile

    rel = str(cache_path).removeprefix("/vol/")
    with tempfile.TemporaryDirectory() as tmp:
        local_path = Path(tmp) / cache_path.name
        proc = subprocess.run(
            [
                "modal",
                "volume",
                "get",
                VOLUME_NAME,
                rel,
                str(local_path),
                "--force",
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"modal volume get failed for {cache_path}: {proc.stderr}"
            )
        out: list[dict[str, Any]] = []
        with open(local_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out


@app.local_entrypoint()
def run_eval(
    baseline: bool = False,
    checkpoints: str = "",
    names: str = "",
    dryrun: bool = False,
) -> None:
    """Run the Stage 8 eval harness across one or more checkpoints.

    Args:
        baseline: include base Gemma 4 E4B as a column.
        checkpoints: comma-separated checkpoint paths on the
            ``judge-from-scratch`` volume (e.g.,
            ``"/vol/checkpoints/sft-final/,/vol/checkpoints/dpo-final/"``).
        names: optional comma-separated display names; must match
            ``checkpoints`` length when supplied.
        dryrun: subset to 25 in_dist + 25 OOD pairs (50 total).
    """
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    from eval.eval_harness import (
        SELF_CONSISTENCY_RUNS,
        Prediction,
        aggregate_metrics,
        assert_no_thinking_in_prompt,
        flip_verdict,
        load_eval_set,
        render_markdown_table,
        select_dryrun_subset,
    )
    from train.modal._cost_ledger import (
        check_budget,
        project_cost,
        record_cost,
        total_spend,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if not LOCAL_EVAL_SET.exists():
        raise FileNotFoundError(
            f"{LOCAL_EVAL_SET} not found. The eval harness reads the "
            "Stage-3b-labeled holdout from this path; do not rename or "
            "delete it. (Decision: legacy filename retained — see plan.)"
        )
    system_prompt = LOCAL_SYSTEM_PROMPT.read_text()
    assert_no_thinking_in_prompt(system_prompt)

    # ----- Spawn-or-resume ----------------------------------------------
    if IN_FLIGHT_PATH.exists():
        state = json.loads(IN_FLIGHT_PATH.read_text())
        print(
            f"[stage8] Found in-flight state at {IN_FLIGHT_PATH}: "
            f"resuming wait on {len(state['calls'])} prior spawn(s)."
        )
        if state["dryrun"] != dryrun:
            print(
                f"[stage8] WARNING: state has dryrun={state['dryrun']}; "
                f"current invocation has dryrun={dryrun}. Using state's value."
            )
            dryrun = bool(state["dryrun"])
        records = load_eval_set(LOCAL_EVAL_SET)
        if dryrun:
            records = select_dryrun_subset(records)
        calls: list[tuple[str, str | None, modal.FunctionCall]] = [
            (
                c["display_name"],
                c["ckpt_path"],
                modal.FunctionCall.from_id(c["call_id"]),
            )
            for c in state["calls"]
        ]
    else:
        records = load_eval_set(LOCAL_EVAL_SET)
        if dryrun:
            records = select_dryrun_subset(records)
        targets = _resolve_targets(baseline, checkpoints, names)
        n_passes_per_ckpt = 2 + SELF_CONSISTENCY_RUNS
        print(f"\n[stage8] Targets: {[t[0] for t in targets]}")
        n_calls_per_ckpt = len(records) * n_passes_per_ckpt
        print(
            f"[stage8] Estimated inferences: {n_calls_per_ckpt} per checkpoint × "
            f"{len(targets)} = {n_calls_per_ckpt * len(targets)} total."
        )
        projected = project_cost(MODAL_GPU, TIMEOUT_S * len(targets))
        spent = total_spend()
        check_budget(
            projected_usd=projected,
            cap_usd=PROJECT_BUDGET_CAP_USD,
            spent_usd=spent,
            label=f"stage8-eval ({len(targets)} ckpts on {MODAL_GPU})",
        )
        print(f"\n[stage8] Spawning {len(targets)} parallel Modal functions…")
        calls = []
        for display_name, ckpt_path in targets:
            shown_id = ckpt_path or BASELINE_MODEL_ID
            call = _run_inference_remote.spawn(ckpt_path, dryrun, display_name)
            print(f"  spawned {display_name} ({shown_id}) -> {call.object_id}")
            calls.append((display_name, ckpt_path, call))
        IN_FLIGHT_PATH.parent.mkdir(parents=True, exist_ok=True)
        IN_FLIGHT_PATH.write_text(
            json.dumps(
                {
                    "dryrun": dryrun,
                    "spawned_at": datetime.utcnow().isoformat() + "Z",
                    "calls": [
                        {
                            "display_name": dn,
                            "ckpt_path": cp,
                            "call_id": c.object_id,
                        }
                        for dn, cp, c in calls
                    ],
                },
                indent=2,
            )
        )
        print(f"[stage8] In-flight state persisted to {IN_FLIGHT_PATH}.")

    # ----- Wait with retry ----------------------------------------------
    t0 = time.time()
    notes_status = "ok"
    remote_results: list[dict[str, Any]] = []
    retryable = (TimeoutError, modal.exception.ConnectionError, ConnectionError)
    try:
        for display_name, ckpt_path, call in calls:
            print(f"\n[stage8] Waiting for {display_name} ({call.object_id})…")
            attempts = 0
            while True:
                try:
                    result = call.get()
                    break
                except retryable as exc:
                    attempts += 1
                    if attempts > 6:
                        raise
                    backoff = min(60 * attempts, 300)
                    print(
                        f"[stage8] Wait attempt {attempts} failed "
                        f"({type(exc).__name__}: {exc}). "
                        f"Sleeping {backoff}s then retrying."
                    )
                    time.sleep(backoff)
            remote_results.append(result)
    except Exception as exc:  # noqa: BLE001
        notes_status = f"failed:{type(exc).__name__}"
        raise
    finally:
        wait_wallclock_s = time.time() - t0
        # Modal billing is per-container-second on the remote side. The
        # local entrypoint just polls; sum returned per-checkpoint
        # wallclock_s for an accurate stage-8 marginal cost. Falls back
        # to local wait time if no results came back.
        remote_total_s = (
            sum(r.get("wallclock_s", 0.0) for r in remote_results)
            if remote_results
            else wait_wallclock_s
        )
        cost_row = record_cost(
            stage="stage8",
            function=("run_eval_dryrun" if dryrun else "run_eval"),
            gpu=MODAL_GPU,
            wallclock_s=remote_total_s,
            notes=(
                f"status={notes_status} targets={len(calls)} "
                f"records={len(records)} dryrun={dryrun} "
                f"wait_s={wait_wallclock_s:.0f}"
            ),
        )
        print(
            f"\n[cost] Recorded ${cost_row['est_cost_usd']:.3f} "
            f"({remote_total_s:.0f}s remote, {notes_status}). "
            f"Cumulative spend: ${total_spend():.2f}"
        )

    # All checkpoints completed cleanly: clear in-flight state so the
    # next invocation does not try to resume.
    IN_FLIGHT_PATH.unlink(missing_ok=True)

    print("\n[stage8] Loading tokenizer locally for verbosity-bias scoring…")
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-4-E4B-it")

    per_checkpoint_metrics: dict[str, dict[str, float]] = {}
    detailed: list[dict[str, Any]] = []
    for result in remote_results:
        preds = [Prediction(**p) for p in result["predictions"]]
        normal = [p for p in preds if not p.swapped and p.temperature == 0.0]
        swapped = [p for p in preds if p.swapped and p.temperature == 0.0]
        consistency_runs = [
            [p for p in preds if p.temperature == 0.3 and p.run_index == i]
            for i in range(SELF_CONSISTENCY_RUNS)
        ]
        per_checkpoint_metrics[result["display_name"]] = aggregate_metrics(
            records, normal, swapped, consistency_runs, tokenizer
        )

        n_index = {p.pair_id: p for p in normal}
        s_index = {p.pair_id: p for p in swapped}
        c_indexes = [{p.pair_id: p for p in run} for run in consistency_runs]
        for record in records:
            pid = record["pair_id"]
            n = n_index.get(pid)
            s = s_index.get(pid)
            cs = [c.get(pid) for c in c_indexes]
            detailed.append(
                {
                    "pair_id": pid,
                    "checkpoint_name": result["display_name"],
                    "human_verdict": record["human_verdict"],
                    "model_verdict": n.verdict if n else None,
                    "swapped_verdict": s.verdict if s else None,
                    "consistency_verdicts": [c.verdict if c else None for c in cs],
                    "raw_reasoning": n.reasoning if n else None,
                    "eval_slice": record.get("eval_slice"),
                    "pair_category": record.get("pair_category"),
                    "confidence": record.get("confidence"),
                    "notes": record.get("notes"),
                }
            )

    table = render_markdown_table(per_checkpoint_metrics)
    print("\n" + ("=" * 60))
    print(f"Stage 8 results ({len(records)} pairs{', dryrun' if dryrun else ''})")
    print("=" * 60)
    print(table)
    print()

    if dryrun:
        _print_dryrun_extras(
            records,
            remote_results,
            remote_total_s,
            cost_row["est_cost_usd"],
            flip_verdict,
        )

    run_id = (
        f"stage8-eval-{'dryrun-' if dryrun else ''}"
        f"{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{run_id}.json"
    payload = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_sha": _git_sha(),
        "modal_gpu": MODAL_GPU,
        "remote_wallclock_s": remote_total_s,
        "local_wait_wallclock_s": wait_wallclock_s,
        "estimated_cost_usd": cost_row["est_cost_usd"],
        "checkpoints": [
            {
                "display_name": r["display_name"],
                "model_name": r["model_name"],
                "checkpoint_path": r["checkpoint_path"],
                "n_records": r["n_records"],
                "wallclock_s": r["wallclock_s"],
            }
            for r in remote_results
        ],
        "metrics": per_checkpoint_metrics,
        "predictions": detailed,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[stage8] Results written to {out_path}")


@app.local_entrypoint()
def cache_status() -> None:
    """Report cached-inference progress per checkpoint.

    Reads ``/vol/eval/cache/*_predictions.jsonl`` and counts how many
    of the 900 inferences-per-checkpoint (300 normal + 300 swapped +
    300 sampled) are already cached. Run after a partial / failed
    invocation to see how much work the next run can skip::

        modal run eval/modal/run_eval.py::cache_status
    """
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from eval.eval_harness import slug

    needed_passes = [(0.0, False, 0), (0.0, True, 0), (0.3, False, 0)]
    target_per_pass = 300

    candidate_names = [
        "Base Gemma 4 E4B",
        "After SFT",
        "After SFT+DPO",
    ]
    print(f"{'Checkpoint':<25} | cached / total | per-pass breakdown")
    print("-" * 80)
    for display_name in candidate_names:
        cache_path = Path(f"/vol/eval/cache/{slug(display_name)}_predictions.jsonl")
        counts = _count_cache_entries(cache_path)
        if counts is None:
            print(f"{display_name:<25} | (no cache file)")
            continue
        useful = sum(min(counts.get(k, 0), target_per_pass) for k in needed_passes)
        total = target_per_pass * len(needed_passes)
        per_pass = " ".join(
            f"{int(t * 10)}/{int(s)}/r{r}={counts.get((t, s, r), 0)}"
            for t, s, r in needed_passes
        )
        print(f"{display_name:<25} | {useful:>3}/{total} | {per_pass}")
    print("\nLegend: per-pass `T*10/swapped/run=count` (e.g., 0/0/r0 = T=0 normal).")


def _count_cache_entries(cache_path: Path) -> dict[tuple[float, bool, int], int] | None:
    """Stream-read ``cache_path`` via Modal volume, return per-(T, swap, run) counts.

    Returns ``None`` when the file does not exist on the volume.
    """
    import tempfile
    from collections import Counter

    # Copy from /vol via the modal CLI, since the volume is only
    # mounted inside @app.function containers.
    rel = str(cache_path).removeprefix("/vol/")
    with tempfile.TemporaryDirectory() as tmp:
        local_path = Path(tmp) / cache_path.name
        proc = subprocess.run(
            [
                "modal",
                "volume",
                "get",
                VOLUME_NAME,
                rel,
                str(local_path),
                "--force",
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            if "not found" in (proc.stderr + proc.stdout).lower():
                return None
            raise RuntimeError(
                f"modal volume get failed for {cache_path}: {proc.stderr}"
            )
        counts: Counter[tuple[float, bool, int]] = Counter()
        with open(local_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                key = (
                    float(r["temperature"]),
                    bool(r["swapped"]),
                    int(r["run_index"]),
                )
                counts[key] += 1
        return dict(counts)


def _print_dryrun_extras(
    records: list[dict[str, Any]],
    remote_results: list[dict[str, Any]],
    wallclock_s: float,
    cost_usd: float,
    flip_verdict: Any,
) -> None:
    """Print sample predictions, position-bias flips, parse-fail tail, cost."""
    from eval.eval_harness import Prediction

    for result in remote_results:
        name = result["display_name"]
        preds = [Prediction(**p) for p in result["predictions"]]
        normal = {p.pair_id: p for p in preds if not p.swapped and p.temperature == 0.0}
        swapped = {p.pair_id: p for p in preds if p.swapped and p.temperature == 0.0}

        print(f"\n--- Sample predictions ({name}) ---")
        for record in records[:5]:
            n = normal.get(record["pair_id"])
            print(
                f"pair={record['pair_id'][:12]} | "
                f"human={record['human_verdict']} | "
                f"model={n.verdict if n else 'MISSING'}"
            )
            if n and n.reasoning:
                trimmed = n.reasoning.strip().replace("\n", " ")
                print(f"  reasoning: {trimmed[:160]}…")

        flips: list[dict[str, str]] = []
        for record in records:
            n = normal.get(record["pair_id"])
            s = swapped.get(record["pair_id"])
            if (
                n is None
                or s is None
                or n.verdict == "PARSE_FAIL"
                or s.verdict == "PARSE_FAIL"
            ):
                continue
            if flip_verdict(n.verdict) != s.verdict:
                flips.append(
                    {
                        "pair_id": record["pair_id"],
                        "normal": n.verdict,
                        "swapped": s.verdict,
                    }
                )
            if len(flips) >= 3:
                break
        print(f"\n--- Position-bias flips ({name}, up to 3 shown) ---")
        if not flips:
            print("(no flips in this subset)")
        for f in flips:
            print(
                f"  pair={f['pair_id'][:12]} normal={f['normal']} swapped={f['swapped']}"
            )

        parse_fails = [p for p in preds if p.verdict == "PARSE_FAIL"]
        print(f"\n--- Parse failures ({name}): {len(parse_fails)}/{len(preds)} ---")
        for pf in parse_fails[:3]:
            tail = pf.raw_output.replace("\n", " ")[:200]
            print(
                f"  pair={pf.pair_id[:12]} swap={pf.swapped} "
                f"run={pf.run_index} raw={tail!r}"
            )

    from eval.eval_harness import SELF_CONSISTENCY_RUNS as _SCR

    n_calls_dryrun = sum(len(r["predictions"]) for r in remote_results)
    full_calls_per_ckpt = 300 * (2 + _SCR)
    full_calls = full_calls_per_ckpt * len(remote_results)
    extrapolated = cost_usd * (full_calls / n_calls_dryrun) if n_calls_dryrun else 0.0
    print(
        f"\n--- Cost ---\n"
        f"  Wallclock: {wallclock_s:.1f}s. Dryrun cost: ${cost_usd:.3f} "
        f"({n_calls_dryrun} calls).\n"
        f"  Extrapolated full-run cost: ${extrapolated:.2f} "
        f"({full_calls} calls; assumes the same per-call rate).\n"
        "Review the table and examples above; run the full eval only "
        "after confirming."
    )
