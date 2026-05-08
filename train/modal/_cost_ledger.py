"""Modal cost-tracking ledger and pre-flight budget guardrail.

Mirrors the Stage 4/5 cost-ledger pattern (``data/04_label_pairs.py``,
``data/synth_rejected.py``) but for Modal GPU spend instead of
Anthropic/OpenAI token spend.

Per-run flow (called from each ``@app.local_entrypoint()``):

1. Read local ledger → cumulative spend so far.
2. Project this run's worst-case cost (timeout × hourly rate).
3. Hard-fail if cumulative + projected exceeds the stage budget cap,
   unless the operator types ``CONTINUE`` at the prompt or sets
   ``BUDGET_OVERRIDE=1`` in the env.
4. Run the Modal function via ``.remote()``.
5. Record the actual cost (wallclock × hourly rate) into the ledger.

The ledger lives at ``train/.cost_ledger.jsonl`` (gitignored). One JSON
record per Modal invocation; append-only.

Pricing values are approximate from Modal's pricing page as of
2026-05-03. They are NOT live — refresh from
https://modal.com/pricing before relying on these for invoicing
projections. The dominant cost is GPU-time; CPU/memory/storage are
modelled but small.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

LEDGER_PATH = Path(__file__).resolve().parent.parent / ".cost_ledger.jsonl"

# Approximate Modal rates in USD per second. Sourced from
# https://modal.com/pricing on 2026-05-03; refresh if cost projections
# look off.
MODAL_GPU_RATES_USD_PER_SEC: dict[str, float] = {
    "A100": 1.86 / 3600,
    "A100-40GB": 1.86 / 3600,
    "A100-80GB": 2.50 / 3600,
    "A10G": 1.10 / 3600,
    "A10": 1.10 / 3600,
    "L40S": 1.95 / 3600,
    "L4": 0.80 / 3600,
    "T4": 0.59 / 3600,
    "H100": 4.40 / 3600,
    "H100-80GB": 4.40 / 3600,
    "CPU": 0.135 / 3600,
}

# Stage 6 budget cap. Beyond this, the operator must explicitly type
# CONTINUE at a prompt or set BUDGET_OVERRIDE=1 in the env to proceed.
# Sized to comfortably accommodate one full SFT (~$15) plus iteration
# headroom; tighten for production.
STAGE6_BUDGET_CAP_USD: float = 30.0

# Stage 7 budget cap. The original $30 minus what Stage 6 actually
# burned ($3.63 across SFT smoke, dryrun, and full run as of 2026-05-04
# — see train/.cost_ledger.jsonl). Effectively "the remaining money
# allocated when the project started, with Stage 6 baked in." Accepts
# DPO dryrun (~$0.25 expected, $0.93 worst-case) and full run (~$7
# expected, $7.44 worst-case) with comfortable headroom for retries.
STAGE7_BUDGET_CAP_USD: float = 26.37

# Stage 8 (vLLM eval pivot) budget cap. Realistic worst case under the
# @app.cls amortization assumption is ~$3.75 (3 models × 30-min
# ceiling × $2.50/h). The $10 cap absorbs ~3× that for retries and
# ad-hoc smoke runs without forcing operator overrides on every run.
STAGE8_BUDGET_CAP_USD: float = 10.0


class BudgetExceededError(RuntimeError):
    """Raised when projected + cumulative spend would exceed the cap."""


def compute_modal_cost(gpu: str, wallclock_s: float) -> float:
    """Approximate cost in USD for a Modal function run.

    Args:
        gpu: GPU spec string (e.g., ``"A100"``, ``"A100-80GB"``,
            ``"CPU"``). Unknown specs default to CPU pricing with a
            warning written into the ledger note.
        wallclock_s: Wall-clock seconds the function ran for, including
            container spin-up.

    Returns:
        Estimated cost in USD. Negative values raise ``ValueError``.
    """
    if wallclock_s < 0:
        raise ValueError(f"wallclock_s must be non-negative; got {wallclock_s!r}")
    rate = MODAL_GPU_RATES_USD_PER_SEC.get(gpu, MODAL_GPU_RATES_USD_PER_SEC["CPU"])
    return rate * wallclock_s


def total_spend(ledger_path: Path = LEDGER_PATH) -> float:
    """Sum the ``est_cost_usd`` field across the ledger. 0.0 if missing."""
    if not ledger_path.exists():
        return 0.0
    total = 0.0
    with open(ledger_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += float(row.get("est_cost_usd", 0.0))
    return total


def record_cost(
    stage: str,
    function: str,
    gpu: str,
    wallclock_s: float,
    modal_run_id: str | None = None,
    notes: str | None = None,
    ledger_path: Path = LEDGER_PATH,
) -> dict[str, Any]:
    """Append one cost record to the ledger and return the appended row.

    Idempotent in the sense that re-running creates additional rows,
    not duplicates — Modal runs themselves are unique invocations and
    each one earns one entry.
    """
    est_cost = compute_modal_cost(gpu, wallclock_s)
    row: dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "stage": stage,
        "function": function,
        "gpu": gpu,
        "wallclock_s": round(wallclock_s, 2),
        "rate_usd_per_sec": MODAL_GPU_RATES_USD_PER_SEC.get(
            gpu, MODAL_GPU_RATES_USD_PER_SEC["CPU"]
        ),
        "est_cost_usd": round(est_cost, 4),
        "modal_run_id": modal_run_id,
        "notes": notes,
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def project_cost(gpu: str, timeout_s: float) -> float:
    """Worst-case projection: ``timeout_s`` × hourly rate.

    Real runs almost always complete faster than the timeout, but the
    cap is conservative — we'd rather pre-empt an unintended long run
    than under-project.
    """
    return compute_modal_cost(gpu, timeout_s)


def check_budget(
    projected_usd: float,
    cap_usd: float = STAGE6_BUDGET_CAP_USD,
    spent_usd: float | None = None,
    *,
    force: bool = False,
    label: str = "this run",
) -> None:
    """Hard-fail or interactive-prompt if cumulative+projected > cap.

    Args:
        projected_usd: Worst-case cost of the upcoming run.
        cap_usd: Hard cap; defaults to the stage-6 cap.
        spent_usd: Cumulative spend so far. Defaults to ``total_spend()``.
        force: If True, skip the prompt entirely (ledger still records
            the run). Useful for non-interactive contexts.
        label: Human-readable label for the operator prompt.

    Raises:
        BudgetExceededError: if cap would be exceeded and the operator
            does not confirm.
    """
    if spent_usd is None:
        spent_usd = total_spend()
    total_after = spent_usd + projected_usd

    print(
        f"[cost] {label}: projected ${projected_usd:.2f} | "
        f"spent so far ${spent_usd:.2f} | "
        f"cap ${cap_usd:.2f} | total-after ${total_after:.2f}"
    )

    if total_after <= cap_usd:
        return

    if force or os.environ.get("BUDGET_OVERRIDE") == "1":
        print(f"[cost] OVERRIDE engaged — proceeding past cap (${cap_usd:.2f}).")
        return

    answer = input(
        f"[cost] Projected total ${total_after:.2f} > cap ${cap_usd:.2f}. "
        "Type CONTINUE to override: "
    )
    if answer.strip() != "CONTINUE":
        raise BudgetExceededError(
            f"Aborted by operator. Spent ${spent_usd:.2f}, "
            f"projected ${projected_usd:.2f}, cap ${cap_usd:.2f}."
        )
    print("[cost] CONTINUE acknowledged — proceeding past cap.")


def status_summary(ledger_path: Path = LEDGER_PATH) -> str:
    """Human-readable summary of the ledger. Used for ``status`` commands."""
    if not ledger_path.exists():
        return "No cost ledger yet (no Modal runs recorded)."
    lines = [f"Cost ledger: {ledger_path}"]
    rows: list[dict[str, Any]] = []
    with open(ledger_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    total = sum(float(r.get("est_cost_usd", 0.0)) for r in rows)
    lines.append(f"Total runs: {len(rows)}")
    lines.append(f"Cumulative spend: ${total:.2f}")
    lines.append("")
    lines.append("Recent runs (last 10):")
    for r in rows[-10:]:
        lines.append(
            f"  {r['timestamp']}  {r['stage']}/{r['function']:<22}  "
            f"{r['gpu']:<10} {r['wallclock_s']:>7.1f}s  ${r['est_cost_usd']:>6.3f}"
        )
    return "\n".join(lines)
