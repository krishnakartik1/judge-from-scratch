"""Modal smoke test for Stage 6 SFT.

Verifies Modal + uv_sync + Unsloth + Gemma 4 E4B + 4-bit
quantization all work together on a Modal A100 before any
training Modal-cost is incurred. Cost ~$0.20–0.50 per run.

Run from project root:
    modal run train/modal/smoke_test.py
"""

from __future__ import annotations

import modal

app = modal.App("judge-smoke-test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
    .add_local_file("uv.lock", "/root/uv.lock", copy=True)
    .uv_sync()
)

# HF_TOKEN attached defensively in case unsloth/gemma-4-E4B-it ever
# becomes a gated mirror; without it, the base-model download would
# 401 from inside the container with no token in env.
secrets = [modal.Secret.from_name("huggingface")]


@app.function(image=image, gpu="A100", timeout=900, secrets=secrets)
def check_env() -> dict[str, object]:
    import bitsandbytes as bnb
    import peft
    import torch
    import transformers
    import trl
    from unsloth import FastLanguageModel

    # Plan extension #1: verify train_on_responses_only is importable
    # in the installed Unsloth version. The SFT script depends on it;
    # catch a renamed/moved symbol here, not mid-training.
    from unsloth.chat_templates import train_on_responses_only  # noqa: F401

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(
        "Total VRAM: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )
    print(f"torch: {torch.__version__}")
    print(f"transformers: {transformers.__version__}")
    print(f"trl: {trl.__version__}")
    print(f"peft: {peft.__version__}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gemma-4-E4B-it-unsloth-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    vram_after_load = torch.cuda.memory_allocated() / 1e9
    print(f"VRAM after model load: {vram_after_load:.2f} GB")

    # Plan extension #2: assert EOS is the literal "<turn|>" token.
    # The SFT text builder appends tokenizer.eos_token to terminate
    # the assistant turn. If a future Unsloth release changes the
    # EOS, the SFT text would be malformed; fail fast here before
    # any training run is attempted.
    assert tokenizer.eos_token == "<turn|>", (
        f"Unexpected eos_token {tokenizer.eos_token!r} (id="
        f"{tokenizer.eos_token_id}); SFT text builder requires '<turn|>'."
    )

    # Plan extension #3: positive 4-bit verification. Don't rely on
    # VRAM math (Matryoshka E4B + Unsloth Dynamic keeps a non-trivial
    # fraction of weights in fp16, pushing VRAM to ~10–11 GB). Instead,
    # confirm 4-bit engaged by counting Linear4bit modules.
    n_4bit = sum(
        1 for m in model.modules() if isinstance(m, bnb.nn.Linear4bit)  # type: ignore[attr-defined]
    )
    print(f"Linear4bit modules: {n_4bit}")
    assert n_4bit > 0, (
        "load_in_4bit silently failed: 0 Linear4bit modules found. "
        "Model loaded as full precision; check model_id and bitsandbytes."
    )

    # VRAM bound widened from (3.5, 6.5) to (3.5, 16.0) to accommodate
    # Unsloth Dynamic on Gemma 4 E4B's Matryoshka architecture (10.85 GB
    # observed empirically). Still well under the A100-40GB ceiling.
    assert 3.5 < vram_after_load < 16.0, (
        f"Unexpected VRAM after load ({vram_after_load:.2f} GB) — "
        "outside the (3.5, 16.0) GB envelope for Gemma 4 E4B 4-bit."
    )

    # Gemma 4 ships a multimodal processor; the positional-arg form
    # routes the first arg to images/videos depending on Unsloth's
    # patch path. Always call with text= keyword.
    inputs = tokenizer(text="Hello world", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    generated = tokenizer.decode(out[0])
    print(f"Generated: {generated!r}")

    return {
        "vram_gb": vram_after_load,
        "device": torch.cuda.get_device_name(0),
        "generated": generated,
        "eos_token": tokenizer.eos_token,
        "linear4bit_count": n_4bit,
    }


SMOKE_TIMEOUT_S = 900
SMOKE_GPU = "A100"


@app.local_entrypoint()
def main() -> None:
    """Budget-gated smoke test entrypoint.

    Run from project root::

        modal run train/modal/smoke_test.py

    Set ``BUDGET_OVERRIDE=1`` in env to skip the interactive prompt.
    """
    import time

    from train.modal._cost_ledger import (
        check_budget,
        project_cost,
        record_cost,
        total_spend,
    )

    projected = project_cost(SMOKE_GPU, SMOKE_TIMEOUT_S)
    spent = total_spend()
    check_budget(projected, spent_usd=spent, label=f"smoke-test on {SMOKE_GPU}")

    t0 = time.time()
    result: dict[str, object] | None = None
    notes_status = "ok"
    try:
        result = check_env.remote()
    except Exception as exc:  # noqa: BLE001
        notes_status = f"failed:{type(exc).__name__}"
        raise
    finally:
        actual_wallclock_s = time.time() - t0
        if result is not None:
            print(f"\nSmoke test PASSED: {result}")
            notes = (
                f"status={notes_status} "
                f"vram_gb={result.get('vram_gb'):.2f} "
                f"linear4bit={result.get('linear4bit_count')}"
            )
        else:
            notes = f"status={notes_status}"
        row = record_cost(
            stage="stage6",
            function="smoke_test",
            gpu=SMOKE_GPU,
            wallclock_s=actual_wallclock_s,
            notes=notes,
        )
        print(
            f"[cost] Smoke recorded: ${row['est_cost_usd']:.3f} "
            f"({actual_wallclock_s:.1f}s wallclock, {notes_status}). "
            f"Cumulative spend: ${total_spend():.2f}"
        )
