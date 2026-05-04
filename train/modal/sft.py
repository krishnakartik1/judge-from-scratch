"""Stage 6 SFT training of Gemma 4 E4B on Modal.

Two entrypoints sharing the same training routine:

- ``train_sft_dryrun`` — first 50 rows × 1 epoch, used as a
  shake-out gate before committing to the full 4–6 h run.
- ``train_sft_full`` — full 3,844 rows × 3 epochs, production run.

Run from project root::

    modal run train/modal/sft.py::train_sft_dryrun
    modal run train/modal/sft.py::train_sft_full

Both invocations read hyperparameters from
``train/configs/sft.yaml``. Override the GPU variant with the
``MODAL_GPU`` env var (local-only; not forwarded to the remote
container)::

    MODAL_GPU=A100-80GB modal run train/modal/sft.py::train_sft_full
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import modal

logger = logging.getLogger(__name__)

# --- Paths inside the Modal container ------------------------------------
CONFIG_REMOTE = "/root/train/configs/sft.yaml"
PROBE_DIR = "/root/probe"
SFT_DATA = "/vol/data/sft.jsonl"
EVAL_DATA = "/vol/data/eval_set_unlabeled.jsonl"
SFT_META = Path("data/formatted/sft.meta.json")  # local; embedded into image

# --- Modal app ------------------------------------------------------------
MODAL_GPU = os.environ.get("MODAL_GPU", "A100")
VOLUME_NAME = "judge-from-scratch"

app = modal.App("judge-sft")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
    .add_local_file("uv.lock", "/root/uv.lock", copy=True)
    .uv_sync()
    .add_local_file("train/configs/sft.yaml", CONFIG_REMOTE, copy=True)
    .add_local_file(
        "data/_format_helpers.py", f"{PROBE_DIR}/_format_helpers.py", copy=True
    )
    .add_local_file(
        "data/judge_system_prompt.md",
        f"{PROBE_DIR}/judge_system_prompt.md",
        copy=True,
    )
    .add_local_file("data/formatted/sft.meta.json", "/root/sft.meta.json", copy=True)
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
secrets = [
    modal.Secret.from_name("wandb"),
    modal.Secret.from_name("huggingface"),
]


# --- Pure helpers (importable for unit tests, no GPU/Modal needed) -------


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and return the SFT YAML config as a dict."""
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} did not parse to a dict.")
    for key in ("model_id", "max_seq_length", "lora", "training", "output"):
        if key not in cfg:
            raise ValueError(f"Config missing required key: {key}")
    return cfg


def make_text(row: dict[str, Any], eos_token: str) -> str:
    """Concatenate Stage 5's ``prompt`` + ``target`` + EOS for SFT.

    Stage 5 emits ``prompt`` (chat-template-rendered, ending at
    ``<|turn>model\\n``) and ``target`` (raw response body). TRL's
    SFTTrainer wants a single ``text`` field. ``eos_token`` for the
    Gemma 4 E4B tokenizer is literally ``<turn|>``, which closes the
    assistant turn — do not substitute or strip.
    """
    prompt = row["prompt"]
    target = row["target"]
    if not isinstance(prompt, str) or not isinstance(target, str):
        raise ValueError(
            f"Row prompt/target must be strings; got {type(prompt).__name__}, "
            f"{type(target).__name__}"
        )
    return prompt + target + eos_token


def assert_no_thinking(row: dict[str, Any]) -> None:
    """Hard-fail if ``<|think|>`` appears in any string-valued field.

    Decision #13: native thinking mode disabled. Stage 5 has a hard
    block at format time; this guard catches any drift between the
    Stage 5 producer and the Stage 6 consumer (e.g., a stale upload
    bypassing Stage 5's verify step).
    """
    for key, value in row.items():
        if isinstance(value, str) and "<|think|>" in value:
            raise AssertionError(
                f"<|think|> token found in row field {key!r}; "
                "training would silently enable Gemma 4 native thinking "
                "mode (decision #13). Re-run Stage 5."
            )


def select_probe_pair_ids(rows: list[dict[str, Any]], n: int = 5) -> list[str]:
    """Return the first ``n`` pair_ids when sha1-sorted.

    Deterministic across dryrun and full runs so the side-by-side
    comparison in Step 4 compares the same 5 rows. Sort key is the
    sha1 hex of the pair_id, not the pair_id itself, so the
    selection is independent of any natural ordering in the source
    file (e.g., grouping by bias_category).
    """
    if len(rows) < n:
        raise ValueError(f"Need at least {n} rows; got {len(rows)}.")
    return [
        r["pair_id"]
        for r in sorted(
            rows, key=lambda r: hashlib.sha1(r["pair_id"].encode()).hexdigest()
        )[:n]
    ]


# --- Modal-side training routine -----------------------------------------


def _build_dataset_and_text(tokenizer: Any, mode: Literal["dryrun", "full"]) -> Any:
    """Load /vol/data/sft.jsonl, run startup checks, build the text field."""
    from datasets import load_dataset

    ds = load_dataset("json", data_files=SFT_DATA, split="train")
    first = ds[0]
    assert_no_thinking(first)

    eos = tokenizer.eos_token
    if eos != "<turn|>":
        raise AssertionError(
            f"Tokenizer eos_token is {eos!r}, expected '<turn|>'. "
            "Stage 5 chat-template parity broken — re-run smoke test."
        )

    ds = ds.map(lambda row: {"text": make_text(row, eos)})

    if mode == "dryrun":
        ds = ds.select(range(min(50, len(ds))))
        logger.info("Dryrun mode: training dataset sliced to %d rows.", len(ds))

    return ds


def _pretokenize(train_ds: Any, tokenizer: Any, max_length: int) -> Any:
    """Tokenize CPU-side, single-process, to bypass Unsloth's forced
    multiproc on ``SFTTrainer._prepare_dataset``.

    Why: Unsloth source-injects into ``SFTTrainer.__init__``
    (``unsloth/models/rl.py:1195``) a check that overrides
    ``dataset_num_proc=None`` to ~21 on fork-style Linux. The patched
    code then calls ``dataset.map`` with ``num_proc>=1``, which (per
    HF datasets ``arrow_dataset.py:3318``) spawns a multiprocess Pool
    that tries to pickle the Unsloth-patched processing class —
    failing on ``torch._dynamo.config.ConfigModuleInstance``.

    By pre-tokenizing here and passing
    ``dataset_kwargs={"skip_prepare_dataset": True}`` to ``SFTConfig``,
    TRL skips ``_prepare_dataset`` entirely and uses our rows verbatim.

    Truncation to ``max_length`` is applied here since TRL's truncation
    step is also skipped under ``skip_prepare_dataset``.
    """
    text_tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer

    def tok_fn(example: dict[str, Any]) -> dict[str, Any]:
        enc = text_tok(
            example["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }

    # No num_proc kwarg → defaults to None → single-process branch
    # in arrow_dataset.py:3330. No Pool, no pickling.
    return train_ds.map(
        tok_fn,
        remove_columns=["text"],
        desc="Pre-tokenize SFT (single-process)",
    )


def _assert_label_masking(trainer: Any, tokenizer: Any) -> None:
    """Verify completion-only loss masking is wired correctly.

    After ``train_on_responses_only`` modifies the data collator,
    feeding the first sample through that collator should produce
    a labels tensor where the prompt portion is masked (-100) and
    the response portion is real token ids. Hard-fail if either
    half is empty — over-masking trains nothing, under-masking
    trains the system prompt.
    """
    sample = trainer.train_dataset[0]
    batch = trainer.data_collator([sample])
    labels = batch["labels"][0].tolist()
    input_ids = batch["input_ids"][0].tolist()

    masked = [i for i, lab in enumerate(labels) if lab == -100]
    unmasked = [i for i, lab in enumerate(labels) if lab != -100]

    if not masked:
        raise AssertionError(
            "Label masking sanity check failed: no -100 labels found. "
            "train_on_responses_only did not mask the prompt portion — "
            "training would learn from the system prompt."
        )
    if not unmasked:
        raise AssertionError(
            "Label masking sanity check failed: all labels are -100. "
            "Over-masking — training has no signal to learn from."
        )

    # Decode boundary tokens for a human-legible audit trail.
    last_masked_idx = masked[-1]
    first_unmasked_idx = unmasked[0]
    text_before = tokenizer.decode(input_ids[: last_masked_idx + 1])
    text_after = tokenizer.decode(input_ids[first_unmasked_idx:])

    logger.info(
        "Label masking OK: %d masked / %d unmasked tokens. "
        "Last masked decodes to ...%r; first unmasked decodes to %r...",
        len(masked),
        len(unmasked),
        text_before[-40:],
        text_after[:40],
    )

    if "<|turn>model" not in text_before[-80:]:
        raise AssertionError(
            "Label masking boundary suspect: <|turn>model not in last 80 "
            f"chars of masked region. Tail: {text_before[-80:]!r}"
        )


def _make_volume_commit_callback() -> Any:
    """Return a TrainerCallback class that commits the volume on save.

    Constructed lazily so unit tests do not pull in transformers.
    """
    from transformers import TrainerCallback

    class VolumeCommitCallback(TrainerCallback):
        """Commits the Modal volume at every save event.

        Without this, intermediate epoch checkpoints written by HF
        Trainer are visible in the container's view of the volume
        but not durably persisted until the next ``volume.commit()``.
        A mid-epoch crash on the final epoch would otherwise lose
        all earlier epochs' adapters.
        """

        def __init__(self, vol: Any) -> None:
            self.vol = vol

        def on_save(self, args, state, control, **kwargs):  # noqa: ANN001
            self.vol.commit()
            logger.info(
                "VolumeCommitCallback: committed at step %d.", state.global_step
            )
            return control

    return VolumeCommitCallback


def _load_probe_rows() -> list[dict[str, Any]]:
    """Read /vol/data/eval_set_unlabeled.jsonl into a list of dicts."""
    rows: list[dict[str, Any]] = []
    with open(EVAL_DATA) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _run_probe(
    model: Any,
    tokenizer: Any,
    probe_rows: list[dict[str, Any]],
    system_prompt: str,
) -> list[dict[str, Any]]:
    """Run greedy inference on the 5 probe rows, return generations.

    Builds prompts via Stage 5's ``apply_chat`` helper (mounted at
    ``/root/probe``) so the inference-time prompt shape matches the
    training-time prompt shape exactly. A suffix-parity assertion
    catches any chat-template drift between the two stages.
    """
    import torch

    if PROBE_DIR not in sys.path:
        sys.path.insert(0, PROBE_DIR)
    from _format_helpers import apply_chat, build_user_message  # noqa: E402

    outputs: list[dict[str, Any]] = []
    for row in probe_rows:
        user_msg = build_user_message(row, swap=False)
        prompt = apply_chat(tokenizer, system_prompt, user_msg)
        if not prompt.endswith("<|turn>model\n"):
            raise AssertionError(
                f"Probe prompt suffix mismatch (expected '<|turn>model\\n'): "
                f"{prompt[-60:]!r}"
            )
        # Gemma 4 multimodal processor: must pass text= as keyword,
        # else the positional arg is mis-routed to images/videos.
        inputs = tokenizer(text=prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            out[0][inputs.input_ids.shape[1] :], skip_special_tokens=False
        )
        outputs.append({"pair_id": row["pair_id"], "generated": generated})
    return outputs


def _hf_push(final_dir: str, cfg: dict[str, Any]) -> str:
    """Push the final adapter to HF Hub (private). Returns the repo URL."""
    from huggingface_hub import HfApi

    repo_id = cfg["hf_push"]["repo_id"]
    private = cfg["hf_push"].get("private", True)

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=final_dir, repo_id=repo_id, repo_type="model")
    url = f"https://huggingface.co/{repo_id}"
    logger.info("Pushed final adapter to %s", url)
    return url


def _train(mode: Literal["dryrun", "full"]) -> dict[str, Any]:
    """Shared SFT training routine for both dryrun and full modes."""
    import time

    import torch
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg = load_config(CONFIG_REMOTE)

    # Confirm the tokenizer.model_id matches what Stage 5 used.
    sft_meta = json.loads(Path("/root/sft.meta.json").read_text())
    if sft_meta["tokenizer_model_id"] != cfg["model_id"]:
        raise AssertionError(
            f"Tokenizer mismatch: Stage 5 used {sft_meta['tokenizer_model_id']!r}, "
            f"Stage 6 config has {cfg['model_id']!r}. Re-run Stage 5 or update "
            "config to match."
        )

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{ts}-sft-{mode}"

    # Wire W&B project/run before trainer init: HF TrainingArguments
    # has no wandb_project arg; without these env vars the run lands
    # in wandb's default project.
    os.environ["WANDB_PROJECT"] = cfg["wandb"]["project"]
    os.environ["WANDB_NAME"] = run_name

    # Force Unsloth to return real logit tensors instead of its lazy
    # cut_cross_entropy callable. TRL 0.24 logs per-token entropy by
    # default in SFTTrainer.compute_loss (sft_trainer.py:1105 calls
    # `entropy_from_logits(outputs.logits)`), which subscripts
    # `.shape`; if logits is the lazy callable, this fails with
    # "TypeError: 'function' object is not subscriptable". Must be
    # set BEFORE FastLanguageModel.from_pretrained reads the env.
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

    torch.cuda.reset_peak_memory_stats()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_id"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        target_modules=cfg["lora"]["target_modules"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        bias=cfg["lora"]["bias"],
        random_state=cfg["lora"]["random_state"],
    )

    train_ds = _build_dataset_and_text(tokenizer, mode)
    train_ds = _pretokenize(train_ds, tokenizer, cfg["max_seq_length"])

    train_cfg = cfg["training"]
    epochs = 1 if mode == "dryrun" else train_cfg["num_train_epochs"]

    # TRL 0.24+: SFT-specific args (dataset_text_field, max_seq_length,
    # packing) live on SFTConfig (extends TrainingArguments). The
    # SFTTrainer kwargs `tokenizer=` and `dataset_text_field=` were
    # removed in favor of `processing_class=` and config-level fields.
    args = SFTConfig(
        output_dir=cfg["output"]["checkpoints_dir"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=epochs,
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        optim=train_cfg["optim"],
        weight_decay=train_cfg["weight_decay"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        report_to="wandb",
        run_name=run_name,
        seed=cfg["lora"]["random_state"],
        dataset_text_field="text",
        max_length=cfg["max_seq_length"],  # renamed from max_seq_length in TRL 0.24
        packing=False,
        # We pre-tokenize CPU-side (see `_pretokenize`) and tell TRL to
        # leave the dataset alone. This bypasses both:
        #   1. Unsloth's source-injected override of `dataset_num_proc`
        #      (`unsloth/models/rl.py:1195`), which forces ~21 procs on
        #      fork-style Linux even when we pass None.
        #   2. HF datasets' `arrow_dataset.py:3318` bug where any int
        #      `num_proc>=1` spawns a multiprocess Pool that pickles
        #      the Unsloth-patched processing class — crashing on
        #      `torch._dynamo.config.ConfigModuleInstance`.
        # With `skip_prepare_dataset=True`, TRL respects our pre-built
        # `input_ids` and `attention_mask` columns and adds nothing.
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=None,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        args=args,
    )

    # Apply Unsloth's response-only loss masking. We bypass the
    # `train_on_responses_only(trainer, ...)` wrapper because it
    # auto-detects num_proc to ~21 on Linux fork-style multiprocessing
    # (unsloth_zoo/dataset_utils.py:354), and even passing num_proc=1
    # hits HF datasets' arrow_dataset.py:3318 bug where any int
    # num_proc spawns a Pool. Both paths pickle the Unsloth-patched
    # closure → ConfigModuleInstance crash.
    #
    # `return_function=True` short-circuits before the auto-num_proc
    # block, returning just the per-batch masking callable. We then
    # apply it via a single-process .map() (no num_proc kwarg → None
    # default → single-process branch in arrow_dataset.py:3330).
    mask_fn = train_on_responses_only(
        trainer=None,
        tokenizer=tokenizer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
        return_function=True,
    )
    trainer.train_dataset = trainer.train_dataset.map(
        mask_fn, batched=True, desc="Applying response-only label mask"
    )
    # Mirror the wrapper's defensive filter: drop samples whose entire
    # label tensor is -100 (would NaN cross-entropy). Unlikely with our
    # ~700-token rows under max_length=2048, but ~free insurance.
    trainer.train_dataset = trainer.train_dataset.filter(
        lambda ex: any(label != -100 for label in ex["labels"]),
        desc="Filtering empty-label samples",
    )

    _assert_label_masking(trainer, tokenizer)

    callback_cls = _make_volume_commit_callback()
    trainer.add_callback(callback_cls(volume))

    t0 = time.time()
    train_result = trainer.train()
    wallclock_s = time.time() - t0

    final_dir = cfg["output"]["final_dir"]
    Path(final_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    volume.commit()

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    logger.info(
        "Training finished. wallclock=%.1fs peak_vram=%.2fGB final_loss=%.4f",
        wallclock_s,
        peak_vram_gb,
        train_result.training_loss,
    )

    # --- Probe inference -------------------------------------------------
    eval_rows = _load_probe_rows()
    probe_pair_ids = select_probe_pair_ids(eval_rows, n=cfg["probe"]["count"])
    probe_rows = [r for r in eval_rows if r["pair_id"] in set(probe_pair_ids)]
    probe_rows.sort(key=lambda r: probe_pair_ids.index(r["pair_id"]))
    Path(cfg["output"]["checkpoints_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["output"]["checkpoints_dir"], "probe_pair_ids.json").write_text(
        json.dumps(probe_pair_ids, indent=2)
    )

    system_prompt = Path(f"{PROBE_DIR}/judge_system_prompt.md").read_text()

    # Reload from final_dir so probe runs against the saved adapter
    # (matches what Stage 9 will publish).
    probe_model, probe_tokenizer = FastLanguageModel.from_pretrained(
        model_name=final_dir,
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
    )
    FastLanguageModel.for_inference(probe_model)
    probe_outputs = _run_probe(probe_model, probe_tokenizer, probe_rows, system_prompt)

    out_filename = "dryrun_outputs.json" if mode == "dryrun" else "full_outputs.json"
    out_target_dir = (
        cfg["output"]["checkpoints_dir"]
        if mode == "dryrun"
        else cfg["output"]["final_dir"]
    )
    Path(out_target_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_target_dir, out_filename)
    out_path.write_text(json.dumps(probe_outputs, indent=2))
    volume.commit()
    logger.info("Probe outputs written to %s", out_path)

    # --- HF push (full mode only) ----------------------------------------
    hf_url: str | None = None
    if mode == "full":
        try:
            hf_url = _hf_push(final_dir, cfg)
        except Exception as e:  # noqa: BLE001
            logger.exception(
                "HF push failed (training artifacts safe on volume): %s", e
            )

    return {
        "mode": mode,
        "run_name": run_name,
        "wallclock_s": wallclock_s,
        "peak_vram_gb": peak_vram_gb,
        "final_loss": train_result.training_loss,
        "final_dir": final_dir,
        "probe_outputs_path": str(out_path),
        "probe_pair_ids": probe_pair_ids,
        "hf_url": hf_url,
    }


# --- Modal remote functions (private; invoked via local entrypoints) -----

DRYRUN_TIMEOUT_S = 1800
FULL_TIMEOUT_S = 21600


@app.function(
    image=image,
    gpu=MODAL_GPU,
    timeout=DRYRUN_TIMEOUT_S,
    volumes={"/vol": volume},
    secrets=secrets,
)
def _train_sft_dryrun_remote() -> dict[str, Any]:
    """50 rows × 1 epoch shake-out run. Gates the full SFT.

    Private remote function. Invoke via the
    ``train_sft_dryrun`` local entrypoint, which adds budget gating.
    """
    result = _train("dryrun")
    print(f"\nDryrun result: {json.dumps(result, indent=2)}")
    return result


@app.function(
    image=image,
    gpu=MODAL_GPU,
    timeout=FULL_TIMEOUT_S,
    volumes={"/vol": volume},
    secrets=secrets,
)
def _train_sft_full_remote() -> dict[str, Any]:
    """Full 3,844 rows × 3 epochs production run.

    Private remote function. Invoke via the
    ``train_sft_full`` local entrypoint, which adds budget gating.
    """
    result = _train("full")
    print(f"\nFull-run result: {json.dumps(result, indent=2)}")
    return result


# --- Local entrypoints (with budget gating) ------------------------------


@app.local_entrypoint()
def train_sft_dryrun() -> None:
    """Budget-gated wrapper around the dryrun. Use this, not the remote.

    Run from project root::

        modal run train/modal/sft.py::train_sft_dryrun

    Set ``BUDGET_OVERRIDE=1`` in env to skip the interactive prompt.
    """
    import time

    from train.modal._cost_ledger import (
        check_budget,
        project_cost,
        record_cost,
        total_spend,
    )

    projected = project_cost(MODAL_GPU, DRYRUN_TIMEOUT_S)
    spent = total_spend()
    check_budget(projected, spent_usd=spent, label=f"sft-dryrun on {MODAL_GPU}")

    t0 = time.time()
    result: dict[str, Any] | None = None
    notes_status = "ok"
    try:
        result = _train_sft_dryrun_remote.remote()
    except Exception as exc:  # noqa: BLE001
        notes_status = f"failed:{type(exc).__name__}"
        raise
    finally:
        actual_wallclock_s = time.time() - t0
        if result is not None:
            notes = (
                f"status={notes_status} "
                f"loss={result.get('final_loss'):.4f} "
                f"train_s={result.get('wallclock_s'):.1f}"
            )
        else:
            notes = f"status={notes_status}"
        row = record_cost(
            stage="stage6",
            function="train_sft_dryrun",
            gpu=MODAL_GPU,
            wallclock_s=actual_wallclock_s,
            notes=notes,
        )
        print(
            f"\n[cost] Dryrun recorded: ${row['est_cost_usd']:.3f} "
            f"({actual_wallclock_s:.1f}s wallclock, {notes_status}). "
            f"Cumulative spend: ${total_spend():.2f}"
        )


@app.local_entrypoint()
def train_sft_full() -> None:
    """Budget-gated wrapper around the full SFT. Use this, not the remote.

    Run from project root::

        modal run train/modal/sft.py::train_sft_full

    Set ``BUDGET_OVERRIDE=1`` in env to skip the interactive prompt.
    """
    import time

    from train.modal._cost_ledger import (
        check_budget,
        project_cost,
        record_cost,
        total_spend,
    )

    projected = project_cost(MODAL_GPU, FULL_TIMEOUT_S)
    spent = total_spend()
    check_budget(projected, spent_usd=spent, label=f"sft-full on {MODAL_GPU}")

    t0 = time.time()
    result: dict[str, Any] | None = None
    notes_status = "ok"
    try:
        result = _train_sft_full_remote.remote()
    except Exception as exc:  # noqa: BLE001
        notes_status = f"failed:{type(exc).__name__}"
        raise
    finally:
        actual_wallclock_s = time.time() - t0
        if result is not None:
            notes = (
                f"status={notes_status} "
                f"loss={result.get('final_loss'):.4f} "
                f"train_s={result.get('wallclock_s'):.1f} "
                f"hf_url={result.get('hf_url')}"
            )
        else:
            notes = f"status={notes_status}"
        row = record_cost(
            stage="stage6",
            function="train_sft_full",
            gpu=MODAL_GPU,
            wallclock_s=actual_wallclock_s,
            notes=notes,
        )
        print(
            f"\n[cost] Full SFT recorded: ${row['est_cost_usd']:.2f} "
            f"({actual_wallclock_s:.1f}s wallclock, {notes_status}). "
            f"Cumulative spend: ${total_spend():.2f}"
        )


# --- HF re-push (recovery path for failed in-band push) ------------------


@app.function(
    image=image,
    timeout=600,
    volumes={"/vol": volume},
    secrets=secrets,
)
def _push_sft_to_hf_remote() -> dict[str, str | None]:
    """Push the existing ``/vol/checkpoints/sft-final/`` to HF Hub.

    CPU-only (no GPU needed for the upload). Reads the same
    ``hf_push`` block from ``train/configs/sft.yaml`` that the full
    training run uses, so updating the config repo_id is all that's
    needed to retarget.

    Hard-fails if the adapter directory is missing — the only legitimate
    reason to call this entrypoint is to recover a failed in-band push,
    which means the training step ran successfully and produced an
    adapter on the volume.
    """
    cfg = load_config(CONFIG_REMOTE)
    final_dir = cfg["output"]["final_dir"]
    if not Path(final_dir, "adapter_config.json").exists():
        raise FileNotFoundError(
            f"No adapter_config.json found at {final_dir}. "
            "Did `train_sft_full` complete? This entrypoint is only for "
            "recovering a failed in-band HF push, not for fresh training."
        )
    url = _hf_push(final_dir, cfg)
    return {"hf_url": url, "final_dir": final_dir}


@app.local_entrypoint()
def push_sft_to_hf() -> None:
    """Re-push the saved SFT adapter to HF Hub.

    Use this after ``train_sft_full`` if the in-band HF push failed
    (e.g., 403 because the token didn't have write access to the repo
    namespace). Reads ``train/configs/sft.yaml``'s ``hf_push.repo_id``
    so updating the config is the canonical way to change targets.

    Run from project root::

        modal run train/modal/sft.py::push_sft_to_hf

    Cost is ~$0.01 (CPU-only, ~30 s wallclock for the upload).
    """
    import time

    from train.modal._cost_ledger import record_cost, total_spend

    t0 = time.time()
    result: dict[str, str | None] | None = None
    notes_status = "ok"
    try:
        result = _push_sft_to_hf_remote.remote()
    except Exception as exc:  # noqa: BLE001
        notes_status = f"failed:{type(exc).__name__}"
        raise
    finally:
        actual_wallclock_s = time.time() - t0
        if result is not None:
            print(f"\nHF push complete: {result.get('hf_url')}")
            notes = (
                f"status={notes_status} hf_url={result.get('hf_url')} "
                f"final_dir={result.get('final_dir')}"
            )
        else:
            notes = f"status={notes_status}"
        row = record_cost(
            stage="stage6",
            function="push_sft_to_hf",
            gpu="CPU",
            wallclock_s=actual_wallclock_s,
            notes=notes,
        )
        print(
            f"[cost] HF push recorded: ${row['est_cost_usd']:.4f} "
            f"({actual_wallclock_s:.1f}s wallclock, {notes_status}). "
            f"Cumulative spend: ${total_spend():.2f}"
        )
