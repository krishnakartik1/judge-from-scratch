"""Stage 7 DPO training of Gemma 4 E4B on Modal, on top of Stage 6's
SFT adapter.

Four entrypoints:

- ``train_dpo_dryrun`` — first 50 rows × 1 epoch, used as a shake-out
  gate before committing to the full run. With 50 rows, batch 2,
  grad_accum 8 the dryrun is only ~3 optim steps, so the
  over-collapse verdict it produces is a smoke check, not a quality
  gate. The detector earns its keep on the full run (~137 steps).
- ``train_dpo_full`` — full DPO pool × 1 epoch, production run.
  Saves an adapter at /vol/checkpoints/dpo-final/. Does NOT produce
  the merged-fp16 artifact — call ``merge_dpo_to_fp16`` separately
  once you've reviewed the metrics.
- ``merge_dpo_to_fp16`` — reloads the saved DPO adapter at fp16,
  folds it into the base, writes /vol/checkpoints/merged-fp16/.
  Standalone so the operator can decide post-hoc whether the run is
  worth deploying; the merge is irrelevant to training/eval through
  the adapter and only matters for vLLM, GGUF (Stage 10), and HF
  publishing (Stage 9).
- ``verify_merged_fp16`` — loads the merged checkpoint as a vanilla
  HF model and greedy-generates one verdict on a pinned probe pair,
  asserting the <reasoning>…</reasoning><verdict>X</verdict> shape
  with no <|think|> leakage. Cheap post-merge sanity check.

Run from project root::

    modal run train/modal/dpo.py::train_dpo_dryrun
    modal run train/modal/dpo.py::train_dpo_full
    modal run train/modal/dpo.py::merge_dpo_to_fp16
    modal run train/modal/dpo.py::verify_merged_fp16

Both invocations read hyperparameters from ``train/configs/dpo.yaml``.
Override the GPU variant with the ``MODAL_GPU`` env var (local-only;
not forwarded to the remote container)::

    MODAL_GPU=A100-80GB modal run train/modal/dpo.py::train_dpo_full

REFERENCE MODEL NOTE (also for the eventual model card methodology):
TRL's PEFT integration uses ``model.disable_adapter()`` for the reference
forward pass. The implicit reference is therefore the **base** Gemma 4
E4B (no SFT adapter), not the SFT-frozen state. This is the canonical
Unsloth + TRL PEFT-DPO recipe; loading a separate ref_model would
double VRAM (~32 GB extra at fp16) and is explicitly avoided here.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import json
import logging
import os
import shutil
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import modal


# TRL 0.24 has broken optional-dep gates: trl/import_utils.py stores the
# tuple returned by transformers.utils.import_utils._is_package_available
# (which in transformers v5+ always returns (available, version), even
# without return_version=True) into _<pkg>_available. Tuples like
# (False, None) are truthy, so is_<pkg>_available() unconditionally fires
# `import <pkg>` at module-load time — failing for any optional dep that
# isn't installed.
#
# DPOTrainer's import chain hits two such gates we don't use:
#   - llm_blender (trl/trainer/judges.py:29 — for BasePairwiseJudge)
#   - weave        (trl/trainer/callbacks.py:58 — for W&B Weave tracing)
# Stubs satisfy the module-load-time `import` only. Class-body attribute
# references inside the gated blocks are lazy and never resolved at
# runtime unless someone instantiates LLMBlenderJudge / WeaveCallback,
# which the DPO training path never does (verified: zero references in
# trl/trainer/dpo_trainer.py).
def _stub_module(name: str, attrs: dict[str, Any] | None = None) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    for k, v in (attrs or {}).items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


_stub_module("llm_blender")
_stub_module("weave", {"EvaluationLogger": object})
_stub_module("weave.trace")
_stub_module("weave.trace.context", {"weave_client_context": object})

logger = logging.getLogger(__name__)

# --- Paths inside the Modal container ------------------------------------
CONFIG_REMOTE = "/root/train/configs/dpo.yaml"
PROBE_DIR = "/root/probe"
DPO_DATA = "/vol/data/dpo.jsonl"
EVAL_DATA = "/vol/data/eval_set_unlabeled.jsonl"
DPO_META_REMOTE = "/root/dpo.meta.json"
DPO_META_LOCAL = Path("data/formatted/dpo.meta.json")  # for image build only

# --- Modal app -----------------------------------------------------------
MODAL_GPU = os.environ.get("MODAL_GPU", "A100")
VOLUME_NAME = "judge-from-scratch"

app = modal.App("judge-dpo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
    .add_local_file("uv.lock", "/root/uv.lock", copy=True)
    .uv_sync()
    .add_local_file("train/configs/dpo.yaml", CONFIG_REMOTE, copy=True)
    .add_local_file(
        "data/_format_helpers.py", f"{PROBE_DIR}/_format_helpers.py", copy=True
    )
    .add_local_file(
        "data/judge_system_prompt.md",
        f"{PROBE_DIR}/judge_system_prompt.md",
        copy=True,
    )
    .add_local_file(str(DPO_META_LOCAL), DPO_META_REMOTE, copy=True)
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
secrets = [
    modal.Secret.from_name("wandb"),
    modal.Secret.from_name("huggingface"),
]


# --- Pure helpers (importable for unit tests, no GPU/Modal needed) -------


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and return the DPO YAML config as a dict."""
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} did not parse to a dict.")
    required = (
        "model_id",
        "sft_adapter_dir",
        "max_seq_length",
        "training",
        "output",
        "probe",
        "min_train_rows",
    )
    for key in required:
        if key not in cfg:
            raise ValueError(f"Config missing required key: {key}")
    return cfg


def assert_no_thinking(row: dict[str, Any]) -> None:
    """Hard-fail if ``<|think|>`` appears in any string-valued field.

    Decision #13: native thinking mode disabled. Stage 5 has a hard
    block at format time; this guard catches any drift between Stage 5
    and Stage 7 (e.g., a stale upload bypassing Stage 5's verify).
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

    Same selection mechanism as Stage 6 SFT (``train/modal/sft.py:127``)
    so the side-by-side probe compares the same five rows the user
    examined after Stage 6.
    """
    if len(rows) < n:
        raise ValueError(f"Need at least {n} rows; got {len(rows)}.")
    return [
        r["pair_id"]
        for r in sorted(
            rows, key=lambda r: hashlib.sha1(r["pair_id"].encode()).hexdigest()
        )[:n]
    ]


def compute_overcollapse_verdict(
    log_history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Diagnose DPO collapse from trainer.state.log_history.

    Computes deltas across the back half of logged optimisation steps
    (skipping warmup volatility) and returns a verdict in
    ``{OK, WARN, ABORT}`` plus the raw deltas for the dryrun report.
    Thresholds are conservative starting points — tune after the first
    real run.
    """
    metric_steps = [
        e for e in log_history if "rewards/margins" in e and "logps/chosen" in e
    ]
    if len(metric_steps) < 2:
        return {
            "verdict": "INSUFFICIENT_DATA",
            "reason": (
                f"only {len(metric_steps)} metric points logged; need >=2 "
                "for delta computation"
            ),
            "n_points": len(metric_steps),
        }

    # Compare the back half against an earlier reference point. For
    # short histories (e.g., the dryrun's ~3 points), fall back to the
    # first point — len // 2 alone can collapse mid == last for len=2,
    # zeroing every delta.
    mid_idx = len(metric_steps) // 2
    if mid_idx >= len(metric_steps) - 1:
        mid_idx = max(0, len(metric_steps) - 2)
    mid = metric_steps[mid_idx]
    last = metric_steps[-1]

    margin_delta = last["rewards/margins"] - mid["rewards/margins"]
    chosen_logp_delta = last["logps/chosen"] - mid["logps/chosen"]
    rejected_logp_delta = last["logps/rejected"] - mid["logps/rejected"]
    final_accuracy = last.get("rewards/accuracies", float("nan"))

    if final_accuracy < 0.50 or margin_delta <= 0.10:
        verdict = "ABORT"
        reason = (
            f"margin_delta={margin_delta:.3f} (need >0.10) "
            f"and/or final_accuracy={final_accuracy:.3f} (need >=0.50)"
        )
    elif (
        chosen_logp_delta < -3.0 and rejected_logp_delta < -3.0 and margin_delta < 0.50
    ):
        verdict = "ABORT"
        reason = (
            f"both logps collapsing (chosen_d={chosen_logp_delta:.3f}, "
            f"rejected_d={rejected_logp_delta:.3f}) "
            f"with margin_delta={margin_delta:.3f} < 0.50 — degenerate"
        )
    elif margin_delta > 0.50 and final_accuracy >= 0.55:
        verdict = "OK"
        reason = (
            f"margin_delta={margin_delta:.3f}, "
            f"final_accuracy={final_accuracy:.3f} — healthy separation"
        )
    elif margin_delta > 0.10 and chosen_logp_delta < -2.0:
        verdict = "WARN"
        reason = (
            f"margin growing (delta={margin_delta:.3f}) but chosen logp "
            f"falling fast ({chosen_logp_delta:.3f}) — proceed with care"
        )
    else:
        verdict = "WARN"
        reason = (
            f"margin_delta={margin_delta:.3f}, "
            f"chosen_d={chosen_logp_delta:.3f}, "
            f"final_accuracy={final_accuracy:.3f} — borderline"
        )

    return {
        "verdict": verdict,
        "reason": reason,
        "n_points": len(metric_steps),
        "margin_delta": margin_delta,
        "chosen_logp_delta": chosen_logp_delta,
        "rejected_logp_delta": rejected_logp_delta,
        "final_accuracy": final_accuracy,
    }


# --- Modal-side training routine -----------------------------------------


def _build_dpo_dataset(
    tokenizer: Any, mode: Literal["dryrun", "full"], min_train_rows: int
) -> Any:
    """Load /vol/data/dpo.jsonl, run startup checks, return prompt/chosen/rejected."""
    from datasets import load_dataset

    meta = json.loads(Path(DPO_META_REMOTE).read_text())
    expected_rows = meta.get("post_position_swap_count")
    logger.info(
        "DPO meta: stage=%s expected_rows=%s tokenizer=%s",
        meta.get("stage"),
        expected_rows,
        meta.get("tokenizer_model_id"),
    )

    ds = load_dataset("json", data_files=DPO_DATA, split="train")
    actual_rows = len(ds)
    logger.info("DPO dataset loaded: %d rows from %s", actual_rows, DPO_DATA)

    if actual_rows < min_train_rows:
        raise AssertionError(
            f"DPO dataset too small: {actual_rows} rows < {min_train_rows} "
            "(Stage 5 safeguard threshold). Re-run Stage 5 format-dpo "
            "with more synthesis rows or a wider verdict-flip share."
        )

    first = ds[0]
    assert_no_thinking(first)

    eos = tokenizer.eos_token
    if eos != "<turn|>":
        raise AssertionError(
            f"Tokenizer eos_token is {eos!r}, expected '<turn|>'. "
            "Stage 5 chat-template parity broken — re-run smoke test."
        )

    if not first["prompt"].endswith("<|turn>model\n"):
        raise AssertionError(
            f"Stage 5 prompt suffix mismatch (expected '<|turn>model\\n'): "
            f"{first['prompt'][-60:]!r}"
        )

    drop_cols = [c for c in ("pair_id", "swap", "source") if c in ds.column_names]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)

    if mode == "dryrun":
        ds = ds.select(range(min(50, len(ds))))
        logger.info("Dryrun mode: training dataset sliced to %d rows.", len(ds))

    return ds


def _load_sft_adapter(cfg: dict[str, Any]) -> tuple[Any, Any]:
    """Load the SFT-trained adapter on top of the 4-bit base, trainable.

    Calls ``FastLanguageModel.from_pretrained`` on the SFT adapter dir;
    Unsloth detects ``adapter_config.json`` and loads it as a trainable
    PEFT adapter (``unsloth/models/loader.py:794-801``). Does NOT call
    ``get_peft_model`` — that would create a *second* fresh adapter.
    """
    from peft import PeftModel
    from unsloth import FastLanguageModel

    sft_dir = cfg["sft_adapter_dir"]
    if not Path(sft_dir, "adapter_config.json").exists():
        raise FileNotFoundError(
            f"No adapter_config.json at {sft_dir}. Run Stage 6 first: "
            "`modal run train/modal/sft.py::train_sft_full`"
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=sft_dir,
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
    )

    if not isinstance(model, PeftModel):
        raise AssertionError(
            "Adapter not loaded as PeftModel — Unsloth's auto-detection "
            f"failed for {sft_dir}. DPO would no-op against base."
        )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_trainable == 0:
        raise AssertionError(
            "Adapter loaded but no parameters are trainable — DPO "
            "would no-op. Reload with is_trainable=True."
        )

    # Adapter-name-agnostic: Unsloth occasionally renames the default key.
    peft_cfg = next(iter(model.peft_config.values()))
    adapter_name = next(iter(model.peft_config))
    expected_audit = cfg.get("lora_audit", {})
    if expected_audit.get("r") and peft_cfg.r != expected_audit["r"]:
        raise AssertionError(
            f"Stale adapter — r={peft_cfg.r}, expected {expected_audit['r']}"
        )
    if (
        expected_audit.get("lora_alpha")
        and peft_cfg.lora_alpha != expected_audit["lora_alpha"]
    ):
        raise AssertionError(
            f"Stale adapter — alpha={peft_cfg.lora_alpha}, "
            f"expected {expected_audit['lora_alpha']}"
        )

    logger.info(
        "Loaded SFT adapter from %s — trainable_params=%d, adapter=%s, "
        "r=%d, alpha=%d",
        sft_dir,
        n_trainable,
        adapter_name,
        peft_cfg.r,
        peft_cfg.lora_alpha,
    )
    return model, tokenizer


def _make_volume_commit_callback() -> Any:
    """Return a TrainerCallback class that commits the volume on save.

    Constructed lazily so unit tests do not pull in transformers.
    """
    from transformers import TrainerCallback

    class VolumeCommitCallback(TrainerCallback):
        """Commits the Modal volume at every save event."""

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
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Run greedy inference on the probe rows, return generations.

    Mirrors Stage 6's ``_run_probe`` (``train/modal/sft.py:308``) but
    parameterises ``max_new_tokens`` so DPO can probe at 384 tokens
    (Stage 6 used 256 and truncated pair c78b5769d2397d62).
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
        # Gemma 4 multimodal processor: must pass text= as keyword.
        inputs = tokenizer(text=prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            out[0][inputs.input_ids.shape[1] :], skip_special_tokens=False
        )
        outputs.append({"pair_id": row["pair_id"], "generated": generated})
    return outputs


def _train(mode: Literal["dryrun", "full"]) -> dict[str, Any]:
    """Shared DPO training routine for both dryrun and full modes."""
    import time

    import torch
    from trl import DPOConfig, DPOTrainer
    from unsloth import FastLanguageModel, is_bfloat16_supported

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg = load_config(CONFIG_REMOTE)

    # Tokenizer ↔ Stage 5 parity check, mirrors sft.py:382-388.
    # Stage 5's format-dpo phase used the upstream `unsloth/gemma-4-E4B-it`
    # tokenizer; Stage 6/7 train on the BNB-4bit variant
    # `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit`. The two share the same
    # tokenizer/vocab/chat-template — BNB only quantizes weights — so
    # accept either by stripping the `-unsloth-bnb-4bit` suffix before
    # comparing.
    def _normalize_model_id(mid: str) -> str:
        return mid.removesuffix("-unsloth-bnb-4bit")

    dpo_meta = json.loads(Path(DPO_META_REMOTE).read_text())
    if _normalize_model_id(
        dpo_meta.get("tokenizer_model_id", "")
    ) != _normalize_model_id(cfg["model_id"]):
        raise AssertionError(
            f"Tokenizer mismatch: Stage 5 used "
            f"{dpo_meta.get('tokenizer_model_id')!r}, Stage 7 config has "
            f"{cfg['model_id']!r}. Re-run Stage 5 or update config to match."
        )

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{ts}-dpo-{mode}"

    os.environ["WANDB_PROJECT"] = cfg["wandb"]["project"]
    os.environ["WANDB_NAME"] = run_name

    # Force Unsloth to return real logit tensors instead of its lazy
    # cut_cross_entropy callable — DPOTrainer.compute_loss subscripts
    # outputs.logits.shape, which fails on the lazy callable.
    # Must be set BEFORE FastLanguageModel.from_pretrained reads the env.
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

    torch.cuda.reset_peak_memory_stats()

    model, tokenizer = _load_sft_adapter(cfg)
    train_ds = _build_dpo_dataset(tokenizer, mode, cfg["min_train_rows"])

    t = cfg["training"]
    epochs = 1 if mode == "dryrun" else t["num_train_epochs"]
    # Dryrun has only ~3 optim steps; logging_steps=1 keeps log_history
    # populated for the over-collapse detector.
    logging_steps = 1 if mode == "dryrun" else t["logging_steps"]

    args = DPOConfig(
        output_dir=cfg["output"]["checkpoints_dir"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        num_train_epochs=epochs,
        learning_rate=t["learning_rate"],
        warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler_type"],
        optim=t["optim"],
        weight_decay=t["weight_decay"],
        logging_steps=logging_steps,
        save_strategy=t["save_strategy"],
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        report_to="wandb",
        run_name=run_name,
        seed=3407,
        beta=t["beta"],
        loss_type=t["loss_type"],
        max_length=t["max_length"],
        max_prompt_length=t["max_prompt_length"],
        # Explicit `1` survives Unsloth's source-injected override at
        # unsloth/models/rl.py:1194-1208 (only triggers when None).
        # HF datasets uses the single-process branch when num_proc=1
        # (arrow_dataset.py:3318 spawns a Pool only when >1).
        dataset_num_proc=t["dataset_num_proc"],
        remove_unused_columns=False,
    )

    # TRL 0.24 + transformers 5.5 compat: DPOTrainer.__init__ writes
    # `model.warnings_issued["estimate_tokens"] = True` (dpo_trainer.py:405)
    # to suppress an FA estimate_tokens warning, but
    # Gemma4ForConditionalGeneration doesn't initialize the `warnings_issued`
    # dict on the conditional-generation path. Walk the PEFT wrappers to
    # the underlying transformers model and seed an empty dict so TRL's
    # indexed write succeeds.
    inner = model
    for attr in ("base_model", "model"):
        candidate = getattr(inner, attr, None)
        if candidate is not None and candidate is not inner:
            inner = candidate
    if not hasattr(inner, "warnings_issued"):
        inner.warnings_issued = {}

    # Force text-only DPO path. Gemma 4 E4B is a multimodal
    # (image+text->text) model — its model_type is in
    # MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES, which makes
    # DPOTrainer set is_vision_model=True (dpo_trainer.py:359) and call
    # `process_row` (line 667) which expects an "images" column on each
    # row. Our DPO data is text-only — the prompts are pre-rendered chat
    # templates with no image tokens. Temporarily mask model_type to
    # "gemma" (text-only Gemma family, not in the multimodal mapping) so
    # TRL routes to the text-only `tokenize_row` path. Restored after
    # init since model_type affects nothing else at training time
    # (forward/loss don't read it).
    original_model_type = inner.config.model_type
    inner.config.model_type = "gemma"

    # Pass the inner text-only tokenizer instead of the multimodal
    # Gemma4Processor. Unsloth monkey-patches the processor's __call__
    # to require keyword args (images=, text=, videos=) per
    # unsloth_zoo/tokenizer_utils.py:702. TRL's tokenize_row
    # (dpo_trainer.py:724) calls `tokenizer(features["prompt"], ...)`
    # positionally — Unsloth's patch then routes the prompt string into
    # `images=` and leaves `text=None`, blowing up downstream. The inner
    # `.tokenizer` attribute on the multimodal processor is a plain
    # Gemma4TextTokenizer that's never patched. Same pattern used in
    # sft.py:192 for pre-tokenization.
    text_tokenizer = (
        tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    )

    try:
        trainer = DPOTrainer(
            model=model,
            # ref_model=None: TRL uses model.disable_adapter() for the
            # reference forward — see module docstring for semantics.
            ref_model=None,
            args=args,
            train_dataset=train_ds,
            processing_class=text_tokenizer,
            # peft_config=None is non-negotiable:
            # DPOTrainer._prepare_peft_model (dpo_trainer.py:578-580)
            # calls merge_and_unload if both a PeftModel and a peft_config
            # are passed — that would erase the SFT adapter.
            peft_config=None,
        )
    finally:
        inner.config.model_type = original_model_type

    if trainer.ref_model is not None:
        raise AssertionError("Expected ref_model=None for PEFT disable-adapter path")
    if not trainer.is_peft_model:
        raise AssertionError(
            "DPOTrainer didn't detect PEFT — disable_adapter path won't fire"
        )
    callback_cls = _make_volume_commit_callback()
    trainer.add_callback(callback_cls(volume))

    # Token-length histogram of the dryrun rows (regression guard for
    # max_prompt_length=1024 — current dataset max is 678 tokens).
    if mode == "dryrun":
        text_tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
        prompt_lens = [len(text_tok(r["prompt"]).input_ids) for r in train_ds]
        max_prompt = max(prompt_lens)
        p99 = sorted(prompt_lens)[int(0.99 * len(prompt_lens))]
        logger.info(
            "Dryrun prompt length: max=%d p99=%d (max_prompt_length=%d)",
            max_prompt,
            p99,
            t["max_prompt_length"],
        )
        if max_prompt > t["max_prompt_length"]:
            raise AssertionError(
                f"Prompt length {max_prompt} exceeds max_prompt_length="
                f"{t['max_prompt_length']}. Bump config or shorten prompts."
            )

    # --- 1. Train -------------------------------------------------------
    t0 = time.time()
    train_result = trainer.train()
    wallclock_s = time.time() - t0

    # --- 2-3. Save adapter (clean dir first) ----------------------------
    final_dir = cfg["output"]["final_dir"]
    shutil.rmtree(final_dir, ignore_errors=True)
    Path(final_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    volume.commit()

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    steps_done = train_result.global_step or 1
    steps_per_sec = steps_done / max(wallclock_s, 1e-6)
    logger.info(
        "Training finished. wallclock=%.1fs peak_vram=%.2fGB final_loss=%.4f "
        "steps=%d steps/s=%.3f",
        wallclock_s,
        peak_vram_gb,
        train_result.training_loss,
        steps_done,
        steps_per_sec,
    )

    # --- 4. Over-collapse verdict --------------------------------------
    log_history = list(trainer.state.log_history)
    verdict_info = compute_overcollapse_verdict(log_history)
    summary_dir = cfg["output"]["checkpoints_dir"]
    Path(summary_dir).mkdir(parents=True, exist_ok=True)
    summary_path = Path(summary_dir, f"{mode}_summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "mode": mode,
                "run_name": run_name,
                "wallclock_s": wallclock_s,
                "peak_vram_gb": peak_vram_gb,
                "final_loss": train_result.training_loss,
                "steps_done": steps_done,
                "steps_per_sec": steps_per_sec,
                "verdict_info": verdict_info,
                "log_history": log_history,
            },
            indent=2,
        )
    )
    volume.commit()
    logger.info("Verdict: %s — %s", verdict_info["verdict"], verdict_info["reason"])
    if mode == "dryrun":
        logger.info(
            "Dryrun caveat: ~%d optim steps → ~%d logged points. Verdict is "
            "a smoke check, not a quality gate. Real signal comes from the "
            "full run (~137 steps).",
            steps_done,
            verdict_info.get("n_points", 0),
        )

    # Free the trainer's references before reloading for inference.
    # Merge to fp16 is decoupled — run `merge_dpo_to_fp16` separately
    # against /vol/checkpoints/dpo-final/ once you've eyeballed the
    # full-run metrics and want a deployable artifact for vLLM/GGUF.
    del model, trainer
    torch.cuda.empty_cache()

    # --- 5. Probe (always; SFT vs DPO at max_new_tokens=384) -----------
    eval_rows = _load_probe_rows()
    probe_pair_ids = select_probe_pair_ids(eval_rows, n=cfg["probe"]["count"])
    probe_rows = [r for r in eval_rows if r["pair_id"] in set(probe_pair_ids)]
    probe_rows.sort(key=lambda r: probe_pair_ids.index(r["pair_id"]))
    Path(summary_dir, "probe_pair_ids.json").write_text(
        json.dumps(probe_pair_ids, indent=2)
    )

    system_prompt = Path(f"{PROBE_DIR}/judge_system_prompt.md").read_text()
    max_new = cfg["probe"]["max_new_tokens"]

    # DPO probe — load the freshly saved DPO adapter.
    dpo_model, dpo_tok = FastLanguageModel.from_pretrained(
        model_name=final_dir,
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
    )
    FastLanguageModel.for_inference(dpo_model)
    dpo_outputs = _run_probe(dpo_model, dpo_tok, probe_rows, system_prompt, max_new)
    del dpo_model, dpo_tok
    torch.cuda.empty_cache()

    # SFT probe — re-run at the same max_new_tokens=384 for fair comparison.
    # Stage 6's saved outputs were at 256 tokens; pair c78b5769d2397d62
    # specifically may differ here due to truncation budget, NOT an SFT
    # regression (the SFT adapter is byte-identical).
    sft_model, sft_tok = FastLanguageModel.from_pretrained(
        model_name=cfg["probe"]["sft_adapter_for_compare"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
    )
    FastLanguageModel.for_inference(sft_model)
    sft_outputs = _run_probe(sft_model, sft_tok, probe_rows, system_prompt, max_new)
    del sft_model, sft_tok
    torch.cuda.empty_cache()

    comparison = []
    by_id_dpo = {o["pair_id"]: o["generated"] for o in dpo_outputs}
    by_id_sft = {o["pair_id"]: o["generated"] for o in sft_outputs}
    for row in probe_rows:
        pid = row["pair_id"]
        comparison.append(
            {
                "pair_id": pid,
                "bias_category": row.get("bias_category"),
                "human_verdict": row.get("human_verdict"),
                "sft_generated": by_id_sft.get(pid, ""),
                "dpo_generated": by_id_dpo.get(pid, ""),
            }
        )
    comparison_path = Path(summary_dir, "comparison.json")
    comparison_path.write_text(json.dumps(comparison, indent=2))

    # --- 6. Final volume commit ----------------------------------------
    volume.commit()

    return {
        "mode": mode,
        "run_name": run_name,
        "wallclock_s": wallclock_s,
        "peak_vram_gb": peak_vram_gb,
        "final_loss": train_result.training_loss,
        "steps_done": steps_done,
        "steps_per_sec": steps_per_sec,
        "verdict": verdict_info["verdict"],
        "verdict_reason": verdict_info["reason"],
        "verdict_info": verdict_info,
        "final_dir": final_dir,
        "summary_path": str(summary_path),
        "comparison_path": str(comparison_path),
        "probe_pair_ids": probe_pair_ids,
    }


# --- Modal remote functions (private; invoked via local entrypoints) -----

DRYRUN_TIMEOUT_S = 1800
FULL_TIMEOUT_S = 14400


@app.function(
    image=image,
    gpu=MODAL_GPU,
    timeout=DRYRUN_TIMEOUT_S,
    volumes={"/vol": volume},
    secrets=secrets,
)
def _train_dpo_dryrun_remote() -> dict[str, Any]:
    """50 rows × 1 epoch shake-out run. Gates the full DPO."""
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
def _train_dpo_full_remote() -> dict[str, Any]:
    """Full DPO pool × 1 epoch production run."""
    result = _train("full")
    print(f"\nFull-run result: {json.dumps(result, indent=2)}")
    return result


# --- Local entrypoints (with budget gating) ------------------------------


@app.local_entrypoint()
def train_dpo_dryrun() -> None:
    """Budget-gated wrapper around the DPO dryrun.

    Run from project root::

        modal run train/modal/dpo.py::train_dpo_dryrun

    Set ``BUDGET_OVERRIDE=1`` in env to skip the interactive prompt.
    """
    import time

    from train.modal._cost_ledger import (
        STAGE7_BUDGET_CAP_USD,
        check_budget,
        project_cost,
        record_cost,
        total_spend,
    )

    projected = project_cost(MODAL_GPU, DRYRUN_TIMEOUT_S)
    spent = total_spend()
    check_budget(
        projected,
        spent_usd=spent,
        cap_usd=STAGE7_BUDGET_CAP_USD,
        label=f"dpo-dryrun on {MODAL_GPU}",
    )

    t0 = time.time()
    result: dict[str, Any] | None = None
    notes_status = "ok"
    try:
        result = _train_dpo_dryrun_remote.remote()
    except Exception as exc:  # noqa: BLE001
        notes_status = f"failed:{type(exc).__name__}"
        raise
    finally:
        actual_wallclock_s = time.time() - t0
        if result is not None:
            notes = (
                f"status={notes_status} "
                f"verdict={result.get('verdict')} "
                f"loss={result.get('final_loss'):.4f} "
                f"train_s={result.get('wallclock_s'):.1f} "
                f"final_dir={result.get('final_dir')}"
            )
        else:
            notes = f"status={notes_status}"
        row = record_cost(
            stage="stage7",
            function="train_dpo_dryrun",
            gpu=MODAL_GPU,
            wallclock_s=actual_wallclock_s,
            notes=notes,
        )
        print(
            f"\n[cost] DPO dryrun recorded: ${row['est_cost_usd']:.3f} "
            f"({actual_wallclock_s:.1f}s wallclock, {notes_status}). "
            f"Cumulative spend: ${total_spend():.2f}"
        )


@app.local_entrypoint()
def train_dpo_full() -> None:
    """Budget-gated wrapper around the full DPO run.

    Run from project root::

        modal run train/modal/dpo.py::train_dpo_full

    Set ``BUDGET_OVERRIDE=1`` in env to skip the interactive prompt.
    """
    import time

    from train.modal._cost_ledger import (
        STAGE7_BUDGET_CAP_USD,
        check_budget,
        project_cost,
        record_cost,
        total_spend,
    )

    projected = project_cost(MODAL_GPU, FULL_TIMEOUT_S)
    spent = total_spend()
    check_budget(
        projected,
        spent_usd=spent,
        cap_usd=STAGE7_BUDGET_CAP_USD,
        label=f"dpo-full on {MODAL_GPU}",
    )

    t0 = time.time()
    result: dict[str, Any] | None = None
    notes_status = "ok"
    try:
        result = _train_dpo_full_remote.remote()
    except Exception as exc:  # noqa: BLE001
        notes_status = f"failed:{type(exc).__name__}"
        raise
    finally:
        actual_wallclock_s = time.time() - t0
        if result is not None:
            notes = (
                f"status={notes_status} "
                f"verdict={result.get('verdict')} "
                f"loss={result.get('final_loss'):.4f} "
                f"train_s={result.get('wallclock_s'):.1f} "
                f"final_dir={result.get('final_dir')}"
            )
        else:
            notes = f"status={notes_status}"
        row = record_cost(
            stage="stage7",
            function="train_dpo_full",
            gpu=MODAL_GPU,
            wallclock_s=actual_wallclock_s,
            notes=notes,
        )
        print(
            f"\n[cost] DPO full recorded: ${row['est_cost_usd']:.2f} "
            f"({actual_wallclock_s:.1f}s wallclock, {notes_status}). "
            f"Cumulative spend: ${total_spend():.2f}"
        )


# --- Standalone merge step (decoupled from training) ---------------------
#
# Loads /vol/checkpoints/dpo-final/ at fp16 (NOT 4-bit — merging into a
# quantized base is unsupported), folds the LoRA matrices into the base
# weights via Unsloth's save_pretrained_merged, and writes the result to
# /vol/checkpoints/merged-fp16/. ~2-3 min wallclock on A100, ~$0.10.
#
# Decoupled from training because:
#   1. Merge produces an artifact for downstream consumers (vLLM,
#      llama.cpp GGUF conversion, HF publishing) — irrelevant to
#      training/eval through the adapter itself.
#   2. The over-collapse verdict heuristic isn't statistically
#      meaningful on small step counts (per-batch reward variance
#      dominates), so gating merge on it produces false negatives
#      on healthy runs. Letting the operator decide post-hoc is
#      cleaner than tuning the heuristic.
MERGE_TIMEOUT_S = 900


def _merge_to_fp16(cfg: dict[str, Any]) -> dict[str, Any]:
    """Reload the DPO adapter at fp16, merge, save, and verify.

    Idempotent: rmtree's the merged dir first so a stale shard layout
    can't survive into the new merge.
    """
    import torch
    from unsloth import FastLanguageModel

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    final_dir = cfg["output"]["final_dir"]
    merged_dir = cfg["output"]["merged_fp16_dir"]
    if not Path(final_dir, "adapter_config.json").exists():
        raise FileNotFoundError(
            f"No adapter_config.json at {final_dir}. Run train_dpo_full "
            "first; the merge consumes the saved adapter."
        )

    shutil.rmtree(merged_dir, ignore_errors=True)
    Path(merged_dir).mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    merge_model, merge_tok = FastLanguageModel.from_pretrained(
        model_name=final_dir,
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=False,  # critical — fp16 merge requires fp16 base
        dtype=torch.bfloat16,
    )
    merge_model.save_pretrained_merged(
        merged_dir,
        merge_tok,
        save_method="merged_16bit",
    )
    volume.commit()

    if not Path(merged_dir, "config.json").exists():
        raise AssertionError("Merged checkpoint missing config.json")
    shards = list(Path(merged_dir).glob("*.safetensors")) or list(
        Path(merged_dir).glob("pytorch_model*.bin")
    )
    if not shards:
        raise AssertionError("Merged checkpoint has no model weights")
    total_bytes = sum(s.stat().st_size for s in shards)
    if total_bytes < 5_000_000_000:
        raise AssertionError(
            f"Merged checkpoint suspiciously small: {total_bytes:,} bytes"
        )
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    logger.info(
        "Merged fp16 checkpoint at %s: %d shards, %.2f GB, peak_vram=%.2fGB",
        merged_dir,
        len(shards),
        total_bytes / 1e9,
        peak_vram_gb,
    )
    return {
        "merged_dir": merged_dir,
        "shard_count": len(shards),
        "total_bytes": total_bytes,
        "peak_vram_gb": peak_vram_gb,
    }


@app.function(
    image=image,
    gpu=MODAL_GPU,
    timeout=MERGE_TIMEOUT_S,
    volumes={"/vol": volume},
    secrets=secrets,
)
def _merge_dpo_to_fp16_remote() -> dict[str, Any]:
    """Standalone merge of the DPO adapter into an fp16 checkpoint."""
    cfg = load_config(CONFIG_REMOTE)
    result = _merge_to_fp16(cfg)
    print(f"\nMerge result: {json.dumps(result, indent=2)}")
    return result


@app.local_entrypoint()
def merge_dpo_to_fp16() -> None:
    """Budget-gated wrapper around the standalone DPO merge.

    Run from project root after a successful train_dpo_full::

        modal run train/modal/dpo.py::merge_dpo_to_fp16

    Set ``BUDGET_OVERRIDE=1`` in env to skip the interactive prompt.
    Cost is ~$0.10 (A100, ~2-3 min wallclock for fp16 reload + merge).
    """
    import time

    from train.modal._cost_ledger import (
        STAGE7_BUDGET_CAP_USD,
        check_budget,
        project_cost,
        record_cost,
        total_spend,
    )

    projected = project_cost(MODAL_GPU, MERGE_TIMEOUT_S)
    spent = total_spend()
    check_budget(
        projected,
        spent_usd=spent,
        cap_usd=STAGE7_BUDGET_CAP_USD,
        label=f"dpo-merge on {MODAL_GPU}",
    )

    t0 = time.time()
    result: dict[str, Any] | None = None
    notes_status = "ok"
    try:
        result = _merge_dpo_to_fp16_remote.remote()
    except Exception as exc:  # noqa: BLE001
        notes_status = f"failed:{type(exc).__name__}"
        raise
    finally:
        actual_wallclock_s = time.time() - t0
        if result is not None:
            notes = (
                f"status={notes_status} "
                f"shards={result.get('shard_count')} "
                f"bytes={result.get('total_bytes')}"
            )
        else:
            notes = f"status={notes_status}"
        row = record_cost(
            stage="stage7",
            function="merge_dpo_to_fp16",
            gpu=MODAL_GPU,
            wallclock_s=actual_wallclock_s,
            notes=notes,
        )
        print(
            f"\n[cost] DPO merge recorded: ${row['est_cost_usd']:.3f} "
            f"({actual_wallclock_s:.1f}s wallclock, {notes_status}). "
            f"Cumulative spend: ${total_spend():.2f}"
        )


# --- Post-merge spot check -----------------------------------------------
#
# Loads /vol/checkpoints/merged-fp16/ as a vanilla HuggingFace fp16 model
# (no PEFT, no Unsloth optimizations needed for one-off generation),
# generates a single verdict on a pinned probe pair, and asserts the
# output has the expected <reasoning>…</reasoning><verdict>X</verdict>
# shape with no <|think|> leakage. Cheap sanity check that the merge
# didn't corrupt anything before we ship the artifact to vLLM/GGUF.
SPOT_CHECK_TIMEOUT_S = 600
SPOT_CHECK_PAIR_ID = "8d2242064d47609d"  # religion bias, easy probe


def _spot_check_merged_fp16(
    cfg: dict[str, Any], pair_id: str, max_new_tokens: int
) -> dict[str, Any]:
    """Greedy-generate one verdict against the merged fp16 model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if PROBE_DIR not in sys.path:
        sys.path.insert(0, PROBE_DIR)
    from _format_helpers import apply_chat, build_user_message  # noqa: E402

    merged_dir = cfg["output"]["merged_fp16_dir"]
    if not Path(merged_dir, "config.json").exists():
        raise FileNotFoundError(
            f"No config.json at {merged_dir}. Run merge_dpo_to_fp16 first."
        )

    eval_rows = _load_probe_rows()
    matches = [r for r in eval_rows if r["pair_id"] == pair_id]
    if not matches:
        raise ValueError(f"pair_id {pair_id} not found in {EVAL_DATA}")
    row = matches[0]

    tokenizer = AutoTokenizer.from_pretrained(merged_dir)
    model = AutoModelForCausalLM.from_pretrained(
        merged_dir,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    system_prompt = Path(f"{PROBE_DIR}/judge_system_prompt.md").read_text()
    user_msg = build_user_message(row, swap=False)
    text_tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    prompt = apply_chat(text_tok, system_prompt, user_msg)
    if not prompt.endswith("<|turn>model\n"):
        raise AssertionError(f"Prompt suffix mismatch: {prompt[-60:]!r}")

    inputs = text_tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=text_tok.eos_token_id,
        )
    generated = text_tok.decode(
        out[0][inputs.input_ids.shape[1] :], skip_special_tokens=False
    )

    if "<|think|>" in generated:
        raise AssertionError("Generated text contains <|think|> — thinking mode leak")

    import re

    has_reasoning = bool(re.search(r"<reasoning>.*?</reasoning>", generated, re.DOTALL))
    verdict_match = re.search(r"<verdict>\s*([ABT][^<]*?)\s*</verdict>", generated)
    if not has_reasoning:
        raise AssertionError("Generated text missing <reasoning>...</reasoning> block")
    if not verdict_match:
        raise AssertionError("Generated text missing <verdict>...</verdict> block")

    return {
        "pair_id": pair_id,
        "bias_category": row.get("bias_category"),
        "verdict": verdict_match.group(1).strip(),
        "format_ok": True,
        "no_thinking_leak": True,
        "generated": generated,
    }


@app.function(
    image=image,
    gpu=MODAL_GPU,
    timeout=SPOT_CHECK_TIMEOUT_S,
    volumes={"/vol": volume},
    secrets=secrets,
)
def _verify_merged_fp16_remote(
    pair_id: str = SPOT_CHECK_PAIR_ID, max_new_tokens: int = 384
) -> dict[str, Any]:
    """Greedy-generate one verdict against the merged fp16 model."""
    cfg = load_config(CONFIG_REMOTE)
    result = _spot_check_merged_fp16(cfg, pair_id, max_new_tokens)
    print("\n=== Merged-fp16 spot check ===")
    print(f"pair_id:        {result['pair_id']}")
    print(f"bias_category:  {result['bias_category']}")
    print(f"verdict:        {result['verdict']}")
    print(f"format_ok:      {result['format_ok']}")
    print(f"thinking_leak:  {not result['no_thinking_leak']}")
    print("\n--- generated text ---")
    print(result["generated"])
    print("--- end generated ---\n")
    return result


@app.local_entrypoint()
def verify_merged_fp16(
    pair_id: str = SPOT_CHECK_PAIR_ID, max_new_tokens: int = 384
) -> None:
    """Spot-check the merged fp16 artifact with a single greedy generation.

    Run from project root after a successful merge_dpo_to_fp16::

        modal run train/modal/dpo.py::verify_merged_fp16
        modal run train/modal/dpo.py::verify_merged_fp16 --pair-id <id>
    """
    import time

    from train.modal._cost_ledger import (
        STAGE7_BUDGET_CAP_USD,
        check_budget,
        project_cost,
        record_cost,
        total_spend,
    )

    projected = project_cost(MODAL_GPU, SPOT_CHECK_TIMEOUT_S)
    spent = total_spend()
    check_budget(
        projected,
        spent_usd=spent,
        cap_usd=STAGE7_BUDGET_CAP_USD,
        label=f"dpo-merge-verify on {MODAL_GPU}",
    )

    t0 = time.time()
    result: dict[str, Any] | None = None
    notes_status = "ok"
    try:
        result = _verify_merged_fp16_remote.remote(pair_id, max_new_tokens)
    except Exception as exc:  # noqa: BLE001
        notes_status = f"failed:{type(exc).__name__}"
        raise
    finally:
        actual_wallclock_s = time.time() - t0
        if result is not None:
            notes = (
                f"status={notes_status} "
                f"pair_id={result.get('pair_id')} "
                f"verdict={result.get('verdict')}"
            )
        else:
            notes = f"status={notes_status}"
        row = record_cost(
            stage="stage7",
            function="verify_merged_fp16",
            gpu=MODAL_GPU,
            wallclock_s=actual_wallclock_s,
            notes=notes,
        )
        print(
            f"\n[cost] DPO merge verify recorded: ${row['est_cost_usd']:.3f} "
            f"({actual_wallclock_s:.1f}s wallclock, {notes_status}). "
            f"Cumulative spend: ${total_spend():.2f}"
        )
