"""Opinionated pipeline for Unsloth-backed QLoRA fine-tuning."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping

try:  # pragma: no cover - optional dependency at runtime
    from llm2vec import LLM2Vec
except Exception:  # pragma: no cover - library not installed during tests
    LLM2Vec = None  # type: ignore[assignment]

from monGARS.mlops.artifacts import WrapperConfig, write_wrapper_bundle
from monGARS.mlops.dataset import (
    prepare_instruction_dataset,
    prepare_local_instruction_dataset,
)
from monGARS.mlops.exporters import merge_lora_adapters
from monGARS.mlops.model import load_4bit_causal_lm, summarise_device_map
from monGARS.mlops.training import (
    LoraHyperParams,
    TrainerConfig,
    disable_training_mode,
    prepare_lora_model_light,
    run_embedding_smoke_test,
    save_lora_artifacts,
    train_qlora,
)
from monGARS.mlops.utils import (
    configure_cuda_allocator,
    describe_environment,
    ensure_dependencies,
    ensure_directory,
)

logger = logging.getLogger(__name__)

REQUIRED_PACKAGES = (
    "torch",
    "transformers>=4.44",
    "datasets",
    "peft>=0.11",
    "bitsandbytes>=0.44.1",
)
OPTIONAL_PACKAGES = ("unsloth", "llm2vec")


def _activate_unsloth(model: object) -> tuple[object, bool]:
    """Attempt to apply Unsloth fast-path hooks to ``model``."""

    try:
        from unsloth import FastLanguageModel  # type: ignore
    except Exception:  # pragma: no cover - dependency optional in CI
        logger.info("Unsloth not available; continuing without fast-path hooks")
        return model, False

    candidates = []
    for attr in ("for_training", "prepare_model_for_training"):
        method = getattr(FastLanguageModel, attr, None)
        if callable(method):
            candidates.append(method)
    for candidate in candidates:
        try:
            new_model = candidate(model)
        except TypeError:
            try:
                new_model = candidate(model=model)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("Unsloth activation failed", exc_info=True)
                return model, False
        except Exception:  # pragma: no cover - defensive guard
            logger.warning("Unsloth activation failed", exc_info=True)
            return model, False
        else:
            logger.info("Applied Unsloth fast-path hooks via %s", candidate.__name__)
            return new_model or model, True
    logger.info("FastLanguageModel hooks unavailable on this Unsloth build")
    return model, False


def _load_dataset(
    *,
    dataset_id: str | None,
    dataset_path: Path | None,
    tokenizer: object,
    max_seq_len: int,
    train_fraction: float | None = None,
):
    if dataset_path is not None:
        return prepare_local_instruction_dataset(dataset_path, tokenizer, max_seq_len)
    if not dataset_id:
        raise ValueError("Either dataset_id or dataset_path must be provided")
    options: dict[str, float] = {}
    if train_fraction is not None:
        options["train_fraction"] = float(train_fraction)
    return prepare_instruction_dataset(dataset_id, tokenizer, max_seq_len, **options)


def _normalise_metrics(metrics: Mapping[str, object] | None) -> dict[str, object]:
    if not metrics:
        return {}

    normalised: dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, (str, bool)) or value is None:
            normalised[key] = value
            continue
        if isinstance(value, (int, float)):
            normalised[key] = value
            continue
        item = getattr(value, "item", None)
        if callable(item):
            try:
                normalised[key] = item()  # type: ignore[assignment]
                continue
            except Exception:  # pragma: no cover - defensive guard
                logger.debug("Unable to coerce metric via item()", exc_info=True)
        try:
            normalised[key] = float(value)  # type: ignore[arg-type]
        except Exception:  # pragma: no cover - fallback to string
            normalised[key] = str(value)
    return normalised


def run_unsloth_finetune(
    *,
    model_id: str,
    output_dir: Path,
    dataset_id: str | None = None,
    dataset_path: Path | None = None,
    max_seq_len: int = 1024,
    vram_budget_mb: int = 8192,
    activation_buffer_mb: int = 1024,
    batch_size: int = 1,
    grad_accum: int = 8,
    learning_rate: float = 2e-4,
    epochs: float = 1.0,
    max_steps: int = -1,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    train_fraction: float | None = None,
    eval_dataset_id: str | None = None,
    eval_dataset_path: Path | None = None,
    eval_batch_size: int | None = None,
    run_smoke_tests: bool = True,
    write_metadata: bool = True,
    merge_to_fp16: bool = True,
) -> Mapping[str, Path]:
    """Execute a deterministic Unsloth-oriented fine-tuning pipeline."""

    configure_cuda_allocator()
    ensure_directory(output_dir)
    offload_dir = output_dir / "offload"
    ensure_directory(offload_dir)
    ensure_dependencies(REQUIRED_PACKAGES, OPTIONAL_PACKAGES)
    describe_environment()

    model, tokenizer = load_4bit_causal_lm(
        model_id,
        vram_budget_mb=vram_budget_mb,
        activation_buffer_mb=activation_buffer_mb,
        offload_dir=offload_dir,
    )
    summarise_device_map(model)

    model, unsloth_active = _activate_unsloth(model)
    model = prepare_lora_model_light(
        model,
        LoraHyperParams(r=lora_rank, alpha=lora_alpha, dropout=lora_dropout),
    )

    dataset = _load_dataset(
        dataset_id=dataset_id,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        train_fraction=train_fraction,
    )
    dataset_size = len(dataset) if hasattr(dataset, "__len__") else None

    trainer = train_qlora(
        model,
        dataset,
        config=TrainerConfig(
            output_dir=output_dir,
            batch_size=batch_size,
            grad_accum=grad_accum,
            learning_rate=learning_rate,
            epochs=epochs,
            max_steps=max_steps,
        ),
        extra_args={
            "per_device_eval_batch_size": eval_batch_size or batch_size,
        },
    )

    chat_lora_dir = output_dir / "chat_lora"
    save_lora_artifacts(trainer.model, tokenizer, chat_lora_dir)
    disable_training_mode(trainer.model)

    merged_dir = output_dir / "merged_fp16"
    merged = False
    if merge_to_fp16:
        try:
            merged = merge_lora_adapters(model_id, output_dir, output_dir=merged_dir)
        except Exception:  # pragma: no cover - defensive guard
            logger.warning("Failed to merge adapters to FP16", exc_info=True)

    evaluation_metrics: dict[str, object] | None = None
    eval_dataset_size: int | None = None
    if eval_dataset_id or eval_dataset_path:
        try:
            eval_dataset = _load_dataset(
                dataset_id=eval_dataset_id,
                dataset_path=eval_dataset_path,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
            )
            eval_dataset_size = (
                len(eval_dataset) if hasattr(eval_dataset, "__len__") else None
            )
            evaluation_metrics = _normalise_metrics(trainer.evaluate(eval_dataset))
            logger.info(
                "Evaluation metrics computed",
                extra={"metrics": evaluation_metrics, "eval_size": eval_dataset_size},
            )
        except Exception:  # pragma: no cover - evaluation best-effort
            logger.warning("Evaluation failed", exc_info=True)

    if run_smoke_tests and LLM2Vec is not None:
        encoder = LLM2Vec(trainer.model, tokenizer, pooling_mode="mean")
        shape = run_embedding_smoke_test(encoder, ["monGARS embedding smoke test"])
        logger.info("Embedding smoke test result", extra={"shape": shape})
    elif run_smoke_tests:
        logger.info("LLM2Vec not installed; skipping embedding smoke test")

    wrapper_config = WrapperConfig(
        base_model_id=model_id,
        lora_dir=chat_lora_dir,
        max_seq_len=max_seq_len,
        vram_budget_mb=vram_budget_mb,
        offload_dir=offload_dir,
        activation_buffer_mb=activation_buffer_mb,
        quantized_4bit=True,
    )
    wrapper_paths = write_wrapper_bundle(wrapper_config, output_dir)

    if write_metadata:
        metadata = {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "dataset_path": str(dataset_path) if dataset_path else None,
            "unsloth_active": unsloth_active,
            "max_seq_len": max_seq_len,
            "vram_budget_mb": vram_budget_mb,
            "train_fraction": train_fraction,
            "dataset_size": dataset_size,
            "eval_dataset_id": eval_dataset_id,
            "eval_dataset_path": str(eval_dataset_path) if eval_dataset_path else None,
            "eval_dataset_size": eval_dataset_size,
            "evaluation_metrics": evaluation_metrics,
        }
        (output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    return {
        "output_dir": output_dir,
        "chat_lora_dir": chat_lora_dir,
        "wrapper_module": wrapper_paths["module"],
        "wrapper_config": wrapper_paths["config"],
        "wrapper_dir": wrapper_paths["module"].parent,
        "merged_dir": merged_dir if merged else None,
        "dataset_size": dataset_size,
        "eval_dataset_size": eval_dataset_size,
        "evaluation_metrics": evaluation_metrics,
    }


__all__ = ["run_unsloth_finetune"]
