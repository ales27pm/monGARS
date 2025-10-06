"""Training helpers for QLoRA fine-tuning."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import Trainer, TrainingArguments, default_data_collator

try:  # pragma: no cover - optional dependency under tests
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except Exception:  # pragma: no cover - tests patch this path
    LoraConfig = get_peft_model = prepare_model_for_kbit_training = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LoraHyperParams:
    """Configuration for LoRA adapters."""

    r: int = 32
    alpha: int = 32
    dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


def prepare_lora_model(model: Any, params: LoraHyperParams | None = None) -> Any:
    """Enable gradient checkpointing and attach LoRA adapters."""

    if prepare_model_for_kbit_training is None or LoraConfig is None:
        raise RuntimeError("PEFT is required to prepare the model for LoRA training")

    params = params or LoraHyperParams()
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    config = LoraConfig(
        r=params.r,
        lora_alpha=params.alpha,
        lora_dropout=params.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(params.target_modules),
    )
    logger.info(
        "Attaching LoRA adapters",
        extra={"rank": params.r, "alpha": params.alpha, "dropout": params.dropout},
    )
    return get_peft_model(model, config)


@dataclass(slots=True)
class TrainerConfig:
    """Arguments required to configure the Hugging Face Trainer."""

    output_dir: Path
    batch_size: int
    grad_accum: int
    learning_rate: float
    epochs: float
    max_steps: int


def train_qlora(
    model: Any,
    dataset: Any,
    *,
    config: TrainerConfig,
    extra_args: dict[str, Any] | None = None,
) -> Trainer:
    """Train the provided model using Hugging Face Trainer."""

    extra_args = extra_args or {}
    bf16_ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    args = TrainingArguments(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        max_steps=config.max_steps if config.max_steps > 0 else -1,
        logging_steps=extra_args.pop("logging_steps", 25),
        save_steps=extra_args.pop("save_steps", 250),
        save_total_limit=extra_args.pop("save_total_limit", 1),
        report_to=extra_args.pop("report_to", []),
        bf16=bf16_ok,
        fp16=not bf16_ok,
        gradient_checkpointing=True,
        optim=extra_args.pop("optim", "adamw_bnb_8bit"),
        torch_empty_cache_steps=extra_args.pop("torch_empty_cache_steps", 50),
        **extra_args,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )
    logger.info("Starting QLoRA fine-tuning")
    trainer.train()
    logger.info("Training completed")
    return trainer


def save_lora_artifacts(model: Any, tokenizer: Any, output_dir: Path) -> None:
    """Persist adapters and tokenizer to ``output_dir``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Saved adapters", extra={"output_dir": str(output_dir)})


def disable_training_mode(model: Any) -> None:
    """Put ``model`` into evaluation mode after training."""

    method = getattr(model, "eval", None)
    if callable(method):
        method()


def run_embedding_smoke_test(encoder: Any, texts: Iterable[str]) -> tuple[int, int] | None:
    """Execute a small embedding test returning the resulting tensor shape."""

    if not texts:
        return None
    try:
        result = encoder.encode(list(texts))
    except Exception:  # pragma: no cover - defensive guard
        logger.warning("Embedding smoke test failed", exc_info=True)
        return None
    shape = getattr(result, "shape", None)
    if shape is None:
        return None
    if isinstance(shape, (tuple, list)) and len(shape) == 2:
        return int(shape[0]), int(shape[1])
    return None
