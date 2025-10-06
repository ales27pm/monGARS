"""Training helpers for QLoRA fine-tuning."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Type

import torch
from transformers import Trainer, TrainingArguments, default_data_collator

try:  # pragma: no cover - optional dependency under tests
    from peft import LoraConfig, get_peft_model
except Exception:  # pragma: no cover - tests patch this path
    LoraConfig = get_peft_model = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LoraHyperParams:
    """Configuration for LoRA adapters."""

    r: int = 16
    alpha: int = 16
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


def _enable_input_require_grads(model: Any) -> None:
    """Ensure model inputs require gradients for LoRA fine-tuning."""

    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:  # pragma: no cover - best effort logging
            logger.debug("enable_input_require_grads failed", exc_info=True)
        return

    embeddings = getattr(model, "get_input_embeddings", None)
    if not callable(embeddings):
        return

    try:
        module = embeddings()
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("Unable to access input embeddings", exc_info=True)
        return

    def _require_grad_hook(_: Any, __: Any, output: Any) -> None:
        if isinstance(output, torch.Tensor):
            output.requires_grad_(True)
        elif isinstance(output, (tuple, list)):
            for item in output:
                if isinstance(item, torch.Tensor):
                    item.requires_grad_(True)

    try:
        module.register_forward_hook(_require_grad_hook)
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("Unable to register input grad hook", exc_info=True)


def prepare_lora_model_light(model: Any, params: LoraHyperParams | None = None) -> Any:
    """Attach LoRA adapters without upcasting the ``lm_head`` to FP32."""

    if LoraConfig is None or get_peft_model is None:
        raise RuntimeError("PEFT is required to prepare the model for LoRA training")

    params = params or LoraHyperParams()
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()
    except AttributeError:  # pragma: no cover - defensive guard
        logger.debug("Model does not support gradient checkpointing", exc_info=True)

    _enable_input_require_grads(model)

    config = LoraConfig(
        r=params.r,
        lora_alpha=params.alpha,
        lora_dropout=params.dropout,
        bias="none",
        target_modules=list(params.target_modules),
        task_type="CAUSAL_LM",
    )
    logger.info(
        "Attaching LoRA adapters",
        extra={"rank": params.r, "alpha": params.alpha, "dropout": params.dropout},
    )
    return get_peft_model(model, config)


def prepare_lora_model(model: Any, params: LoraHyperParams | None = None) -> Any:
    """Backward-compatible wrapper around :func:`prepare_lora_model_light`."""

    return prepare_lora_model_light(model, params)


@dataclass(slots=True)
class TrainerConfig:
    """Arguments required to configure the Hugging Face Trainer."""

    output_dir: Path
    batch_size: int
    grad_accum: int
    learning_rate: float
    epochs: float
    max_steps: int


def _is_cuda_oom(exc: BaseException) -> bool:
    """Return ``True`` when ``exc`` represents a CUDA out-of-memory error."""

    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    return False


def _maybe_empty_cuda_cache() -> None:
    """Attempt to release cached CUDA memory."""

    empty_cache = getattr(torch.cuda, "empty_cache", None)
    if callable(empty_cache):  # pragma: no branch - attribute lookup guard
        try:
            empty_cache()
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Unable to empty CUDA cache", exc_info=True)


def train_qlora(
    model: Any,
    dataset: Any,
    *,
    config: TrainerConfig,
    extra_args: dict[str, Any] | None = None,
    trainer_cls: Type[Trainer] = Trainer,
) -> Trainer:
    """Train the provided model using Hugging Face Trainer.

    When CUDA reports an out-of-memory error, the function retries the training run with a
    reduced per-device batch size and, if needed, smaller gradient accumulation settings.
    This makes the helper resilient on constrained GPUs where the default configuration is
    too aggressive.
    """

    extra_args = dict(extra_args or {})
    bf16_ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    logging_steps = extra_args.pop("logging_steps", 25)
    save_steps = extra_args.pop("save_steps", 250)
    save_total_limit = extra_args.pop("save_total_limit", 1)
    report_to = extra_args.pop("report_to", [])
    optim = extra_args.pop("optim", "adamw_bnb_8bit")
    torch_empty_cache_steps = extra_args.pop("torch_empty_cache_steps", 50)
    oom_retries = int(extra_args.pop("oom_retries", 2))

    batch_size = max(1, config.batch_size)
    grad_accum = max(1, config.grad_accum)
    attempt = 0

    while True:
        args = TrainingArguments(
            output_dir=str(config.output_dir),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=config.learning_rate,
            num_train_epochs=config.epochs,
            max_steps=config.max_steps if config.max_steps > 0 else -1,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            report_to=report_to,
            bf16=bf16_ok,
            fp16=not bf16_ok,
            gradient_checkpointing=True,
            optim=optim,
            torch_empty_cache_steps=torch_empty_cache_steps,
            **extra_args,
        )
        trainer = trainer_cls(
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=default_data_collator,
        )

        logger.info(
            "Starting QLoRA fine-tuning",
            extra={
                "attempt": attempt + 1,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
            },
        )

        try:
            trainer.train()
        except BaseException as exc:  # pragma: no cover - covered via unit tests
            if not _is_cuda_oom(exc):
                raise

            _maybe_empty_cuda_cache()

            if attempt >= oom_retries or (batch_size == 1 and grad_accum == 1):
                logger.error(
                    "Training failed due to CUDA OOM",
                    extra={
                        "attempt": attempt + 1,
                        "batch_size": batch_size,
                        "grad_accum": grad_accum,
                    },
                )
                raise

            previous_batch_size = batch_size
            previous_grad_accum = grad_accum

            if batch_size > 1:
                batch_size = max(1, batch_size // 2)
            elif grad_accum > 1:
                grad_accum = max(1, grad_accum // 2)

            attempt += 1

            logger.warning(
                "CUDA OOM encountered during training; retrying with reduced settings",
                extra={
                    "attempt": attempt + 1,
                    "prev_batch_size": previous_batch_size,
                    "prev_grad_accum": previous_grad_accum,
                    "batch_size": batch_size,
                    "grad_accum": grad_accum,
                },
            )

            del trainer
            continue

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


def run_embedding_smoke_test(
    encoder: Any, texts: Iterable[str]
) -> tuple[int, int] | None:
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
