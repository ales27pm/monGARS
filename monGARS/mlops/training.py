"""Training helpers for QLoRA fine-tuning."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Type

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


@dataclass(slots=True)
class OOMRetryEvent:
    """Structured payload describing a CUDA OOM retry decision."""

    exception: BaseException
    attempt: int
    remaining_retries: int
    batch_size: int
    grad_accum: int
    next_batch_size: int
    next_grad_accum: int
    will_retry: bool


OOMEventHook = Callable[[OOMRetryEvent], None]


def _is_cuda_oom(exc: BaseException) -> bool:
    """Return ``True`` when ``exc`` represents a CUDA out-of-memory error."""

    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _maybe_empty_cuda_cache() -> None:
    """Attempt to release cached CUDA memory."""

    empty_cache = getattr(torch.cuda, "empty_cache", None)
    if callable(empty_cache):  # pragma: no branch - attribute lookup guard
        try:
            empty_cache()
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Unable to empty CUDA cache", exc_info=True)


def _reset_cuda_peak_memory_stats() -> None:
    """Reset CUDA peak memory statistics when the API is available."""

    reset_stats = getattr(torch.cuda, "reset_peak_memory_stats", None)
    if callable(reset_stats):  # pragma: no branch - attribute lookup guard
        try:
            reset_stats()
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Unable to reset CUDA peak memory stats", exc_info=True)


def _zero_trainer_optimizer(trainer: Any) -> None:
    """Clear gradients held by the trainer optimizer, if present."""

    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is None:  # pragma: no cover - accessor guard
        return

    try:
        optimizer.zero_grad(set_to_none=True)
    except TypeError:
        optimizer.zero_grad()
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("Unable to zero trainer optimizer", exc_info=True)


def _dispatch_oom_hooks(hooks: Iterable[OOMEventHook], event: OOMRetryEvent) -> None:
    """Call registered OOM hooks, swallowing their exceptions."""

    for hook in hooks:
        try:
            hook(event)
        except Exception:  # pragma: no cover - hooks are best effort
            logger.debug("OOM event hook raised", exc_info=True)


def _apply_backoff(value: int, factor: float) -> int:
    """Reduce ``value`` using ``factor`` while ensuring progress."""

    if value <= 1:
        return 1

    reduced = max(1, int(value * factor))
    if reduced == value:
        reduced = max(1, value - 1)
    return reduced


def _build_training_arguments(
    cfg: TrainerConfig,
    base_args: dict[str, Any],
    batch_size: int,
    grad_accum: int,
    bf16_ok: bool,
) -> TrainingArguments:
    """Create ``TrainingArguments`` with shared defaults."""

    return TrainingArguments(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps if cfg.max_steps > 0 else -1,
        bf16=bf16_ok,
        fp16=not bf16_ok,
        gradient_checkpointing=True,
        **base_args,
    )


def _coerce_oom_hooks(raw_hooks: Any) -> tuple[OOMEventHook, ...]:
    """Normalise hook configuration into an immutable tuple."""

    if raw_hooks is None:
        return ()
    if callable(raw_hooks):
        return (raw_hooks,)
    if isinstance(raw_hooks, Iterable) and not isinstance(raw_hooks, (str, bytes)):
        hooks: list[OOMEventHook] = []
        for hook in raw_hooks:
            if hook is None:
                continue
            if not callable(hook):
                raise TypeError("OOM event hooks must be callables")
            hooks.append(hook)
        return tuple(hooks)
    raise TypeError("OOM event hooks must be a callable or iterable of callables")


def _sanitize_backoff_factor(raw_factor: Any) -> float:
    """Validate and return a usable OOM backoff factor."""

    try:
        factor = float(raw_factor)
    except (TypeError, ValueError):  # pragma: no cover - defensive guard
        logger.warning("Invalid OOM backoff factor %r; falling back to 0.5", raw_factor)
        return 0.5

    if not 0 < factor < 1:
        logger.warning("OOM backoff factor %.3f is outside (0, 1); defaulting to 0.5", factor)
        return 0.5
    return factor


def _handle_cuda_oom(
    *,
    trainer: Any,
    exc: BaseException,
    attempt: int,
    max_retries: int,
    batch_size: int,
    grad_accum: int,
    backoff_factor: float,
    hooks: Iterable[OOMEventHook],
) -> tuple[bool, int, int]:
    """Process a CUDA OOM exception and decide whether to retry."""

    _maybe_empty_cuda_cache()
    _reset_cuda_peak_memory_stats()
    _zero_trainer_optimizer(trainer)

    next_batch = batch_size
    next_grad = grad_accum

    if batch_size > 1:
        next_batch = _apply_backoff(batch_size, backoff_factor)
    if next_batch == batch_size and grad_accum > 1:
        next_grad = _apply_backoff(grad_accum, backoff_factor)

    maxed_attempts = attempt >= max_retries
    can_reduce = (next_batch < batch_size) or (next_grad < grad_accum)
    exhausted = batch_size == 1 and grad_accum == 1
    will_retry = not (maxed_attempts or exhausted) and can_reduce

    log_payload = {
        "attempt": attempt + 1,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "next_batch_size": next_batch,
        "next_grad_accum": next_grad,
        "remaining_retries": max(0, max_retries - attempt),
    }

    if will_retry:
        logger.warning(
            "CUDA OOM encountered during training; retrying with reduced settings",
            extra=log_payload,
        )
    else:
        logger.error("Training failed due to CUDA OOM", extra=log_payload)

    event = OOMRetryEvent(
        exception=exc,
        attempt=attempt + 1,
        remaining_retries=max(0, max_retries - attempt),
        batch_size=batch_size,
        grad_accum=grad_accum,
        next_batch_size=next_batch,
        next_grad_accum=next_grad,
        will_retry=will_retry,
    )
    _dispatch_oom_hooks(hooks, event)

    if not will_retry:
        return False, batch_size, grad_accum

    return True, next_batch, next_grad


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

    oom_retries = int(extra_args.pop("oom_retries", 2))
    backoff_factor = _sanitize_backoff_factor(extra_args.pop("oom_backoff_factor", 0.5))
    oom_hooks = _coerce_oom_hooks(extra_args.pop("oom_event_hooks", ()))

    base_args = {
        "logging_steps": extra_args.pop("logging_steps", 25),
        "save_steps": extra_args.pop("save_steps", 250),
        "save_total_limit": extra_args.pop("save_total_limit", 1),
        "report_to": extra_args.pop("report_to", []),
        "optim": extra_args.pop("optim", "adamw_bnb_8bit"),
        "torch_empty_cache_steps": extra_args.pop("torch_empty_cache_steps", 50),
    }
    base_args.update(extra_args)

    batch_size = max(1, config.batch_size)
    grad_accum = max(1, config.grad_accum)
    attempt = 0

    while True:
        args = _build_training_arguments(config, base_args, batch_size, grad_accum, bf16_ok)
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

            should_retry, next_batch, next_grad = _handle_cuda_oom(
                trainer=trainer,
                exc=exc,
                attempt=attempt,
                max_retries=oom_retries,
                batch_size=batch_size,
                grad_accum=grad_accum,
                backoff_factor=backoff_factor,
                hooks=oom_hooks,
            )

            if not should_retry:
                raise

            batch_size = next_batch
            grad_accum = next_grad
            attempt += 1
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
