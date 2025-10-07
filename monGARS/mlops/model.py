"""Model loading helpers for fine-tuning pipelines."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


def _compute_weight_budget(
    vram_budget_mb: int, activation_buffer_mb: int, runtime_buffer_mb: int
) -> int:
    """Return the VRAM allocation reserved for model weights."""

    if vram_budget_mb <= 0:
        raise ValueError("vram_budget_mb must be positive")

    activation_buffer_mb = max(0, activation_buffer_mb)
    runtime_buffer_mb = max(0, runtime_buffer_mb)
    effective_budget = vram_budget_mb - activation_buffer_mb - runtime_buffer_mb

    if effective_budget < 512:
        logger.warning(
            "Buffer configuration leaves little room for model weights",
            extra={
                "vram_budget_mb": vram_budget_mb,
                "activation_buffer_mb": activation_buffer_mb,
                "runtime_buffer_mb": runtime_buffer_mb,
            },
        )
        effective_budget = max(vram_budget_mb // 2, 512)

    return min(vram_budget_mb, effective_budget)


def load_4bit_causal_lm(
    model_id: str,
    *,
    vram_budget_mb: int = 7100,
    activation_buffer_mb: int = 1024,
    runtime_buffer_mb: int = 768,
    offload_dir: str | Path = "./offload",
    trust_remote_code: bool = True,
    dtype: Optional[torch.dtype] = None,
    compute_dtype: Optional[torch.dtype] = None,
    attention_implementation: str | None = None,
) -> tuple[Any, Any]:
    """Load a causal LM in 4-bit precision while keeping ``lm_head`` on CPU."""

    offload_path = Path(offload_dir)
    offload_path.mkdir(parents=True, exist_ok=True)

    target_dtype = dtype or torch.float16
    compute_dtype = compute_dtype or target_dtype

    bnb_common: dict[str, Any] = {
        "load_in_4bit": True,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": compute_dtype,
    }
    try:
        bnb_cfg = BitsAndBytesConfig(
            **bnb_common, llm_int8_enable_fp32_cpu_offload=True
        )
    except TypeError:
        bnb_cfg = BitsAndBytesConfig(**bnb_common)

    device_map = {
        "model.embed_tokens": 0,
        "model.layers": 0,
        "model.norm": 0,
        "lm_head": "cpu",
    }
    weight_budget_mb = _compute_weight_budget(
        vram_budget_mb, activation_buffer_mb, runtime_buffer_mb
    )
    max_memory = {0: f"{weight_budget_mb}MiB", "cpu": "64GiB"}

    logger.info(
        "Loading base model",
        extra={
            "model": model_id,
            "vram_budget_mb": vram_budget_mb,
            "activation_buffer_mb": activation_buffer_mb,
            "runtime_buffer_mb": runtime_buffer_mb,
            "weight_budget_mb": weight_budget_mb,
            "offload_dir": str(offload_path),
        },
    )

    model_kwargs: dict[str, Any] = {
        "device_map": device_map,
        "max_memory": max_memory,
        "offload_folder": str(offload_path),
        "quantization_config": bnb_cfg,
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, **model_kwargs, torch_dtype=target_dtype
        )
    except TypeError:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, **model_kwargs, dtype=target_dtype
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False
    attn_impl = attention_implementation or "eager"
    try:  # pragma: no cover - depends on HF version
        model.config.attn_implementation = attn_impl
    except Exception:
        pass

    try:  # pragma: no cover - depends on torch build
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:  # pragma: no cover - best effort configuration
        pass

    return model, tokenizer


def summarise_device_map(model: Any) -> dict[str, int] | None:
    """Return a summary of the device map for logging or debugging."""

    mapping = getattr(model, "hf_device_map", None)
    if not mapping:
        logger.info("Model loaded without a device map")
        return None
    counts: dict[str, int] = {}
    for device in mapping.values():
        counts[str(device)] = counts.get(str(device), 0) + 1
    logger.info("Device map summary", extra=counts)
    return counts


def move_to_cpu(model: Any) -> None:
    """Attempt to move ``model`` to CPU for graceful cleanup."""

    mover = getattr(model, "to", None)
    if callable(mover):
        try:  # pragma: no cover - best effort cleanup
            mover("cpu")
        except Exception:
            logger.debug("Unable to move model to CPU", exc_info=True)


def detach_sequences(sequences: Iterable[Any]) -> list[Any]:
    """Detach tensors from the computation graph for downstream processing."""

    detached: list[Any] = []
    for tensor in sequences:
        current = tensor
        for attr in ("detach", "cpu"):
            method = getattr(current, attr, None)
            if callable(method):
                try:
                    current = method()
                except Exception:  # pragma: no cover - defensive guard
                    break
        detached.append(current)
    return detached
