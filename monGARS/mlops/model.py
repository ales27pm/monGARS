"""Model loading helpers for fine-tuning pipelines."""

from __future__ import annotations

import inspect
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from packaging.version import InvalidVersion, Version
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
    """Load a causal LM with 4-bit quantization when possible.

    When CUDA or a compatible bitsandbytes build is unavailable, the loader
    gracefully falls back to a CPU execution path so CI can still exercise the
    fine-tuning pipeline.
    """

    offload_path = Path(offload_dir)
    offload_path.mkdir(parents=True, exist_ok=True)

    target_dtype = dtype or torch.float16
    compute_dtype = compute_dtype or target_dtype

    if not _is_quantization_available():
        return _load_cpu_causal_lm(
            model_id,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            attention_implementation=attention_implementation,
        )

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
        "Loading base model with 4-bit quantization",
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
        model = _load_with_optional_kwargs(
            AutoModelForCausalLM.from_pretrained,
            model_id,
            base_kwargs=model_kwargs,
            optional_kwargs=(
                {"torch_dtype": target_dtype},
                {"dtype": target_dtype},
            ),
        )
    except Exception:
        logger.error(
            "Failed to load model with 4-bit quantization",
            extra={"model": model_id},
            exc_info=True,
        )
        raise

    tokenizer = _initialise_tokenizer(model_id, trust_remote_code=trust_remote_code)
    _configure_model_attention(
        model, attention_implementation=attention_implementation
    )

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


_MIN_BITSANDBYTES_VERSION_ENV = "MONGARS_MIN_BITSANDBYTES_VERSION"
_DEFAULT_MIN_BITSANDBYTES_VERSION = Version("0.44.1")


@lru_cache(maxsize=1)
def _resolve_bitsandbytes_version() -> Version | None:
    """Return the installed bitsandbytes version if available."""

    try:
        import bitsandbytes as bnb  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "bitsandbytes not installed; falling back to CPU execution",
            extra={"package": "bitsandbytes"},
        )
        return None

    raw_version = getattr(bnb, "__version__", "")
    try:
        return Version(raw_version)
    except InvalidVersion:
        logger.warning(
            "Unable to parse bitsandbytes version; disabling 4-bit quantization",
            extra={"package": "bitsandbytes", "version": raw_version},
        )
        return None


def _is_quantization_available() -> bool:
    """Return ``True`` when 4-bit quantization requirements are satisfied."""

    if not torch.cuda.is_available():
        logger.warning(
            "CUDA unavailable; using CPU execution without 4-bit quantization",
            extra={"cuda_available": False},
        )
        return False

    version_info = _resolve_bitsandbytes_version()
    if version_info is None:
        return False

    min_version = _get_min_bitsandbytes_version()
    if version_info < min_version:
        logger.warning(
            "bitsandbytes upgrade required for 4-bit quantization",
            extra={
                "detected_version": str(version_info),
                "required_version": str(min_version),
            },
        )
        return False

    return True


def _load_cpu_causal_lm(
    model_id: str,
    *,
    trust_remote_code: bool,
    dtype: Optional[torch.dtype],
    attention_implementation: str | None,
) -> tuple[Any, Any]:
    """Load a causal LM directly on CPU when quantization is unavailable."""

    cpu_dtype = dtype or torch.float32
    if cpu_dtype == torch.float16:
        logger.info(
            "Promoting float16 dtype to float32 for CPU execution",
            extra={"requested_dtype": "float16", "resolved_dtype": "float32"},
        )
        cpu_dtype = torch.float32

    logger.info(
        "Loading base model on CPU fallback",
        extra={"model": model_id, "dtype": str(cpu_dtype)},
    )

    cpu_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    try:
        model = _load_with_optional_kwargs(
            AutoModelForCausalLM.from_pretrained,
            model_id,
            base_kwargs=cpu_kwargs,
            optional_kwargs=(
                {"torch_dtype": cpu_dtype},
                {"dtype": cpu_dtype},
            ),
        )
    except Exception:
        logger.error(
            "Failed to load model on CPU fallback",
            extra={"model": model_id},
            exc_info=True,
        )
        raise

    try:  # pragma: no cover - defensive cleanup for limited builds
        model.to("cpu")
    except Exception:
        logger.debug("Unable to move model to CPU", exc_info=True)

    tokenizer = _initialise_tokenizer(model_id, trust_remote_code=trust_remote_code)
    _configure_model_attention(
        model, attention_implementation=attention_implementation
    )

    return model, tokenizer


def _load_with_optional_kwargs(
    loader: Any,
    model_id: str,
    *,
    base_kwargs: dict[str, Any],
    optional_kwargs: Iterable[dict[str, Any]],
) -> Any:
    """Call ``loader`` with optional kwargs, gracefully handling unsupported keys."""

    candidates = _build_model_kwargs_candidates(
        loader, base_kwargs=base_kwargs, optional_kwargs=optional_kwargs
    )
    last_error: Exception | None = None

    for candidate in candidates:
        try:
            return loader(model_id, **candidate)
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            last_error = exc
            logger.debug(
                "Retrying model load without unsupported kwargs",
                extra={
                    "loader": getattr(loader, "__name__", str(loader)),
                    "attempted_kwargs": sorted(candidate.keys()),
                },
            )

    if last_error is not None:
        raise last_error

    return loader(model_id, **base_kwargs)


def _build_model_kwargs_candidates(
    loader: Any,
    *,
    base_kwargs: dict[str, Any],
    optional_kwargs: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return keyword argument candidates ordered by preference for ``loader``."""

    signature = inspect.signature(loader)
    parameters = signature.parameters.values()
    accepts_var_kw = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters
    )

    extras_sequence = [dict(extra) for extra in optional_kwargs]
    candidates: list[dict[str, Any]] = []

    if accepts_var_kw:
        for extras in extras_sequence:
            candidate = dict(base_kwargs)
            candidate.update(extras)
            candidates.append(candidate)
        candidates.append(dict(base_kwargs))
        return candidates

    allowed_keys = {
        param.name
        for param in parameters
        if param.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }

    rejected = set()
    for extras in extras_sequence:
        supported = {key: value for key, value in extras.items() if key in allowed_keys}
        rejected.update(key for key in extras if key not in allowed_keys)
        if supported:
            candidate = dict(base_kwargs)
            candidate.update(supported)
            candidates.append(candidate)

    if rejected:
        logger.debug(
            "Skipping unsupported model kwargs",
            extra={
                "rejected_kwargs": sorted(rejected),
                "loader": getattr(loader, "__name__", str(loader)),
            },
        )

    candidates.append(dict(base_kwargs))
    return candidates


def _initialise_tokenizer(model_id: str, *, trust_remote_code: bool) -> Any:
    """Create a tokenizer with consistent padding defaults across loader paths."""

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _configure_model_attention(
    model: Any, *, attention_implementation: str | None
) -> None:
    """Apply consistent attention configuration hints to ``model``."""

    config = getattr(model, "config", None)
    if config is None:
        return

    if hasattr(config, "use_cache"):
        config.use_cache = False

    attn_impl = attention_implementation or "eager"
    for attr in ("attn_impl", "attn_implementation"):
        if hasattr(config, attr):
            setattr(config, attr, attn_impl)


def _get_min_bitsandbytes_version() -> Version:
    """Return the configured minimum supported bitsandbytes version."""

    override = os.getenv(_MIN_BITSANDBYTES_VERSION_ENV)
    if not override:
        return _DEFAULT_MIN_BITSANDBYTES_VERSION

    try:
        return Version(override)
    except InvalidVersion:
        logger.warning(
            "Invalid bitsandbytes minimum version override; using default",
            extra={
                "env_var": _MIN_BITSANDBYTES_VERSION_ENV,
                "value": override,
                "default": str(_DEFAULT_MIN_BITSANDBYTES_VERSION),
            },
        )
        return _DEFAULT_MIN_BITSANDBYTES_VERSION
