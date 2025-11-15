"""Model loading helpers for fine-tuning pipelines."""

from __future__ import annotations

# isort: off
from ._unsloth_bootstrap import UNSLOTH_AVAILABLE

# isort: on
import inspect
import logging
import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from packaging.version import InvalidVersion, Version
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

try:  # pragma: no cover - accelerate optional during some tests
    from accelerate.hooks import remove_hook_from_module as _ACCELERATE_REMOVE_HOOK
except Exception:  # pragma: no cover - fallback path exercised in unit tests
    _ACCELERATE_REMOVE_HOOK = None  # type: ignore[assignment]

try:  # pragma: no cover - accelerate optional during some tests
    from accelerate.state import AcceleratorState as _ACCELERATE_STATE
except Exception:  # pragma: no cover - fallback path exercised in unit tests
    _ACCELERATE_STATE = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

if not UNSLOTH_AVAILABLE:
    logger.debug(
        "Unsloth package not preloaded; 4-bit loader will rely on standard Transformers kernels"
    )


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
        model, tokenizer = _load_cpu_causal_lm(
            model_id,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            attention_implementation=attention_implementation,
        )
        setattr(model, "_mongars_quantized_4bit", False)
        return model, tokenizer

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
                {"dtype": target_dtype},
                {"torch_dtype": target_dtype},
            ),
        )
    except TypeError:
        logger.error(
            "Failed to load model with 4-bit quantization",
            extra={"model": model_id},
            exc_info=True,
        )
        raise

    tokenizer = _initialise_tokenizer(model_id, trust_remote_code=trust_remote_code)

    _configure_model_post_load(model, attention_implementation=attention_implementation)

    # ``AutoModelForCausalLM.from_pretrained`` may replace the explicit device map
    # with the string "auto" when quantization hooks request dynamic placement.
    # Accelerate refuses to train models that advertise ``device_map='auto'`` in any
    # distributed context, which surfaces as a hard error during Trainer setup even
    # when we only intend to train on a single process.  Normalise the attribute to
    # the deterministic mapping we computed so Accelerate recognises that placement
    # is intentional and bypass-safe.
    current_map = getattr(model, "hf_device_map", None)
    if isinstance(current_map, str) and current_map.lower() == "auto":
        try:
            setattr(model, "hf_device_map", device_map)
            setattr(model, "_hf_device_map", device_map)
            logger.info(
                "Normalised model hf_device_map from 'auto' to explicit mapping"
            )
        except Exception:  # pragma: no cover - best effort correction
            logger.warning(
                "Failed to override hf_device_map='auto' with explicit mapping",
                exc_info=True,
            )

    _cache_device_map_hint(model, device_map)

    try:  # pragma: no cover - depends on torch build
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:  # pragma: no cover - best effort configuration
        pass

    setattr(model, "_mongars_quantized_4bit", True)
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


def _cache_device_map_hint(model: Any, mapping: Mapping[str, Any]) -> None:
    """Persist the last known explicit mapping on ``model`` for future reuse."""

    try:
        setattr(model, "_mongars_device_map_hint", dict(mapping))
    except Exception:  # pragma: no cover - best effort hint caching
        logger.debug(
            "Unable to cache device map hint on %s", type(model), exc_info=True
        )


_DEVICE_MAP_PROXY_ATTRS: tuple[str, ...] = (
    "base_model",
    "model",
    "pretrained_model",
    "inner_model",
    "llm",
)


def _resolve_device_map_candidate(model: Any) -> Mapping[str, Any] | None:
    for attr in ("_hf_device_map", "_mongars_device_map_hint"):
        candidate = getattr(model, attr, None)
        if isinstance(candidate, Mapping):
            return candidate

    for attr in _DEVICE_MAP_PROXY_ATTRS:
        proxy = getattr(model, attr, None)
        if proxy is None or proxy is model:
            continue
        mapping = getattr(proxy, "hf_device_map", None)
        if isinstance(mapping, Mapping):
            return mapping
        hint = getattr(proxy, "_mongars_device_map_hint", None)
        if isinstance(hint, Mapping):
            return hint
    return None


def ensure_explicit_device_map(model: Any) -> bool:
    """Ensure ``hf_device_map`` attributes are mappings instead of the string ``'auto'``."""

    processed: set[int] = set()
    stack: list[tuple[Any, bool]] = [(model, False)]
    updated = False

    while stack:
        current, children_processed = stack.pop()
        if current is None:
            continue

        pointer = id(current)
        if children_processed:
            if pointer in processed:
                continue
            processed.add(pointer)
        elif pointer in processed:
            continue

        if not children_processed:
            stack.append((current, True))
            for attr in ("base_model", "model"):
                nested = getattr(current, attr, None)
                if nested is not None:
                    stack.append((nested, False))
            continue

        mapping = getattr(current, "hf_device_map", None)
        if isinstance(mapping, Mapping):
            _cache_device_map_hint(current, mapping)
            continue

        candidate = _resolve_device_map_candidate(current)
        if candidate is None:
            continue

        try:
            normalised = dict(candidate)
        except Exception:  # pragma: no cover - fallback to original object
            normalised = candidate  # type: ignore[assignment]

        try:
            setattr(current, "hf_device_map", normalised)
            updated = True
        except Exception:  # pragma: no cover - best effort enforcement
            logger.debug(
                "Unable to normalise hf_device_map on %s", type(current), exc_info=True
            )
            continue

        try:
            setattr(current, "_hf_device_map", normalised)
        except Exception:  # pragma: no cover - best effort enforcement
            logger.debug(
                "Unable to refresh _hf_device_map on %s", type(current), exc_info=True
            )

        _cache_device_map_hint(current, normalised)

    return updated


def move_to_cpu(model: Any) -> None:
    """Attempt to move ``model`` to CPU for graceful cleanup."""

    mover = getattr(model, "to", None)
    if callable(mover):
        try:  # pragma: no cover - best effort cleanup
            mover("cpu")
        except Exception:
            logger.debug("Unable to move model to CPU", exc_info=True)

    _remove_accelerate_hooks(model)
    _normalise_device_map_to_cpu(model)
    _reset_accelerate_state()


def _remove_accelerate_hooks(model: Any) -> None:
    """Detach Accelerate hooks so CPU fallbacks do not reattach CUDA inputs."""

    if _ACCELERATE_REMOVE_HOOK is not None:
        try:
            _ACCELERATE_REMOVE_HOOK(model, recurse=True)
        except Exception:  # pragma: no cover - best effort cleanup
            logger.debug("Unable to remove accelerate hooks", exc_info=True)
        else:
            return

    # Fallback path when accelerate is unavailable—manually clear common attributes.
    hook = getattr(model, "_hf_hook", None)
    if hook is not None:
        try:
            detach = getattr(hook, "detach_hook", None)
            if callable(detach):
                detach(model)
        except Exception:  # pragma: no cover - best effort cleanup
            logger.debug("Unable to detach manual accelerate hook", exc_info=True)
        try:
            delattr(model, "_hf_hook")
        except Exception:
            logger.debug(
                "Unable to delete manual accelerate hook attribute", exc_info=True
            )

    old_forward = getattr(model, "_old_forward", None)
    if old_forward is not None:
        try:
            model.forward = old_forward  # type: ignore[attr-defined]
        except Exception:
            logger.debug("Unable to restore original forward method", exc_info=True)
        try:
            delattr(model, "_old_forward")
        except Exception:
            logger.debug("Unable to delete cached forward attribute", exc_info=True)

    for attr in ("cuda", "npu", "xpu", "mlu", "sdaa", "musa"):
        try:
            model.__dict__.pop(attr, None)
        except Exception:
            logger.debug(
                "Unable to remove auxiliary accelerate attribute %s",
                attr,
                exc_info=True,
            )

    children = getattr(model, "children", None)
    if callable(children):
        for child in children():
            _remove_accelerate_hooks(child)


def _normalise_device_map_to_cpu(model: Any) -> None:
    """Update any recorded device maps to reflect CPU execution."""

    mapping = getattr(model, "hf_device_map", None)
    if not mapping:
        return

    items: Iterable[tuple[str, Any]]
    if isinstance(mapping, Mapping):
        items = mapping.items()
    else:
        try:
            items = list(mapping.items())  # type: ignore[attr-defined]
        except Exception:
            logger.debug("Model device map is not a mapping; skipping normalisation")
            return

    new_map = {key: "cpu" for key, _ in items}

    try:
        setattr(model, "hf_device_map", new_map)
    except Exception:  # pragma: no cover - best effort cleanup
        logger.debug(
            "Unable to update model hf_device_map for CPU fallback", exc_info=True
        )

    if hasattr(model, "_hf_device_map"):
        try:
            setattr(model, "_hf_device_map", new_map)
        except Exception:  # pragma: no cover - best effort cleanup
            logger.debug(
                "Unable to update model _hf_device_map for CPU fallback", exc_info=True
            )


def _reset_accelerate_state() -> None:
    """Reset Accelerate's global state so future trainers re-evaluate devices."""

    if _ACCELERATE_STATE is None:
        return

    try:
        _ACCELERATE_STATE._reset_state()
    except Exception:  # pragma: no cover - best effort cleanup
        logger.debug("Unable to reset accelerate state", exc_info=True)


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
_DEFAULT_MIN_BITSANDBYTES_VERSION = Version("0.48.0")


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

    config_dtype = _resolve_config_dtype(model_id, trust_remote_code=trust_remote_code)
    candidate_dtypes = _cpu_dtype_candidates(requested=dtype, configured=config_dtype)
    cpu_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }

    last_error: Exception | None = None
    tried: list[str] = []

    for cpu_dtype in candidate_dtypes:
        tried.append(str(cpu_dtype))
        logger.info(
            "Loading base model on CPU fallback",
            extra={"model": model_id, "dtype": str(cpu_dtype)},
        )
        try:
            model = _load_with_optional_kwargs(
                AutoModelForCausalLM.from_pretrained,
                model_id,
                base_kwargs=cpu_kwargs,
                optional_kwargs=(
                    {"dtype": cpu_dtype},
                    {"torch_dtype": cpu_dtype},
                ),
            )
        except Exception as exc:  # pragma: no cover - relies on runtime backends
            last_error = exc
            logger.warning(
                "CPU fallback load failed for dtype",  #
                extra={"model": model_id, "dtype": str(cpu_dtype)},
                exc_info=True,
            )
            continue

        try:  # pragma: no cover - defensive cleanup for limited builds
            model.to("cpu")
        except Exception:
            logger.debug("Unable to move model to CPU", exc_info=True)

        tokenizer = _initialise_tokenizer(model_id, trust_remote_code=trust_remote_code)

        _configure_model_post_load(
            model, attention_implementation=attention_implementation
        )

        setattr(model, "_mongars_quantized_4bit", False)
        return model, tokenizer

    logger.error(
        "Failed to load model on CPU fallback",  #
        extra={"model": model_id, "dtypes_tried": tried},
        exc_info=last_error,
    )
    if last_error is not None:
        raise last_error
    raise RuntimeError("Unable to load model on CPU fallback")


def _cpu_dtype_candidates(
    *, requested: Optional[torch.dtype], configured: Optional[torch.dtype]
) -> list[torch.dtype]:
    """Return preferred CPU dtype candidates ordered by desirability."""

    candidates: list[torch.dtype] = []

    def _append(candidate: Optional[torch.dtype]) -> None:
        if candidate is None or candidate in candidates:
            return
        candidates.append(candidate)

    _append(requested)
    _append(configured)
    _append(torch.float16)
    _append(getattr(torch, "bfloat16", None))
    _append(torch.float32)
    return candidates


def _resolve_config_dtype(
    model_id: str, *, trust_remote_code: bool
) -> Optional[torch.dtype]:
    """Return the dtype declared by the model config when available."""

    try:
        config = AutoConfig.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
    except Exception:  # pragma: no cover - network/backends may be unavailable
        logger.debug(
            "Unable to resolve model config dtype; falling back to defaults",
            extra={"model": model_id},
            exc_info=True,
        )
        return None

    for attr in ("torch_dtype", "dtype"):
        dtype = _normalise_torch_dtype(getattr(config, attr, None))
        if dtype is not None:
            logger.debug(
                "Resolved model config dtype",
                extra={"model": model_id, "dtype": str(dtype)},
            )
            return dtype

    return None


def _normalise_torch_dtype(value: Any) -> Optional[torch.dtype]:
    """Normalise string or ``torch.dtype`` values into ``torch.dtype`` objects."""

    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": getattr(torch, "bfloat16", None),
            "bf16": getattr(torch, "bfloat16", None),
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        key = value.lower()
        return mapping.get(key)
    return None


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
            candidates.append(base_kwargs | extras)
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
            candidates.append(base_kwargs | supported)

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


def _configure_model_post_load(
    model: Any, *, attention_implementation: str | None
) -> None:
    """Apply shared configuration to models loaded for fine-tuning."""

    model.config.use_cache = False
    attn_impl = attention_implementation or "eager"
    for attr in ("attn_impl", "attn_implementation"):
        if hasattr(model.config, attr):
            setattr(model.config, attr, attn_impl)


def _initialise_tokenizer(model_id: str, *, trust_remote_code: bool) -> Any:
    """Create a tokenizer with consistent padding defaults across loader paths."""

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


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
