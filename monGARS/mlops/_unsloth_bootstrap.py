"""Eagerly import Unsloth before Transformers bindings when available."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
from types import ModuleType

logger = logging.getLogger(__name__)

UNSLOTH_AVAILABLE = False
UNSLOTH_DISABLED_REASON: str | None = None
UNSLOTH_IMPORT_ERROR: Exception | None = None
_UNSLOTH_MODULE: ModuleType | None = None


def _allow_forced_import() -> bool:
    value = os.getenv("MONGARS_FORCE_UNSLOTH", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _has_supported_accelerator() -> tuple[bool, str | None]:
    if _allow_forced_import():
        return True, None

    torch_module = sys.modules.get("torch")
    if torch_module is None and importlib.util.find_spec("torch") is not None:
        try:
            torch_module = importlib.import_module("torch")
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.info(
                "Unsloth bootstrap skipped while probing torch support",
                extra={"reason": str(exc)[:200]},
            )
            return False, "torch_probe_failed"

    if torch_module is None:
        return False, "torch_missing"

    accelerators = (
        getattr(getattr(torch_module, "cuda", None), "is_available", None),
        getattr(getattr(torch_module, "xpu", None), "is_available", None),
    )
    for checker in accelerators:
        if not callable(checker):
            continue
        try:
            if bool(checker()):
                return True, None
        except Exception:  # pragma: no cover - backend specific
            logger.debug(
                "Unsloth accelerator probe failed",
                exc_info=True,
            )
    return False, "no_supported_accelerator"


def _bootstrap_unsloth() -> None:
    global UNSLOTH_AVAILABLE
    global UNSLOTH_DISABLED_REASON
    global UNSLOTH_IMPORT_ERROR
    global _UNSLOTH_MODULE

    unsloth_spec = importlib.util.find_spec("unsloth")
    if unsloth_spec is None:
        UNSLOTH_DISABLED_REASON = "package_not_found"
        logger.debug("Unsloth package not found; using standard Transformers path")
        return

    accelerator_available, accelerator_reason = _has_supported_accelerator()
    if not accelerator_available:
        UNSLOTH_DISABLED_REASON = accelerator_reason
        logger.info(
            "Skipping Unsloth bootstrap on this host",
            extra={"reason": accelerator_reason},
        )
        return

    try:
        _UNSLOTH_MODULE = importlib.import_module("unsloth")
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        UNSLOTH_IMPORT_ERROR = exc
        UNSLOTH_DISABLED_REASON = "import_failed"
        logger.info(
            "Unable to import Unsloth package; continuing without fast-path hooks",
            extra={"reason": str(exc)[:200]},
        )
        # Guard against partially imported modules poisoning future imports.
        sys.modules.pop("unsloth", None)
        _UNSLOTH_MODULE = None
        return

    UNSLOTH_AVAILABLE = True
    logger.debug(
        "Imported Unsloth before Transformers modules", extra={"module": "unsloth"}
    )

    if importlib.util.find_spec("unsloth_zoo") is None:
        logger.info(
            "Unsloth zoo extension not installed; fast-path kernels may be limited"
        )


def get_unsloth_module() -> ModuleType | None:
    """Return the loaded Unsloth module when the fast-path is available."""

    module = _UNSLOTH_MODULE
    if module is not None:
        return module
    existing = sys.modules.get("unsloth")
    if isinstance(existing, ModuleType):
        return existing
    return None


def get_fast_language_model() -> object | None:
    """Return ``unsloth.FastLanguageModel`` when available."""

    module = get_unsloth_module()
    if module is None:
        return None
    return getattr(module, "FastLanguageModel", None)


_bootstrap_unsloth()

__all__ = [
    "UNSLOTH_AVAILABLE",
    "UNSLOTH_DISABLED_REASON",
    "UNSLOTH_IMPORT_ERROR",
    "get_fast_language_model",
    "get_unsloth_module",
]
