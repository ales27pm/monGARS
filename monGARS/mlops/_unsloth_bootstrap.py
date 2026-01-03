"""Eagerly import Unsloth before Transformers bindings when available."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys

logger = logging.getLogger(__name__)

_UNSLOTH_SPEC = importlib.util.find_spec("unsloth")

if _UNSLOTH_SPEC is None:
    UNSLOTH_AVAILABLE = False
    logger.debug("Unsloth package not found; using standard Transformers path")
else:
    try:
        importlib.import_module("unsloth")
    except Exception:  # pragma: no cover - depends on runtime environment
        UNSLOTH_AVAILABLE = False
        logger.warning(
            "Unable to import Unsloth package; continuing without fast-path hooks",
            exc_info=True,
        )
        # Guard against partially imported modules poisoning future imports.
        sys.modules.pop("unsloth", None)
    else:
        UNSLOTH_AVAILABLE = True
        logger.debug(
            "Imported Unsloth before Transformers modules", extra={"module": "unsloth"}
        )

        if importlib.util.find_spec("unsloth_zoo") is None:
            logger.info(
                "Unsloth zoo extension not installed; fast-path kernels may be limited"
            )

__all__ = ["UNSLOTH_AVAILABLE"]
