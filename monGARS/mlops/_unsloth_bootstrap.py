"""Eagerly import Unsloth before Transformers bindings when available."""

from __future__ import annotations

import importlib.util
import logging

logger = logging.getLogger(__name__)

_UNSLOTH_SPEC = importlib.util.find_spec("unsloth")
_UNSLOTH_ZOO_SPEC = importlib.util.find_spec("unsloth_zoo")

if _UNSLOTH_SPEC is not None and _UNSLOTH_ZOO_SPEC is not None:
    import unsloth  # type: ignore  # noqa: F401

    UNSLOTH_AVAILABLE = True
    logger.debug(
        "Imported Unsloth before Transformers modules", extra={"module": "unsloth"}
    )
else:
    UNSLOTH_AVAILABLE = False
    if _UNSLOTH_SPEC is not None and _UNSLOTH_ZOO_SPEC is None:
        logger.info(
            "Detected Unsloth without unsloth_zoo; skipping eager import to avoid runtime errors"
        )
    else:
        logger.debug("Unsloth package not found; using standard Transformers path")

__all__ = ["UNSLOTH_AVAILABLE"]
