"""Interpreter-level bootstrap hooks for monGARS."""

from __future__ import annotations

import importlib
import logging
import sys
from types import ModuleType

logger = logging.getLogger("monGARS.bootstrap")


def _import_unsloth() -> ModuleType | None:
    """Attempt to import Unsloth before ``transformers`` loads.

    Importing Unsloth early allows it to patch the Hugging Face stack for
    improved performance.  The helper mirrors the optional import guards used
    throughout the codebase and keeps failure states silent unless debugging is
    enabled.
    """

    if "unsloth" in sys.modules:
        return sys.modules["unsloth"]
    try:
        module = importlib.import_module("unsloth")
    except ModuleNotFoundError:
        logger.debug("Unsloth is not installed; skipping performance bootstrap.")
        return None
    except Exception as exc:  # pragma: no cover - defensive guardrail
        logger.warning("Failed to import Unsloth early", exc_info=exc)
        return None
    else:
        return module


_unsloth_module = _import_unsloth()

if _unsloth_module is None and "transformers" in sys.modules:
    logger.debug(
        "Transformers imported before Unsloth; optimisation patches may be inactive.",
    )
