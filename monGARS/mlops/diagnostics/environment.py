"""Environment inspection helpers for diagnostics utilities."""

from __future__ import annotations

import logging
import platform
import time
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)


def configure_logging(verbose: bool) -> None:
    """Configure logging for diagnostics entrypoints."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def import_optional(name: str) -> ModuleType | None:
    """Attempt to import a module without raising on failure."""

    try:
        return __import__(name, fromlist=["*"])
    except ModuleNotFoundError:
        logger.debug("optional dependency %s missing", name)
        return None
    except Exception:  # pragma: no cover - defensive guardrail
        logger.exception("unexpected error importing %s", name)
        return None


def gather_environment(torch_module: ModuleType | None) -> dict[str, Any]:
    """Collect runtime metadata for diagnostics output."""

    python_impl = platform.python_implementation()
    environment: dict[str, Any] = {
        "timestamp": time.time(),
        "python": {
            "implementation": python_impl,
            "version": platform.python_version(),
            "compiler": platform.python_compiler(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "torch": {
            "available": torch_module is not None,
            "version": (
                getattr(torch_module, "__version__", None) if torch_module else None
            ),
            "cuda_version": (
                getattr(getattr(torch_module, "version", None), "cuda", None)
                if torch_module
                else None
            ),
        },
        "unsloth": None,
    }

    if torch_module is not None and hasattr(torch_module, "cuda"):
        try:
            environment["torch"]["device_count"] = torch_module.cuda.device_count()
        except Exception:  # pragma: no cover - defensive guardrail
            logger.debug("unable to query CUDA device count", exc_info=True)

    unsloth_module = import_optional("unsloth")
    if unsloth_module is not None:
        environment["unsloth"] = {
            "available": True,
            "version": getattr(unsloth_module, "__version__", None),
            "module_path": getattr(unsloth_module, "__file__", None),
        }
    else:
        environment["unsloth"] = {"available": False}

    return environment
