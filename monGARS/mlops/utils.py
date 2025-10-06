"""Runtime helpers shared across fine-tuning pipelines."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from importlib.util import find_spec
from typing import Iterable, Sequence

try:  # pragma: no cover - optional during lightweight tests
    import torch
except Exception:  # pragma: no cover - torch not installed
    torch = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _validate_spec(spec: str) -> str:
    """Ensure the pip requirement specification is safe to pass to subprocess."""

    if not spec or any(ch.isspace() for ch in spec.strip("\n\r")):
        raise ValueError(f"Invalid requirement specification: {spec!r}")
    if "\x00" in spec:
        raise ValueError("Requirement specification contains NUL byte")
    return spec


def _import_target(spec: str) -> str:
    base = spec.split(";", 1)[0]
    base = base.split("[", 1)[0]
    for token in ("==", ">=", "<=", "!=", "~=", ">", "<"):
        if token in base:
            base = base.split(token, 1)[0]
            break
    return base


def ensure_dependencies(
    required: Sequence[str],
    optional: Sequence[str] | None = None,
    *,
    auto_install: bool = True,
) -> None:
    """Install dependencies on demand.

    Parameters
    ----------
    required:
        Packages that must be importable. Each entry should be a string accepted by
        ``pip install`` (for example ``"transformers>=4.44"``).
    optional:
        Packages that enable additional functionality. Missing optional packages are
        logged instead of raising when ``auto_install`` is ``False``.
    auto_install:
        When ``True`` (default) missing packages are installed automatically. When
        ``False`` the function raises ``ImportError``.
    """

    optional = optional or []
    for spec in required:
        if not find_spec(_import_target(spec)):
            if not auto_install:
                raise ImportError(f"Required dependency {spec!r} not installed")
            _pip_install(_validate_spec(spec))
    for spec in optional:
        if find_spec(_import_target(spec)):
            continue
        if not auto_install:
            logger.info("Optional dependency missing: %s", spec)
            continue
        try:
            _pip_install(_validate_spec(spec))
        except subprocess.CalledProcessError as exc:  # pragma: no cover - best effort
            logger.warning("Unable to install optional dependency %s: %s", spec, exc)


def _pip_install(spec: str) -> None:
    """Invoke pip in a subprocess with hardened arguments."""

    logger.info("Installing dependency: %s", spec)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-U", spec],
        check=True,
    )


def configure_cuda_allocator(default: str = "expandable_segments:True,max_split_size_mb:64") -> None:
    """Set CUDA allocator defaults if they are not already configured."""

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", default)


def ensure_directory(path: os.PathLike[str] | str) -> None:
    """Create ``path`` (and parents) when it does not exist."""

    os.makedirs(path, exist_ok=True)


def chunked(iterable: Iterable, size: int) -> list[list]:
    """Return ``iterable`` grouped into chunks of ``size`` (primarily for tests)."""

    chunk: list = []
    result: list[list] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            result.append(chunk)
            chunk = []
    if chunk:
        result.append(chunk)
    return result


def describe_environment() -> None:
    """Print interpreter and CUDA/GPU information when available."""

    logger.info("Python runtime", extra={"version": sys.version})
    if torch is None:
        logger.info("Torch unavailable; GPU status unknown")
        return
    if not torch.cuda.is_available():  # pragma: no cover - depends on hardware
        logger.info("CUDA not available")
        return
    props = torch.cuda.get_device_properties(0)
    logger.info(
        "CUDA environment",
        extra={
            "device": props.name,
            "vram_gb": round(props.total_memory / (1024**3), 2),
            "cuda_version": getattr(torch.version, "cuda", "unknown"),
            "torch_version": torch.__version__,
        },
    )
