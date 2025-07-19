"""Hardware detection utilities for resource-constrained devices."""

from __future__ import annotations

import platform
from typing import Optional

import psutil


def detect_embedded_device() -> Optional[str]:
    """Return architecture name if running on an embedded ARM board."""
    arch = platform.machine().lower()
    if arch in {"armv7l", "aarch64", "arm64"}:
        return arch
    return None


def recommended_worker_count(default: int = 4) -> int:
    """Return suggested FastAPI worker count based on hardware."""
    device = detect_embedded_device()
    if not device:
        return default

    cores = psutil.cpu_count(logical=False)
    if cores is None:
        cores = psutil.cpu_count(logical=True)
    if cores is None or cores == 0:
        cores = 1
    if device == "armv7l":
        return 1
    return max(1, min(2, cores))
