"""CUDA diagnostics helpers shared across build and CLI flows."""

from __future__ import annotations

import logging
import os
from types import ModuleType
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _format_bytes(num_bytes: int) -> dict[str, float]:
    kib = num_bytes / 1024
    mib = kib / 1024
    gib = mib / 1024
    return {"bytes": float(num_bytes), "mib": float(mib), "gib": float(gib)}


def gather_cuda_metrics(
    torch_module: ModuleType,
    device_selector: Callable[[], list[int]],
) -> dict[str, Any] | None:
    """Collect memory metrics for selected CUDA devices."""

    if not hasattr(torch_module, "cuda") or not torch_module.cuda.is_available():
        logger.info("CUDA is not available; GPU diagnostics skipped")
        return None

    requested_indices = device_selector()
    if not requested_indices:
        return None

    device_count = torch_module.cuda.device_count()
    valid_indices: list[int] = []
    invalid_indices: list[int] = []
    for idx in requested_indices:
        if 0 <= idx < device_count:
            valid_indices.append(idx)
        else:
            invalid_indices.append(idx)

    for idx in invalid_indices:
        logger.warning(
            "requested CUDA device %s is out of range (available: %s)",
            idx,
            device_count,
        )

    if not valid_indices:
        return None

    def _inspect_device(device: int) -> dict[str, Any] | None:
        with torch_module.cuda.device(device):
            try:
                free_bytes, total_bytes = torch_module.cuda.mem_get_info()
            except Exception:  # pragma: no cover - defensive guardrail
                logger.exception("failed to query CUDA memory usage")
                return None

            try:
                properties = torch_module.cuda.get_device_properties(device)
                device_name = properties.name
                capability = f"{properties.major}.{properties.minor}"
                total_memory = int(properties.total_memory)
            except Exception:  # pragma: no cover - defensive guardrail
                logger.exception("failed to read CUDA device properties")
                device_name = None
                capability = None
                total_memory = int(total_bytes)

            try:
                memory_stats = torch_module.cuda.memory_stats()
            except Exception:  # pragma: no cover - defensive guardrail
                logger.debug(
                    "unable to collect extended CUDA memory stats", exc_info=True
                )
                memory_stats = None

            metrics = {
                "index": device,
                "name": device_name,
                "compute_capability": capability,
                "memory_bytes": {
                    "free": _format_bytes(int(free_bytes)),
                    "total": _format_bytes(int(total_bytes)),
                    "reserved": _format_bytes(int(torch_module.cuda.memory_reserved())),
                    "allocated": _format_bytes(
                        int(torch_module.cuda.memory_allocated())
                    ),
                    "total_reported": _format_bytes(total_memory),
                },
                "stats": memory_stats,
            }

            total_bytes_reported = metrics["memory_bytes"]["total"]["bytes"]
            try:
                metrics["utilisation"] = {
                    "allocated_fraction": (
                        float(metrics["memory_bytes"]["allocated"]["bytes"])
                        / total_bytes_reported
                        if total_bytes_reported
                        else 0.0
                    ),
                    "reserved_fraction": (
                        float(metrics["memory_bytes"]["reserved"]["bytes"])
                        / total_bytes_reported
                        if total_bytes_reported
                        else 0.0
                    ),
                }
            except ZeroDivisionError:  # pragma: no cover - defensive guardrail
                metrics["utilisation"] = {
                    "allocated_fraction": 0.0,
                    "reserved_fraction": 0.0,
                }

            return metrics

    devices = [report for idx in valid_indices if (report := _inspect_device(idx))]
    if not devices:
        return None

    return {
        "driver_version": getattr(torch_module.version, "cuda", None),
        "device_count": device_count,
        "devices": devices,
        "visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "invalid_indices": invalid_indices or None,
    }
