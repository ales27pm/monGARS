"""Utility to inspect Unsloth integration and GPU memory headroom."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

logger = logging.getLogger("scripts.unsloth.diagnose")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _import_optional(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        logger.debug("optional dependency %s missing", name)
        return None
    except Exception:  # pragma: no cover - defensive guardrail
        logger.exception("unexpected error importing %s", name)
        return None


def _format_bytes(num_bytes: int) -> dict[str, float]:
    kib = num_bytes / 1024
    mib = kib / 1024
    gib = mib / 1024
    return {"bytes": float(num_bytes), "mib": float(mib), "gib": float(gib)}


def _gather_cuda_metrics(
    torch_module: ModuleType,
    device_selector: Callable[[], list[int]],
) -> dict[str, Any] | None:
    if not hasattr(torch_module, "cuda") or not torch_module.cuda.is_available():
        logger.info("CUDA is not available; GPU diagnostics skipped")
        return None

    device_indices = device_selector()
    if not device_indices:
        return None

    device_count = torch_module.cuda.device_count()
    for idx in device_indices:
        if idx < 0 or idx >= device_count:
            logger.warning(
                "requested CUDA device %s is out of range (available: %s)",
                idx,
                device_count,
            )
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

            try:
                metrics["utilisation"] = {
                    "allocated_fraction": (
                        float(metrics["memory_bytes"]["allocated"]["bytes"])
                        / float(metrics["memory_bytes"]["total"]["bytes"])
                        if metrics["memory_bytes"]["total"]["bytes"]
                        else 0.0
                    ),
                    "reserved_fraction": (
                        float(metrics["memory_bytes"]["reserved"]["bytes"])
                        / float(metrics["memory_bytes"]["total"]["bytes"])
                        if metrics["memory_bytes"]["total"]["bytes"]
                        else 0.0
                    ),
                }
            except ZeroDivisionError:  # pragma: no cover - defensive guardrail
                metrics["utilisation"] = {
                    "allocated_fraction": 0.0,
                    "reserved_fraction": 0.0,
                }

            return metrics

    devices = [report for idx in device_indices if (report := _inspect_device(idx))]
    if not devices:
        return None

    return {
        "driver_version": getattr(torch_module.version, "cuda", None),
        "device_count": device_count,
        "devices": devices,
        "visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _gather_environment(torch_module: ModuleType | None) -> dict[str, Any]:
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

    unsloth_module = _import_optional("unsloth")
    if unsloth_module is not None:
        environment["unsloth"] = {
            "available": True,
            "version": getattr(unsloth_module, "__version__", None),
            "module_path": getattr(unsloth_module, "__file__", None),
        }
    else:
        environment["unsloth"] = {"available": False}

    return environment


def _analyse_cuda_state(
    cuda_payload: dict[str, Any] | None,
    *,
    min_free_gib: float,
    min_free_ratio: float,
    skip_reason: str | None,
) -> dict[str, Any]:
    thresholds = {
        "min_free_gib": float(min_free_gib),
        "min_free_ratio": float(min_free_ratio),
    }

    if not cuda_payload:
        reason = skip_reason or "cuda_unavailable"
        return {
            "status": "unknown",
            "reason": reason,
            "thresholds": thresholds,
        }

    devices_payload = cuda_payload.get("devices") or []
    if not devices_payload:
        return {
            "status": "unknown",
            "reason": "no_devices_reported",
            "thresholds": thresholds,
            "devices": [],
        }

    severity_order = {"ok": 0, "warning": 1, "critical": 2, "unknown": 3}
    worst_status = "ok"
    device_reports: list[dict[str, Any]] = []

    for device in devices_payload:
        memory_bytes = device.get("memory_bytes", {})
        free = float(memory_bytes.get("free", {}).get("gib", 0.0))
        total = float(memory_bytes.get("total", {}).get("gib", 0.0))
        reserved_bytes = float(memory_bytes.get("reserved", {}).get("bytes", 0.0))
        allocated_bytes = float(memory_bytes.get("allocated", {}).get("bytes", 0.0))

        recommendations: list[str] = []
        if total <= 0:
            status = "unknown"
            free_ratio = 0.0
        else:
            free_ratio = free / total if total else 0.0
            status = "ok"
            if free_ratio < min_free_ratio or free < min_free_gib:
                status = "critical"
                recommendations.extend(
                    [
                        "Reduce ModelSlotManager max_seq_length to decrease context VRAM usage.",
                        "Lower offload_threshold so activations spill to CPU earlier.",
                        "Use gradient accumulation or smaller per-device batches during fine-tuning.",
                    ]
                )
            elif free_ratio < min_free_ratio * 1.5 or free < min_free_gib * 1.5:
                status = "warning"
                recommendations.extend(
                    [
                        "Trim prompt lengths or enable adapter offloading to keep headroom.",
                        "Prefer 4-bit weights and disable eager weight loading where possible.",
                    ]
                )

        fragmentation_ratio = 0.0
        if reserved_bytes > 0 and reserved_bytes > allocated_bytes:
            total_bytes = (
                float(memory_bytes.get("total", {}).get("bytes", 0.0)) or reserved_bytes
            )
            fragmentation_ratio = (reserved_bytes - allocated_bytes) / total_bytes
            if fragmentation_ratio > 0.2:
                recommendations.append(
                    "Call torch.cuda.empty_cache() or restart the worker to defragment the allocator."
                )

        device_report = {
            "index": device.get("index"),
            "status": status,
            "free_gib": free,
            "free_ratio": free_ratio,
            "fragmentation_ratio": fragmentation_ratio,
            "recommendations": recommendations,
        }
        device_reports.append(device_report)

        if severity_order[status] > severity_order[worst_status]:
            worst_status = status

    return {
        "status": worst_status,
        "thresholds": thresholds,
        "devices": device_reports,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose whether Unsloth is available and inspect CUDA memory headroom.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run the Unsloth patch even if it was previously initialised.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index to inspect (default: 0).",
    )
    parser.add_argument(
        "--all-devices",
        action="store_true",
        help="Collect diagnostics for every visible CUDA device instead of a single index.",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Skip CUDA diagnostics to avoid touching GPU state.",
    )
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=1.0,
        help="Minimum free VRAM (GiB) required before flagging OOM risk (default: 1.0).",
    )
    parser.add_argument(
        "--min-free-ratio",
        type=float,
        default=0.1,
        help="Minimum free/total VRAM ratio required before flagging OOM risk (default: 0.1).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output for troubleshooting.",
    )

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    os.environ.setdefault("DEBUG", "true")
    os.environ.setdefault("SECRET_KEY", "diagnostics-only")

    try:
        from monGARS.core.llm_integration import initialize_unsloth
    except Exception:  # pragma: no cover - defensive guardrail
        logger.exception("unable to import Unsloth integration helper")
        return 2

    try:
        unsloth_state = initialize_unsloth(force=args.force)
    except Exception:  # pragma: no cover - defensive guardrail
        logger.exception("initialising Unsloth failed")
        return 3

    torch_module = _import_optional("torch")

    payload: dict[str, Any] = {
        "environment": _gather_environment(torch_module),
        "unsloth": {**unsloth_state},
    }

    skip_reason: str | None = None
    if args.no_cuda:
        payload["cuda"] = None
        skip_reason = "cuda_diagnostics_disabled"
    elif torch_module is None:
        payload["cuda"] = None
        skip_reason = "torch_missing"
    else:
        selector = (
            (lambda: list(range(torch_module.cuda.device_count())))
            if args.all_devices
            else (lambda: [args.device])
        )
        payload["cuda"] = _gather_cuda_metrics(torch_module, selector)

    payload.setdefault("analysis", {})["oom_risk"] = _analyse_cuda_state(
        payload.get("cuda"),
        min_free_gib=args.min_free_gib,
        min_free_ratio=args.min_free_ratio,
        skip_reason=skip_reason,
    )

    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no branch
    raise SystemExit(main())
