"""CUDA memory headroom analysis utilities."""

from __future__ import annotations

from typing import Any

SEVERITY_ORDER = {"ok": 0, "warning": 1, "critical": 2, "unknown": 3}


def _classify_device(
    device: dict[str, Any],
    *,
    min_free_gib: float,
    min_free_ratio: float,
) -> dict[str, Any]:
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
        free_ratio = free / total
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
    total_bytes_raw = float(memory_bytes.get("total", {}).get("bytes", 0.0))
    denominator = total_bytes_raw or reserved_bytes
    if reserved_bytes > 0 and reserved_bytes > allocated_bytes and denominator > 0:
        fragmentation_ratio = (reserved_bytes - allocated_bytes) / denominator
        if fragmentation_ratio > 0.2:
            recommendations.append(
                "Call torch.cuda.empty_cache() or restart the worker to defragment the allocator."
            )

    return {
        "index": device.get("index"),
        "status": status,
        "free_gib": free,
        "free_ratio": free_ratio,
        "fragmentation_ratio": fragmentation_ratio,
        "recommendations": recommendations,
    }


def _derive_overall_status(devices: list[dict[str, Any]]) -> str:
    status = "ok"
    for device in devices:
        device_status = device.get("status", "unknown")
        if SEVERITY_ORDER.get(device_status, 3) > SEVERITY_ORDER.get(status, 3):
            status = device_status
    return status


def analyse_cuda_state(
    cuda_payload: dict[str, Any] | None,
    *,
    min_free_gib: float,
    min_free_ratio: float,
    skip_reason: str | None,
) -> dict[str, Any]:
    """Classify CUDA headroom risk for diagnostics output."""

    thresholds = {"min_free_gib": min_free_gib, "min_free_ratio": min_free_ratio}

    if not cuda_payload:
        reason = skip_reason or "cuda_unavailable"
        return {"status": "unknown", "reason": reason, "thresholds": thresholds}

    devices_payload = list(cuda_payload.get("devices") or [])
    if not devices_payload:
        return {
            "status": "unknown",
            "reason": "no_devices_reported",
            "thresholds": thresholds,
            "devices": [],
        }

    device_reports = [
        _classify_device(
            device, min_free_gib=min_free_gib, min_free_ratio=min_free_ratio
        )
        for device in devices_payload
    ]
    overall_status = _derive_overall_status(device_reports)

    result: dict[str, Any] = {
        "status": overall_status,
        "thresholds": thresholds,
        "devices": device_reports,
    }

    invalid_indices = cuda_payload.get("invalid_indices")
    if invalid_indices:
        result["invalid_indices"] = invalid_indices

    return result
