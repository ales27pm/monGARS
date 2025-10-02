"""Hardware awareness helpers for the evolution engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import psutil

from monGARS.config import HardwareHeuristics, get_settings

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import GPUtil
except Exception:  # pragma: no cover - dependency may be unavailable
    GPUtil = None  # type: ignore


@dataclass(frozen=True)
class HardwareProfile:
    """Snapshot of host hardware traits used for scaling heuristics."""

    physical_cores: int
    logical_cpus: int
    total_memory_gb: float
    gpu_count: int
    heuristics: HardwareHeuristics = field(
        default_factory=lambda: get_settings().hardware_heuristics,
        repr=False,
        compare=False,
    )

    @classmethod
    def detect(cls, heuristics: HardwareHeuristics | None = None) -> "HardwareProfile":
        """Inspect the host to derive a hardware profile."""

        try:
            physical = psutil.cpu_count(logical=False) or 1
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "hardware_profile.cpu_detect_failed", extra={"error": str(exc)}
            )
            physical = 1

        try:
            logical = psutil.cpu_count(logical=True) or physical
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "hardware_profile.logical_cpu_detect_failed",
                extra={"error": str(exc)},
            )
            logical = physical

        try:
            memory = psutil.virtual_memory().total / (1024**3)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "hardware_profile.memory_detect_failed", extra={"error": str(exc)}
            )
            memory = 1.0

        gpu_count = 0
        if GPUtil is not None:  # pragma: no cover - optional dependency branch
            try:
                gpu_count = len(GPUtil.getGPUs())
            except Exception as exc:
                logger.warning(
                    "hardware_profile.gpu_detect_failed", extra={"error": str(exc)}
                )

        heuristics = heuristics or get_settings().hardware_heuristics

        return cls(
            physical_cores=max(1, physical),
            logical_cpus=max(physical, logical),
            total_memory_gb=max(0.5, float(memory)),
            gpu_count=max(0, gpu_count),
            heuristics=heuristics,
        )

    def estimate_training_power_draw(self) -> float:
        """Estimate watts consumed by a typical training run on this host."""

        heuristics = self.heuristics
        base = (
            heuristics.base_power_draw + heuristics.power_per_core * self.physical_cores
        )
        if self.gpu_count:
            base += heuristics.power_per_gpu * self.gpu_count
        if self.total_memory_gb < heuristics.low_memory_power_threshold_gb:
            base *= heuristics.low_memory_power_scale
        return max(heuristics.minimum_power_draw, base)

    def max_recommended_workers(self, configured_default: int) -> int:
        """Return an upper bound for worker replicas based on hardware."""

        heuristics = self.heuristics
        baseline = max(heuristics.warm_pool_floor, configured_default)
        cpu_capacity = max(
            heuristics.warm_pool_floor,
            self.logical_cpus // heuristics.cpu_capacity_divisor,
        )
        gpu_bonus = self.gpu_count * heuristics.gpu_worker_bonus
        cap = max(baseline, cpu_capacity + gpu_bonus)
        if self.total_memory_gb < heuristics.worker_low_memory_soft_limit_gb:
            memory_increment = (
                heuristics.worker_low_memory_increment
                if self.total_memory_gb < heuristics.worker_memory_floor_gb
                else heuristics.worker_default_increment
            )
            cap = min(cap, baseline + memory_increment)
        return max(heuristics.warm_pool_floor, cap)

    def min_recommended_workers(self) -> int:
        """Return the minimum number of replicas to keep warm."""

        heuristics = self.heuristics
        if self.total_memory_gb <= heuristics.warm_pool_memory_threshold_gb:
            return heuristics.warm_pool_floor
        warm_candidates = max(
            heuristics.warm_pool_floor,
            self.physical_cores // heuristics.warm_pool_divisor,
        )
        warm_cap = max(heuristics.warm_pool_floor, heuristics.warm_pool_cap)
        return min(warm_cap, warm_candidates)

    def to_snapshot(self) -> dict[str, float | int]:
        """Return a telemetry-friendly representation of the profile."""

        return {
            "physical_cores": self.physical_cores,
            "logical_cpus": self.logical_cpus,
            "total_memory_gb": self.total_memory_gb,
            "gpu_count": self.gpu_count,
        }
