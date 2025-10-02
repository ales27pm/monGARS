"""Hardware awareness helpers for the evolution engine."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import psutil

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

    @classmethod
    def detect(cls) -> "HardwareProfile":
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

        return cls(
            physical_cores=max(1, physical),
            logical_cpus=max(physical, logical),
            total_memory_gb=max(0.5, float(memory)),
            gpu_count=max(0, gpu_count),
        )

    def estimate_training_power_draw(self) -> float:
        """Estimate watts consumed by a typical training run on this host."""

        base = 20.0 + 5.0 * self.physical_cores
        if self.gpu_count:
            base += 75.0 * self.gpu_count
        if self.total_memory_gb < 8.0:
            base *= 0.8
        return max(15.0, base)

    def max_recommended_workers(self, configured_default: int) -> int:
        """Return an upper bound for worker replicas based on hardware."""

        baseline = max(1, configured_default)
        cpu_capacity = max(1, self.logical_cpus // 2)
        memory_limited = 1 if self.total_memory_gb < 4.0 else 2
        gpu_bonus = self.gpu_count * 2
        cap = max(baseline, cpu_capacity + gpu_bonus)
        if self.total_memory_gb < 8.0:
            cap = min(cap, baseline + memory_limited)
        return max(1, cap)

    def min_recommended_workers(self) -> int:
        """Return the minimum number of replicas to keep warm."""

        if self.total_memory_gb <= 2.0:
            return 1
        return min(2, max(1, self.physical_cores // 4 or 1))

    def to_snapshot(self) -> dict[str, float | int]:
        """Return a telemetry-friendly representation of the profile."""

        return asdict(self)
