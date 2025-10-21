"""Energy usage tracking utilities for evolution training runs."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Iterator

import psutil

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from codecarbon import EmissionsTracker as CodeCarbonTracker
else:
    CodeCarbonTracker = object

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency for detailed emissions data
    from codecarbon import EmissionsTracker
except Exception:  # pragma: no cover - dependency not available in many envs
    EmissionsTracker = None  # type: ignore


@dataclass(frozen=True)
class EnergyUsageReport:
    """Structured summary of energy consumption for a training run."""

    energy_wh: float
    duration_seconds: float
    cpu_seconds: float
    baseline_cpu_power_watts: float
    backend: str
    emissions_grams: float | None = None
    carbon_intensity_g_co2_per_kwh: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable representation for persistence/logging."""

        return asdict(self)


class EnergyTracker:
    """Track energy usage during a training session with graceful fallbacks."""

    def __init__(
        self,
        *,
        baseline_cpu_power_watts: float = 45.0,
        process: psutil.Process | None = None,
    ) -> None:
        self._baseline_cpu_power_watts = max(1.0, float(baseline_cpu_power_watts))
        self._process = process or psutil.Process()
        self._start_cpu: float | None = None
        self._start_time: float | None = None
        self._tracker: CodeCarbonTracker | None = None
        self._backend = "psutil"
        self._last_report: EnergyUsageReport | None = None

    def start(self) -> None:
        """Begin tracking energy usage for the current process."""

        self._start_time = time.perf_counter()
        try:
            cpu_times = self._process.cpu_times()
            self._start_cpu = float(cpu_times.user + cpu_times.system)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning(
                "energy_tracker.cpu_times_unavailable",
                extra={"error": str(exc)},
            )
            self._start_cpu = None

        if EmissionsTracker is None:
            return

        try:  # pragma: no cover - optional dependency branch
            tracker = EmissionsTracker(
                measure_power_secs=1,
                tracking_mode="process",
                save_to_file=False,
            )
            tracker.start()
            self._tracker = tracker
            self._backend = "codecarbon"
        except Exception as exc:  # pragma: no cover - optional dependency branch
            logger.warning(
                "energy_tracker.emissions_start_failed",
                extra={"error": str(exc)},
            )
            self._tracker = None
            self._backend = "psutil"

    def stop(self) -> EnergyUsageReport:
        """Stop tracking and return an energy usage report."""

        duration = max(
            0.0, time.perf_counter() - (self._start_time or time.perf_counter())
        )
        cpu_seconds = 0.0
        if self._start_cpu is not None:
            try:
                cpu_times = self._process.cpu_times()
                cpu_seconds = max(
                    0.0,
                    float(cpu_times.user + cpu_times.system) - self._start_cpu,
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning(
                    "energy_tracker.cpu_times_stop_failed",
                    extra={"error": str(exc)},
                )

        energy_wh = (cpu_seconds * self._baseline_cpu_power_watts) / 3600.0
        emissions_grams: float | None = None
        carbon_intensity: float | None = None
        backend = self._backend

        if self._tracker is not None:
            try:  # pragma: no cover - optional dependency branch
                data = self._tracker.stop()
                backend = "codecarbon"
                if data is not None:
                    energy_kwh = getattr(data, "energy_consumed", None)
                    if energy_kwh is not None:
                        energy_wh = float(energy_kwh) * 1000.0
                    emissions_kg = getattr(data, "emissions", None)
                    if emissions_kg is not None:
                        emissions_grams = float(emissions_kg) * 1000.0
                    emissions_rate = getattr(
                        data, "emissions_rate", carbon_intensity or 0.0
                    )
                    carbon_intensity = float(emissions_rate) * 1000.0
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning(
                    "energy_tracker.emissions_stop_failed",
                    extra={"error": str(exc)},
                )
                backend = "psutil"

        report = EnergyUsageReport(
            energy_wh=energy_wh,
            duration_seconds=duration,
            cpu_seconds=cpu_seconds,
            baseline_cpu_power_watts=self._baseline_cpu_power_watts,
            backend=backend,
            emissions_grams=emissions_grams,
            carbon_intensity_g_co2_per_kwh=carbon_intensity,
        )
        self._last_report = report
        return report

    @contextmanager
    def track(self) -> Iterator["EnergyTracker"]:
        """Context manager to track energy usage for a block."""

        self.start()
        try:
            yield self
        finally:
            try:
                self.stop()
            except Exception:  # pragma: no cover - defensive guard
                logger.exception("energy_tracker.stop_failed")

    @property
    def last_report(self) -> EnergyUsageReport | None:
        """Return the most recent energy usage report, if available."""

        return self._last_report
