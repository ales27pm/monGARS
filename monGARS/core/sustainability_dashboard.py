"""Bridge evolution energy telemetry with sustainability dashboards."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from opentelemetry import metrics

from modules.evolution_engine.energy import EnergyUsageReport
from monGARS.core.long_haul_validation import (
    LongHaulCycleReport,
    LongHaulValidationSummary,
)

logger = logging.getLogger(__name__)

_meter = metrics.get_meter(__name__)

_energy_wh_histogram = _meter.create_histogram(
    "llm.sustainability.energy.wh",
    unit="Wh",
    description="Energy consumed by reinforcement and evolution workloads.",
)
_energy_duration_histogram = _meter.create_histogram(
    "llm.sustainability.energy.duration_seconds",
    unit="s",
    description="Wall clock duration associated with recorded energy reports.",
)
_cpu_seconds_histogram = _meter.create_histogram(
    "llm.sustainability.energy.cpu_seconds",
    unit="s",
    description="CPU seconds captured by the energy tracker.",
)
_emissions_histogram = _meter.create_histogram(
    "llm.sustainability.energy.emissions_grams",
    unit="g",
    description="Carbon emissions associated with recorded energy usage.",
)
_carbon_intensity_histogram = _meter.create_histogram(
    "llm.sustainability.energy.carbon_intensity_gco2_per_kwh",
    unit="gCO2/kWh",
    description="Carbon intensity observed during recorded energy usage.",
)
_success_rate_histogram = _meter.create_histogram(
    "llm.sustainability.reinforcement.success_rate",
    description="Success rate from reinforcement long-haul validation runs.",
)
_reward_histogram = _meter.create_histogram(
    "llm.sustainability.reinforcement.total_reward",
    description="Total reward accumulated during reinforcement validation runs.",
)
_reinforcement_energy_histogram = _meter.create_histogram(
    "llm.sustainability.reinforcement.energy_wh",
    unit="Wh",
    description="Total energy consumed during reinforcement validation runs.",
)
_approval_histogram = _meter.create_histogram(
    "llm.sustainability.reinforcement.approval_pending",
    description="Pending approval counts observed at the end of validation runs.",
)
_incident_histogram = _meter.create_histogram(
    "llm.sustainability.reinforcement.incident_count",
    description="Incident counts surfaced during reinforcement validation runs.",
)
_mnpt_runs_histogram = _meter.create_histogram(
    "llm.sustainability.reinforcement.mnpt_runs",
    description="Number of MNTP runs executed inside reinforcement validation cycles.",
)
_replica_peak_histogram = _meter.create_histogram(
    "llm.sustainability.reinforcement.replica_peak",
    description="Peak worker counts observed in reinforcement validation runs.",
)
_replica_average_histogram = _meter.create_histogram(
    "llm.sustainability.reinforcement.replica_average",
    description="Average worker counts observed in reinforcement validation runs.",
)


class SustainabilityDashboardBridge:
    """Persist sustainability telemetry and emit metrics for dashboards."""

    def __init__(
        self,
        storage_path: str | Path,
        *,
        observability_path: str | Path | None = None,
        max_energy_reports: int = 200,
        max_reinforcement_records: int = 50,
    ) -> None:
        self._path = Path(storage_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._observability_path = (
            Path(observability_path) if observability_path is not None else None
        )
        self._max_energy_reports = max(1, int(max_energy_reports))
        self._max_reinforcement_records = max(1, int(max_reinforcement_records))

    def record_energy_report(
        self,
        report: EnergyUsageReport,
        *,
        scope: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist ``report`` and emit metrics for dashboards."""

        entry: MutableMapping[str, Any] = {
            "scope": scope,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "energy_wh": float(report.energy_wh),
            "duration_seconds": float(report.duration_seconds),
            "cpu_seconds": float(report.cpu_seconds),
            "baseline_cpu_power_watts": float(report.baseline_cpu_power_watts),
            "backend": report.backend,
        }
        if report.emissions_grams is not None:
            entry["emissions_grams"] = float(report.emissions_grams)
        if report.carbon_intensity_g_co2_per_kwh is not None:
            entry["carbon_intensity_g_co2_per_kwh"] = float(
                report.carbon_intensity_g_co2_per_kwh
            )
        if metadata:
            normalised = self._normalise_metadata(metadata)
            if normalised:
                entry["metadata"] = normalised

        self._record_energy_metrics(entry)
        payload = self._load()
        reports = payload.setdefault("energy_reports", [])
        reports.append(entry)
        if len(reports) > self._max_energy_reports:
            payload["energy_reports"] = reports[-self._max_energy_reports:]
        payload.setdefault("meta", {})["updated_at"] = datetime.now(
            timezone.utc
        ).isoformat()
        self._write(payload)

    def record_reinforcement_summary(
        self,
        summary: LongHaulValidationSummary,
        *,
        scope: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist ``summary`` highlights and emit reinforcement metrics."""

        record = self._summarise_summary(summary, scope, metadata)
        self._record_summary_metrics(summary, record)

        payload = self._load()
        runs = payload.setdefault("reinforcement_runs", [])
        runs.append(record)
        if len(runs) > self._max_reinforcement_records:
            payload["reinforcement_runs"] = runs[-self._max_reinforcement_records:]
        payload["latest_reinforcement_summary"] = record
        meta = payload.setdefault("meta", {})
        meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        if self._observability_path is not None:
            payload.setdefault("references", {})["reinforcement_observability_path"] = (
                str(self._observability_path)
            )
        self._write(payload)

    def _record_energy_metrics(self, entry: Mapping[str, Any]) -> None:
        try:
            _energy_wh_histogram.record(float(entry["energy_wh"]))
            _energy_duration_histogram.record(float(entry["duration_seconds"]))
            _cpu_seconds_histogram.record(float(entry["cpu_seconds"]))
            emissions = entry.get("emissions_grams")
            if emissions is not None:
                _emissions_histogram.record(float(emissions))
            carbon = entry.get("carbon_intensity_g_co2_per_kwh")
            if carbon is not None:
                _carbon_intensity_histogram.record(float(carbon))
        except Exception:  # pragma: no cover - metrics recording best effort
            logger.debug(
                "sustainability.energy.metrics_failed",
                extra={"scope": entry.get("scope", "unknown")},
                exc_info=True,
            )

    def _record_summary_metrics(
        self,
        summary: LongHaulValidationSummary,
        record: Mapping[str, Any],
    ) -> None:
        try:
            _success_rate_histogram.record(float(summary.success_rate))
            _reward_histogram.record(float(summary.total_reward))
            if summary.energy_wh is not None:
                _reinforcement_energy_histogram.record(float(summary.energy_wh))
            if summary.approval_pending_final is not None:
                _approval_histogram.record(float(summary.approval_pending_final))
            _incident_histogram.record(float(len(summary.incidents)))
            _mnpt_runs_histogram.record(float(summary.mnpt_runs))
            replica_overview = record.get("replica_overview") or {}
            peak = replica_overview.get("peak")
            average = replica_overview.get("average")
            if peak is not None:
                _replica_peak_histogram.record(float(peak))
            if average is not None:
                _replica_average_histogram.record(float(average))
        except Exception:  # pragma: no cover - metrics recording best effort
            logger.debug(
                "sustainability.reinforcement.metrics_failed",
                extra={"scope": record.get("scope", "unknown")},
                exc_info=True,
            )

    def _summarise_summary(
        self,
        summary: LongHaulValidationSummary,
        scope: str,
        metadata: Mapping[str, Any] | None,
    ) -> MutableMapping[str, Any]:
        record: MutableMapping[str, Any] = {
            "scope": scope,
            "started_at": summary.started_at,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": summary.duration_seconds,
            "total_cycles": summary.total_cycles,
            "total_episodes": summary.total_episodes,
            "total_reward": summary.total_reward,
            "average_reward": summary.average_reward,
            "total_failures": summary.total_failures,
            "success_rate": summary.success_rate,
            "energy_wh": summary.energy_wh,
            "approval_pending_final": summary.approval_pending_final,
            "mnpt_runs": summary.mnpt_runs,
            "incidents": list(summary.incidents),
        }
        if metadata:
            normalised = self._normalise_metadata(metadata)
            if normalised:
                record["metadata"] = normalised
        record["energy_per_cycle"] = [
            cycle.energy_wh for cycle in summary.cycles if cycle.energy_wh is not None
        ]
        record["approvals_per_cycle"] = [
            cycle.approval_pending
            for cycle in summary.cycles
            if cycle.approval_pending is not None
        ]
        record["replica_overview"] = self._summarise_replica_overview(summary.cycles)
        return record

    def _summarise_replica_overview(
        self, cycles: Sequence[LongHaulCycleReport]
    ) -> MutableMapping[str, Any]:
        counts: list[float] = []
        reasons: dict[str, int] = {}
        cycles_reporting = 0
        for cycle in cycles:
            load = cycle.replica_load
            timeline = getattr(load, "timeline", ())
            if timeline:
                cycles_reporting += 1
            for entry in timeline:
                counts.append(float(getattr(entry, "worker_count", 0)))
            for reason, value in getattr(load, "reasons", {}).items():
                reasons[str(reason)] = reasons.get(str(reason), 0) + int(value)
        if not counts:
            return {}
        average = sum(counts) / len(counts)
        return {
            "peak": max(counts),
            "low": min(counts),
            "average": average,
            "events": len(counts),
            "reasons": reasons,
            "cycles_reporting": cycles_reporting,
        }

    def _normalise_metadata(
        self, metadata: Mapping[str, Any]
    ) -> MutableMapping[str, Any]:
        normalised: MutableMapping[str, Any] = {}
        for key, value in metadata.items():
            key_str = str(key)
            if value is None or isinstance(value, (bool, int, float, str)):
                normalised[key_str] = value
                continue
            try:
                normalised[key_str] = float(value)  # type: ignore[assignment]
            except (TypeError, ValueError):
                normalised[key_str] = str(value)
        return normalised

    def _load(self) -> MutableMapping[str, Any]:
        if not self._path.exists():
            return self._bootstrap_payload()
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return raw
        except Exception:  # pragma: no cover - defensive guard
            logger.debug(
                "sustainability.dashboard.load_failed",
                extra={"path": str(self._path)},
                exc_info=True,
            )
        return self._bootstrap_payload()

    def _write(self, payload: Mapping[str, Any]) -> None:
        meta = payload.setdefault("meta", {})
        meta.setdefault("version", 1)
        meta.setdefault("updated_at", datetime.now(timezone.utc).isoformat())
        if self._observability_path is not None:
            payload.setdefault("references", {})["reinforcement_observability_path"] = (
                str(self._observability_path)
            )
        self._path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _bootstrap_payload(self) -> MutableMapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "meta": {
                "version": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            "energy_reports": [],
            "reinforcement_runs": [],
        }
        if self._observability_path is not None:
            payload["references"] = {
                "reinforcement_observability_path": str(self._observability_path)
            }
        return payload


__all__ = ["SustainabilityDashboardBridge"]
