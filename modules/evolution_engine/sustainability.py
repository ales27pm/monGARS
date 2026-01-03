"""Carbon-aware scheduling heuristics for evolution engine rollouts."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CarbonAwareDecision:
    """Outcome of evaluating whether a rollout should proceed."""

    should_proceed: bool
    reason: str
    carbon_intensity_g_co2_per_kwh: float | None = None
    energy_window_wh: float | None = None
    approvals_pending: int | None = None
    incidents: int | None = None
    recommended_delay: timedelta | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_logging_context(self) -> dict[str, Any]:
        """Return a log-friendly representation of the decision."""

        payload: dict[str, Any] = {
            "should_proceed": self.should_proceed,
            "reason": self.reason,
        }
        if self.carbon_intensity_g_co2_per_kwh is not None:
            payload["carbon_intensity_g_co2_per_kwh"] = round(
                float(self.carbon_intensity_g_co2_per_kwh), 2
            )
        if self.energy_window_wh is not None:
            payload["energy_window_wh"] = round(float(self.energy_window_wh), 2)
        if self.approvals_pending is not None:
            payload["approvals_pending"] = int(self.approvals_pending)
        if self.incidents is not None:
            payload["incidents"] = int(self.incidents)
        if self.recommended_delay is not None:
            payload["recommended_delay_seconds"] = int(
                self.recommended_delay.total_seconds()
            )
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


class CarbonAwarePolicy:
    """Interpret sustainability telemetry to gate training rollouts."""

    def __init__(
        self,
        dashboard_path: str | Path,
        *,
        carbon_pause_threshold: float = 550.0,
        carbon_caution_threshold: float = 400.0,
        energy_budget_wh: float = 7500.0,
        energy_window: timedelta = timedelta(hours=12),
        approvals_threshold: int = 10,
        cooldown: timedelta = timedelta(hours=2),
        caution_delay_factor: float = 0.5,
        stale_data_threshold: timedelta = timedelta(hours=6),
        incident_blocks: bool = True,
    ) -> None:
        if carbon_pause_threshold <= 0:
            raise ValueError("carbon_pause_threshold must be positive")
        if carbon_caution_threshold <= 0:
            raise ValueError("carbon_caution_threshold must be positive")
        if carbon_pause_threshold < carbon_caution_threshold:
            raise ValueError(
                "carbon_pause_threshold must be greater than caution threshold"
            )
        self._path = Path(dashboard_path)
        self._carbon_pause_threshold = float(carbon_pause_threshold)
        self._carbon_caution_threshold = float(carbon_caution_threshold)
        self._energy_budget_wh = float(energy_budget_wh)
        self._energy_window = energy_window
        self._approvals_threshold = int(max(0, approvals_threshold))
        self._cooldown = cooldown
        self._caution_delay_factor = max(0.1, float(caution_delay_factor))
        self._stale_data_threshold = stale_data_threshold
        self._incident_blocks = incident_blocks

    def evaluate(
        self,
        scope: str | None = None,
        *,
        now: datetime | None = None,
    ) -> CarbonAwareDecision:
        """Decide whether a rollout should proceed based on telemetry."""

        snapshot = self._load_dashboard()
        now = now or datetime.now(timezone.utc)
        reports = self._prepare_reports(snapshot.get("energy_reports", ()))
        scoped_reports = self._filter_by_scope(reports, scope)
        summaries = self._prepare_summaries(snapshot)
        latest_summary = self._select_latest_summary(summaries)

        latest_carbon_intensity, latest_timestamp = self._latest_carbon(scoped_reports)
        energy_window_wh = self._energy_in_window(scoped_reports, now)
        approvals_pending, incident_count = self._summarise_summary(latest_summary)

        reasons: list[str] = []
        should_proceed = True
        recommended_delay: timedelta | None = None

        if (
            latest_timestamp is not None
            and self._stale_data_threshold.total_seconds() > 0
            and now - latest_timestamp > self._stale_data_threshold
        ):
            reasons.append(
                "sustainability telemetry stale; using caution thresholds only"
            )

        if latest_carbon_intensity is not None:
            if latest_carbon_intensity >= self._carbon_pause_threshold:
                should_proceed = False
                recommended_delay = self._max_delay(recommended_delay, self._cooldown)
                reasons.append(
                    (
                        "carbon intensity %.1f gCO2/kWh exceeds pause threshold %.1f"
                        % (latest_carbon_intensity, self._carbon_pause_threshold)
                    )
                )
            elif latest_carbon_intensity >= self._carbon_caution_threshold:
                should_proceed = False
                caution_delay_seconds = max(
                    60,
                    int(self._cooldown.total_seconds() * self._caution_delay_factor),
                )
                recommended_delay = self._max_delay(
                    recommended_delay, timedelta(seconds=caution_delay_seconds)
                )
                reasons.append(
                    (
                        "carbon intensity %.1f gCO2/kWh above caution threshold %.1f"
                        % (latest_carbon_intensity, self._carbon_caution_threshold)
                    )
                )

        if (
            self._energy_window.total_seconds() > 0
            and energy_window_wh is not None
            and energy_window_wh > self._energy_budget_wh
        ):
            should_proceed = False
            recommended_delay = self._max_delay(recommended_delay, self._cooldown)
            reasons.append(
                (
                    "energy budget exceeded: %.1f Wh in window (limit %.1f Wh)"
                    % (energy_window_wh, self._energy_budget_wh)
                )
            )

        if (
            approvals_pending is not None
            and approvals_pending > self._approvals_threshold
        ):
            should_proceed = False
            recommended_delay = self._max_delay(recommended_delay, self._cooldown)
            reasons.append(
                (
                    "pending approvals %d exceed threshold %d"
                    % (approvals_pending, self._approvals_threshold)
                )
            )

        if self._incident_blocks and incident_count:
            should_proceed = False
            recommended_delay = self._max_delay(recommended_delay, self._cooldown)
            reasons.append(
                ("recent reinforcement incidents detected (%d)" % incident_count)
            )

        if not reasons:
            reasons.append("carbon-aware policy clear")

        metadata: dict[str, Any] = {
            "energy_window_hours": round(
                self._energy_window.total_seconds() / 3600.0, 2
            ),
            "energy_budget_wh": self._energy_budget_wh,
            "carbon_pause_threshold": self._carbon_pause_threshold,
            "carbon_caution_threshold": self._carbon_caution_threshold,
            "approvals_threshold": self._approvals_threshold,
        }
        if latest_timestamp is not None:
            metadata["latest_recorded_at"] = latest_timestamp.isoformat()
        if scope:
            metadata["scope"] = scope

        decision = CarbonAwareDecision(
            should_proceed=should_proceed,
            reason="; ".join(reasons),
            carbon_intensity_g_co2_per_kwh=latest_carbon_intensity,
            energy_window_wh=energy_window_wh,
            approvals_pending=approvals_pending,
            incidents=incident_count,
            recommended_delay=recommended_delay,
            metadata=metadata,
        )
        logger.debug("carbon_policy.evaluate", extra=decision.as_logging_context())
        return decision

    def _load_dashboard(self) -> MutableMapping[str, Any]:
        if not self._path.exists():
            return {}
        try:
            content = self._path.read_text(encoding="utf-8")
            data = json.loads(content)
            if isinstance(data, Mapping):
                return dict(data)
        except Exception:  # pragma: no cover - defensive guard
            logger.debug(
                "carbon_policy.dashboard_read_failed",
                extra={"path": str(self._path)},
                exc_info=True,
            )
        return {}

    def _prepare_reports(
        self, entries: Iterable[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            mapped = dict(entry)
            timestamp = self._parse_timestamp(mapped.get("recorded_at"))
            if timestamp is None:
                continue
            mapped["recorded_at"] = timestamp
            try:
                mapped["energy_wh"] = float(mapped.get("energy_wh", 0.0))
            except (TypeError, ValueError):
                mapped["energy_wh"] = 0.0
            try:
                carbon = mapped.get("carbon_intensity_g_co2_per_kwh")
                mapped["carbon_intensity_g_co2_per_kwh"] = (
                    float(carbon) if carbon is not None else None
                )
            except (TypeError, ValueError):
                mapped["carbon_intensity_g_co2_per_kwh"] = None
            reports.append(mapped)
        reports.sort(key=lambda item: item["recorded_at"])
        return reports

    def _filter_by_scope(
        self, reports: Sequence[dict[str, Any]], scope: str | None
    ) -> list[dict[str, Any]]:
        if not scope:
            return list(reports)
        matched = [
            report
            for report in reports
            if str(report.get("scope", "")).startswith(scope)
        ]
        return matched or list(reports)

    def _prepare_summaries(self, snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        latest = snapshot.get("latest_reinforcement_summary")
        runs = snapshot.get("reinforcement_runs", [])
        candidates: list[Mapping[str, Any]] = []
        if isinstance(latest, Mapping):
            candidates.append(latest)
        if isinstance(runs, Sequence):
            candidates.extend(entry for entry in runs if isinstance(entry, Mapping))
        for entry in candidates:
            mapped = dict(entry)
            timestamp = self._parse_timestamp(
                mapped.get("recorded_at") or mapped.get("started_at")
            )
            if timestamp is None:
                continue
            mapped["recorded_at"] = timestamp
            mapped["approval_pending_final"] = _coerce_optional_int(
                mapped.get("approval_pending_final")
            )
            incidents = mapped.get("incidents")
            if isinstance(incidents, Sequence) and not isinstance(incidents, str):
                mapped["incidents"] = list(incidents)
            else:
                mapped["incidents"] = []
            summaries.append(mapped)
        summaries.sort(key=lambda item: item["recorded_at"])
        return summaries

    def _select_latest_summary(
        self, summaries: Sequence[dict[str, Any]]
    ) -> dict[str, Any] | None:
        if not summaries:
            return None
        return summaries[-1]

    def _latest_carbon(
        self, reports: Sequence[Mapping[str, Any]]
    ) -> tuple[float | None, datetime | None]:
        for report in reversed(reports):
            intensity = report.get("carbon_intensity_g_co2_per_kwh")
            if intensity is not None:
                return float(intensity), report.get("recorded_at")
        timestamp = reports[-1]["recorded_at"] if reports else None
        return None, timestamp

    def _energy_in_window(
        self, reports: Sequence[Mapping[str, Any]], now: datetime
    ) -> float | None:
        if not reports or self._energy_window.total_seconds() <= 0:
            return None
        window_start = now - self._energy_window
        total = 0.0
        for report in reports:
            recorded_at = report.get("recorded_at")
            if recorded_at is None or recorded_at < window_start:
                continue
            try:
                total += float(report.get("energy_wh", 0.0))
            except (TypeError, ValueError):
                continue
        return total

    def _summarise_summary(
        self, summary: Mapping[str, Any] | None
    ) -> tuple[int | None, int | None]:
        if not summary:
            return None, None
        approvals = _coerce_optional_int(summary.get("approval_pending_final"))
        incidents = summary.get("incidents")
        incident_count: int | None
        if isinstance(incidents, Sequence) and not isinstance(incidents, str):
            incident_count = len(tuple(incidents))
        else:
            incident_count = None
        return approvals if approvals is not None else None, incident_count

    def _parse_timestamp(self, value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc)
        if not value:
            return None
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value), tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                return None
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError:
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        return None

    def _max_delay(self, current: timedelta | None, candidate: timedelta) -> timedelta:
        if current is None:
            return candidate
        return max(current, candidate)


__all__ = ["CarbonAwareDecision", "CarbonAwarePolicy"]


def _coerce_optional_int(value: Any) -> int | None:
    """Best-effort conversion of telemetry counters to integers."""

    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except (OverflowError, ValueError):
            return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None
