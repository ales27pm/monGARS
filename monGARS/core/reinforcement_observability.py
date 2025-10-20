"""Durable observability store for reinforcement-learning validation runs."""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from monGARS.core.long_haul_validation import (
    LongHaulCycleReport,
    LongHaulValidationSummary,
    ReplicaLoadReport,
    ReplicaTimelineEntry,
)

logger = logging.getLogger(__name__)


class ReinforcementObservabilityStore:
    """Persist correlated telemetry for reinforcement-learning runs."""

    def __init__(self, storage_path: str | Path, *, max_records: int = 50) -> None:
        self._path = Path(storage_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._max_records = max(1, int(max_records))

    def record_summary(self, summary: LongHaulValidationSummary) -> None:
        """Persist ``summary`` for downstream dashboards."""

        try:
            runs = self._load().get("runs", [])
            runs.append(self._build_record(summary))
            if len(runs) > self._max_records:
                runs = runs[-self._max_records:]
            payload = {
                "meta": {
                    "version": 1,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                "runs": runs,
            }
            self._write(payload)
        except Exception:  # pragma: no cover - persistence must not break validation
            logger.exception(
                "reinforcement.observability.persist_failed",
                extra={"path": str(self._path)},
            )

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return {"runs": []}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning(
                "reinforcement.observability.load_failed",
                extra={"error": str(exc), "path": str(self._path)},
            )
            return {"runs": []}
        if not isinstance(raw, Mapping):
            return {"runs": []}
        runs = raw.get("runs")
        if isinstance(runs, list):
            return {"runs": runs}
        return {"runs": []}

    def _write(self, payload: Mapping[str, Any]) -> None:
        self._path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _build_record(self, summary: LongHaulValidationSummary) -> dict[str, Any]:
        cycles = [self._serialise_cycle(cycle) for cycle in summary.cycles]
        energy_series = [cycle.get("energy_wh") for cycle in cycles]
        approvals_series = [cycle.get("approval_pending") for cycle in cycles]
        record = {
            "started_at": summary.started_at,
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
            "energy_per_cycle": energy_series,
            "approvals_per_cycle": approvals_series,
            "replica_overview": self._aggregate_replica_overview(summary.cycles),
            "cycles": cycles,
        }
        return record

    def _serialise_cycle(self, cycle: LongHaulCycleReport) -> dict[str, Any]:
        payload = {
            "index": cycle.index,
            "status": cycle.status,
            "episodes": cycle.episodes,
            "total_reward": cycle.total_reward,
            "average_reward": cycle.average_reward,
            "failures": cycle.failures,
            "duration_seconds": cycle.duration_seconds,
            "energy_wh": cycle.energy_wh,
            "approval_pending": cycle.approval_pending,
            "mnpt_executed": cycle.mnpt_executed,
            "incidents": list(cycle.incidents),
        }
        replica_payload = self._serialise_replica_load(cycle.replica_load)
        if replica_payload is not None:
            payload["replica_load"] = replica_payload
        return payload

    def _serialise_replica_load(
        self, load: ReplicaLoadReport | None
    ) -> dict[str, Any] | None:
        if load is None:
            return None
        timeline = [
            {
                "batch_index": entry.batch_index,
                "worker_count": entry.worker_count,
                "reason": entry.reason,
            }
            for entry in load.timeline
        ]
        if not any(
            [
                load.peak is not None,
                load.low is not None,
                load.average is not None,
                load.events,
                timeline,
            ]
        ):
            return None
        return {
            "peak": load.peak,
            "low": load.low,
            "average": load.average,
            "events": load.events,
            "reasons": dict(load.reasons),
            "timeline": timeline,
        }

    def _aggregate_replica_overview(
        self, cycles: Sequence[LongHaulCycleReport]
    ) -> dict[str, Any]:
        counts: list[int] = []
        reasons: Counter[str] = Counter()
        cycles_reporting = 0
        for cycle in cycles:
            load = cycle.replica_load
            if load is None:
                continue
            timeline: Sequence[ReplicaTimelineEntry] = load.timeline
            if timeline:
                cycles_reporting += 1
            for entry in timeline:
                counts.append(int(entry.worker_count))
            for reason, value in load.reasons.items():
                reasons[str(reason)] += int(value)
        if not counts:
            return {}
        average = sum(counts) / len(counts)
        return {
            "peak": max(counts),
            "low": min(counts),
            "average": average,
            "events": len(counts),
            "reasons": dict(reasons),
            "cycles_reporting": cycles_reporting,
        }


__all__ = ["ReinforcementObservabilityStore"]
