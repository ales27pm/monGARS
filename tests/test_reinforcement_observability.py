from __future__ import annotations

import json
from pathlib import Path

from monGARS.core.long_haul_validation import (
    LongHaulCycleReport,
    LongHaulValidationSummary,
    ReplicaLoadReport,
    ReplicaTimelineEntry,
)
from monGARS.core.reinforcement_observability import ReinforcementObservabilityStore


def _make_cycle(index: int) -> LongHaulCycleReport:
    return LongHaulCycleReport(
        index=index,
        status="completed",
        episodes=4,
        total_reward=3.2 + index,
        average_reward=0.8 + index * 0.01,
        failures=0,
        duration_seconds=1.5 + index,
        energy_wh=0.5 + (index * 0.1),
        approval_pending=2 - index,
        incidents=(),
        mnpt_executed=bool(index % 2 == 0),
        replica_load=ReplicaLoadReport(
            peak=3 + index,
            low=2,
            average=2.5 + index * 0.1,
            events=2,
            reasons={"initial": 1, "scale_up": 1 + index},
            timeline=(
                ReplicaTimelineEntry(batch_index=0, worker_count=2, reason="initial"),
                ReplicaTimelineEntry(
                    batch_index=1, worker_count=3 + index, reason="scale_up"
                ),
            ),
        ),
    )


def _make_summary() -> LongHaulValidationSummary:
    return LongHaulValidationSummary(
        started_at="2024-01-01T00:00:00+00:00",
        duration_seconds=3.4,
        total_cycles=2,
        total_episodes=8,
        total_reward=6.8,
        average_reward=0.85,
        total_failures=0,
        success_rate=1.0,
        energy_wh=1.1,
        approval_pending_final=1,
        mnpt_runs=2,
        cycles=[_make_cycle(0), _make_cycle(1)],
        incidents=(),
    )


def test_observability_store_persists_runs(tmp_path: Path) -> None:
    storage = tmp_path / "reinforcement_observability.json"
    store = ReinforcementObservabilityStore(storage, max_records=1)

    summary = _make_summary()
    store.record_summary(summary)

    payload = json.loads(storage.read_text(encoding="utf-8"))
    assert payload["runs"], "expected at least one run persisted"
    run = payload["runs"][0]
    assert run["energy_wh"] == summary.energy_wh
    assert run["replica_overview"]["peak"] == 4
    assert run["replica_overview"]["cycles_reporting"] == 2
    assert run["cycles"][0]["replica_load"]["timeline"][0]["worker_count"] == 2

    # Recording another summary should respect the max_records limit
    store.record_summary(summary)
    payload = json.loads(storage.read_text(encoding="utf-8"))
    assert len(payload["runs"]) == 1
    assert payload["runs"][0]["mnpt_runs"] == summary.mnpt_runs
