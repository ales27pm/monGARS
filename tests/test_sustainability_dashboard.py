from __future__ import annotations

import json
from pathlib import Path

import pytest

from modules.evolution_engine.energy import EnergyUsageReport
from monGARS.core.long_haul_validation import (
    LongHaulCycleReport,
    LongHaulValidationSummary,
    ReplicaLoadReport,
    ReplicaTimelineEntry,
)
from monGARS.core.sustainability_dashboard import SustainabilityDashboardBridge


def _make_summary() -> LongHaulValidationSummary:
    cycle0 = LongHaulCycleReport(
        index=0,
        status="completed",
        episodes=4,
        total_reward=2.4,
        average_reward=0.6,
        failures=1,
        duration_seconds=1.2,
        energy_wh=0.5,
        approval_pending=2,
        incidents=(),
        mnpt_executed=True,
        replica_load=ReplicaLoadReport(
            peak=4,
            low=2,
            average=3.0,
            events=3,
            reasons={"initial": 1, "scale_up": 2},
            timeline=(
                ReplicaTimelineEntry(batch_index=0, worker_count=2, reason="initial"),
                ReplicaTimelineEntry(batch_index=1, worker_count=4, reason="scale_up"),
            ),
        ),
    )
    cycle1 = LongHaulCycleReport(
        index=1,
        status="completed",
        episodes=4,
        total_reward=3.2,
        average_reward=0.8,
        failures=0,
        duration_seconds=1.1,
        energy_wh=0.7,
        approval_pending=1,
        incidents=(),
        mnpt_executed=False,
        replica_load=ReplicaLoadReport(
            peak=3,
            low=2,
            average=2.5,
            events=2,
            reasons={"initial": 1, "scale_down": 1},
            timeline=(
                ReplicaTimelineEntry(batch_index=0, worker_count=3, reason="initial"),
                ReplicaTimelineEntry(
                    batch_index=1, worker_count=2, reason="scale_down"
                ),
            ),
        ),
    )
    return LongHaulValidationSummary(
        started_at="2024-01-01T00:00:00+00:00",
        duration_seconds=2.3,
        total_cycles=2,
        total_episodes=8,
        total_reward=5.6,
        average_reward=0.7,
        total_failures=1,
        success_rate=0.875,
        energy_wh=1.2,
        approval_pending_final=1,
        mnpt_runs=2,
        cycles=[cycle0, cycle1],
        incidents=(),
    )


def test_bridge_records_energy_and_summary(tmp_path: Path) -> None:
    storage = tmp_path / "sustainability.json"
    observability = tmp_path / "reinforcement_observability.json"
    observability.write_text("{}", encoding="utf-8")
    bridge = SustainabilityDashboardBridge(
        storage,
        observability_path=observability,
        max_energy_reports=2,
        max_reinforcement_records=2,
    )

    energy_report = EnergyUsageReport(
        energy_wh=0.9,
        duration_seconds=1.5,
        cpu_seconds=0.8,
        baseline_cpu_power_watts=25.0,
        backend="test",
        emissions_grams=12.0,
        carbon_intensity_g_co2_per_kwh=350.0,
    )

    bridge.record_energy_report(
        energy_report,
        scope="reinforcement.longhaul.cycle",
        metadata={"cycle_index": 0, "status": "completed"},
    )

    summary = _make_summary()
    bridge.record_reinforcement_summary(
        summary,
        scope="reinforcement.longhaul.summary",
        metadata={"cycles": summary.total_cycles},
    )

    payload = json.loads(storage.read_text(encoding="utf-8"))
    assert payload["energy_reports"], "energy reports should be persisted"
    energy_entry = payload["energy_reports"][0]
    assert energy_entry["energy_wh"] == pytest.approx(0.9)
    assert energy_entry["metadata"]["cycle_index"] == 0
    assert payload["latest_reinforcement_summary"]["total_reward"] == pytest.approx(5.6)
    replica_overview = payload["latest_reinforcement_summary"]["replica_overview"]
    assert replica_overview["peak"] == 4
    assert payload["references"]["reinforcement_observability_path"].endswith(
        "reinforcement_observability.json"
    )


def test_bridge_enforces_history_limits(tmp_path: Path) -> None:
    storage = tmp_path / "sustainability.json"
    bridge = SustainabilityDashboardBridge(
        storage, max_energy_reports=2, max_reinforcement_records=2
    )

    for idx in range(3):
        report = EnergyUsageReport(
            energy_wh=idx + 1.0,
            duration_seconds=1.0,
            cpu_seconds=0.5,
            baseline_cpu_power_watts=20.0,
            backend="test",
        )
        bridge.record_energy_report(
            report, scope="cycle", metadata={"cycle_index": idx}
        )

    for idx in range(3):
        summary = _make_summary()
        bridge.record_reinforcement_summary(summary, scope=f"summary-{idx}")

    payload = json.loads(storage.read_text(encoding="utf-8"))
    assert len(payload["energy_reports"]) == 2
    assert payload["energy_reports"][0]["metadata"]["cycle_index"] == 1
    assert len(payload["reinforcement_runs"]) == 2
    assert payload["reinforcement_runs"][0]["scope"] == "summary-1"
