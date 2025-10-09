from __future__ import annotations

import contextlib
from collections import deque
from typing import Any

import pytest

from modules.evolution_engine.energy import EnergyUsageReport
from modules.neurons.training.reinforcement_loop import (
    ReinforcementLearningSummary,
    WorkerAdjustment,
)
from monGARS.core.long_haul_validation import (
    LongHaulCycleReport,
    LongHaulValidationSummary,
    ResearchLoopLongHaulValidator,
)
from monGARS.core.operator_approvals import OperatorApprovalRegistry


class _RecordingSpan:
    def __init__(self, name: str) -> None:
        self.name = name
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.attributes: dict[str, Any] = {}

    def __enter__(self) -> "_RecordingSpan":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append((name, attributes or {}))

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value


class _RecordingTracer:
    def __init__(self) -> None:
        self.spans: list[_RecordingSpan] = []

    def start_as_current_span(self, name: str) -> _RecordingSpan:
        span = _RecordingSpan(name)
        self.spans.append(span)
        return span


class _StubReinforcementLoop:
    def __init__(
        self, expected_episodes: int, summary: ReinforcementLearningSummary
    ) -> None:
        self.expected_episodes = expected_episodes
        self.summary = summary
        self.calls = 0

    def run(self, total_episodes: int) -> ReinforcementLearningSummary:
        if total_episodes != self.expected_episodes:
            raise AssertionError(
                f"Expected {self.expected_episodes} episodes but received {total_episodes}"
            )
        self.calls += 1
        return self.summary


class _EnergyTracker:
    def __init__(self, energy_queue: deque[float]) -> None:
        self._energy_queue = energy_queue
        self.last_report: EnergyUsageReport | None = None

    @contextlib.contextmanager
    def track(self) -> Any:
        try:
            yield self
        finally:
            energy = self._energy_queue.popleft()
            self.last_report = EnergyUsageReport(
                energy_wh=energy,
                duration_seconds=1.0,
                cpu_seconds=0.5,
                baseline_cpu_power_watts=10.0,
                backend="test",
            )


class _RecordingObservabilityStore:
    def __init__(self) -> None:
        self.records: list[LongHaulValidationSummary] = []

    def record_summary(self, summary: LongHaulValidationSummary) -> None:
        self.records.append(summary)


@pytest.mark.asyncio
async def test_long_haul_validator_collects_cycle_metrics(tmp_path) -> None:
    approvals_path = tmp_path / "approvals.json"
    registry = OperatorApprovalRegistry(approvals_path)
    registry.submit(source="reinforcement.reasoning", payload={"version": 1})

    summaries = deque(
        [
            ReinforcementLearningSummary(
                total_episodes=12,
                total_reward=7.2,
                average_reward=0.72,
                episode_results=[],
                failures=2,
                wall_clock_seconds=0.4,
                worker_history=[
                    WorkerAdjustment(batch_index=0, worker_count=2, reason="initial"),
                    WorkerAdjustment(batch_index=1, worker_count=3, reason="scale_up"),
                ],
            ),
            ReinforcementLearningSummary(
                total_episodes=12,
                total_reward=8.4,
                average_reward=0.84,
                episode_results=[],
                failures=1,
                wall_clock_seconds=0.5,
                worker_history=[
                    WorkerAdjustment(batch_index=0, worker_count=3, reason="initial"),
                    WorkerAdjustment(
                        batch_index=1, worker_count=2, reason="scale_down"
                    ),
                ],
            ),
        ]
    )

    def loop_factory() -> _StubReinforcementLoop:
        summary = summaries.popleft()
        return _StubReinforcementLoop(expected_episodes=12, summary=summary)

    energy_values = deque([1.2, 1.6])

    def energy_factory() -> _EnergyTracker:
        return _EnergyTracker(energy_values)

    metrics: list[tuple[str, dict[str, float | int]]] = []

    def metrics_sink(name: str, payload: dict[str, float | int]) -> None:
        metrics.append((name, dict(payload)))

    tracer = _RecordingTracer()
    mnpt_calls = 0

    async def mnpt_callback() -> None:
        nonlocal mnpt_calls
        mnpt_calls += 1

    observability_store = _RecordingObservabilityStore()

    validator = ResearchLoopLongHaulValidator(
        reinforcement_loop_factory=loop_factory,
        approval_registry=registry,
        energy_tracker_factory=energy_factory,
        metrics_sink=metrics_sink,
        tracer_factory=lambda _: tracer,
        mnpt_callback=mnpt_callback,
        observability_store=observability_store,
    )

    summary = await validator.execute(
        cycles=2, episodes_per_cycle=12, cooldown_seconds=0.0
    )

    assert isinstance(summary, LongHaulValidationSummary)
    assert summary.total_cycles == 2
    assert summary.total_episodes == 24
    assert summary.total_reward == pytest.approx(15.6)
    assert summary.total_failures == 3
    assert summary.energy_wh == pytest.approx(2.8)
    assert summary.approval_pending_final == 1
    assert summary.mnpt_runs == 2
    assert summary.incidents == ()
    assert mnpt_calls == 2

    assert all(isinstance(report, LongHaulCycleReport) for report in summary.cycles)
    assert summary.cycles[0].mnpt_executed is True
    assert summary.cycles[1].energy_wh == pytest.approx(1.6)
    assert summary.cycles[0].replica_load.peak == 3
    assert summary.cycles[1].replica_load.low == 2
    assert summary.cycles[0].replica_load.reasons["scale_up"] == 1
    assert observability_store.records and observability_store.records[0] is summary

    cycle_metrics = [name for name, _ in metrics if name == "research.longhaul.cycle"]
    assert len(cycle_metrics) == 2
    assert metrics[-1][0] == "research.longhaul.summary"

    assert tracer.spans and tracer.spans[0].events
    assert tracer.spans[0].events[0][0] == "cycle.completed"


@pytest.mark.asyncio
async def test_long_haul_validator_records_failures(tmp_path) -> None:
    approvals_path = tmp_path / "approvals.json"
    registry = OperatorApprovalRegistry(approvals_path)

    def failing_factory() -> _StubReinforcementLoop:
        raise RuntimeError("failed to construct loop")

    metrics: list[tuple[str, dict[str, float | int]]] = []

    def metrics_sink(name: str, payload: dict[str, float | int]) -> None:
        metrics.append((name, dict(payload)))

    energy_values = deque([0.0])

    def energy_factory() -> _EnergyTracker:
        return _EnergyTracker(energy_values)

    observability_store = _RecordingObservabilityStore()

    validator = ResearchLoopLongHaulValidator(
        reinforcement_loop_factory=failing_factory,
        approval_registry=registry,
        energy_tracker_factory=energy_factory,
        metrics_sink=metrics_sink,
        tracer_factory=lambda _: _RecordingTracer(),
        observability_store=observability_store,
    )

    summary = await validator.execute(
        cycles=1, episodes_per_cycle=8, cooldown_seconds=0.0
    )

    assert summary.total_cycles == 1
    assert summary.total_episodes == 0
    assert summary.total_reward == 0
    assert summary.total_failures == 0
    assert summary.energy_wh is None or summary.energy_wh == pytest.approx(0.0)
    assert summary.success_rate == 0.0
    assert summary.incidents
    assert summary.cycles[0].status == "failed"
    assert any(name == "research.longhaul.cycle" for name, _ in metrics)


@pytest.mark.asyncio
async def test_long_haul_validator_captures_multi_replica_soak(tmp_path) -> None:
    approvals_path = tmp_path / "approvals.json"
    registry = OperatorApprovalRegistry(approvals_path)
    registry.submit(source="reinforcement.reasoning", payload={"version": 2})

    cycles = deque(
        [
            {
                "episodes": 6,
                "reward": 3.5,
                "average": 0.7,
                "failures": 1,
                "wall_clock": 0.35,
                "workers": [
                    WorkerAdjustment(batch_index=0, worker_count=2, reason="initial"),
                    WorkerAdjustment(batch_index=1, worker_count=4, reason="scale_up"),
                    WorkerAdjustment(batch_index=2, worker_count=3, reason="stabilise"),
                ],
            },
            {
                "episodes": 6,
                "reward": 4.2,
                "average": 0.7,
                "failures": 0,
                "wall_clock": 0.4,
                "workers": [
                    WorkerAdjustment(batch_index=0, worker_count=3, reason="initial"),
                    WorkerAdjustment(batch_index=1, worker_count=5, reason="scale_up"),
                    WorkerAdjustment(batch_index=2, worker_count=4, reason="stabilise"),
                    WorkerAdjustment(batch_index=3, worker_count=5, reason="burst"),
                ],
            },
            {
                "episodes": 6,
                "reward": 2.4,
                "average": 0.6,
                "failures": 2,
                "wall_clock": 0.38,
                "workers": [
                    WorkerAdjustment(batch_index=0, worker_count=4, reason="initial"),
                    WorkerAdjustment(batch_index=1, worker_count=3, reason="scale_down"),
                    WorkerAdjustment(batch_index=2, worker_count=2, reason="stabilise"),
                ],
            },
        ]
    )

    def loop_factory() -> _StubReinforcementLoop:
        cycle = cycles.popleft()
        summary = ReinforcementLearningSummary(
            total_episodes=cycle["episodes"],
            total_reward=cycle["reward"],
            average_reward=cycle["average"],
            episode_results=[],
            failures=cycle["failures"],
            wall_clock_seconds=cycle["wall_clock"],
            worker_history=list(cycle["workers"]),
        )
        return _StubReinforcementLoop(
            expected_episodes=cycle["episodes"], summary=summary
        )

    energy_values = deque([0.8, 0.9, 0.7])

    def energy_factory() -> _EnergyTracker:
        return _EnergyTracker(energy_values)

    metrics: list[tuple[str, dict[str, float | int]]] = []

    def metrics_sink(name: str, payload: dict[str, float | int]) -> None:
        metrics.append((name, dict(payload)))

    tracer = _RecordingTracer()
    mnpt_calls = 0

    async def mnpt_callback() -> None:
        nonlocal mnpt_calls
        mnpt_calls += 1

    observability_store = _RecordingObservabilityStore()

    validator = ResearchLoopLongHaulValidator(
        reinforcement_loop_factory=loop_factory,
        approval_registry=registry,
        energy_tracker_factory=energy_factory,
        metrics_sink=metrics_sink,
        tracer_factory=lambda _: tracer,
        mnpt_callback=mnpt_callback,
        observability_store=observability_store,
    )

    summary = await validator.execute(
        cycles=3, episodes_per_cycle=6, cooldown_seconds=0.0
    )

    assert summary.total_cycles == 3
    assert summary.total_episodes == 18
    assert summary.total_reward == pytest.approx(10.1)
    assert summary.total_failures == 3
    assert summary.energy_wh == pytest.approx(2.4)
    assert summary.mnpt_runs == 3
    assert summary.incidents == ()
    assert mnpt_calls == 3
    assert observability_store.records and observability_store.records[0] is summary

    cycle_metrics = [payload for name, payload in metrics if name == "research.longhaul.cycle"]
    assert len(cycle_metrics) == 3
    assert {payload["cycle"] for payload in cycle_metrics} == {0, 1, 2}
    assert metrics[-1][0] == "research.longhaul.summary"

    cycle0 = summary.cycles[0]
    assert cycle0.replica_load.events == 3
    assert cycle0.replica_load.peak == 4
    assert cycle0.replica_load.low == 2
    assert cycle0.replica_load.average == pytest.approx(3.0)
    assert cycle0.replica_load.reasons["initial"] == 1
    assert cycle0.replica_load.timeline[1].reason == "scale_up"

    cycle1 = summary.cycles[1]
    assert cycle1.replica_load.events == 4
    assert cycle1.replica_load.peak == 5
    assert cycle1.replica_load.low == 3
    assert cycle1.replica_load.timeline[-1].reason == "burst"
    assert cycle1.replica_load.reasons["scale_up"] == 1

    cycle2 = summary.cycles[2]
    assert cycle2.replica_load.events == 3
    assert cycle2.replica_load.low == 2
    assert cycle2.replica_load.reasons["scale_down"] == 1
    assert cycle2.replica_load.timeline[0].batch_index == 0
    assert cycle2.replica_load.timeline[-1].worker_count == 2

    event_payloads = [attrs for span in tracer.spans for _, attrs in span.events]
    assert any(payload.get("mnpt_executed") == 1 for payload in event_payloads)
