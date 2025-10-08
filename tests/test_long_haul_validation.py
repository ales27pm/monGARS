from __future__ import annotations

import contextlib
from collections import deque
from typing import Any

import pytest

from modules.evolution_engine.energy import EnergyUsageReport
from modules.neurons.training.reinforcement_loop import (
    ReinforcementLearningSummary,
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
                worker_history=[],
            ),
            ReinforcementLearningSummary(
                total_episodes=12,
                total_reward=8.4,
                average_reward=0.84,
                episode_results=[],
                failures=1,
                wall_clock_seconds=0.5,
                worker_history=[],
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

    validator = ResearchLoopLongHaulValidator(
        reinforcement_loop_factory=loop_factory,
        approval_registry=registry,
        energy_tracker_factory=energy_factory,
        metrics_sink=metrics_sink,
        tracer_factory=lambda _: tracer,
        mnpt_callback=mnpt_callback,
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

    validator = ResearchLoopLongHaulValidator(
        reinforcement_loop_factory=failing_factory,
        approval_registry=registry,
        energy_tracker_factory=energy_factory,
        metrics_sink=metrics_sink,
        tracer_factory=lambda _: _RecordingTracer(),
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
