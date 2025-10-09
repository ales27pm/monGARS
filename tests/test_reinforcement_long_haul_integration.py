import asyncio
import contextlib
import queue
from collections import deque
from typing import Any

import pytest

from modules.evolution_engine.energy import EnergyUsageReport
from modules.neurons.training.reinforcement_loop import ReinforcementLearningLoop
from monGARS.core.long_haul_validation import ResearchLoopLongHaulValidator
from monGARS.core.operator_approvals import OperatorApprovalRegistry
from monGARS.core.research_validation import ResearchLongHaulService


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


class _RecordingObservabilityStore:
    def __init__(self) -> None:
        self.records: list[Any] = []

    def record_summary(self, summary: Any) -> None:
        self.records.append(summary)


class _FixedScalingStrategy:
    def recommend_worker_count(
        self, current_workers: int, batch_index: int, stats: Any
    ):
        return current_workers, "fixed"


class _ConstantPolicy:
    def select_action(self, state: Any) -> int:
        return 0

    def update(self, transitions: list[Any]) -> None:
        return None

    def clone(self) -> "_ConstantPolicy":
        return _ConstantPolicy()


class _RewardingEnvironment:
    def __init__(self, reward: float) -> None:
        self._reward = float(reward)

    def reset(self) -> int:
        return 0

    def step(self, action: int) -> tuple[int, float, bool, dict[str, float]]:
        return 0, self._reward, True, {"reward": self._reward}


class _StubEnergyTracker:
    def __init__(self, values: deque[float]) -> None:
        self._values = values
        self.last_report: EnergyUsageReport | None = None

    @contextlib.contextmanager
    def track(self):
        try:
            yield self
        finally:
            value = self._values.popleft() if self._values else 0.0
            self.last_report = EnergyUsageReport(
                energy_wh=float(value),
                duration_seconds=1.0,
                cpu_seconds=0.4,
                baseline_cpu_power_watts=10.0,
                backend="test",
            )


@pytest.mark.asyncio
async def test_long_haul_service_executes_reinforcement_loop(tmp_path) -> None:
    approvals_path = tmp_path / "approvals.json"
    registry = OperatorApprovalRegistry(approvals_path)
    registry.submit(source="reinforcement.reasoning", payload={"version": "v1"})
    registry.submit(source="reinforcement.reasoning", payload={"version": "v2"})

    reward_sequences_data = [
        [1.0, 0.9, 0.95, 0.85, 0.92, 0.88],
        [1.0, 0.97, 0.93, 0.9, 0.88, 0.9],
    ]
    reward_sequences = deque(reward_sequences_data)
    energy_values_data = [0.45, 0.6]
    energy_values = deque(energy_values_data)

    tracer = _RecordingTracer()
    metrics: list[tuple[str, dict[str, float | int]]] = []
    summary_ready = asyncio.Event()

    def metrics_sink(name: str, payload: dict[str, float | int]) -> None:
        metrics.append((name, dict(payload)))
        if name == "research.longhaul.summary":
            summary_ready.set()

    mnpt_calls = 0

    async def mnpt_callback() -> None:
        nonlocal mnpt_calls
        await asyncio.sleep(0)
        mnpt_calls += 1

    def make_loop() -> ReinforcementLearningLoop:
        rewards = reward_sequences.popleft() if reward_sequences else [1.0] * 6
        reward_queue: queue.Queue[float] = queue.Queue()
        for value in rewards:
            reward_queue.put(value)

        policy = _ConstantPolicy()

        def environment_factory() -> _RewardingEnvironment:
            try:
                reward = reward_queue.get_nowait()
            except queue.Empty:
                reward = rewards[-1]
            return _RewardingEnvironment(reward)

        return ReinforcementLearningLoop(
            environment_factory=environment_factory,
            policy=policy,
            max_steps=1,
            scaling_strategy=_FixedScalingStrategy(),
            initial_workers=2,
            max_workers=2,
            tracer_factory=lambda _: tracer,
            metrics_sink=metrics_sink,
        )

    observability_store = _RecordingObservabilityStore()

    validator = ResearchLoopLongHaulValidator(
        reinforcement_loop_factory=make_loop,
        approval_registry=registry,
        energy_tracker_factory=lambda: _StubEnergyTracker(energy_values),
        metrics_sink=metrics_sink,
        tracer_factory=lambda _: tracer,
        mnpt_callback=mnpt_callback,
        observability_store=observability_store,
    )

    service = ResearchLongHaulService(
        validator_factory=lambda: validator,
        enabled=True,
        interval_seconds=60.0,
        jitter_seconds=0.0,
    )

    await service.schedule_once(
        reason="integration-test",
        cycles=2,
        episodes_per_cycle=6,
        cooldown_seconds=0.0,
    )

    await asyncio.wait_for(summary_ready.wait(), timeout=5.0)
    await service.stop()

    summary = service.last_summary
    assert summary is not None
    assert summary.total_cycles == 2
    assert summary.total_episodes == 12
    expected_reward = sum(sum(seq) for seq in reward_sequences_data)
    assert summary.total_reward == pytest.approx(expected_reward)
    assert summary.total_failures == 0
    assert summary.average_reward == pytest.approx(expected_reward / 12)
    assert summary.energy_wh == pytest.approx(sum(energy_values_data))
    assert summary.approval_pending_final == 2
    assert summary.mnpt_runs == 2
    assert mnpt_calls == 2
    assert summary.success_rate == pytest.approx(1.0)
    assert summary.incidents == ()
    assert observability_store.records and observability_store.records[0] is summary
    assert len(summary.cycles) == 2
    assert all(report.status == "completed" for report in summary.cycles)
    assert summary.cycles[0].episodes == 6
    assert summary.cycles[0].energy_wh == pytest.approx(energy_values_data[0])
    assert metrics[-1][0] == "research.longhaul.summary"

    rl_summary_count = sum(
        1 for name, _ in metrics if name == "reinforcement.loop.summary"
    )
    assert rl_summary_count == 2
    cycle_metrics = [name for name, _ in metrics if name == "research.longhaul.cycle"]
    assert len(cycle_metrics) == 2

    span_names = [span.name for span in tracer.spans]
    assert "reinforcement.loop.run" in span_names
    assert "research.longhaul.execute" in span_names
    assert any(
        event_name == "cycle.completed"
        for span in tracer.spans
        for event_name, _ in span.events
    )

    assert service.last_reason == "integration-test"
