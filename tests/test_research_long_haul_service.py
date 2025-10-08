import asyncio
from typing import Any, Awaitable, Callable

import pytest

from monGARS.core.long_haul_validation import (
    LongHaulCycleReport,
    LongHaulValidationSummary,
)
from monGARS.core.research_validation import ResearchLongHaulService


class _ImmediateScheduler:
    def __init__(self) -> None:
        self.calls = 0
        self.last_task: Callable[[], Awaitable[None]] | None = None

    async def add_task(self, task: Callable[[], Awaitable[None]]) -> None:
        self.calls += 1
        self.last_task = task
        await task()


def _make_summary() -> LongHaulValidationSummary:
    cycle = LongHaulCycleReport(
        index=0,
        status="completed",
        episodes=4,
        total_reward=2.5,
        average_reward=0.625,
        failures=0,
        duration_seconds=0.5,
        energy_wh=None,
        approval_pending=None,
        incidents=(),
        mnpt_executed=False,
    )
    return LongHaulValidationSummary(
        started_at="2025-01-01T00:00:00+00:00",
        duration_seconds=0.5,
        total_cycles=1,
        total_episodes=4,
        total_reward=2.5,
        average_reward=0.625,
        total_failures=0,
        success_rate=1.0,
        energy_wh=None,
        approval_pending_final=None,
        mnpt_runs=0,
        cycles=[cycle],
        incidents=(),
    )


@pytest.mark.asyncio
async def test_schedule_once_runs_validator_without_scheduler() -> None:
    triggered = asyncio.Event()
    summary = _make_summary()

    class _Validator:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, **_: Any) -> LongHaulValidationSummary:
            self.calls += 1
            triggered.set()
            return summary

    validator = _Validator()
    service = ResearchLongHaulService(
        validator_factory=lambda: validator,
        enabled=True,
        interval_seconds=5.0,
        jitter_seconds=0.0,
    )

    await service.schedule_once(reason="unit-test")
    await triggered.wait()

    assert validator.calls == 1
    assert service.last_summary is summary
    assert service.last_reason == "unit-test"
    await service.stop()


@pytest.mark.asyncio
async def test_schedule_once_uses_scheduler() -> None:
    summary = _make_summary()
    scheduler = _ImmediateScheduler()
    executed = asyncio.Event()

    class _Validator:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, **_: Any) -> LongHaulValidationSummary:
            self.calls += 1
            executed.set()
            return summary

    validator = _Validator()
    service = ResearchLongHaulService(
        validator_factory=lambda: validator,
        scheduler=scheduler,
        enabled=True,
        interval_seconds=10.0,
        jitter_seconds=0.0,
    )

    await service.schedule_once(reason="scheduled")
    await executed.wait()

    assert scheduler.calls == 1
    assert validator.calls == 1
    assert service.last_summary is summary
    await service.stop()


@pytest.mark.asyncio
async def test_service_prevents_duplicate_runs() -> None:
    release = asyncio.Event()
    started = asyncio.Event()
    summary = _make_summary()

    class _Validator:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, **_: Any) -> LongHaulValidationSummary:
            self.calls += 1
            started.set()
            await release.wait()
            return summary

    validator = _Validator()
    service = ResearchLongHaulService(
        validator_factory=lambda: validator,
        enabled=True,
        interval_seconds=1.0,
        jitter_seconds=0.0,
    )

    await service.schedule_once(reason="first")
    await started.wait()
    await service.schedule_once(reason="second")
    release.set()
    await asyncio.sleep(0)  # allow task cleanup

    assert validator.calls == 1
    assert service.last_reason == "first"
    await service.stop()


@pytest.mark.asyncio
async def test_periodic_loop_schedules_runs() -> None:
    summaries = []
    summary = _make_summary()
    completion = asyncio.Event()

    class _Validator:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, **_: Any) -> LongHaulValidationSummary:
            self.calls += 1
            summaries.append(self.calls)
            if self.calls >= 2:
                completion.set()
            return summary

    validator = _Validator()
    service = ResearchLongHaulService(
        validator_factory=lambda: validator,
        enabled=True,
        interval_seconds=0.05,
        jitter_seconds=0.0,
    )

    service.start()
    await asyncio.wait_for(completion.wait(), timeout=1.0)
    await service.stop()

    assert len(summaries) >= 2
