import os

os.environ.setdefault("DEBUG", "1")
os.environ.setdefault("SECRET_KEY", "test-secret")

from unittest.mock import AsyncMock

import pytest

from monGARS.core.evolution_engine import EvolutionEngine, PerformanceIssue
from monGARS.core.monitor import SystemStats


@pytest.mark.asyncio
async def test_diagnose_performance_detects_cpu_pressure() -> None:
    engine = EvolutionEngine()
    engine._stat_history.clear()
    engine._stat_history.extend(
        [
            SystemStats(cpu_usage=90.0, memory_usage=65.0, disk_usage=40.0),
            SystemStats(cpu_usage=92.0, memory_usage=66.0, disk_usage=41.0),
        ]
    )

    engine.monitor = AsyncMock()
    engine.monitor.get_system_stats = AsyncMock(
        return_value=SystemStats(
            cpu_usage=96.0,
            memory_usage=67.0,
            disk_usage=42.0,
            gpu_usage=None,
            gpu_memory_usage=None,
        )
    )

    issues = await engine.diagnose_performance()
    identifiers = {issue.identifier for issue in issues}

    assert "cpu_sustained_high" in identifiers


@pytest.mark.asyncio
async def test_apply_optimizations_scales_down_when_underutilized() -> None:
    engine = EvolutionEngine()
    engine._last_scale_timestamp = 0.0

    engine.diagnose_performance = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            PerformanceIssue(
                "workers_underutilized",
                "info",
                {"cpu_average": 10.0, "mem_average": 20.0, "window": 3},
            )
        ]
    )
    engine._get_worker_replicas = AsyncMock(return_value=3)  # type: ignore[method-assign]
    engine._scale_workers = AsyncMock()  # type: ignore[method-assign]
    engine._clear_caches = AsyncMock()  # type: ignore[method-assign]

    await engine.apply_optimizations()

    engine._scale_workers.assert_awaited_once()
    await_call = engine._scale_workers.await_args
    assert await_call.args[0] == -1


@pytest.mark.asyncio
async def test_apply_optimizations_clears_cache_for_memory_pressure() -> None:
    engine = EvolutionEngine()
    engine._last_scale_timestamp = 0.0

    engine.diagnose_performance = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            PerformanceIssue(
                "memory_pressure",
                "high",
                {"average": 93.0, "latest": 94.0},
            )
        ]
    )
    engine._get_worker_replicas = AsyncMock(return_value=2)  # type: ignore[method-assign]
    engine._scale_workers = AsyncMock()  # type: ignore[method-assign]
    engine._clear_caches = AsyncMock()  # type: ignore[method-assign]

    await engine.apply_optimizations()

    engine._clear_caches.assert_awaited_once()
