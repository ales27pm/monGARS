import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("SECRET_KEY", "test")

from monGARS.core.evolution_engine import EvolutionEngine


@pytest.mark.asyncio
async def test_safe_apply_success(monkeypatch):
    engine = EvolutionEngine()

    async def fake_apply():
        engine.applied = True

    monkeypatch.setattr(engine, "apply_optimizations", fake_apply)
    result = await engine.safe_apply_optimizations()
    assert result is True
    assert getattr(engine, "applied", False)


@pytest.mark.asyncio
async def test_safe_apply_failure(monkeypatch):
    engine = EvolutionEngine()

    async def fail():
        raise RuntimeError("boom")

    monkeypatch.setattr(engine, "apply_optimizations", fail)
    result = await engine.safe_apply_optimizations()
    assert result is False


@pytest.mark.asyncio
async def test_apply_optimizations_clears_cache_on_memory_spike(monkeypatch):
    engine = EvolutionEngine()

    async def fake_stats():
        return SimpleNamespace(cpu_usage=20.0, memory_usage=95.0, gpu_usage=None)

    calls: dict[str, object] = {}

    async def fake_clear() -> None:
        calls["cleared"] = True

    async def fake_scale(_delta: int) -> None:
        calls.setdefault("scaled", True)

    monkeypatch.setattr(engine.monitor, "get_system_stats", fake_stats)
    monkeypatch.setattr(engine, "_clear_caches", fake_clear)
    monkeypatch.setattr(engine, "_scale_workers", fake_scale)

    await engine.apply_optimizations()

    assert calls.get("cleared") is True
    assert "scaled" not in calls
