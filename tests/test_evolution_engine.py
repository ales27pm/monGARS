import pytest

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
