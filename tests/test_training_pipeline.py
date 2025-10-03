from typing import Any

import pytest

from monGARS.config import get_settings
from monGARS.mlops.training_pipeline import training_workflow


class _StubEngine:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def train_cycle(self, *, user_id: str | None = None, version: str) -> None:
        self.calls.append({"user_id": user_id, "version": version})


@pytest.mark.asyncio
async def test_training_workflow_runs_requested_cycles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = get_settings().model_copy(update={"training_cycle_jitter_seconds": 0})
    engine = _StubEngine()

    await training_workflow(
        engine_factory=lambda: engine,
        max_cycles=2,
        settings_override=settings,
        interval_override=0,
        jitter_override=0,
    )

    assert len(engine.calls) == 2
    user_ids = {call["user_id"] for call in engine.calls}
    assert user_ids == {settings.training_pipeline_user_id}
    versions = [call["version"] for call in engine.calls]
    assert len({v for v in versions}) == len(versions)
    assert all(settings.training_pipeline_version_prefix in v for v in versions)


@pytest.mark.asyncio
async def test_training_workflow_skips_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAINING_PIPELINE_ENABLED", "false")
    get_settings.cache_clear()
    settings = get_settings()

    called = False

    def _factory() -> _StubEngine:
        nonlocal called
        called = True
        return _StubEngine()

    await training_workflow(engine_factory=_factory, settings_override=settings)

    assert settings.training_pipeline_enabled is False
    assert called is False
