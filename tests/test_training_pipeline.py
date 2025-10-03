from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import pytest

from monGARS.config import get_settings
from monGARS.mlops import training_pipeline
from monGARS.mlops.training_pipeline import (
    _compute_delay,
    _generate_version,
    training_workflow,
)


@pytest.fixture(autouse=True)
def _reset_settings_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SECRET_KEY", "test")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class _StubEngine:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def train_cycle(self, *, version: str, user_id: str | None = None) -> None:
        self.calls.append({"user_id": user_id, "version": version})


@pytest.mark.asyncio
async def test_training_workflow_runs_requested_cycles() -> None:
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
    assert len(set(versions)) == len(versions)
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
    assert not called


@pytest.mark.asyncio
async def test_training_workflow_exits_on_shutdown_event() -> None:
    settings = get_settings().model_copy(update={"training_cycle_jitter_seconds": 0})
    engine = _StubEngine()
    shutdown_event = asyncio.Event()
    shutdown_event.set()

    await training_workflow(
        engine_factory=lambda: engine,
        max_cycles=2,
        settings_override=settings,
        interval_override=0,
        jitter_override=0,
        shutdown_event=shutdown_event,
    )

    assert engine.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", (0, -1))
async def test_training_workflow_exits_immediately_on_non_positive_max_cycles(
    limit: int,
) -> None:
    settings = get_settings().model_copy(update={"training_cycle_jitter_seconds": 0})
    engine = _StubEngine()

    def _engine_factory(_engine: _StubEngine = engine) -> _StubEngine:
        return _engine

    await training_workflow(
        engine_factory=_engine_factory,
        max_cycles=limit,
        settings_override=settings,
        interval_override=0,
        jitter_override=0,
    )
    assert engine.calls == []


def test_compute_delay_respects_interval_and_jitter_bounds() -> None:
    cases = (
        {"interval": 10.0, "jitter": 0.0, "expected_min": 10.0, "expected_max": 10.0},
        {"interval": 10.0, "jitter": 2.0, "expected_min": 8.0, "expected_max": 12.0},
        {"interval": 5.0, "jitter": 5.0, "expected_min": 0.0, "expected_max": 10.0},
        {"interval": 0.0, "jitter": 3.0, "expected_min": 0.0, "expected_max": 0.0},
    )

    for case in cases:
        delays = {_compute_delay(case["interval"], case["jitter"]) for _ in range(100)}
        assert min(delays) >= case["expected_min"]
        assert max(delays) <= case["expected_max"]
        if case["jitter"] > 0.0 and case["interval"] > 0.0:
            assert len(delays) > 1


@pytest.mark.parametrize(
    ("raw_prefix", "expected_prefix"),
    (
        ("enc", "enc"),
        ("  custom prefix  ", "custom-prefix"),
        ("@@@", "enc"),
    ),
)
def test_generate_version_sanitizes_prefix(
    monkeypatch: pytest.MonkeyPatch, raw_prefix: str, expected_prefix: str
) -> None:
    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return datetime(2024, 1, 2, 3, 4, 5, tzinfo=tz)

    monkeypatch.setattr(training_pipeline, "datetime", _FixedDatetime)

    version = _generate_version(raw_prefix, 3)

    assert version.startswith(f"{expected_prefix}-20240102T030405Z-0003")
