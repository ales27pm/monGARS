"""Asynchronous orchestration for the evolution training pipeline."""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime

try:  # Python 3.11+
    from datetime import UTC
except ImportError:  # pragma: no cover - Python 3.10 fallback
    from datetime import timezone

    UTC = timezone.utc  # type: ignore[assignment]

from typing import Callable

from monGARS.config import Settings, get_settings
from monGARS.core.evolution_engine import EvolutionEngine

logger = logging.getLogger(__name__)


def _generate_version(prefix: str, iteration: int) -> str:
    """Generate a unique training version identifier."""

    sanitized = "-".join(part for part in prefix.strip().split() if part) or "enc"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{sanitized}-{timestamp}-{iteration:04d}"


def _compute_delay(interval: float, jitter: float) -> float:
    """Return the delay before the next training cycle respecting jitter bounds."""

    interval = max(0.0, float(interval))
    jitter = max(0.0, float(jitter))
    if interval == 0.0:
        return 0.0
    spread = min(jitter, interval)
    if spread == 0.0:
        return interval
    return max(0.0, interval + random.uniform(-spread, spread))


async def _wait_for_delay(
    duration: float, shutdown_event: asyncio.Event | None
) -> None:
    """Sleep for ``duration`` seconds unless ``shutdown_event`` fires sooner."""

    if duration <= 0:
        return
    if shutdown_event is None:
        await asyncio.sleep(duration)
        return
    if shutdown_event.is_set():
        return
    try:
        await asyncio.wait_for(shutdown_event.wait(), timeout=duration)
    except asyncio.TimeoutError:
        return


async def training_workflow(
    *,
    engine_factory: Callable[[], EvolutionEngine] | None = None,
    shutdown_event: asyncio.Event | None = None,
    max_cycles: int | None = None,
    settings_override: Settings | None = None,
    interval_override: float | None = None,
    jitter_override: float | None = None,
) -> None:
    """Run the background evolution training workflow.

    The workflow coordinates :class:`~monGARS.core.evolution_engine.EvolutionEngine`
    to periodically execute training cycles. The behaviour is primarily driven by
    configuration returned from :func:`monGARS.config.get_settings`, but optional
    overrides make the routine deterministic under tests.

    Args:
        engine_factory: Optional callable that returns an ``EvolutionEngine``
            instance. Defaults to ``EvolutionEngine``.
        shutdown_event: When provided, the workflow exits early once the event is
            set. Useful for orderly shutdowns in service managers.
        max_cycles: Optional hard limit on the number of cycles to execute. When
            ``None`` the workflow runs indefinitely until cancelled or the
            shutdown event is triggered.
        settings_override: Supply a preconfigured ``Settings`` instance to avoid
            polluting the global cache during tests.
        interval_override: Explicit interval between cycles in seconds. When
            omitted, ``training_cycle_interval_seconds`` from settings is used.
        jitter_override: Explicit jitter window applied to the interval. When
            omitted, ``training_cycle_jitter_seconds`` from settings is used.
    """

    settings = settings_override or get_settings()
    if not settings.training_pipeline_enabled:
        logger.info("Training pipeline disabled via configuration; skipping workflow.")
        return

    if max_cycles is not None and max_cycles <= 0:
        logger.info(
            "Training workflow requested with non-positive max_cycles; exiting immediately."
        )
        return

    engine_ctor = engine_factory or EvolutionEngine
    engine = engine_ctor()

    interval = (
        float(interval_override)
        if interval_override is not None
        else float(settings.training_cycle_interval_seconds)
    )
    jitter = (
        float(jitter_override)
        if jitter_override is not None
        else float(settings.training_cycle_jitter_seconds)
    )

    user_id = settings.training_pipeline_user_id or None
    prefix = settings.training_pipeline_version_prefix or "enc"

    iteration = 0
    while True:
        if shutdown_event is not None and shutdown_event.is_set():
            logger.info(
                "Training workflow shutdown signal received; terminating loop before next cycle."
            )
            break
        if max_cycles is not None and iteration >= max_cycles:
            logger.info("Training workflow completed %s cycles; exiting.", iteration)
            break

        iteration += 1
        version = _generate_version(prefix, iteration)
        logger.info(
            "Starting training cycle",
            extra={"version": version, "iteration": iteration},
        )
        try:
            await engine.train_cycle(user_id=user_id, version=version)
        except asyncio.CancelledError:
            logger.info(
                "Training workflow cancelled during cycle; propagating cancellation."
            )
            raise
        except (
            Exception
        ):  # pragma: no cover - defensive guard against unexpected failures
            logger.exception(
                "Training cycle failed",
                extra={"version": version, "iteration": iteration},
            )

        if shutdown_event is not None and shutdown_event.is_set():
            logger.info(
                "Training workflow shutdown signal received after cycle %s; exiting.",
                iteration,
            )
            break

        if max_cycles is not None and iteration >= max_cycles:
            logger.info("Training workflow completed %s cycles; exiting.", iteration)
            break

        delay = _compute_delay(interval, jitter)
        try:
            await _wait_for_delay(delay, shutdown_event)
        except asyncio.CancelledError:
            logger.info(
                "Training workflow cancelled during backoff; propagating cancellation."
            )
            raise
