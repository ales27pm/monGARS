"""Background service for research long-haul validation.

This module coordinates :class:`monGARS.core.long_haul_validation.
ResearchLoopLongHaulValidator` executions so the reinforcement-learning
research loop is exercised on a cadence even when no operator manually
triggers it.  The service keeps scheduling logic isolated from orchestrators
while providing hooks for integration with :class:`DistributedScheduler`.

The service intentionally keeps the interface simple:

* ``schedule_once`` queues a single validation run when none are active.
* ``start`` launches a periodic loop that repeatedly schedules runs using
  configured intervals and jitter.
* ``stop`` cancels the periodic loop and any in-flight background tasks.

Implementations that provide a fully featured scheduler (like
``DistributedScheduler``) can be injected, but the service also works without a
dedicated scheduler by spawning background tasks directly on the current event
loop.  This keeps tests lightweight and allows deployments without the
distributed scheduler to benefit from long-haul checks.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from monGARS.config import get_settings
from monGARS.core.long_haul_validation import LongHaulValidationSummary

logger = logging.getLogger(__name__)


class SchedulerProtocol(Protocol):
    """Minimal interface expected from a scheduler implementation."""

    async def add_task(self, task: Callable[[], Awaitable[None]]) -> None:
        """Queue ``task`` for execution."""


class LongHaulValidatorProtocol(Protocol):
    """Protocol describing the validator interface used by the service."""

    async def execute(
        self,
        *,
        cycles: int | None = None,
        episodes_per_cycle: int | None = None,
        cooldown_seconds: float | None = None,
    ) -> LongHaulValidationSummary:
        """Run the long-haul validation and return the aggregated summary."""


class ResearchLongHaulService:
    """Coordinate recurring research long-haul validation runs."""

    def __init__(
        self,
        *,
        validator_factory: Callable[[], LongHaulValidatorProtocol],
        scheduler: SchedulerProtocol | None = None,
        enabled: bool | None = None,
        interval_seconds: float | None = None,
        jitter_seconds: float | None = None,
    ) -> None:
        if validator_factory is None:
            raise ValueError("validator_factory is required")

        settings = None
        if enabled is None or interval_seconds is None or jitter_seconds is None:
            settings = get_settings()

        if enabled is None:
            enabled = bool(getattr(settings, "research_long_haul_enabled", True))
        if interval_seconds is None:
            interval_seconds = float(
                getattr(settings, "research_long_haul_interval_seconds", 3600.0)
            )
        if jitter_seconds is None:
            jitter_seconds = float(
                getattr(settings, "research_long_haul_jitter_seconds", 300.0)
            )

        self._validator_factory = validator_factory
        self._scheduler = scheduler
        self._enabled = bool(enabled)
        self._interval = max(0.0, float(interval_seconds))
        self._jitter = max(0.0, float(jitter_seconds))
        self._lock = asyncio.Lock()
        self._pending_runs = 0
        self._active_runs = 0
        self._last_summary: LongHaulValidationSummary | None = None
        self._last_reason: str | None = None
        self._background_task: asyncio.Task[Any] | None = None
        self._shutdown_event = asyncio.Event()
        self._inflight_tasks: set[asyncio.Task[Any]] = set()

    @property
    def enabled(self) -> bool:
        """Return whether the service will schedule validation runs."""

        return self._enabled

    @property
    def last_summary(self) -> LongHaulValidationSummary | None:
        """Return the most recent validation summary, if available."""

        return self._last_summary

    @property
    def last_reason(self) -> str | None:
        """Return the reason supplied for the most recent validation run."""

        return self._last_reason

    async def schedule_once(
        self,
        *,
        reason: str = "manual",
        cycles: int | None = None,
        episodes_per_cycle: int | None = None,
        cooldown_seconds: float | None = None,
    ) -> None:
        """Schedule a validation run when no other run is active or pending."""

        async with self._lock:
            if not self._enabled:
                logger.info(
                    "research.longhaul.schedule_skipped",
                    extra={"reason": reason, "enabled": False},
                )
                return

            if self._pending_runs > 0 or self._active_runs > 0:
                logger.debug(
                    "research.longhaul.schedule_deduplicated",
                    extra={
                        "reason": reason,
                        "pending": self._pending_runs,
                        "active": self._active_runs,
                    },
                )
                return

            self._pending_runs += 1

        async def _task() -> None:
            await self._execute_run(
                reason=reason,
                cycles=cycles,
                episodes_per_cycle=episodes_per_cycle,
                cooldown_seconds=cooldown_seconds,
            )

        if self._scheduler is not None:
            await self._scheduler.add_task(_task)
            return

        loop = asyncio.get_running_loop()
        task = loop.create_task(_task())
        self._track_inflight(task)

    def start(self) -> None:
        """Start the periodic scheduling loop."""

        if not self._enabled:
            logger.info("research.longhaul.periodic_disabled")
            return

        if self._background_task is not None and not self._background_task.done():
            return

        loop = asyncio.get_running_loop()
        self._shutdown_event.clear()
        self._background_task = loop.create_task(self._periodic_loop())

    async def stop(self) -> None:
        """Stop the periodic loop and cancel background tasks."""

        self._shutdown_event.set()
        if self._background_task is not None:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            except Exception:  # pragma: no cover - defensive logging
                logger.exception(
                    "research.longhaul.periodic_stop_failed", exc_info=True
                )
            finally:
                self._background_task = None

        if self._inflight_tasks:
            for task in list(self._inflight_tasks):
                task.cancel()
            await asyncio.gather(*self._inflight_tasks, return_exceptions=True)
            self._inflight_tasks.clear()

    def _track_inflight(self, task: asyncio.Task[Any]) -> None:
        self._inflight_tasks.add(task)

        def _cleanup(completed: asyncio.Task[Any]) -> None:
            self._inflight_tasks.discard(completed)

        task.add_done_callback(_cleanup)

    async def _periodic_loop(self) -> None:
        try:
            while not self._shutdown_event.is_set():
                await self.schedule_once(reason="periodic")
                delay = self._compute_delay()
                if delay <= 0:
                    await asyncio.sleep(0)
                    continue
                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:  # pragma: no cover - task cancellation
            raise

    async def _execute_run(
        self,
        *,
        reason: str,
        cycles: int | None,
        episodes_per_cycle: int | None,
        cooldown_seconds: float | None,
    ) -> None:
        async with self._lock:
            self._pending_runs = max(0, self._pending_runs - 1)
            self._active_runs += 1

        try:
            validator = self._validator_factory()
        except Exception:
            logger.exception(
                "research.longhaul.validator_factory_failed", extra={"reason": reason}
            )
            return

        try:
            summary = await validator.execute(
                cycles=cycles,
                episodes_per_cycle=episodes_per_cycle,
                cooldown_seconds=cooldown_seconds,
            )
        except Exception:
            logger.exception(
                "research.longhaul.validation_failed", extra={"reason": reason}
            )
        else:
            self._last_summary = summary
            self._last_reason = reason
            logger.info(
                "research.longhaul.validation_completed",
                extra={
                    "reason": reason,
                    "cycles": summary.total_cycles,
                    "episodes": summary.total_episodes,
                    "success_rate": round(summary.success_rate, 4),
                },
            )
        finally:
            async with self._lock:
                self._active_runs = max(0, self._active_runs - 1)

    def _compute_delay(self) -> float:
        if self._interval <= 0:
            return 0.0
        jitter = min(self._jitter, self._interval)
        if jitter <= 0:
            return self._interval
        return max(0.0, self._interval + random.uniform(-jitter, jitter))
