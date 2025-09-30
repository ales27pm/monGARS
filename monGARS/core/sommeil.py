from __future__ import annotations

import asyncio
import logging
from typing import Any

from .distributed_scheduler import DistributedScheduler
from .evolution_engine import EvolutionEngine
from .ui_events import event_bus, make_event

logger = logging.getLogger(__name__)


class SommeilParadoxal:
    """Background optimizations triggered when the system is idle."""

    def __init__(
        self,
        scheduler: DistributedScheduler,
        evolution: EvolutionEngine | None = None,
        check_interval: int = 60,
    ) -> None:
        self.scheduler = scheduler
        self.evolution = evolution or EvolutionEngine()
        self.check_interval = check_interval
        self._task: asyncio.Task[Any] | None = None
        self._running = False

    async def run(self, user_id: str | None = None) -> None:
        """Execute a single idle-time optimization cycle."""

        await event_bus().publish(
            make_event(
                "sleep_time_compute.phase_start",
                user_id,
                {"phase": "deep"},
            )
        )
        logger.info(
            "sommeil.phase.start",
            extra={"user_id": user_id},
        )

        try:
            success = await self.evolution.safe_apply_optimizations()
        except Exception:
            logger.exception(
                "sommeil.optimization.error",
                extra={"user_id": user_id},
            )
            success = False

        await event_bus().publish(
            make_event(
                "sleep_time_compute.creative_phase",
                user_id,
                {"ideas": 3},
            )
        )
        logger.info(
            "sommeil.phase.complete",
            extra={"user_id": user_id, "success": success},
        )

    async def _loop(self) -> None:
        try:
            while self._running:
                await self.scheduler.queue.join()
                await self.evolution.safe_apply_optimizations()
                await asyncio.sleep(self.check_interval)
        except asyncio.CancelledError:
            pass

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        try:
            self._task = asyncio.create_task(self._loop())
        except Exception as exc:  # pragma: no cover - unlikely
            self._running = False
            logger.error("Failed to start SommeilParadoxal: %s", exc)
            raise

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # pragma: no cover - unexpected
                logger.error("Error stopping SommeilParadoxal: %s", exc)
