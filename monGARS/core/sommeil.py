from __future__ import annotations

import asyncio
import logging
from typing import Any

from .distributed_scheduler import DistributedScheduler
from .evolution_engine import EvolutionEngine

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
