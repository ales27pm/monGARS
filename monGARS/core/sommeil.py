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

    async def _loop(self) -> None:
        try:
            while True:
                await self.scheduler.queue.join()
                await self.evolution.safe_apply_optimizations()
                await asyncio.sleep(self.check_interval)
        except asyncio.CancelledError:
            pass

    def start(self) -> None:
        if not self._task or self._task.done():
            self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        if self._task:
            self._task.cancel()
