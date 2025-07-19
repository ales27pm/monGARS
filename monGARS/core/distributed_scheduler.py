from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine, Optional

from .peer import PeerCommunicator

logger = logging.getLogger(__name__)


class DistributedScheduler:
    """Simple scheduler that distributes tasks across peer nodes."""

    def __init__(
        self,
        communicator: PeerCommunicator,
        concurrency: int = 1,
    ) -> None:
        self.communicator = communicator
        self.concurrency = max(1, concurrency)
        self.queue: asyncio.Queue[Callable[[], Coroutine[Any, Any, Any]]] = (
            asyncio.Queue()
        )
        self._workers: list[asyncio.Task[Any]] = []
        self._running = asyncio.Event()

    async def add_task(self, task: Callable[[], Coroutine[Any, Any, Any]]) -> None:
        """Queue a coroutine factory for execution."""
        await self.queue.put(task)

    async def _worker(self) -> None:
        while self._running.is_set():
            try:
                factory = await asyncio.wait_for(self.queue.get(), timeout=1)
            except asyncio.TimeoutError:
                continue
            try:
                result = await factory()
                await self.communicator.send({"result": result})
            except Exception as exc:  # pragma: no cover - unexpected errors
                logger.error("Task failed: %s", exc)
            finally:
                self.queue.task_done()

    async def run(self) -> None:
        self._running.set()
        self._workers = [
            asyncio.create_task(self._worker()) for _ in range(self.concurrency)
        ]
        await self.queue.join()
        self._running.clear()
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)

    def stop(self) -> None:
        self._running.clear()
