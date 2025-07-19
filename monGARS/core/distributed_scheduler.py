from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine

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
        self._running: bool = False
        self._stopping: bool = False

    async def add_task(self, task: Callable[[], Coroutine[Any, Any, Any]]) -> None:
        """Queue a coroutine factory for execution."""
        if self._stopping:
            raise RuntimeError("Scheduler is stopping")
        await self.queue.put(task)

    async def _worker(self) -> None:
        while self._running or not self.queue.empty():
            try:
                factory = await asyncio.wait_for(self.queue.get(), timeout=1)
            except asyncio.TimeoutError:
                if not self._running and self.queue.empty():
                    break
                continue
            try:
                result = await factory()
                await self.communicator.send({"result": result})
            except Exception as exc:  # pragma: no cover - unexpected errors
                logger.error("Task failed: %s", exc)
            finally:
                self.queue.task_done()

    async def run(self) -> None:
        if self._running:
            return
        self._stopping = False
        self._running = True
        self._workers = [
            asyncio.create_task(self._worker()) for _ in range(self.concurrency)
        ]
        while self._running:
            await asyncio.sleep(0.1)
        await self.queue.join()
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)

    def stop(self) -> None:
        self._stopping = True
        self._running = False
