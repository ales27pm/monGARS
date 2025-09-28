from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable, Coroutine, Iterable
from threading import Lock
from typing import Any
from weakref import WeakSet

from opentelemetry import metrics
from opentelemetry.metrics import CallbackOptions, Observation

from monGARS.config import get_settings

from .peer import PeerCommunicator

logger = logging.getLogger(__name__)

meter = metrics.get_meter(__name__)

_scheduler_registry: "WeakSet[DistributedScheduler]" = WeakSet()
_settings: Any | None = None
_metrics_registered = False
_metrics_enabled = False


def _ensure_metrics_registered() -> bool:
    """Initialise OpenTelemetry instruments if metrics are enabled."""

    global _metrics_registered, _metrics_enabled, _settings
    if _metrics_registered:
        return _metrics_enabled

    try:
        _settings = get_settings()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to load settings for metrics: %s", exc, exc_info=True)
        _metrics_registered = True
        _metrics_enabled = False
        return False

    if not getattr(_settings, "otel_metrics_enabled", False):
        _metrics_registered = True
        _metrics_enabled = False
        return False

    meter.create_observable_gauge(
        "distributed_scheduler_queue_depth",
        callbacks=[_observe_queue_depth],
        description="Current number of pending tasks per scheduler instance.",
    )
    meter.create_observable_gauge(
        "distributed_scheduler_worker_uptime_seconds",
        callbacks=[_observe_worker_uptime],
        description="Aggregate uptime in seconds for active scheduler workers.",
    )
    meter.create_observable_gauge(
        "distributed_scheduler_task_failure_rate",
        callbacks=[_observe_failure_rate],
        description="Rolling task failure rate for scheduler instances.",
    )
    _metrics_registered = True
    _metrics_enabled = True
    return True


def _observe_queue_depth(options: CallbackOptions) -> Iterable[Observation]:
    return tuple(
        Observation(scheduler.queue_depth, scheduler.metric_attributes)
        for scheduler in list(_scheduler_registry)
    )


def _observe_worker_uptime(options: CallbackOptions) -> Iterable[Observation]:
    return tuple(
        Observation(scheduler.worker_uptime_seconds, scheduler.metric_attributes)
        for scheduler in list(_scheduler_registry)
    )


def _observe_failure_rate(options: CallbackOptions) -> Iterable[Observation]:
    return tuple(
        Observation(scheduler.task_failure_rate, scheduler.metric_attributes)
        for scheduler in list(_scheduler_registry)
    )


class DistributedScheduler:
    """Simple scheduler that distributes tasks across peer nodes."""

    def __init__(
        self,
        communicator: PeerCommunicator,
        concurrency: int = 1,
        metrics_interval: float = 5.0,
    ) -> None:
        self.communicator = communicator
        self.concurrency = max(1, concurrency)
        self.queue: asyncio.Queue[Callable[[], Coroutine[Any, Any, Any]]] = (
            asyncio.Queue()
        )
        self._workers: list[asyncio.Task[Any]] = []
        self._running: bool = False
        self._stopping: bool = False
        self._metrics_lock = Lock()
        self._worker_start_times: dict[int, float] = {}
        self._worker_uptime_total: float = 0.0
        self._processed_tasks: int = 0
        self._failed_tasks: int = 0
        self._last_metrics_emit: float = 0.0
        self.metrics_interval = max(0.5, metrics_interval)
        self.instance_id = uuid.uuid4().hex
        self._metric_attributes = {
            "scheduler_id": self.instance_id,
            "concurrency": self.concurrency,
        }
        if _ensure_metrics_registered():
            _scheduler_registry.add(self)

    @property
    def metric_attributes(self) -> dict[str, int | str]:
        return self._metric_attributes

    @property
    def queue_depth(self) -> int:
        return self.queue.qsize()

    @property
    def worker_uptime_seconds(self) -> float:
        with self._metrics_lock:
            now = time.monotonic()
            uptime = self._worker_uptime_total
            for start_time in self._worker_start_times.values():
                uptime += now - start_time
            return uptime

    @property
    def task_failure_rate(self) -> float:
        with self._metrics_lock:
            if self._processed_tasks == 0:
                return 0.0
            return self._failed_tasks / self._processed_tasks

    def get_metrics_snapshot(self) -> dict[str, float | int]:
        """Return a point-in-time view of scheduler health metrics."""

        with self._metrics_lock:
            processed = self._processed_tasks
            failed = self._failed_tasks
            now = time.monotonic()
            uptime = self._worker_uptime_total
            for start_time in self._worker_start_times.values():
                uptime += now - start_time
            return {
                "queue_depth": self.queue.qsize(),
                "active_workers": len(self._worker_start_times),
                "concurrency": self.concurrency,
                "worker_uptime_seconds": uptime,
                "tasks_processed": processed,
                "tasks_failed": failed,
                "task_failure_rate": 0.0 if processed == 0 else failed / processed,
            }

    async def add_task(self, task: Callable[[], Coroutine[Any, Any, Any]]) -> None:
        """Queue a coroutine factory for execution."""
        if self._stopping:
            raise RuntimeError("Scheduler is stopping")
        await self.queue.put(task)

    async def _worker(self, worker_id: int) -> None:
        with self._metrics_lock:
            self._worker_start_times[worker_id] = time.monotonic()
        try:
            while self._running or not self.queue.empty():
                try:
                    factory = await asyncio.wait_for(self.queue.get(), timeout=1)
                except asyncio.TimeoutError:
                    if not self._running and self.queue.empty():
                        break
                    continue
                try:
                    result = await factory()
                except Exception as exc:  # pragma: no cover - unexpected errors
                    with self._metrics_lock:
                        self._processed_tasks += 1
                        self._failed_tasks += 1
                    logger.error(
                        "scheduler.task_failure",
                        extra={
                            "scheduler_id": self.instance_id,
                            "worker_id": worker_id,
                        },
                        exc_info=True,
                    )
                else:
                    with self._metrics_lock:
                        self._processed_tasks += 1
                    await self.communicator.send({"result": result})
                finally:
                    self.queue.task_done()
        finally:
            with self._metrics_lock:
                start_time = self._worker_start_times.pop(worker_id, None)
                if start_time is not None:
                    self._worker_uptime_total += time.monotonic() - start_time

    def _emit_metrics_log(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_metrics_emit < self.metrics_interval:
            return
        snapshot = self.get_metrics_snapshot()
        logger.info(
            "scheduler.metrics",
            extra={"scheduler_id": self.instance_id, **snapshot},
        )
        self._last_metrics_emit = now

    async def run(self) -> None:
        if self._running:
            return
        self._stopping = False
        self._running = True
        self._last_metrics_emit = time.monotonic()
        self._workers = [
            asyncio.create_task(self._worker(index))
            for index in range(self.concurrency)
        ]
        try:
            while self._running:
                await asyncio.sleep(0.1)
                self._emit_metrics_log()
        finally:
            await self.queue.join()
            for worker in self._workers:
                worker.cancel()
            await asyncio.gather(*self._workers, return_exceptions=True)
            self._emit_metrics_log(force=True)

    def stop(self) -> None:
        self._stopping = True
        self._running = False
