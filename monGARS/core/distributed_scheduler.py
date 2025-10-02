from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
import uuid
from collections.abc import Callable, Coroutine, Iterable
from datetime import datetime, timezone
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
_metrics_registration_lock = threading.Lock()


def _load_settings() -> Any | None:
    try:
        return get_settings()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to load settings for metrics: %s", exc, exc_info=True)
        return None


def _disable_metrics() -> bool:
    global _metrics_registered, _metrics_enabled
    _metrics_registered = True
    _metrics_enabled = False
    return False


def _ensure_metrics_registered() -> bool:
    """Initialise OpenTelemetry instruments if metrics are enabled."""

    global _metrics_registered, _metrics_enabled, _settings
    with _metrics_registration_lock:
        if _metrics_registered:
            return _metrics_enabled

        settings = _load_settings()
        if settings is None:
            return _disable_metrics()

        _settings = settings
        if not getattr(settings, "otel_metrics_enabled", False):
            return _disable_metrics()

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
        self._metrics_lock: asyncio.Lock | None = None
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
        self._metrics_cache = {
            "queue_depth": 0,
            "active_workers": 0,
            "concurrency": self.concurrency,
            "worker_uptime_seconds": 0.0,
            "tasks_processed": 0,
            "tasks_failed": 0,
            "task_failure_rate": 0.0,
            "load_factor": 0.0,
        }
        self._loop: asyncio.AbstractEventLoop | None = None
        self._metrics_registered_with_meter = False
        self._telemetry_last_broadcast: float = 0.0
        self._register_with_communicator()

    @property
    def metric_attributes(self) -> dict[str, int | str]:
        return self._metric_attributes

    @property
    def queue_depth(self) -> int:
        return self._metrics_cache["queue_depth"]

    @property
    def worker_uptime_seconds(self) -> float:
        if self._in_event_loop_thread():
            return self._compute_worker_uptime(time.monotonic())
        loop = self._loop
        if not loop or not loop.is_running():
            return self._metrics_cache["worker_uptime_seconds"]
        future = asyncio.run_coroutine_threadsafe(self._update_metrics(), loop)
        return future.result()["worker_uptime_seconds"]

    @property
    def task_failure_rate(self) -> float:
        if self._in_event_loop_thread():
            return self._compute_failure_rate()
        loop = self._loop
        if not loop or not loop.is_running():
            return self._metrics_cache["task_failure_rate"]
        future = asyncio.run_coroutine_threadsafe(self._update_metrics(), loop)
        return future.result()["task_failure_rate"]

    async def get_metrics_snapshot(self) -> dict[str, float | int]:
        """Return a point-in-time view of scheduler health metrics."""

        return await self._update_metrics()

    async def add_task(self, task: Callable[[], Coroutine[Any, Any, Any]]) -> None:
        """Queue a coroutine factory for execution."""
        if self._stopping:
            raise RuntimeError("Scheduler is stopping")
        await self.queue.put(task)
        await self._update_metrics()

    async def _worker(self, worker_id: int) -> None:
        await self._update_metrics(
            lambda: self._worker_start_times.__setitem__(worker_id, time.monotonic())
        )
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
                    await self._update_metrics(self._mark_task_failure)
                    logger.error(
                        "scheduler.task_failure",
                        extra={
                            "scheduler_id": self.instance_id,
                            "worker_id": worker_id,
                        },
                        exc_info=True,
                    )
                else:
                    snapshot = await self._update_metrics(self._mark_task_success)
                    await self._dispatch_result(result, snapshot)
                finally:
                    self.queue.task_done()
                    await self._update_metrics()
        except asyncio.CancelledError:
            pass
        finally:
            await self._update_metrics(lambda: self._finalise_worker(worker_id))

    async def _emit_metrics_log(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_metrics_emit < self.metrics_interval:
            return
        snapshot = await self.get_metrics_snapshot()
        logger.info(
            "scheduler.metrics",
            extra={"scheduler_id": self.instance_id, **snapshot},
        )
        self._last_metrics_emit = now
        await self._publish_telemetry(snapshot, force=force)

    async def _publish_telemetry(
        self, snapshot: dict[str, float | int], force: bool = False
    ) -> None:
        payload = dict(snapshot)
        payload["scheduler_id"] = self.instance_id
        payload.setdefault("observed_at", datetime.now(timezone.utc).isoformat())

        update_local = getattr(self.communicator, "update_local_telemetry", None)
        if update_local is not None:
            try:
                update_local(payload)
            except Exception:  # pragma: no cover - defensive
                logger.warning(
                    "scheduler.telemetry_cache_failed",
                    extra={"scheduler_id": self.instance_id},
                    exc_info=True,
                )

        now = time.monotonic()
        if not force and now - self._telemetry_last_broadcast < self.metrics_interval:
            return

        broadcast = getattr(self.communicator, "broadcast_telemetry", None)
        if broadcast is None:
            return
        success = False
        try:
            success = await broadcast(payload)
        except Exception:  # pragma: no cover - defensive
            logger.warning(
                "scheduler.telemetry_broadcast_failed",
                extra={"scheduler_id": self.instance_id},
                exc_info=True,
            )
            return

        if success:
            self._telemetry_last_broadcast = time.monotonic()

    async def run(self) -> None:
        if self._running:
            return
        self._stopping = False
        self._running = True
        self._loop = asyncio.get_running_loop()
        self._metrics_registered_with_meter = self.configure_metrics()
        if self._metrics_registered_with_meter:
            _scheduler_registry.add(self)
        self._last_metrics_emit = time.monotonic()
        self._workers = [
            asyncio.create_task(self._worker(index))
            for index in range(self.concurrency)
        ]
        try:
            await self._emit_metrics_log(force=True)
            while self._running:
                await asyncio.sleep(0.1)
                await self._update_metrics()
                await self._emit_metrics_log()
        finally:
            await self.queue.join()
            for worker in self._workers:
                worker.cancel()
            await asyncio.gather(*self._workers, return_exceptions=True)
            await self._update_metrics()
            await self._emit_metrics_log(force=True)
            if self._metrics_registered_with_meter:
                _scheduler_registry.discard(self)
            self._metrics_registered_with_meter = False
            self._loop = None

    def stop(self) -> None:
        self._stopping = True
        if self._running:
            self._running = False
            for worker in list(self._workers):
                worker.cancel()

    @classmethod
    def configure_metrics(cls) -> bool:
        return _ensure_metrics_registered()

    def _in_event_loop_thread(self) -> bool:
        loop = self._loop
        if loop is None:
            return False
        try:
            return asyncio.get_running_loop() is loop
        except RuntimeError:
            return False

    def _get_metrics_lock(self) -> asyncio.Lock:
        if self._metrics_lock is None:
            self._metrics_lock = asyncio.Lock()
        return self._metrics_lock

    async def _update_metrics(
        self, mutate: Callable[[], None] | None = None
    ) -> dict[str, float | int]:
        lock = self._get_metrics_lock()
        async with lock:
            if mutate:
                mutate()
            snapshot = self._compute_metrics_snapshot(time.monotonic())
            self._metrics_cache = snapshot
            return snapshot

    def _compute_metrics_snapshot(self, now: float) -> dict[str, float | int]:
        uptime = self._compute_worker_uptime(now)
        processed = self._processed_tasks
        failed = self._failed_tasks
        failure_rate = 0.0 if processed == 0 else failed / processed
        queue_depth = self.queue.qsize()
        active_workers = len(self._worker_start_times)
        load_factor = self._calculate_load(queue_depth, active_workers)
        return {
            "queue_depth": queue_depth,
            "active_workers": active_workers,
            "concurrency": self.concurrency,
            "worker_uptime_seconds": uptime,
            "tasks_processed": processed,
            "tasks_failed": failed,
            "task_failure_rate": failure_rate,
            "load_factor": load_factor,
        }

    def _compute_worker_uptime(self, now: float) -> float:
        uptime = self._worker_uptime_total
        for start_time in self._worker_start_times.values():
            uptime += now - start_time
        return uptime

    def _compute_failure_rate(self) -> float:
        processed = self._metrics_cache["tasks_processed"]
        failed = self._metrics_cache["tasks_failed"]
        return 0.0 if processed == 0 else failed / processed

    def _mark_task_success(self) -> None:
        self._processed_tasks += 1

    def _mark_task_failure(self) -> None:
        self._processed_tasks += 1
        self._failed_tasks += 1

    def _finalise_worker(self, worker_id: int) -> None:
        start_time = self._worker_start_times.pop(worker_id, None)
        if start_time is not None:
            self._worker_uptime_total += time.monotonic() - start_time

    def _register_with_communicator(self) -> None:
        register = getattr(self.communicator, "register_load_provider", None)
        if register is None:
            return
        try:
            register(self.get_load_snapshot)
        except Exception:  # pragma: no cover - defensive
            logger.warning(
                "scheduler.load_provider_registration_failed",
                extra={"scheduler_id": self.instance_id},
                exc_info=True,
            )

    async def _dispatch_result(
        self,
        result: Any,
        metrics_snapshot: dict[str, float | int] | None = None,
    ) -> None:
        payload = {"result": result, "origin": self.instance_id}
        snapshot = metrics_snapshot or await self.get_metrics_snapshot()
        local_load = float(snapshot.get("load_factor", 0.0))
        peer_loads = await self._get_peer_loads()
        targets = self._select_peer_targets(peer_loads, local_load)
        if targets:
            await self._send_to_targets(targets, payload)
        else:
            await self._send_to_targets(None, payload)

    async def _send_to_targets(
        self, targets: Iterable[str] | None, payload: dict[str, Any]
    ) -> None:
        send_to = getattr(self.communicator, "send_to", None)
        if targets:
            if send_to is not None:
                await send_to(targets, payload)
                return
        await self.communicator.send(payload)

    async def _get_peer_loads(self) -> dict[str, float]:
        loads: dict[str, float] = {}
        get_cached = getattr(self.communicator, "get_cached_peer_loads", None)
        if get_cached is not None:
            try:
                horizon = max(10.0, self.metrics_interval * 3)
                cached = get_cached(max_age=horizon)
                if cached:
                    loads.update({k: float(v) for k, v in cached.items()})
            except Exception:  # pragma: no cover - defensive
                logger.warning(
                    "scheduler.peer_load_cache_failed",
                    extra={"scheduler_id": self.instance_id},
                    exc_info=True,
                )

        peers = getattr(self.communicator, "peers", None)
        needs_refresh = not loads or (
            peers and any(peer not in loads for peer in peers)
        )
        fetch = getattr(self.communicator, "fetch_peer_loads", None)
        if fetch is None:
            return loads
        if not needs_refresh:
            return loads
        try:
            fresh = await fetch()
        except Exception:  # pragma: no cover - defensive
            logger.warning(
                "scheduler.peer_load_fetch_failed",
                extra={"scheduler_id": self.instance_id},
                exc_info=True,
            )
            return loads
        loads.update({k: float(v) for k, v in fresh.items()})
        return loads

    def _select_peer_targets(
        self, peer_loads: dict[str, float], local_load: float
    ) -> list[str]:
        if not peer_loads:
            return []
        telemetry_map: dict[str, dict[str, Any]] = {}
        get_map = getattr(self.communicator, "get_peer_telemetry_map", None)
        horizon = max(30.0, self.metrics_interval * 5)
        if get_map is not None:
            try:
                telemetry_map = get_map(max_age=horizon)
            except Exception:  # pragma: no cover - defensive
                logger.warning(
                    "scheduler.peer_telemetry_read_failed",
                    extra={"scheduler_id": self.instance_id},
                    exc_info=True,
                )

        if not telemetry_map and peer_loads:
            logger.warning(
                "scheduler.selection_fallback_no_telemetry",
                extra={"scheduler_id": self.instance_id},
            )
            telemetry_map = {}
            for peer, load in peer_loads.items():
                if not isinstance(load, (int, float)):
                    continue
                load_value = float(load)
                if not math.isfinite(load_value) or load_value < 0.0:
                    continue
                telemetry_map[peer] = {"load_factor": load_value}

        candidates: list[tuple[str, float, float, float]] = []
        for peer, load in peer_loads.items():
            if (
                not isinstance(load, (int, float))
                or not math.isfinite(load)
                or load < 0
            ):
                continue
            metadata = telemetry_map.get(peer, {})
            base_load = metadata.get("load_factor")
            if not isinstance(base_load, (int, float)) or not math.isfinite(base_load):
                base_load = float(load)
            failure_rate = metadata.get("task_failure_rate") or 0.0
            try:
                failure_rate = float(failure_rate)
            except (TypeError, ValueError):
                failure_rate = 0.0
            queue_depth = metadata.get("queue_depth") or 0
            concurrency = metadata.get("concurrency") or 1
            try:
                queue_depth = int(queue_depth)
            except (TypeError, ValueError):
                queue_depth = 0
            try:
                concurrency = int(concurrency)
            except (TypeError, ValueError):
                concurrency = 1
            concurrency = max(1, concurrency)
            queue_pressure = queue_depth / concurrency
            queue_penalty = min(0.5, queue_pressure * 0.05)
            age = metadata.get("age_seconds") or 0.0
            try:
                age = float(age)
            except (TypeError, ValueError):
                age = 0.0
            age_penalty = min(0.5, age / horizon)
            effective_load = (
                float(base_load) + max(0.0, failure_rate) + queue_penalty + age_penalty
            )
            candidates.append((peer, float(load), effective_load, float(failure_rate)))

        if not candidates:
            return []

        candidates.sort(key=lambda item: item[2])
        tolerance = max(0.05, min(0.35, local_load * 0.2))
        selected: list[str] = []
        for peer, reported_load, effective_load, failure_rate in candidates:
            if failure_rate >= 0.9:
                continue
            if effective_load + tolerance < local_load:
                selected.append(peer)
            elif (
                effective_load <= candidates[0][2] + tolerance
                and reported_load + tolerance < local_load
            ):
                selected.append(peer)
            if len(selected) >= self.concurrency:
                break

        if selected:
            return selected

        best_peer, _, best_effective, failure_rate = candidates[0]
        if failure_rate < 0.9 and best_effective + tolerance < local_load:
            return [best_peer]
        return []

    def _calculate_load(self, queue_depth: int, active_workers: int) -> float:
        concurrency = max(1, self.concurrency)
        load = (queue_depth + active_workers) / concurrency
        return load

    async def get_load_snapshot(self) -> dict[str, float | int]:
        metrics = await self.get_metrics_snapshot()
        return {
            "scheduler_id": self.instance_id,
            "queue_depth": int(metrics["queue_depth"]),
            "active_workers": int(metrics["active_workers"]),
            "concurrency": int(metrics["concurrency"]),
            "load_factor": float(metrics["load_factor"]),
        }
