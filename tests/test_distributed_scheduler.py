import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from monGARS.core import distributed_scheduler
from monGARS.core.distributed_scheduler import DistributedScheduler
from monGARS.core.peer import PeerCommunicator


@pytest.mark.asyncio
async def test_scheduler_broadcasts(monkeypatch):
    async def fake_send(msg):
        calls.append(msg)
        return [True]

    communicator = PeerCommunicator([])
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(communicator, "send", fake_send)
    scheduler = DistributedScheduler(communicator)

    async def task():
        return "done"

    await scheduler.add_task(task)
    run_task = asyncio.create_task(scheduler.run())
    await asyncio.sleep(0.05)
    scheduler.stop()
    await run_task
    assert calls[0] == {"result": "done"}


@pytest.mark.asyncio
async def test_scheduler_concurrent(monkeypatch):
    communicator = PeerCommunicator([])
    results: list[int] = []

    async def fake_send(msg):
        results.append(msg["result"])
        return [True]

    monkeypatch.setattr(communicator, "send", fake_send)
    scheduler = DistributedScheduler(communicator, concurrency=2)

    def make_task(n: int):
        async def _task():
            await asyncio.sleep(0.01)
            return n

        return _task

    for i in range(5):
        await scheduler.add_task(make_task(i))

    run = asyncio.create_task(scheduler.run())
    await asyncio.sleep(0.1)
    scheduler.stop()
    await run
    assert sorted(results) == list(range(5))


@pytest.mark.asyncio
async def test_scheduler_metrics_snapshot(monkeypatch):
    communicator = PeerCommunicator([])
    sent: list[dict[str, Any]] = []

    async def fake_send(msg):
        sent.append(msg)
        return [True]

    monkeypatch.setattr(communicator, "send", fake_send)
    scheduler = DistributedScheduler(communicator, concurrency=1, metrics_interval=0.1)

    async def task():
        await asyncio.sleep(0.01)
        return "payload"

    await scheduler.add_task(task)
    runner = asyncio.create_task(scheduler.run())
    await asyncio.sleep(0.2)
    scheduler.stop()
    await runner

    snapshot = await scheduler.get_metrics_snapshot()
    assert sent == [{"result": "payload"}]
    assert snapshot["queue_depth"] == 0
    assert snapshot["tasks_processed"] == 1
    assert snapshot["tasks_failed"] == 0
    assert snapshot["task_failure_rate"] == 0.0
    assert snapshot["worker_uptime_seconds"] >= 0.0


@pytest.mark.asyncio
async def test_scheduler_failure_metrics(monkeypatch):
    communicator = PeerCommunicator([])
    sent: list[dict[str, Any]] = []

    async def fake_send(msg):
        sent.append(msg)
        return [True]

    monkeypatch.setattr(communicator, "send", fake_send)
    scheduler = DistributedScheduler(communicator, concurrency=1, metrics_interval=0.1)

    async def success():
        return "ok"

    async def failure():
        raise RuntimeError("boom")

    await scheduler.add_task(success)
    await scheduler.add_task(failure)

    runner = asyncio.create_task(scheduler.run())
    await asyncio.sleep(0.2)
    scheduler.stop()
    await runner

    snapshot = await scheduler.get_metrics_snapshot()
    assert sent == [{"result": "ok"}]
    assert snapshot["tasks_processed"] == 2
    assert snapshot["tasks_failed"] == 1
    assert snapshot["task_failure_rate"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_scheduler_metrics_logging(monkeypatch):
    communicator = PeerCommunicator([])
    sent: list[dict[str, Any]] = []

    async def fake_send(msg):
        sent.append(msg)
        return [True]

    monkeypatch.setattr(communicator, "send", fake_send)
    scheduler = DistributedScheduler(communicator, concurrency=1, metrics_interval=0.1)

    async def success():
        return "ok"

    async def failure():
        raise RuntimeError("boom")

    await scheduler.add_task(success)
    await scheduler.add_task(failure)

    metrics_logs: list[dict[str, Any]] = []

    def capture_metrics_log(*args, **kwargs):
        extra = kwargs.get("extra")
        if extra is not None:
            metrics_logs.append(extra)

    with patch.object(
        distributed_scheduler.logger, "info", side_effect=capture_metrics_log
    ):
        runner = asyncio.create_task(scheduler.run())
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 1.0
        while len(metrics_logs) < 2 and loop.time() < deadline:
            await asyncio.sleep(0.05)
        assert len(metrics_logs) >= 2
        scheduler.stop()
        await runner

    snapshot = await scheduler.get_metrics_snapshot()
    assert sent == [{"result": "ok"}]
    assert snapshot["tasks_processed"] == 2
    assert snapshot["tasks_failed"] == 1
    assert snapshot["task_failure_rate"] == pytest.approx(0.5)

    assert len(metrics_logs) >= 3
    assert "tasks_processed" in metrics_logs[-1]
    assert metrics_logs[-1]["tasks_processed"] == 2
