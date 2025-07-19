import asyncio

import pytest

from monGARS.core.distributed_scheduler import DistributedScheduler
from monGARS.core.peer import PeerCommunicator


@pytest.mark.asyncio
async def test_scheduler_broadcasts(monkeypatch):
    async def fake_send(msg):
        calls.append(msg)
        return [True]

    communicator = PeerCommunicator([])
    calls: list[dict] = []
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
