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
