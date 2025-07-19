import asyncio

import pytest

from monGARS.core.distributed_scheduler import DistributedScheduler
from monGARS.core.peer import PeerCommunicator
from monGARS.core.sommeil import SommeilParadoxal


@pytest.mark.asyncio
async def test_sommeil_runs_when_idle(monkeypatch):
    communicator = PeerCommunicator([])
    scheduler = DistributedScheduler(communicator)

    called = {}

    async def fake_apply():
        called["ok"] = True

    class FakeEngine:
        async def safe_apply_optimizations(self):
            await fake_apply()
            return True

    sommeil = SommeilParadoxal(scheduler, FakeEngine(), check_interval=0)
    sommeil.start()
    await asyncio.sleep(0.02)
    await sommeil.stop()
    assert called.get("ok")


@pytest.mark.asyncio
async def test_sommeil_skips_when_busy(monkeypatch):
    communicator = PeerCommunicator([])
    scheduler = DistributedScheduler(communicator)

    called = False

    async def fake_apply():
        nonlocal called
        called = True

    class FakeEngine:
        async def safe_apply_optimizations(self):
            await fake_apply()
            return True

    async def busy_task():
        await asyncio.sleep(0.05)

    await scheduler.add_task(busy_task)
    run = asyncio.create_task(scheduler.run())
    sommeil = SommeilParadoxal(scheduler, FakeEngine(), check_interval=0)
    sommeil.start()
    await asyncio.sleep(0.01)
    assert not called
    scheduler.stop()
    await run
    await sommeil.stop()
