import asyncio
from datetime import datetime, timezone
from typing import Any, Iterable
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
    assert calls[0]["result"] == "done"
    assert "origin" in calls[0]


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
    assert sent[0]["result"] == "payload"
    assert sent[0]["origin"]
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
    assert sent[0]["result"] == "ok"
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
    assert sent[0]["result"] == "ok"
    assert snapshot["tasks_processed"] == 2
    assert snapshot["tasks_failed"] == 1
    assert snapshot["task_failure_rate"] == pytest.approx(0.5)

    assert len(metrics_logs) >= 3
    assert "tasks_processed" in metrics_logs[-1]
    assert metrics_logs[-1]["tasks_processed"] == 2


@pytest.mark.asyncio
async def test_scheduler_routes_to_least_loaded(monkeypatch):
    communicator = PeerCommunicator(
        [
            "http://peer1/api/v1/peer/message",
            "http://peer2/api/v1/peer/message",
        ]
    )

    async def noop_broadcast(payload):
        communicator.update_local_telemetry(payload)
        return True

    monkeypatch.setattr(communicator, "broadcast_telemetry", noop_broadcast)
    routed: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    async def fake_send_to(peers, message):
        routed.append((tuple(peers), message))
        return [True] * len(tuple(peers))

    async def fake_send(message):
        routed.append((("broadcast",), message))
        return [True] * len(communicator.peers)

    async def fake_fetch_peer_loads():
        return {
            "http://peer1/api/v1/peer/message": 0.2,
            "http://peer2/api/v1/peer/message": 0.9,
        }

    monkeypatch.setattr(communicator, "send_to", fake_send_to)
    monkeypatch.setattr(communicator, "send", fake_send)
    monkeypatch.setattr(communicator, "fetch_peer_loads", fake_fetch_peer_loads)

    scheduler = DistributedScheduler(communicator, concurrency=1)

    async def task():
        return "payload"

    await scheduler.add_task(task)
    runner = asyncio.create_task(scheduler.run())
    await asyncio.sleep(0.1)
    scheduler.stop()
    await runner

    assert routed
    first_route, message = routed[0]
    assert first_route == ("http://peer1/api/v1/peer/message",)
    assert message["result"] == "payload"
    assert "origin" in message


@pytest.mark.asyncio
async def test_scheduler_falls_back_to_broadcast_when_no_better_peer(monkeypatch):
    communicator = PeerCommunicator(["http://peer/api/v1/peer/message"])
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    async def noop_broadcast(payload):
        communicator.update_local_telemetry(payload)
        return True

    monkeypatch.setattr(communicator, "broadcast_telemetry", noop_broadcast)

    async def fake_send(message):
        calls.append((("broadcast",), message))
        return [True]

    async def fake_send_to(peers, message):
        calls.append((tuple(peers), message))
        return [True] * len(tuple(peers))

    async def fake_fetch_peer_loads():
        return {"http://peer/api/v1/peer/message": 5.0}

    monkeypatch.setattr(communicator, "send", fake_send)
    monkeypatch.setattr(communicator, "send_to", fake_send_to)
    monkeypatch.setattr(communicator, "fetch_peer_loads", fake_fetch_peer_loads)

    scheduler = DistributedScheduler(communicator, concurrency=1)

    async def task():
        return "payload"

    await scheduler.add_task(task)
    runner = asyncio.create_task(scheduler.run())
    await asyncio.sleep(0.1)
    scheduler.stop()
    await runner

    assert calls
    route, message = calls[0]
    assert route == ("broadcast",)
    assert message["result"] == "payload"


@pytest.mark.asyncio
async def test_scheduler_prefers_cached_peer_load(monkeypatch):
    communicator = PeerCommunicator(["http://peer/api/v1/peer/message"])
    communicator.ingest_remote_telemetry(
        "http://peer/api/v1/peer/message",
        {
            "scheduler_id": "remote",
            "queue_depth": 0,
            "active_workers": 0,
            "concurrency": 1,
            "load_factor": 0.1,
            "worker_uptime_seconds": 1.0,
            "tasks_processed": 5,
            "tasks_failed": 0,
            "task_failure_rate": 0.0,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "source": "http://peer/api/v1/peer/message",
        },
    )

    routed: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    async def fake_send_to(peers, message):
        routed.append((tuple(peers), message))
        return [True] * len(tuple(peers))

    async def fake_send(message):
        routed.append((("broadcast",), message))
        return [True]

    async def fail_fetch():
        raise AssertionError("fetch should not be called")

    async def noop_broadcast(payload):
        communicator.update_local_telemetry(payload)
        return True

    monkeypatch.setattr(communicator, "send_to", fake_send_to)
    monkeypatch.setattr(communicator, "send", fake_send)
    monkeypatch.setattr(communicator, "fetch_peer_loads", fail_fetch)
    monkeypatch.setattr(communicator, "broadcast_telemetry", noop_broadcast)

    scheduler = DistributedScheduler(communicator, concurrency=1, metrics_interval=0.1)

    async def task():
        await asyncio.sleep(0.01)
        return "payload"

    await scheduler.add_task(task)
    runner = asyncio.create_task(scheduler.run())
    await asyncio.sleep(0.2)
    scheduler.stop()
    await runner

    assert routed
    first_route, message = routed[0]
    assert first_route == ("http://peer/api/v1/peer/message",)
    assert message["result"] == "payload"


@pytest.mark.asyncio
async def test_scheduler_registers_load_provider_for_telemetry() -> None:
    communicator = PeerCommunicator(["http://peer/api/v1/peer/message"])
    scheduler = DistributedScheduler(communicator, concurrency=3)

    snapshot = await communicator.get_local_load()
    assert snapshot["scheduler_id"] == scheduler.instance_id
    assert snapshot["concurrency"] == 3
    assert snapshot["queue_depth"] == 0


@pytest.mark.asyncio
async def test_scheduler_dispatch_targets_low_load_peers(monkeypatch):
    communicator = PeerCommunicator(
        [
            "http://peer1/api/v1/peer/message",
            "http://peer2/api/v1/peer/message",
            "http://peer3/api/v1/peer/message",
        ]
    )

    scheduler = DistributedScheduler(communicator, concurrency=2, metrics_interval=0.1)

    async def fake_get_peer_loads() -> dict[str, float]:
        return {
            "http://peer1/api/v1/peer/message": 0.1,
            "http://peer2/api/v1/peer/message": 0.25,
            "http://peer3/api/v1/peer/message": 0.4,
        }

    def fake_get_peer_telemetry_map(
        max_age: float = 0.0, include_self: bool = False
    ) -> dict[str, dict[str, Any]]:
        return {
            "http://peer1/api/v1/peer/message": {
                "load_factor": 0.1,
                "task_failure_rate": 0.0,
                "queue_depth": 0,
                "concurrency": 2,
                "age_seconds": 0.0,
            },
            "http://peer2/api/v1/peer/message": {
                "load_factor": 0.25,
                "task_failure_rate": 0.05,
                "queue_depth": 1,
                "concurrency": 1,
                "age_seconds": 0.0,
            },
            "http://peer3/api/v1/peer/message": {
                "load_factor": 0.4,
                "task_failure_rate": 0.95,
                "queue_depth": 0,
                "concurrency": 1,
                "age_seconds": 0.0,
            },
        }

    send_calls: list[tuple[tuple[str, ...] | None, dict[str, Any]]] = []

    async def capture_send(
        targets: Iterable[str] | None, payload: dict[str, Any]
    ) -> None:
        send_calls.append((tuple(targets) if targets else None, payload))

    monkeypatch.setattr(scheduler, "_get_peer_loads", fake_get_peer_loads)
    monkeypatch.setattr(
        communicator, "get_peer_telemetry_map", fake_get_peer_telemetry_map
    )
    monkeypatch.setattr(scheduler, "_send_to_targets", capture_send)

    await scheduler._dispatch_result("payload", {"load_factor": 2.5})

    assert send_calls, "Scheduler did not dispatch the result"
    targets, payload = send_calls[0]
    assert targets == (
        "http://peer1/api/v1/peer/message",
        "http://peer2/api/v1/peer/message",
    )
    assert payload["origin"] == scheduler.instance_id
