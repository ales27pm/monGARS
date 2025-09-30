from __future__ import annotations

import asyncio
from typing import List

import pytest

from monGARS.api.ws_manager import WebSocketManager
from monGARS.core.ui_events import Event


class DummyWebSocket:
    def __init__(self, *, fail: bool = False) -> None:
        self.accepted = False
        self.closed = False
        self.sent: List[str] = []
        self.fail = fail
        self.event = asyncio.Event()

    async def accept(self) -> None:
        self.accepted = True

    async def send_text(self, message: str) -> None:
        if self.fail:
            raise RuntimeError("send failure")
        self.sent.append(message)
        self.event.set()

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_connect_and_disconnect_cleans_up() -> None:
    manager = WebSocketManager()
    ws = DummyWebSocket()

    await manager.connect(ws, "user-1")

    assert ws.accepted
    async with manager._lock:  # noqa: SLF001 - test-only inspection
        assert ws in manager.active["user-1"]

    await manager.disconnect(ws, "user-1")

    assert ws.closed
    async with manager._lock:  # noqa: SLF001 - test-only inspection
        assert "user-1" not in manager.active


@pytest.mark.asyncio
async def test_send_event_routes_by_user() -> None:
    manager = WebSocketManager()
    alice = DummyWebSocket()
    bob = DummyWebSocket()
    await manager.connect(alice, "alice")
    await manager.connect(bob, "bob")

    event = Event(id="1", type="chat.message", ts=1.0, user="alice", data={"x": 1})
    await manager.send_event(event)

    assert alice.sent, "Expected Alice to receive the event"
    assert not bob.sent, "Bob should not receive Alice's event"


@pytest.mark.asyncio
async def test_send_event_broadcasts_when_user_missing() -> None:
    manager = WebSocketManager()
    sockets = [DummyWebSocket() for _ in range(3)]
    for idx, ws in enumerate(sockets):
        await manager.connect(ws, f"user-{idx}")

    event = Event(id="2", type="system.broadcast", ts=2.0, user=None, data={"ok": True})
    await manager.send_event(event)

    for ws in sockets:
        assert ws.sent, "All sockets should receive broadcast events"


@pytest.mark.asyncio
async def test_send_event_removes_failing_sockets() -> None:
    manager = WebSocketManager()
    healthy = DummyWebSocket()
    failing = DummyWebSocket(fail=True)
    await manager.connect(healthy, "user")
    await manager.connect(failing, "user")

    event = Event(id="3", type="chat.message", ts=3.0, user="user", data={})
    await manager.send_event(event)

    assert healthy.sent
    assert failing.closed
    async with manager._lock:  # noqa: SLF001 - test-only inspection
        assert failing not in manager.active.get("user", set())


@pytest.mark.asyncio
async def test_background_fanout_consumes_bus(monkeypatch) -> None:
    manager = WebSocketManager()
    ws = DummyWebSocket()
    await manager.connect(ws, "user")

    class DummyBus:
        def __init__(self) -> None:
            self.queue: asyncio.Queue[Event] = asyncio.Queue()

        async def publish(self, ev: Event) -> None:
            await self.queue.put(ev)

        def subscribe(self):  # type: ignore[override]
            async def iterator():
                while True:
                    yield await self.queue.get()

            return iterator()

    bus = DummyBus()
    monkeypatch.setattr("monGARS.api.ws_manager.event_bus", lambda: bus)

    manager.ensure_background_fanout()

    event = Event(id="4", type="chat.message", ts=4.0, user="user", data={})
    await bus.publish(event)

    await asyncio.wait_for(ws.event.wait(), timeout=0.5)

    await manager.reset()
