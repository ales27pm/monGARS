from __future__ import annotations

import asyncio
import contextlib
from typing import List

import pytest

import monGARS.api.ws_manager as ws_module
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

    async def close(
        self, code: int | None = None
    ) -> None:  # noqa: ARG002 - parity with FastAPI API
        self.closed = True


@pytest.mark.asyncio
async def test_connect_and_disconnect_cleans_up() -> None:
    manager = WebSocketManager()
    ws = DummyWebSocket()

    state = await manager.connect(ws, "user-1")

    assert ws.accepted
    async with manager._lock:  # noqa: SLF001 - test-only inspection
        assert ws in manager.active["user-1"]
        assert manager._state[ws] is state  # noqa: SLF001 - test-only inspection

    await manager.disconnect(ws, "user-1")

    assert ws.closed
    async with manager._lock:  # noqa: SLF001 - test-only inspection
        assert "user-1" not in manager.active
        assert ws not in manager._state  # noqa: SLF001 - test-only inspection


@pytest.mark.asyncio
async def test_send_event_routes_by_user() -> None:
    manager = WebSocketManager()
    alice = DummyWebSocket()
    bob = DummyWebSocket()
    state_alice = await manager.connect(alice, "alice")
    state_bob = await manager.connect(bob, "bob")
    state_alice.sender_task = asyncio.create_task(ws_module._sender_loop(state_alice))
    state_bob.sender_task = asyncio.create_task(ws_module._sender_loop(state_bob))

    event = Event(id="1", type="chat.message", ts=1.0, user="alice", data={"x": 1})
    await manager.send_event(event)

    await asyncio.wait_for(alice.event.wait(), timeout=0.5)
    assert alice.sent, "Expected Alice to receive the event"
    assert not bob.sent, "Bob should not receive Alice's event"

    for task in (state_alice.sender_task, state_bob.sender_task):
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    await manager.reset()


@pytest.mark.asyncio
async def test_send_event_broadcasts_when_user_missing() -> None:
    manager = WebSocketManager()
    sockets = [DummyWebSocket() for _ in range(3)]
    states = []
    for idx, ws in enumerate(sockets):
        state = await manager.connect(ws, f"user-{idx}")
        state.sender_task = asyncio.create_task(ws_module._sender_loop(state))
        states.append(state)

    event = Event(id="2", type="system.broadcast", ts=2.0, user=None, data={"ok": True})
    await manager.send_event(event)

    for ws in sockets:
        await asyncio.wait_for(ws.event.wait(), timeout=0.5)
        assert ws.sent, "All sockets should receive broadcast events"

    for state in states:
        if state.sender_task:
            state.sender_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await state.sender_task

    await manager.reset()


@pytest.mark.asyncio
async def test_send_event_queue_overflow_disconnects(monkeypatch) -> None:
    manager = WebSocketManager()
    monkeypatch.setattr(ws_module.settings, "WS_CONNECTION_QUEUE_SIZE", 1)
    ws = DummyWebSocket()
    await manager.connect(ws, "user")

    event = Event(id="3", type="chat.message", ts=3.0, user="user", data={})
    await manager.send_event(event)

    # queue has one item and is not drained; second event should overflow and disconnect
    overflow = Event(id="4", type="chat.message", ts=4.0, user="user", data={})
    await manager.send_event(overflow)

    assert ws.closed
    async with manager._lock:  # noqa: SLF001 - test-only inspection
        assert "user" not in manager.active

    await manager.reset()


@pytest.mark.asyncio
async def test_background_fanout_consumes_bus(monkeypatch) -> None:
    manager = WebSocketManager()
    ws = DummyWebSocket()
    state = await manager.connect(ws, "user")
    state.sender_task = asyncio.create_task(ws_module._sender_loop(state))

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
    if state.sender_task:
        state.sender_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await state.sender_task
