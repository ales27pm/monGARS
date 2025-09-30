from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest
from fastapi import WebSocketDisconnect

from monGARS.api.ws_manager import WebSocketManager


class DummyWebSocket:
    def __init__(
        self,
        *,
        fail_after: int | None = None,
        fail_with_disconnect: bool = False,
    ) -> None:
        self.accepted = False
        self.closed = False
        self.sent: List[Dict[str, Any]] = []
        self.fail_after = fail_after
        self.fail_with_disconnect = fail_with_disconnect
        self.close_code: int | None = None

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, message: Dict[str, Any]) -> None:
        if self.fail_after is not None and len(self.sent) >= self.fail_after:
            if self.fail_with_disconnect:
                raise WebSocketDisconnect()
            raise RuntimeError("send failure")
        self.sent.append(message)

    async def close(self, code: int | None = None) -> None:
        self.closed = True
        self.close_code = code


@pytest.mark.asyncio
async def test_broadcast_queue_and_replay_on_reconnect() -> None:
    manager = WebSocketManager(heartbeat_interval=0.1, max_offline_messages=10)

    message = {"payload": "data"}
    await manager.broadcast("user", message)

    reconnect_ws = DummyWebSocket()
    await manager.connect("user", reconnect_ws)
    await manager.flush_offline("user")

    assert reconnect_ws.sent[0]["payload"] == "data"

    await manager.disconnect("user", reconnect_ws)


@pytest.mark.asyncio
async def test_disconnect_and_queue_on_subsequent_broadcast() -> None:
    manager = WebSocketManager(heartbeat_interval=0.1)
    active_ws = DummyWebSocket()
    await manager.connect("user", active_ws)

    await manager.broadcast("user", {"payload": "online"})
    assert any(msg["payload"] == "online" for msg in active_ws.sent)

    await manager.disconnect("user", active_ws)

    await manager.broadcast("user", {"payload": "queued"})

    new_ws = DummyWebSocket()
    await manager.connect("user", new_ws)
    await manager.flush_offline("user")
    assert any(msg["payload"] == "queued" for msg in new_ws.sent)

    await manager.disconnect("user", new_ws)


@pytest.mark.asyncio
async def test_heartbeat_detects_stale_connection() -> None:
    manager = WebSocketManager(heartbeat_interval=0.05)
    failing_ws = DummyWebSocket(fail_after=0)

    await manager.connect("user", failing_ws)

    await asyncio.sleep(0.15)

    async with manager._lock:  # noqa: SLF001 - accessing for assertion in tests
        assert "user" not in manager.connections or not manager.connections["user"]
