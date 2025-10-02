from __future__ import annotations

import asyncio
import json
import socket
import time
import uuid
from dataclasses import dataclass

import pytest
import uvicorn
import websockets

from app import main as app_main


@dataclass
class _ServerHandle:
    url: str
    _server: uvicorn.Server
    _task: asyncio.Task[None]
    _original_values: dict[str, object]

    async def stop(self) -> None:
        if not self._server.should_exit:
            self._server.should_exit = True
            await self._task

    async def __aenter__(self) -> "_ServerHandle":  # pragma: no cover - convenience
        return self

    async def __aexit__(
        self, exc_type, exc, tb
    ) -> None:  # pragma: no cover - convenience
        await self.stop()
        for key, value in self._original_values.items():
            setattr(app_main, key, value)


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def _start_server(**overrides: object) -> _ServerHandle:
    original_values: dict[str, object] = {}
    for name, value in overrides.items():
        original_values[name] = getattr(app_main, name)
        setattr(app_main, name, value)

    port = _get_free_port()
    config = uvicorn.Config(
        app_main.app,
        host="127.0.0.1",
        port=port,
        log_level="error",
        loop="asyncio",
        lifespan="on",
    )
    server = uvicorn.Server(config)
    task: asyncio.Task[None] = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    return _ServerHandle(
        url=f"ws://127.0.0.1:{port}/ws",
        _server=server,
        _task=task,
        _original_values=original_values,
    )


@pytest.mark.asyncio
async def test_round_trip_ack_latency() -> None:
    handle = await _start_server()
    try:
        async with websockets.connect(handle.url) as ws:
            ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=1))
            assert ready["type"] == "ready"

            payload = {
                "id": str(uuid.uuid4()),
                "type": "message",
                "payload": {"msg": "hi"},
            }
            start = time.perf_counter()
            await ws.send(json.dumps(payload))

            ack = json.loads(await asyncio.wait_for(ws.recv(), timeout=1))
            latency = time.perf_counter() - start
            assert ack["type"] == "ack"
            assert ack["id"] == payload["id"]
            assert ack["payload"]["status"] == "ok"
            assert latency < 0.1

            echo = json.loads(await asyncio.wait_for(ws.recv(), timeout=1))
            assert echo["type"] == "echo"
            assert echo["payload"] == payload["payload"]
    finally:
        await handle.stop()
        for key, value in handle._original_values.items():
            setattr(app_main, key, value)


@pytest.mark.asyncio
async def test_heartbeat_pong_response_keeps_connection_alive() -> None:
    handle = await _start_server(
        HEARTBEAT_INTERVAL=0.05,
        HEARTBEAT_TIMEOUT=0.2,
    )
    try:
        async with websockets.connect(handle.url) as ws:
            ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=1))
            assert ready["type"] == "ready"

            ping = json.loads(await asyncio.wait_for(ws.recv(), timeout=1))
            assert ping["type"] == "ping"

            pong = {"id": ping["id"], "type": "pong", "payload": None}
            await ws.send(json.dumps(pong))

            payload = {
                "id": str(uuid.uuid4()),
                "type": "message",
                "payload": {"ok": True},
            }
            await ws.send(json.dumps(payload))

            ack = json.loads(await asyncio.wait_for(ws.recv(), timeout=1))
            assert ack["type"] == "ack"
            echo = json.loads(await asyncio.wait_for(ws.recv(), timeout=1))
            assert echo["type"] == "echo"
            assert echo["payload"] == payload["payload"]
    finally:
        await handle.stop()
        for key, value in handle._original_values.items():
            setattr(app_main, key, value)


@pytest.mark.asyncio
async def test_shutdown_closes_client_without_hanging() -> None:
    handle = await _start_server(
        HEARTBEAT_INTERVAL=0.5,
        HEARTBEAT_TIMEOUT=1.0,
    )
    try:
        async with websockets.connect(handle.url) as ws:
            ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=1))
            assert ready["type"] == "ready"

            await handle.stop()
            await asyncio.wait_for(ws.wait_closed(), timeout=1)
            assert ws.close_code in {1000, 1001, 1012}
    finally:
        for key, value in handle._original_values.items():
            setattr(app_main, key, value)
