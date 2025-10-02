from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("ws.config.invalid_int", extra={"name": name, "value": raw})
        return default
    return max(1, value)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("ws.config.invalid_float", extra={"name": name, "value": raw})
        return default
    return max(0.01, value)


MAX_QUEUE_SIZE = _env_int("WS_MAX_QUEUE_SIZE", 128)
HEARTBEAT_INTERVAL = _env_float("WS_HEARTBEAT_INTERVAL", 20.0)
HEARTBEAT_TIMEOUT = _env_float("WS_HEARTBEAT_TIMEOUT", 60.0)


@dataclass(slots=True)
class Envelope:
    """A simple message envelope."""

    id: str
    type: str
    payload: Any


class Connection:
    """State and helpers for a single WebSocket connection."""

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self.send_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(
            maxsize=MAX_QUEUE_SIZE
        )
        self.sender_task: asyncio.Task[None] | None = None
        self.heartbeat_task: asyncio.Task[None] | None = None
        self.alive = True
        self.last_pong = time.monotonic()
        self.close_code = 1000
        self.close_reason = "normal closure"
        self._close_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start background tasks for outbound sends and heartbeat."""

        self.sender_task = asyncio.create_task(self._sender())
        self.heartbeat_task = asyncio.create_task(self._heartbeat())

    async def send(self, message: dict[str, Any]) -> None:
        """Queue a message for delivery or close on backpressure."""

        if not self.alive:
            raise RuntimeError("connection is closed")
        try:
            self.send_queue.put_nowait(message)
        except asyncio.QueueFull:
            logger.warning("ws.backpressure", extra={"policy": "close"})
            await self.close(code=1013, reason="backpressure limit exceeded")
            raise RuntimeError("connection closed due to backpressure") from None

    async def close(self, code: int = 1000, reason: str = "") -> None:
        """Close the connection gracefully."""

        async with self._close_lock:
            if not self.alive:
                return
            self.alive = False
            self.close_code = code
            self.close_reason = reason or self.close_reason
            if self.heartbeat_task and not self.heartbeat_task.done():
                self.heartbeat_task.cancel()
            # Ensure the sender loop can exit.
            inserted = False
            while not inserted:
                try:
                    self.send_queue.put_nowait(None)
                    inserted = True
                except asyncio.QueueFull:
                    try:
                        self.send_queue.get_nowait()
                    except asyncio.QueueEmpty:  # pragma: no cover - defensive
                        inserted = True
            await self.wait_closed()

    async def wait_closed(self) -> None:
        """Wait for background tasks to finish."""

        tasks = [t for t in (self.sender_task, self.heartbeat_task) if t is not None]
        for task in tasks:
            if task is None:
                continue
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("ws.connection.task_failed")

    def mark_pong(self) -> None:
        self.last_pong = time.monotonic()

    async def _sender(self) -> None:
        try:
            while True:
                payload = await self.send_queue.get()
                if payload is None:
                    break
                await self.websocket.send_json(payload)
        except WebSocketDisconnect:
            logger.info("ws.sender.disconnect")
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("ws.sender.failed")
        finally:
            with contextlib.suppress(Exception):
                await self.websocket.close(
                    code=self.close_code, reason=self.close_reason
                )

    async def _heartbeat(self) -> None:
        try:
            while self.alive:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                if not self.alive:
                    break
                if time.monotonic() - self.last_pong > HEARTBEAT_TIMEOUT:
                    logger.warning("ws.heartbeat.timeout")
                    await self.close(code=1011, reason="missed heartbeats")
                    break
                ping = {
                    "id": str(uuid.uuid4()),
                    "type": "ping",
                    "payload": {"ts": time.time()},
                }
                try:
                    await self.send(ping)
                except RuntimeError:
                    break
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("ws.heartbeat.failed")
            await self.close(code=1011, reason="heartbeat failure")


class ConnectionRegistry:
    """Track live connections for graceful shutdown."""

    def __init__(self) -> None:
        self._connections: set[Connection] = set()
        self._lock = asyncio.Lock()

    async def add(self, connection: Connection) -> None:
        async with self._lock:
            self._connections.add(connection)

    async def remove(self, connection: Connection) -> None:
        async with self._lock:
            self._connections.discard(connection)

    async def close_all(self, *, code: int, reason: str) -> None:
        async with self._lock:
            current = list(self._connections)
            self._connections.clear()
        await asyncio.gather(
            *(conn.close(code=code, reason=reason) for conn in current),
            return_exceptions=True,
        )


registry = ConnectionRegistry()


@asynccontextmanager
async def _lifespan(_: FastAPI):
    try:
        yield
    finally:
        await registry.close_all(code=1001, reason="server shutdown")


app = FastAPI(title="Realtime WebSocket Service", lifespan=_lifespan)


def _parse_envelope(data: Any) -> Envelope:
    if not isinstance(data, dict):
        raise ValueError("Envelope must be a JSON object")
    msg_id = data.get("id")
    msg_type = data.get("type")
    if not isinstance(msg_id, str) or not msg_id:
        raise ValueError("Field 'id' must be a non-empty string")
    if not isinstance(msg_type, str) or not msg_type:
        raise ValueError("Field 'type' must be a non-empty string")
    payload = data.get("payload")
    return Envelope(id=msg_id, type=msg_type, payload=payload)


@app.websocket("/ws")
async def websocket_handler(websocket: WebSocket) -> None:
    await websocket.accept()
    connection = Connection(websocket)
    await registry.add(connection)
    await connection.start()

    ready_message = {
        "id": str(uuid.uuid4()),
        "type": "ready",
        "payload": {"message": "connected"},
    }
    try:
        await connection.send(ready_message)
    except RuntimeError:
        await registry.remove(connection)
        await connection.close(code=1011, reason="failed to queue ready message")
        return

    try:
        while True:
            try:
                message = await websocket.receive_json()
            except WebSocketDisconnect:
                raise
            except json.JSONDecodeError as exc:
                error_id = str(uuid.uuid4())
                try:
                    await connection.send(
                        {
                            "id": error_id,
                            "type": "ack",
                            "payload": {
                                "status": "error",
                                "message": "invalid JSON",
                            },
                        }
                    )
                except RuntimeError:
                    pass
                logger.debug("ws.invalid_json", exc_info=exc)
                continue

            try:
                envelope = _parse_envelope(message)
            except ValueError as exc:
                message_id = (
                    message["id"]
                    if isinstance(message, dict) and isinstance(message.get("id"), str)
                    else str(uuid.uuid4())
                )
                try:
                    await connection.send(
                        {
                            "id": message_id,
                            "type": "ack",
                            "payload": {
                                "status": "error",
                                "message": str(exc),
                            },
                        }
                    )
                except RuntimeError:
                    pass
                continue

            if envelope.type == "pong":
                connection.mark_pong()
                continue

            if envelope.type == "ack":
                continue

            if envelope.type == "close":
                try:
                    await connection.send(
                        {
                            "id": envelope.id,
                            "type": "ack",
                            "payload": {"status": "closing"},
                        }
                    )
                except RuntimeError:
                    pass
                await connection.close(code=1000, reason="client requested close")
                break

            try:
                await connection.send(
                    {
                        "id": envelope.id,
                        "type": "ack",
                        "payload": {
                            "status": "ok",
                            "type": envelope.type,
                        },
                    }
                )
            except RuntimeError:
                break

            response = {
                "id": envelope.id,
                "type": "echo",
                "payload": envelope.payload,
            }
            try:
                await connection.send(response)
            except RuntimeError:
                break
    except WebSocketDisconnect:
        logger.info("ws.disconnect")
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("ws.handler.failed")
        await connection.close(code=1011, reason="unexpected error")
    finally:
        await registry.remove(connection)
        await connection.close(code=1000, reason="connection closed")
