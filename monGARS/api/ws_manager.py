from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from sqlalchemy.exc import SQLAlchemyError

from monGARS.api.dependencies import get_hippocampus
from monGARS.api.ws_ticket import verify_ws_ticket
from monGARS.config import get_settings
from monGARS.core.ui_events import BackendUnavailable, Event, event_bus, make_event

log = logging.getLogger(__name__)
settings = get_settings()


class _TokenBucket:
    """Token bucket used to throttle per-user WebSocket fan-out."""

    __slots__ = ("capacity", "refill_seconds", "tokens", "updated_at")

    def __init__(self, *, capacity: int, refill_seconds: float) -> None:
        self.capacity = capacity
        self.refill_seconds = refill_seconds
        self.tokens = float(capacity)
        self.updated_at = time.monotonic()

    def consume(self, cost: float = 1.0) -> bool:
        now = time.monotonic()
        elapsed = now - self.updated_at
        if elapsed > 0 and self.refill_seconds > 0:
            refill = elapsed / self.refill_seconds
            if refill > 0:
                self.tokens = min(float(self.capacity), self.tokens + refill)
        self.updated_at = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


@dataclass(slots=True)
class _ConnectionState:
    """Track per-connection queues and liveness metadata."""

    ws: WebSocket
    user_id: str
    queue: asyncio.Queue[str]
    sender_task: Optional[asyncio.Task[None]] = None
    closed: bool = False
    last_pong: float = field(default_factory=time.monotonic)
    expected_pong: Optional[str] = None
    requested_close_code: Optional[int] = None

    def mark_pong(self) -> None:
        self.last_pong = time.monotonic()


class WebSocketManager:
    """Manage active WebSocket connections and stream UI events."""

    def __init__(self) -> None:
        self.active: Dict[str, Set[WebSocket]] = {}
        self._reverse: Dict[WebSocket, str] = {}
        self._state: Dict[WebSocket, _ConnectionState] = {}
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._buckets: Dict[str, "_TokenBucket"] = {}

    @staticmethod
    def _rate_limit_enabled() -> bool:
        return (
            settings.WS_RATE_LIMIT_MAX_TOKENS > 0
            and settings.WS_RATE_LIMIT_REFILL_SECONDS > 0
        )

    def _bucket_for(self, user_id: str) -> "_TokenBucket":
        bucket = self._buckets.get(user_id)
        if bucket is None:
            bucket = _TokenBucket(
                capacity=settings.WS_RATE_LIMIT_MAX_TOKENS,
                refill_seconds=settings.WS_RATE_LIMIT_REFILL_SECONDS,
            )
            self._buckets[user_id] = bucket
        return bucket

    @property
    def connections(self) -> Dict[str, Set[WebSocket]]:
        """Backwards compatible alias for tests importing ``connections``."""

        return self.active

    async def connect(self, ws: WebSocket, user_id: str) -> _ConnectionState:
        await ws.accept()
        state = _ConnectionState(
            ws=ws,
            user_id=user_id,
            queue=asyncio.Queue(maxsize=settings.WS_CONNECTION_QUEUE_SIZE),
        )
        async with self._lock:
            self.active.setdefault(user_id, set()).add(ws)
            self._reverse[ws] = user_id
            self._state[ws] = state
        return state

    async def disconnect(
        self, ws: WebSocket, user_id: str, *, code: int | None = None
    ) -> None:
        async with self._lock:
            sockets = self.active.get(user_id)
            if sockets and ws in sockets:
                sockets.remove(ws)
                if not sockets:
                    self.active.pop(user_id, None)
                    self._buckets.pop(user_id, None)
            self._reverse.pop(ws, None)
            state = self._state.pop(ws, None)
        if state:
            state.closed = True
            sender = state.sender_task
            if sender:
                current = asyncio.current_task()
                sender.cancel()
                if sender is not current:
                    with contextlib.suppress(asyncio.CancelledError):
                        await sender
        with contextlib.suppress(Exception):
            await ws.close(code=code)

    async def send_event(self, ev: Event) -> None:
        rate_limited = self._rate_limit_enabled()
        async with self._lock:
            if ev.user is None:
                candidates = [
                    state for state in self._state.values() if not state.closed
                ]
            else:
                sockets = self.active.get(ev.user, set())
                candidates = [
                    self._state[ws]
                    for ws in sockets
                    if ws in self._state and not self._state[ws].closed
                ]

            if not candidates:
                return

            if rate_limited:
                permitted: list[_ConnectionState] = []
                for state in candidates:
                    bucket = self._bucket_for(state.user_id)
                    if bucket.consume():
                        permitted.append(state)
                    else:
                        log.warning(
                            "ws_manager.rate_limited",
                            extra={"user_id": state.user_id, "event_type": ev.type},
                        )
                targets = permitted
            else:
                targets = candidates

        if not targets:
            return

        payload = ev.to_json()
        overflow: list[_ConnectionState] = []
        for state in targets:
            if state.closed:
                continue
            try:
                state.queue.put_nowait(payload)
            except asyncio.QueueFull:
                overflow.append(state)

        if not overflow:
            return

        for state in overflow:
            log.warning(
                "ws_manager.backpressure_closed",
                extra={"user_id": state.user_id, "event_type": ev.type},
            )
            await self.disconnect(state.ws, state.user_id, code=1013)

    def ensure_background_fanout(self) -> None:
        if self._task and not self._task.done():
            return

        async def _run() -> None:
            try:
                async for ev in event_bus().subscribe():
                    await self.send_event(ev)
            except asyncio.CancelledError:
                raise
            except BackendUnavailable:  # pragma: no cover - defensive logging
                log.exception("ws_manager.background_failed")
                raise
            except (
                RuntimeError,
                OSError,
                ValueError,
            ):  # pragma: no cover - defensive logging
                log.exception("ws_manager.background_failed")
                raise
            finally:
                self._task = None

        self._task = asyncio.create_task(_run())

    async def reset(self) -> None:
        task = self._task
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        async with self._lock:
            sockets = list(self._reverse.keys())
            states = list(self._state.values())
            self.active.clear()
            self._reverse.clear()
            self._state.clear()
            self._buckets.clear()
        current = asyncio.current_task()
        for state in states:
            sender = state.sender_task
            if sender:
                sender.cancel()
                if sender is current:
                    continue
                with contextlib.suppress(asyncio.CancelledError):
                    await sender
        for ws in sockets:
            with contextlib.suppress(Exception):
                await ws.close()


ws_manager = WebSocketManager()
router = APIRouter()


async def _enqueue_payload(
    state: _ConnectionState, payload: dict[str, Any], purpose: str
) -> bool:
    """Serialise ``payload`` and enqueue it for transmission."""

    message = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    try:
        state.queue.put_nowait(message)
        return True
    except asyncio.QueueFull:
        log.warning(
            "ws_manager.queue_overflow",
            extra={"user_id": state.user_id, "purpose": purpose},
        )
        await ws_manager.disconnect(state.ws, state.user_id, code=1013)
        return False


async def _sender_loop(state: _ConnectionState) -> None:
    """Drain the per-connection queue and push payloads to the client."""

    try:
        while True:
            payload = await state.queue.get()
            await state.ws.send_text(payload)
    except asyncio.CancelledError:
        raise
    except WebSocketDisconnect:
        raise
    except (OSError, RuntimeError, ValueError):  # pragma: no cover - defensive logging
        if not state.closed:
            log.exception(
                "ws_manager.send_failed",
                extra={"user_id": state.user_id},
            )
        raise


async def _heartbeat_loop(state: _ConnectionState) -> None:
    """Emit periodic pings and ensure clients respond within the timeout."""

    interval = settings.WS_HEARTBEAT_INTERVAL_SECONDS
    timeout = settings.WS_HEARTBEAT_TIMEOUT_SECONDS
    while True:
        await asyncio.sleep(interval)
        if time.monotonic() - state.last_pong > timeout:
            raise TimeoutError("heartbeat timeout")
        ping_id = str(uuid.uuid4())
        state.expected_pong = ping_id
        queued = await _enqueue_payload(
            state,
            {"id": ping_id, "type": "ping", "payload": None},
            "heartbeat",
        )
        if not queued:
            return


def _prepare_ack(
    state: _ConnectionState, raw: str
) -> tuple[dict[str, Any], Optional[int]]:
    """Validate ``raw`` and produce an acknowledgement envelope."""

    ack_id = str(uuid.uuid4())
    try:
        message = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "id": ack_id,
            "type": "ack",
            "payload": {"status": "error", "detail": "invalid_json"},
        }, None

    if not isinstance(message, dict):
        return {
            "id": ack_id,
            "type": "ack",
            "payload": {"status": "error", "detail": "invalid_envelope"},
        }, None

    errors: list[str] = []
    msg_type = message.get("type")
    if not isinstance(msg_type, str) or not msg_type:
        errors.append("missing_type")
        return {
            "id": ack_id,
            "type": "ack",
            "payload": {"status": "error", "detail": ",".join(errors)},
        }, None

    raw_id = message.get("id")
    if isinstance(raw_id, str) and raw_id:
        ack_id = raw_id
    elif msg_type != "client.ping":
        errors.append("missing_id")

    if errors:
        return {
            "id": ack_id,
            "type": "ack",
            "payload": {"status": "error", "detail": ",".join(errors)},
        }, None

    if msg_type == "pong":
        state.mark_pong()
        expected = state.expected_pong
        if expected and expected != raw_id:
            log.debug(
                "ws_manager.pong_mismatch",
                extra={
                    "user_id": state.user_id,
                    "expected": expected,
                    "received": raw_id,
                },
            )
        state.expected_pong = None
        return {"id": ack_id, "type": "ack", "payload": {"status": "ok"}}, None

    if msg_type == "client.ping":
        state.mark_pong()
        if state.expected_pong and raw_id == state.expected_pong:
            state.expected_pong = None
        return {
            "id": ack_id,
            "type": "ack",
            "payload": {"status": "ok", "detail": "client.ping"},
        }, None

    if msg_type == "close":
        return {"id": ack_id, "type": "ack", "payload": {"status": "ok"}}, 1000

    return {"id": ack_id, "type": "ack", "payload": {"status": "ok"}}, None


async def _receiver_loop(state: _ConnectionState) -> None:
    """Read client frames and queue acknowledgements."""

    while True:
        raw = await state.ws.receive_text()
        ack, close_code = _prepare_ack(state, raw)
        queued = await _enqueue_payload(state, ack, "ack")
        if not queued:
            return
        if close_code is not None:
            state.requested_close_code = close_code
            return


@router.websocket("/ws/chat/")
async def ws_chat(ws: WebSocket, ticket: str = Query(..., alias="t")) -> None:
    allowed = {str(origin) for origin in settings.WS_ALLOWED_ORIGINS}
    origin = ws.headers.get("origin", "")
    if allowed and origin not in allowed:
        await ws.close(code=4403)
        return

    if not settings.WS_ENABLE_EVENTS:
        await ws.close(code=4403)
        return

    try:
        user_id = verify_ws_ticket(ticket)
    except HTTPException as exc:
        close_code = 4401 if exc.status_code == status.HTTP_401_UNAUTHORIZED else 4403
        log.warning(
            "ws_manager.ticket_rejected",
            extra={"status_code": exc.status_code},
        )
        await ws.close(code=close_code)
        return
    except (RuntimeError, UnicodeError, ValueError):
        log.exception("ws_manager.ticket_verification_failed")
        await ws.close(code=1011)
        return

    state = await ws_manager.connect(ws, user_id)
    ws_manager.ensure_background_fanout()

    connected_event = make_event(
        "ws.connected",
        user_id,
        {"origin": origin},
    )
    await ws.send_text(connected_event.to_json())

    try:
        store = get_hippocampus()
        history = await store.history(user_id, limit=10)
        payload = [
            {
                "query": item.query,
                "response": item.response,
                "timestamp": item.timestamp.isoformat(),
            }
            for item in history
        ]
        snapshot = make_event(
            "history.snapshot",
            user_id,
            {"items": payload},
        )
        await ws.send_text(snapshot.to_json())
    except (RuntimeError, SQLAlchemyError):  # pragma: no cover - defensive logging
        log.exception(
            "ws_manager.history_failed",
            extra={"user_id": user_id},
        )

    state.sender_task = asyncio.create_task(_sender_loop(state))
    receiver_task = asyncio.create_task(_receiver_loop(state))
    heartbeat_task = asyncio.create_task(_heartbeat_loop(state))
    close_code: int | None = None
    try:
        done, pending = await asyncio.wait(
            [state.sender_task, receiver_task, heartbeat_task],
            return_when=asyncio.FIRST_EXCEPTION,
        )
        for task in pending:
            task.cancel()
        for task in done:
            exc = task.exception()
            if exc is None or isinstance(exc, asyncio.CancelledError):
                continue
            if isinstance(exc, WebSocketDisconnect):
                close_code = getattr(exc, "code", close_code)
                break
            if isinstance(exc, TimeoutError):
                close_code = 1011
                log.warning(
                    "ws_manager.heartbeat_timeout",
                    extra={"user_id": user_id},
                )
                break
            log.exception(
                "ws_manager.connection_task_failed",
                extra={"user_id": user_id, "task": task.get_name() or "unknown"},
            )
    except WebSocketDisconnect as exc:
        close_code = getattr(exc, "code", close_code)
    finally:
        for task in (receiver_task, heartbeat_task):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect):
                await task
        if close_code is None:
            close_code = state.requested_close_code
        await ws_manager.disconnect(ws, user_id, code=close_code)
