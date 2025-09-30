from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Dict, Set

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from monGARS.api.dependencies import get_hippocampus
from monGARS.api.ws_ticket import verify_ws_ticket
from monGARS.config import get_settings
from monGARS.core.ui_events import Event, event_bus, make_event

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


class WebSocketManager:
    """Manage active WebSocket connections and stream UI events."""

    def __init__(self) -> None:
        self.active: Dict[str, Set[WebSocket]] = {}
        self._reverse: Dict[WebSocket, str] = {}
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

    async def connect(self, ws: WebSocket, user_id: str) -> None:
        await ws.accept()
        async with self._lock:
            self.active.setdefault(user_id, set()).add(ws)
            self._reverse[ws] = user_id

    async def disconnect(self, ws: WebSocket, user_id: str) -> None:
        async with self._lock:
            sockets = self.active.get(user_id)
            if sockets and ws in sockets:
                sockets.remove(ws)
                if not sockets:
                    self.active.pop(user_id, None)
                    self._buckets.pop(user_id, None)
            self._reverse.pop(ws, None)
        with contextlib.suppress(Exception):
            await ws.close()

    async def send_event(self, ev: Event) -> None:
        rate_limited = self._rate_limit_enabled()
        async with self._lock:
            if ev.user is None:
                raw_targets = [
                    (ws, self._reverse.get(ws)) for ws in list(self._reverse.keys())
                ]
            else:
                raw_targets = [
                    (ws, ev.user) for ws in list(self.active.get(ev.user, set()))
                ]

            if not raw_targets:
                targets: list[WebSocket] = []
            elif not rate_limited:
                targets = [ws for ws, _ in raw_targets]
            else:
                permitted: list[WebSocket] = []
                for ws, user_id in raw_targets:
                    if not user_id:
                        permitted.append(ws)
                        continue
                    bucket = self._bucket_for(user_id)
                    if bucket.consume():
                        permitted.append(ws)
                    else:
                        log.warning(
                            "ws_manager.rate_limited",
                            extra={"user_id": user_id, "event_type": ev.type},
                        )
                targets = permitted

        if not targets:
            return

        payload = ev.to_json()
        stale: list[WebSocket] = []
        for ws in targets:
            try:
                await ws.send_text(payload)
            except Exception:
                stale.append(ws)

        if not stale:
            return

        async with self._lock:
            for ws in stale:
                uid = self._reverse.pop(ws, None)
                if uid is None:
                    continue
                sockets = self.active.get(uid)
                if sockets and ws in sockets:
                    sockets.remove(ws)
                    if not sockets:
                        self.active.pop(uid, None)
        for ws in stale:
            with contextlib.suppress(Exception):
                await ws.close()

    def ensure_background_fanout(self) -> None:
        if self._task and not self._task.done():
            return

        async def _run() -> None:
            try:
                async for ev in event_bus().subscribe():
                    await self.send_event(ev)
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - defensive logging
                log.exception("ws_manager.background_failed")
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
            self.active.clear()
            self._reverse.clear()
            self._buckets.clear()
        for ws in sockets:
            with contextlib.suppress(Exception):
                await ws.close()


ws_manager = WebSocketManager()
router = APIRouter()


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
    except Exception:
        await ws.close(code=4401)
        return

    await ws_manager.connect(ws, user_id)
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
    except Exception:  # pragma: no cover - defensive logging
        log.exception("ws_manager.history_failed", extra={"user_id": user_id})

    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect(ws, user_id)
