from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Set

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manage WebSocket connections per user."""

    def __init__(
        self,
        *,
        heartbeat_interval: float = 30.0,
        max_offline_messages: int = 100,
    ) -> None:
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be positive")
        if max_offline_messages <= 0:
            raise ValueError("max_offline_messages must be positive")
        self.connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()
        self._heartbeat_interval = heartbeat_interval
        self._max_offline_messages = max_offline_messages
        self._heartbeat_tasks: Dict[WebSocket, asyncio.Task[None]] = {}
        self._offline_queues: Dict[str, Deque[Dict[str, Any]]] = {}

    async def connect(self, user_id: str, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self.connections.setdefault(user_id, set()).add(ws)
        heartbeat = asyncio.create_task(self._heartbeat(user_id, ws))
        self._heartbeat_tasks[ws] = heartbeat

    async def disconnect(self, user_id: str, ws: WebSocket) -> None:
        await self._remove_connection(user_id, ws)
        task = self._heartbeat_tasks.pop(ws, None)
        current = asyncio.current_task()
        if task and task is not current:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        try:
            await ws.close()
        except RuntimeError:
            # The WebSocket might already be closed by the caller or the client.
            pass
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Failed to close websocket", exc_info=exc)

    async def broadcast(self, user_id: str, message: Dict[str, Any]) -> None:
        await self._send_message(user_id, message, queue_if_offline=True)

    async def flush_offline(self, user_id: str) -> None:
        await self._flush_offline_queue(user_id)

    async def reset(self) -> None:
        tasks = list(self._heartbeat_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._heartbeat_tasks.clear()
        self.connections.clear()
        self._offline_queues.clear()

    async def _remove_connection(self, user_id: str, ws: WebSocket) -> None:
        async with self._lock:
            conns = self.connections.get(user_id)
            if conns and ws in conns:
                conns.remove(ws)
                if not conns:
                    self.connections.pop(user_id, None)

    async def _heartbeat(self, user_id: str, ws: WebSocket) -> None:
        try:
            while True:
                await asyncio.sleep(self._heartbeat_interval)
                try:
                    await ws.send_json(
                        {
                            "type": "ping",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                except WebSocketDisconnect:
                    logger.info(
                        "WebSocket disconnected during heartbeat",
                        extra={"user_id": user_id},
                    )
                    break
                except Exception as exc:
                    logger.warning(
                        "Heartbeat ping failed; scheduling disconnect",
                        exc_info=exc,
                        extra={"user_id": user_id},
                    )
                    break
        except asyncio.CancelledError:
            logger.debug(
                "Heartbeat cancelled for websocket", extra={"user_id": user_id}
            )
            raise
        finally:
            self._heartbeat_tasks.pop(ws, None)
            await self._remove_connection(user_id, ws)

    async def _send_message(
        self,
        user_id: str,
        message: Dict[str, Any],
        *,
        queue_if_offline: bool,
    ) -> bool:
        async with self._lock:
            conns: List[WebSocket] = list(self.connections.get(user_id, set()))
        if not conns:
            if queue_if_offline:
                await self._queue_message(user_id, message)
            return False

        failures: Set[WebSocket] = set()
        delivered = False
        for ws in conns:
            try:
                await ws.send_json(message)
                delivered = True
            except WebSocketDisconnect:
                failures.add(ws)
            except Exception as exc:
                failures.add(ws)
                logger.warning(
                    "Failed to deliver websocket message",
                    exc_info=exc,
                    extra={"user_id": user_id},
                )

        for ws in failures:
            await self.disconnect(user_id, ws)

        if not delivered and queue_if_offline:
            await self._queue_message(user_id, message)

        return delivered

    async def _queue_message(self, user_id: str, message: Dict[str, Any]) -> None:
        payload = dict(message)
        async with self._lock:
            queue = self._offline_queues.setdefault(
                user_id, deque(maxlen=self._max_offline_messages)
            )
            dropped = queue[0] if len(queue) == queue.maxlen else None
            queue.append(payload)
        if dropped is not None:
            logger.warning(
                "Dropping oldest queued message due to capacity",
                extra={"user_id": user_id, "dropped": dropped},
            )

    async def _flush_offline_queue(self, user_id: str) -> None:
        while True:
            async with self._lock:
                queue = self._offline_queues.get(user_id)
                if not queue:
                    return
                message = dict(queue[0])
            delivered = await self._send_message(
                user_id, message, queue_if_offline=False
            )
            if not delivered:
                return
            async with self._lock:
                queue = self._offline_queues.get(user_id)
                if not queue:
                    continue
                if queue:
                    queue.popleft()
                if not queue:
                    self._offline_queues.pop(user_id, None)
