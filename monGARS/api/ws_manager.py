from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import UTC, datetime
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

    def reset(self) -> None:
        for task in self._heartbeat_tasks.values():
            task.cancel()
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
                            "timestamp": datetime.now(UTC).isoformat(),
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

    async def _queue_message(
        self, user_id: str, message: Dict[str, Any], *, to_left: bool = False
    ) -> None:
        async with self._lock:
            queue = self._offline_queues.setdefault(
                user_id, deque(maxlen=self._max_offline_messages)
            )
            if len(queue) >= self._max_offline_messages:
                dropped = queue.popleft()
                logger.warning(
                    "Dropping oldest queued message due to capacity",
                    extra={"user_id": user_id, "dropped": dropped},
                )
            if to_left:
                queue.appendleft(dict(message))
            else:
                queue.append(dict(message))

    async def _flush_offline_queue(self, user_id: str) -> None:
        queued_messages = await self._drain_offline_queue(user_id)
        if not queued_messages:
            return
        for message in queued_messages:
            delivered = await self._send_message(
                user_id, message, queue_if_offline=False
            )
            if not delivered:
                await self._queue_message(user_id, message, to_left=True)
                break

    async def _drain_offline_queue(self, user_id: str) -> List[Dict[str, Any]]:
        async with self._lock:
            queue = self._offline_queues.pop(user_id, None)
            if not queue:
                return []
            return list(queue)
