import asyncio
from typing import Any, Dict, Set

from fastapi import WebSocket, WebSocketDisconnect


class WebSocketManager:
    """Manage WebSocket connections per user."""

    def __init__(self) -> None:
        self.connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, user_id: str, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self.connections.setdefault(user_id, set()).add(ws)

    async def disconnect(self, user_id: str, ws: WebSocket) -> None:
        async with self._lock:
            conns = self.connections.get(user_id)
            if conns:
                conns.discard(ws)
                if not conns:
                    self.connections.pop(user_id, None)

    async def broadcast(self, user_id: str, message: Dict[str, Any]) -> None:
        async with self._lock:
            conns = list(self.connections.get(user_id, set()))
        to_remove: Set[WebSocket] = set()
        for ws in conns:
            try:
                await ws.send_json(message)
            except WebSocketDisconnect:
                to_remove.add(ws)
            except Exception:
                to_remove.add(ws)
        for ws in to_remove:
            await self.disconnect(user_id, ws)
