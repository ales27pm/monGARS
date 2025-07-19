import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    user_id: str
    query: str
    response: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class Hippocampus:
    """Simple in-memory store for conversation history."""

    MAX_HISTORY = 100

    def __init__(self) -> None:
        self._memory: Dict[str, Deque[MemoryItem]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    def _get_lock(self, user_id: str) -> asyncio.Lock:
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    async def store(self, user_id: str, query: str, response: str) -> None:
        """Persist a query/response pair for a user."""
        logger.debug("Storing interaction for %s", user_id)
        async with self._get_lock(user_id):
            history = self._memory.setdefault(user_id, deque(maxlen=self.MAX_HISTORY))
            history.append(MemoryItem(user_id=user_id, query=query, response=response))

    async def history(self, user_id: str, limit: int = 10) -> List[MemoryItem]:
        """Return recent conversation history."""
        if limit <= 0:
            return []
        limit = min(limit, self.MAX_HISTORY)
        async with self._get_lock(user_id):
            history = self._memory.get(user_id, deque())
            items = list(history)[-limit:]
            return list(reversed(items))
