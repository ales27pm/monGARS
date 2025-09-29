from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List, Set

from monGARS.core.persistence import PersistenceRepository

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    user_id: str
    query: str
    response: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class Hippocampus:
    """Hybrid in-memory/persistent store for conversation history."""

    MAX_HISTORY = 100

    def __init__(
        self,
        persistence: PersistenceRepository | None = None,
        *,
        persist_on_store: bool = False,
    ) -> None:
        self._memory: Dict[str, Deque[MemoryItem]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._hydrated_users: Set[str] = set()
        self._persistence = persistence or PersistenceRepository()
        self._persist_on_store = persist_on_store and self._persistence is not None

    def _get_lock(self, user_id: str) -> asyncio.Lock:
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    @staticmethod
    def _normalise_timestamp(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    async def _hydrate_from_persistence(self, user_id: str) -> None:
        if user_id in self._hydrated_users:
            return
        if self._persistence is None:
            return

        records = await self._persistence.get_history(user_id, limit=self.MAX_HISTORY)
        if not records:
            async with self._get_lock(user_id):
                self._hydrated_users.add(user_id)
            return

        items = [
            MemoryItem(
                user_id=record.user_id,
                query=record.query,
                response=record.response,
                timestamp=self._normalise_timestamp(record.timestamp),
            )
            for record in reversed(records)
        ]

        async with self._get_lock(user_id):
            if user_id in self._hydrated_users:
                return
            history = self._memory.setdefault(user_id, deque(maxlen=self.MAX_HISTORY))
            history.clear()
            history.extend(items)
            self._hydrated_users.add(user_id)

    async def store(self, user_id: str, query: str, response: str) -> None:
        """Persist a query/response pair for a user."""
        logger.debug("Storing interaction for %s", user_id)
        item = MemoryItem(user_id=user_id, query=query, response=response)
        async with self._get_lock(user_id):
            history = self._memory.setdefault(user_id, deque(maxlen=self.MAX_HISTORY))
            history.append(item)
            self._hydrated_users.add(user_id)
        if self._persist_on_store and self._persistence is not None:
            try:
                await self._persistence.save_history_entry(
                    user_id=user_id, query=query, response=response
                )
            except Exception:
                logger.exception(
                    "hippocampus.persistence_failed",
                    extra={"user_id": user_id},
                )

    async def history(self, user_id: str, limit: int = 10) -> List[MemoryItem]:
        """Return recent conversation history."""
        if limit <= 0:
            return []
        limit = min(limit, self.MAX_HISTORY)
        await self._hydrate_from_persistence(user_id)
        async with self._get_lock(user_id):
            history = self._memory.get(user_id, deque())
            items = list(history)[-limit:]
            return list(reversed(items))
