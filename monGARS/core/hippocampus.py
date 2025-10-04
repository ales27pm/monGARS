from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Deque, Dict, List, Set

try:  # pragma: no cover - optional dependency for scheduled flushes
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
except ModuleNotFoundError:  # pragma: no cover - scheduler is optional during tests
    AsyncIOScheduler = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - used only for typing helpers
    from apscheduler.schedulers.asyncio import AsyncIOScheduler as _AsyncIOScheduler
from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError

from monGARS.core.persistence import PersistenceRepository
from monGARS.db import MemoryEntry
from monGARS.utils.database import AsyncSessionFactory, session_scope

try:  # pragma: no cover - optional fallback during tests
    from monGARS.init_db import async_session_factory as _default_async_session_factory
except Exception:  # pragma: no cover - init_db may be unavailable in some contexts
    _default_async_session_factory = None

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return the current UTC datetime."""

    return datetime.now(timezone.utc)


@dataclass
class MemoryItem:
    user_id: str
    query: str
    response: str
    timestamp: datetime = field(default_factory=_utcnow)
    expires_at: datetime | None = None


class Hippocampus:
    """Hybrid in-memory/persistent store for conversation history."""

    MAX_HISTORY = 100

    def __init__(
        self,
        persistence: PersistenceRepository | None = None,
        *,
        persist_on_store: bool = False,
        session_factory: AsyncSessionFactory | None = None,
        default_ttl: timedelta = timedelta(hours=24),
        flush_interval_seconds: int = 300,
        enable_scheduler: bool = True,
    ) -> None:
        self._memory: Dict[str, Deque[MemoryItem]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._hydrated_users: Set[str] = set()
        self._persistence = persistence or PersistenceRepository()
        self._persist_on_store = persist_on_store and self._persistence is not None
        self._session_factory: AsyncSessionFactory | None = session_factory
        self._session_factory_fallback: AsyncSessionFactory | None = (
            _default_async_session_factory
        )
        self._default_ttl = default_ttl
        self._flush_interval_seconds = flush_interval_seconds
        scheduler_available = AsyncIOScheduler is not None
        if enable_scheduler and not scheduler_available:
            logger.warning(
                "hippocampus.scheduler_unavailable",
                extra={"reason": "apscheduler missing"},
            )
        self._enable_scheduler = enable_scheduler and scheduler_available
        self._scheduler: "_AsyncIOScheduler" | None = (
            AsyncIOScheduler(timezone=timezone.utc)
            if self._enable_scheduler and AsyncIOScheduler is not None
            else None
        )
        self._scheduler_lock = asyncio.Lock()
        self._scheduler_started = False
        self._flush_job_id = "hippocampus.flush"

    def _get_lock(self, user_id: str) -> asyncio.Lock:
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    def _resolve_session_factory(self) -> AsyncSessionFactory | None:
        if self._session_factory is not None:
            return self._session_factory
        if self._session_factory_fallback is None:
            return None
        self._session_factory = self._session_factory_fallback
        return self._session_factory

    @staticmethod
    def _normalise_timestamp(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _now(self) -> datetime:
        return _utcnow()

    def _compute_expiry(self, base: datetime, ttl: timedelta | None = None) -> datetime:
        duration = ttl or self._default_ttl
        if duration.total_seconds() <= 0:
            return base
        return base + duration

    def _prune_user_history(self, history: Deque[MemoryItem], now: datetime) -> int:
        if not history:
            return 0
        filtered = [
            item for item in history if item.expires_at is None or item.expires_at > now
        ]
        removed = len(history) - len(filtered)
        if removed:
            history.clear()
            history.extend(filtered)
        return removed

    async def _start_scheduler_if_needed(self) -> None:
        if not self._enable_scheduler or self._scheduler is None:
            return
        if self._scheduler_started:
            return
        async with self._scheduler_lock:
            if self._scheduler_started:
                return
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            self._scheduler.configure(event_loop=loop)
            if self._scheduler.get_job(self._flush_job_id) is None:
                self._scheduler.add_job(
                    self.flush_now,
                    trigger="interval",
                    seconds=self._flush_interval_seconds,
                    id=self._flush_job_id,
                    coalesce=True,
                    max_instances=1,
                )
            if not self._scheduler.running:
                self._scheduler.start()
            self._scheduler_started = True

    async def _hydrate_from_persistence(self, user_id: str) -> None:
        if user_id in self._hydrated_users:
            return

        now = self._now()
        items: list[MemoryItem] = []

        session_factory = self._resolve_session_factory()
        if session_factory is not None:
            try:
                async with session_scope(session_factory) as session:
                    stmt = (
                        select(MemoryEntry)
                        .where(MemoryEntry.user_id == user_id)
                        .where(MemoryEntry.ttl > now)
                        .order_by(MemoryEntry.timestamp.desc())
                        .limit(self.MAX_HISTORY)
                    )
                    result = await session.execute(stmt)
                    records = result.scalars().all()
            except SQLAlchemyError:
                logger.exception(
                    "hippocampus.hydrate_failed",
                    extra={"user_id": user_id},
                )
                records = []
            else:
                if records:
                    items.extend(
                        MemoryItem(
                            user_id=record.user_id,
                            query=record.query,
                            response=record.response,
                            timestamp=self._normalise_timestamp(record.timestamp),
                            expires_at=self._normalise_timestamp(record.ttl),
                        )
                        for record in reversed(records)
                    )

        if not items and self._persistence is not None:
            try:
                history_records = await self._persistence.get_history(
                    user_id, limit=self.MAX_HISTORY
                )
            except Exception:
                logger.exception(
                    "hippocampus.persistence_fetch_failed",
                    extra={"user_id": user_id},
                )
                history_records = []
            if history_records:
                items.extend(
                    MemoryItem(
                        user_id=record.user_id,
                        query=record.query,
                        response=record.response,
                        timestamp=self._normalise_timestamp(record.timestamp),
                        expires_at=self._compute_expiry(
                            self._normalise_timestamp(record.timestamp)
                        ),
                    )
                    for record in reversed(history_records)
                )

        async with self._get_lock(user_id):
            if user_id in self._hydrated_users:
                return
            history = self._memory.setdefault(user_id, deque(maxlen=self.MAX_HISTORY))
            history.clear()
            if items:
                history.extend(items)
            self._prune_user_history(history, now)
            self._hydrated_users.add(user_id)

    async def _persist_memory_entry(self, item: MemoryItem) -> None:
        session_factory = self._resolve_session_factory()
        if session_factory is None:
            return
        expires_at = item.expires_at or self._compute_expiry(item.timestamp)
        entry = MemoryEntry(
            user_id=item.user_id,
            query=item.query,
            response=item.response,
            timestamp=item.timestamp,
            # ``ttl`` stores the absolute expiration timestamp for legacy schema compatibility.
            ttl=expires_at,
        )
        try:
            async with session_scope(session_factory, commit=True) as session:
                session.add(entry)
        except SQLAlchemyError:
            logger.exception(
                "hippocampus.memory_entry_persist_failed",
                extra={"user_id": item.user_id},
            )

    async def store(
        self,
        user_id: str,
        query: str,
        response: str,
        *,
        ttl: timedelta | None = None,
    ) -> MemoryItem:
        """Persist a query/response pair for a user."""

        await self._start_scheduler_if_needed()
        logger.debug("Storing interaction for %s", user_id)
        timestamp = self._now()
        expires_at = self._compute_expiry(timestamp, ttl)
        item = MemoryItem(
            user_id=user_id,
            query=query,
            response=response,
            timestamp=timestamp,
            expires_at=expires_at,
        )
        async with self._get_lock(user_id):
            history = self._memory.setdefault(user_id, deque(maxlen=self.MAX_HISTORY))
            history.append(item)
            self._prune_user_history(history, self._now())
            self._hydrated_users.add(user_id)

        await self._persist_memory_entry(item)

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
        return item

    async def history(self, user_id: str, limit: int = 10) -> List[MemoryItem]:
        """Return recent conversation history."""

        if limit <= 0:
            return []

        await self._start_scheduler_if_needed()
        limit = min(limit, self.MAX_HISTORY)
        await self._hydrate_from_persistence(user_id)
        async with self._get_lock(user_id):
            history = self._memory.get(user_id, deque())
            self._prune_user_history(history, self._now())
            items = list(history)[-limit:]
            return list(reversed(items))

    async def flush_now(self) -> int:
        """Flush expired memory entries from cache and persistence."""

        now = self._now()
        removed = 0

        for user_id in list(self._memory.keys()):
            async with self._get_lock(user_id):
                history = self._memory.get(user_id)
                if history is None:
                    continue
                removed += self._prune_user_history(history, now)
                if not history:
                    self._memory.pop(user_id, None)
                    self._hydrated_users.discard(user_id)
                    self._locks.pop(user_id, None)

        session_factory = self._resolve_session_factory()
        if session_factory is None:
            return removed

        deleted_ids: list[int] = []
        affected_users: Set[str] = set()
        try:
            async with session_scope(session_factory, commit=True) as session:
                stmt = select(MemoryEntry.id, MemoryEntry.user_id).where(
                    MemoryEntry.ttl <= now
                )
                result = await session.execute(stmt)
                rows = result.all()
                candidate_ids = [row.id for row in rows]
                if candidate_ids:
                    deleted_ids = candidate_ids
                    affected_users = {row.user_id for row in rows}
                    await session.execute(
                        delete(MemoryEntry).where(MemoryEntry.id.in_(deleted_ids))
                    )
        except SQLAlchemyError:
            logger.exception(
                "hippocampus.flush_failed",
                extra={"removed_in_memory": removed},
            )
            return removed

        if deleted_ids:
            removed += len(deleted_ids)
            for user in affected_users:
                self._hydrated_users.discard(user)
                if user not in self._memory:
                    self._locks.pop(user, None)

        logger.debug(
            "hippocampus.flush", extra={"removed": removed, "expired_ids": deleted_ids}
        )
        return removed

    def shutdown(self) -> None:
        """Stop the background scheduler."""

        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            self._scheduler_started = False
