"""Lightweight async database primitives for unit tests and local runs."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator
from weakref import WeakKeyDictionary

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

try:  # pragma: no cover - optional dependency during tests
    import aiosqlite  # noqa: F401

    HAS_AIOSQLITE = True
except ModuleNotFoundError:  # pragma: no cover - fallback to sync engine
    HAS_AIOSQLITE = False


class Base(DeclarativeBase):
    """Base class for all lightweight ORM models."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ConversationHistory(Base):
    __tablename__ = "conversation_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    query: Mapped[str] = mapped_column(String)
    response: Mapped[str] = mapped_column(String)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    vector: Mapped[list[float] | None] = mapped_column(JSON, default=list)


class Interaction(Base):
    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    session_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    input_data: Mapped[dict] = mapped_column(JSON, default=dict)
    output_data: Mapped[dict] = mapped_column(JSON, default=dict)
    message: Mapped[str] = mapped_column(String)
    response: Mapped[str] = mapped_column(String)
    personality: Mapped[dict] = mapped_column(JSON, default=dict)
    context: Mapped[dict] = mapped_column(JSON, default=dict)
    meta_data: Mapped[str | None] = mapped_column(String, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    processing_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )


class UserPreferences(Base):
    __tablename__ = "user_preferences"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    interaction_style: Mapped[dict] = mapped_column(JSON, default=dict)
    preferred_topics: Mapped[dict] = mapped_column(JSON, default=dict)


class UserPersonality(Base):
    __tablename__ = "user_personality"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    traits: Mapped[dict] = mapped_column(JSON, default=dict)
    interaction_style: Mapped[dict] = mapped_column(JSON, default=dict)
    context_preferences: Mapped[dict] = mapped_column(JSON, default=dict)
    adaptation_rate: Mapped[float] = mapped_column(Float, default=0.1)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class UserAccount(Base):
    __tablename__ = "user_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(150), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )


if HAS_AIOSQLITE:
    DATABASE_URL = "sqlite+aiosqlite:///./mongars_local.db"
    _async_engine = create_async_engine(DATABASE_URL, future=True, echo=False)
    _async_session_maker = async_sessionmaker(_async_engine, expire_on_commit=False)
    _sync_engine = None
    _sync_session_maker = None
else:
    DATABASE_URL = "sqlite:///./mongars_local.db"
    _sync_engine = create_engine(
        DATABASE_URL,
        future=True,
        echo=False,
        connect_args={"check_same_thread": False},
    )
    _sync_session_maker = sessionmaker(_sync_engine, expire_on_commit=False)
    _async_engine = None
    _async_session_maker = None

_init_locks: "WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = (
    WeakKeyDictionary()
)
_initialized = False


def _get_loop_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _init_locks.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _init_locks[loop] = lock
    return lock


class _AsyncSessionProxy:
    """Minimal async facade over a synchronous SQLAlchemy session."""

    def __init__(self, session):
        self._session = session

    class _AsyncTransactionProxy:
        """Async-compatible wrapper around ``Session.begin`` transactions."""

        def __init__(self, transaction):
            self._transaction = transaction

        async def __aenter__(self):
            return await asyncio.to_thread(self._transaction.__enter__)

        async def __aexit__(self, exc_type, exc, tb):
            return await asyncio.to_thread(
                self._transaction.__exit__, exc_type, exc, tb
            )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    def add(self, obj) -> None:
        self._session.add(obj)

    def begin(self):
        return self._AsyncTransactionProxy(self._session.begin())

    def in_transaction(self) -> bool:
        return self._session.in_transaction()

    async def merge(self, obj):
        return await asyncio.to_thread(self._session.merge, obj)

    async def execute(self, statement, params=None):
        if params is None:
            return await asyncio.to_thread(self._session.execute, statement)
        return await asyncio.to_thread(self._session.execute, statement, params)

    async def commit(self) -> None:
        await asyncio.to_thread(self._session.commit)

    async def rollback(self) -> None:
        await asyncio.to_thread(self._session.rollback)

    async def close(self) -> None:
        await asyncio.to_thread(self._session.close)


async def _ensure_schema() -> None:
    """Create the lightweight schema once per process."""

    global _initialized
    if _initialized:
        return
    lock = _get_loop_lock()
    async with lock:
        if _initialized:
            return
        if HAS_AIOSQLITE:
            assert _async_engine is not None
            async with _async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        else:
            assert _sync_engine is not None
            await asyncio.to_thread(Base.metadata.create_all, _sync_engine)
        _initialized = True


@asynccontextmanager
async def async_session_factory() -> AsyncIterator[AsyncSession]:
    """Provide an ``AsyncSession`` with an initialized schema."""

    await _ensure_schema()
    if HAS_AIOSQLITE:
        assert _async_session_maker is not None
        async with _async_session_maker() as session:
            yield session
    else:
        assert _sync_session_maker is not None
        session = _sync_session_maker()
        proxy = _AsyncSessionProxy(session)
        try:
            yield proxy
        finally:
            await proxy.close()


async def reset_database() -> None:
    """Utility for tests that need a clean in-memory database."""

    global _initialized
    lock = _get_loop_lock()
    async with lock:
        if HAS_AIOSQLITE:
            assert _async_engine is not None
            async with _async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
                await conn.run_sync(Base.metadata.create_all)
        else:
            assert _sync_engine is not None
            await asyncio.to_thread(Base.metadata.drop_all, _sync_engine)
            await asyncio.to_thread(Base.metadata.create_all, _sync_engine)
        _initialized = True
