"""Lightweight async database primitives for unit tests and local runs."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator
from weakref import WeakKeyDictionary

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

try:  # pragma: no cover - optional dependency during tests
    import aiosqlite  # noqa: F401

    HAS_AIOSQLITE = True
except ModuleNotFoundError:  # pragma: no cover - fallback to sync engine
    HAS_AIOSQLITE = False

from monGARS.config import get_settings
from monGARS.db import (
    Base,
    ConversationHistory,
    Interaction,
    UserAccount,
    UserPersonality,
    UserPreferences,
)

logger = logging.getLogger(__name__)

_settings = get_settings()
_raw_database_url = os.environ.get("DATABASE_URL") or str(_settings.database_url)

try:
    _database_url_obj: URL = make_url(_raw_database_url)
except Exception:  # pragma: no cover - invalid URL fallback
    logger.warning(
        "Invalid DATABASE_URL '%s'; falling back to sqlite+aiosqlite", _raw_database_url
    )
    _database_url_obj = make_url("sqlite+aiosqlite:///./mongars_local.db")

_async_engine = None
_async_session_maker: async_sessionmaker[AsyncSession] | None = None
_sync_engine = None
_sync_session_maker: sessionmaker | None = None
_using_async_engine = False

if _database_url_obj.drivername.startswith("postgresql"):
    if not _database_url_obj.drivername.startswith("postgresql+"):
        _database_url_obj = _database_url_obj.set(drivername="postgresql+asyncpg")
    _async_engine = create_async_engine(
        str(_database_url_obj),
        future=True,
        echo=_settings.debug,
        pool_size=_settings.db_pool_size,
        max_overflow=_settings.db_max_overflow,
        pool_timeout=_settings.db_pool_timeout,
    )
    _async_session_maker = async_sessionmaker(_async_engine, expire_on_commit=False)
    _using_async_engine = True
elif _database_url_obj.drivername.startswith("sqlite+aiosqlite") and HAS_AIOSQLITE:
    _async_engine = create_async_engine(str(_database_url_obj), future=True, echo=False)
    _async_session_maker = async_sessionmaker(_async_engine, expire_on_commit=False)
    _using_async_engine = True
else:
    sqlite_url = (
        _database_url_obj.set(drivername="sqlite")
        if _database_url_obj.drivername.startswith("sqlite")
        else make_url("sqlite:///./mongars_local.db")
    )
    _sync_engine = create_engine(
        str(sqlite_url),
        future=True,
        echo=False,
        connect_args={"check_same_thread": False},
    )
    _sync_session_maker = sessionmaker(_sync_engine, expire_on_commit=False)

_init_locks: "WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = (
    WeakKeyDictionary()
)
_initialized = False

__all__ = [
    "Base",
    "ConversationHistory",
    "Interaction",
    "UserAccount",
    "UserPersonality",
    "UserPreferences",
    "async_session_factory",
    "reset_database",
]


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
        if _using_async_engine:
            assert _async_engine is not None
            async with _async_engine.begin() as conn:
                if conn.dialect.name == "postgresql":
                    try:
                        await conn.execute(
                            text("CREATE EXTENSION IF NOT EXISTS vector")
                        )
                    except Exception as exc:  # pragma: no cover - extension optional
                        logger.warning("Unable to ensure pgvector extension: %s", exc)
                await conn.run_sync(Base.metadata.create_all)
        else:
            assert _sync_engine is not None
            await asyncio.to_thread(Base.metadata.create_all, _sync_engine)
        _initialized = True


@asynccontextmanager
async def async_session_factory() -> AsyncIterator[AsyncSession]:
    """Provide an ``AsyncSession`` with an initialized schema."""

    await _ensure_schema()
    if _using_async_engine:
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
        if _using_async_engine:
            assert _async_engine is not None
            async with _async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
                await conn.run_sync(Base.metadata.create_all)
        else:
            assert _sync_engine is not None
            await asyncio.to_thread(Base.metadata.drop_all, _sync_engine)
            await asyncio.to_thread(Base.metadata.create_all, _sync_engine)
        _initialized = True
