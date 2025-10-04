"""Database URL helpers and session utilities."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Mapping, Protocol

from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncSession


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def apply_database_url_overrides(
    url: URL,
    *,
    username: str | None = None,
    password: str | None = None,
    host: str | None = None,
    port: int | str | None = None,
    database: str | None = None,
    logger: logging.Logger | None = None,
    field_sources: Mapping[str, str] | None = None,
) -> URL:
    """Return a URL with discrete overrides applied.

    Parameters mirror PostgreSQL connection attributes. Values that are ``None`` or empty
    strings are ignored. When ``logger`` is provided, debug statements indicate which
    non-sensitive components were overridden and invalid port inputs are reported at
    warning level without echoing the original value.
    """

    field_sources = field_sources or {}

    overrides: dict[str, object] = {}

    norm_username = _normalize_text(username)
    if norm_username and norm_username != url.username:
        overrides["username"] = norm_username
        if logger:
            logger.debug(
                "Applying %s override to database username.",
                field_sources.get("username", "username"),
            )

    norm_password = _normalize_text(password)
    if norm_password and norm_password != url.password:
        overrides["password"] = norm_password

    norm_host = _normalize_text(host)
    if norm_host and norm_host != url.host:
        overrides["host"] = norm_host
        if logger:
            logger.debug(
                "Applying %s override to database host.",
                field_sources.get("host", "host"),
            )

    if port is not None:
        try:
            port_int = int(port)
        except (TypeError, ValueError):
            if logger:
                logger.warning("Invalid database port override provided; ignoring.")
        else:
            if url.port != port_int:
                overrides["port"] = port_int
                if logger:
                    logger.debug(
                        "Applying %s override to database port.",
                        field_sources.get("port", "port"),
                    )

    norm_database = _normalize_text(database)
    if norm_database and norm_database != url.database:
        overrides["database"] = norm_database
        if logger:
            logger.debug(
                "Applying %s override to database name.",
                field_sources.get("database", "database"),
            )

    if not overrides:
        return url

    return url.set(**overrides)


class AsyncSessionFactory(Protocol):
    """Protocol describing the async session factory used across the app."""

    def __call__(self) -> AsyncIterator[AsyncSession]:  # pragma: no cover - typing aid
        ...


@asynccontextmanager
async def session_scope(
    factory: AsyncSessionFactory, *, commit: bool = False
) -> AsyncIterator[AsyncSession]:
    """Yield an :class:`AsyncSession` with optional commit semantics."""

    async with factory() as session:
        try:
            yield session
            if commit:
                await session.commit()
        except Exception:
            try:
                await session.rollback()
            except Exception:  # pragma: no cover - defensive rollback
                pass
            raise
