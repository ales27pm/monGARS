#!/usr/bin/env python3
"""Programmatic Alembic runner."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import time
from pathlib import Path
from typing import Final

from alembic import command
from alembic.config import Config
from passlib.context import CryptContext
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import SQLAlchemyError

from monGARS.utils.database import apply_database_url_overrides

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_DB_STARTUP_TIMEOUT: Final[float] = 120.0
DEFAULT_DB_STARTUP_RETRY_INTERVAL: Final[float] = 3.0
_TRUTHY_VALUES: Final[set[str]] = {"1", "true", "yes", "on"}
_BOOTSTRAP_PASSWORD_CONTEXT = CryptContext(
    schemes=["pbkdf2_sha256"],
    deprecated="auto",
    default="pbkdf2_sha256",
    pbkdf2_sha256__rounds=390000,
)


def _determine_sync_driver(
    candidates: tuple[str, ...] = ("postgresql+psycopg", "postgresql+psycopg2"),
    *,
    logger: logging.Logger | None = None,
) -> str:
    """Return the first available synchronous PostgreSQL driver."""

    for driver in candidates:
        try:
            backend = driver.split("+", 1)[1]
        except IndexError:  # pragma: no cover - defensive guard
            backend = driver
        if importlib.util.find_spec(backend) is not None:
            if logger:
                logger.debug("Using PostgreSQL driver %s", driver)
            return driver

    if logger:
        logger.warning(
            "Falling back to generic 'postgresql' driver; install psycopg for optimal support.",
        )
    return "postgresql"


SYNC_DRIVERNAME = _determine_sync_driver(logger=logger)


def build_sync_url() -> URL:
    """Build a PostgreSQL SQLAlchemy URL using the available psycopg driver."""

    raw = os.getenv("DATABASE_URL") or os.getenv("DJANGO_DATABASE_URL")
    if raw:
        url = make_url(raw)
        url = apply_database_url_overrides(
            url,
            username=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
            logger=logger,
            field_sources={
                "username": "DB_USER",
                "password": "DB_PASSWORD",
                "host": "DB_HOST",
                "port": "DB_PORT",
                "database": "DB_NAME",
            },
        )
        return url.set(drivername=SYNC_DRIVERNAME)

    return URL.create(
        drivername=SYNC_DRIVERNAME,
        username=os.getenv("DB_USER", "mongars"),
        password=os.getenv("DB_PASSWORD", "changeme"),
        host=os.getenv("DB_HOST", "postgres"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME", "mongars_db"),
    )


def render_url(url: URL, *, hide_password: bool) -> str:
    return url.render_as_string(hide_password=hide_password)


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY_VALUES


def _bootstrap_admin_credentials() -> tuple[str, str] | None:
    if not _is_truthy(os.getenv("ENABLE_ADMIN_BOOTSTRAP")):
        return None

    username = (os.getenv("BOOTSTRAP_ADMIN_USERNAME") or "").strip()
    password = os.getenv("BOOTSTRAP_ADMIN_PASSWORD") or ""

    if not username or not password:
        logger.info(
            "Admin bootstrap enabled but BOOTSTRAP_ADMIN_USERNAME/BOOTSTRAP_ADMIN_PASSWORD are incomplete; skipping initial admin seed"
        )
        return None

    return username, password


def ensure_bootstrap_admin(url: URL) -> bool:
    """Create the first persisted admin account when explicitly configured."""

    credentials = _bootstrap_admin_credentials()
    if credentials is None:
        return False

    username, password = credentials
    engine = create_engine(render_url(url, hide_password=False))

    try:
        with engine.connect() as conn:
            admin_count = conn.execute(
                text("SELECT COUNT(*) FROM user_accounts WHERE is_admin = :is_admin"),
                {"is_admin": True},
            ).scalar_one()
            if admin_count:
                logger.info("Admin user already exists; skipping initial admin seed")
                return False

            existing = conn.execute(
                text(
                    "SELECT is_admin FROM user_accounts WHERE username = :username"
                ),
                {"username": username},
            ).scalar_one_or_none()
            if existing is not None:
                logger.warning(
                    "Configured bootstrap username '%s' already exists; skipping initial admin seed",
                    username,
                )
                return False

            password_hash = _BOOTSTRAP_PASSWORD_CONTEXT.hash(password)
            conn.execute(
                text(
                    "INSERT INTO user_accounts (username, password_hash, is_admin) "
                    "VALUES (:username, :password_hash, :is_admin)"
                ),
                {
                    "username": username,
                    "password_hash": password_hash,
                    "is_admin": True,
                },
            )
            conn.commit()
    finally:
        engine.dispose()

    logger.info("Bootstrapped initial admin user '%s'", username)
    return True


def ensure_extensions(url: URL) -> None:
    """Create required PostgreSQL extensions prior to migrations."""

    if not url.drivername.startswith("postgresql"):
        return

    engine = create_engine(render_url(url, hide_password=False))
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
    finally:
        engine.dispose()


def coerce_positive_float(
    value: str | None, *, default: float, minimum: float
) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        logger.warning(
            "Invalid database startup value '%s'; falling back to default %s",
            value,
            default,
        )
        return default
    if parsed <= minimum:
        logger.warning(
            "Database startup value %s must be greater than %s; using default %s",
            parsed,
            minimum,
            default,
        )
        return default
    return parsed


def wait_for_database(url: URL, *, timeout: float, retry_interval: float) -> None:
    deadline = time.monotonic() + timeout
    attempt = 1
    last_error: Exception | None = None
    rendered_url = render_url(url, hide_password=True)
    logger.info(
        "Waiting for database %s (timeout=%ss, interval=%ss)",
        rendered_url,
        timeout,
        retry_interval,
    )
    while True:
        engine = create_engine(
            render_url(url, hide_password=False),
            pool_pre_ping=True,
        )
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                logger.info("Database %s is available", rendered_url)
                return
        except SQLAlchemyError as exc:
            last_error = exc
            now = time.monotonic()
            if now >= deadline:
                break
            logger.warning(
                "Database not ready yet (%s). Retrying in %ss (attempt %s)",
                exc.__class__.__name__,
                retry_interval,
                attempt,
            )
            logger.debug("Database connection attempt failed", exc_info=exc)
            attempt += 1
            time.sleep(retry_interval)
        finally:
            engine.dispose()

    assert (
        last_error is not None
    )  # for mypy/pylint; loop guarantees assignment on failure
    raise TimeoutError(  # noqa: TRY003
        f"Timed out waiting for database {rendered_url} after {timeout}s",
    ) from last_error


async def init_db() -> None:
    repo_root = Path(__file__).resolve().parent
    cfg = Config(str(repo_root / "alembic.ini"))
    sync_url = build_sync_url()
    sync_url_str = render_url(sync_url, hide_password=False)
    cfg.set_main_option("sqlalchemy.url", sync_url_str)
    timeout = coerce_positive_float(
        os.getenv("DB_STARTUP_TIMEOUT"),
        default=DEFAULT_DB_STARTUP_TIMEOUT,
        minimum=0.0,
    )
    retry_interval = coerce_positive_float(
        os.getenv("DB_STARTUP_RETRY_INTERVAL"),
        default=DEFAULT_DB_STARTUP_RETRY_INTERVAL,
        minimum=0.0,
    )
    wait_for_database(sync_url, timeout=timeout, retry_interval=retry_interval)
    ensure_extensions(sync_url)
    logger.info(
        "Running alembic upgrade head using %s",
        render_url(sync_url, hide_password=True),
    )
    await asyncio.to_thread(command.upgrade, cfg, "head")
    logger.info("Alembic upgrade to head completed")
    await asyncio.to_thread(ensure_bootstrap_admin, sync_url)


def main() -> None:
    try:
        asyncio.run(init_db())
    except Exception:  # pragma: no cover - surface failure details to logs
        logger.exception("DB init failed")
        raise


if __name__ == "__main__":
    main()
