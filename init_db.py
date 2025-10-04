#!/usr/bin/env python3
"""Programmatic Alembic runner."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def build_sync_url() -> URL:
    """Build a psycopg2 SQLAlchemy URL from environment configuration."""

    raw = os.getenv("DATABASE_URL") or os.getenv("DJANGO_DATABASE_URL")
    if raw:
        url = make_url(raw)
        return url.set(drivername="postgresql+psycopg2")

    return URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("DB_USER", "mongars"),
        password=os.getenv("DB_PASSWORD", "changeme"),
        host=os.getenv("DB_HOST", "postgres"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME", "mongars_db"),
    )


def render_url(url: URL, *, hide_password: bool) -> str:
    return url.render_as_string(hide_password=hide_password)


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


async def init_db() -> None:
    repo_root = Path(__file__).resolve().parent
    cfg = Config(str(repo_root / "alembic.ini"))
    sync_url = build_sync_url()
    sync_url_str = render_url(sync_url, hide_password=False)
    cfg.set_main_option("sqlalchemy.url", sync_url_str)
    ensure_extensions(sync_url)
    logger.info(
        "Running alembic upgrade head using %s",
        render_url(sync_url, hide_password=True),
    )
    await asyncio.to_thread(command.upgrade, cfg, "head")
    logger.info("Alembic upgrade to head completed")


def main() -> None:
    try:
        asyncio.run(init_db())
    except Exception:  # pragma: no cover - surface failure details to logs
        logger.exception("DB init failed")
        raise


if __name__ == "__main__":
    main()
