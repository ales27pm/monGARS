#!/usr/bin/env python3
"""Programmatic Alembic runner."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Union

from alembic import command
from alembic.config import Config
from sqlalchemy.engine import URL, make_url

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _coerce_port(value: str) -> Union[int, str]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def make_sync_url() -> str:
    """Build a psycopg2 SQLAlchemy URL from env configuration."""

    raw = os.getenv("DATABASE_URL") or os.getenv("DJANGO_DATABASE_URL")
    if raw:
        url = make_url(raw)
        # Force psycopg2 driver while retaining the original database settings.
        return url.set(drivername="postgresql+psycopg2").render_as_string(
            hide_password=False
        )

    return URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("DB_USER", "mongars"),
        password=os.getenv("DB_PASSWORD", "changeme"),
        host=os.getenv("DB_HOST", "postgres"),
        port=_coerce_port(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", "mongars_db"),
    ).render_as_string(hide_password=False)


def redact_url(url: str) -> str:
    try:
        return make_url(url).render_as_string(hide_password=True)
    except Exception:  # pragma: no cover - defensive logging helper
        return "***"


async def init_db() -> None:
    repo_root = Path(__file__).resolve().parent
    cfg = Config(str(repo_root / "alembic.ini"))
    sync_url = make_sync_url()
    cfg.set_main_option("sqlalchemy.url", sync_url)
    logger.info("Running alembic upgrade head using %s", redact_url(sync_url))
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
