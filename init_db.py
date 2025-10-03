"""Database bootstrap and session utilities for the production stack."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from alembic import command
from alembic.config import Config as AlembicConfig
from config import get_settings
from monGARS.db import (
    Base,
    ConversationHistory,
    Interaction,
    UserAccount,
    UserPersonality,
    UserPreferences,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()

async_engine = create_async_engine(
    str(settings.database_url),
    echo=settings.debug,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_timeout=settings.db_pool_timeout,
)
async_session_factory = async_sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)

__all__ = [
    "Base",
    "ConversationHistory",
    "Interaction",
    "UserAccount",
    "UserPersonality",
    "UserPreferences",
    "async_session_factory",
    "init_db",
]


def _alembic_config() -> AlembicConfig:
    config_path = Path(__file__).resolve().parent / "alembic.ini"
    alembic_config = AlembicConfig(str(config_path))
    alembic_config.set_main_option("sqlalchemy.url", str(settings.database_url))
    alembic_config.attributes["configure_logger"] = False
    return alembic_config


async def init_db() -> None:
    """Apply database migrations and ensure required extensions are present."""

    try:
        if async_engine.dialect.name == "postgresql":
            async with async_engine.begin() as conn:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except Exception as exc:  # pragma: no cover - extension optional in tests
        logger.warning("Unable to ensure pgvector extension: %s", exc)

    config = _alembic_config()
    await asyncio.to_thread(command.upgrade, config, "head")
    logger.info("Database migrations applied successfully")


async def main() -> None:
    try:
        await init_db()
    except Exception:
        logger.exception("Failed to initialize database")
        raise


if __name__ == "__main__":
    asyncio.run(main())
