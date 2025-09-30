import asyncio
import logging
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Column, DateTime, Float, Index, Integer, String, func, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import Mapped, declarative_base, mapped_column, sessionmaker

from config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
Base = declarative_base()
settings = get_settings()


class ConversationHistory(Base):
    __tablename__ = "conversation_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    query = Column(String)
    response = Column(String)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    vector = Column(Vector(3072))
    __table_args__ = (Index("idx_user_timestamp", "user_id", "timestamp"),)


class ConversationSession(Base):
    __tablename__ = "conversation_sessions"
    user_id = Column(String, primary_key=True)
    session_data = Column(JSON)
    last_active = Column(DateTime(timezone=True), default=datetime.utcnow)


class UserPreferences(Base):
    __tablename__ = "user_preferences"
    user_id = Column(String, primary_key=True)
    interaction_style = Column(JSON)
    preferred_topics = Column(JSON)


class UserPersonality(Base):
    __tablename__ = "user_personality"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String, index=True, unique=True)
    traits: Mapped[dict] = mapped_column(JSON)
    interaction_style: Mapped[dict] = mapped_column(JSON)
    context_preferences: Mapped[dict] = mapped_column(JSON, default=lambda: {})
    adaptation_rate: Mapped[float] = mapped_column(Float, default=0.1)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class EmotionTrend(Base):
    __tablename__ = "emotion_trends"
    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True)
    emotion = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)


class Interaction(Base):
    __tablename__ = "interactions"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    session_id: Mapped[str] = mapped_column(String, index=True)
    input_data: Mapped[dict] = mapped_column(JSON)
    output_data: Mapped[dict] = mapped_column(JSON)
    message: Mapped[str] = mapped_column(String)
    response: Mapped[str] = mapped_column(String)
    personality: Mapped[dict] = mapped_column(JSON, nullable=True)
    context: Mapped[dict] = mapped_column(JSON, nullable=True)
    meta_data: Mapped[str] = mapped_column(String, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=True)
    processing_time: Mapped[float] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )
    __table_args__ = (
        Index("idx_user_session_created", "user_id", "session_id", "created_at"),
    )


async_engine = create_async_engine(
    str(settings.database_url),
    echo=settings.debug,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_timeout=settings.db_pool_timeout,
)
async_session_factory = sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)


async def init_db():
    try:
        async with async_engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


async def main():
    try:
        await init_db()
    except Exception:
        logger.exception("Failed to initialize database")


if __name__ == "__main__":
    asyncio.run(main())
