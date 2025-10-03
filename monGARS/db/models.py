"""Canonical SQLAlchemy models shared across runtime and migrations."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

_JSON = JSON().with_variant(JSONB, "postgresql")


class Base(DeclarativeBase):
    """Base class for ORM models."""


class ConversationHistory(Base):
    __tablename__ = "conversation_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str | None] = mapped_column(String, index=True)
    query: Mapped[str | None] = mapped_column(String)
    response: Mapped[str | None] = mapped_column(String)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    vector: Mapped[list[float] | None] = mapped_column(_JSON, default=list)

    __table_args__ = (Index("idx_user_timestamp", "user_id", "timestamp"),)


class Interaction(Base):
    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    session_id: Mapped[str | None] = mapped_column(String, index=True)
    input_data: Mapped[dict] = mapped_column(_JSON, default=dict)
    output_data: Mapped[dict] = mapped_column(_JSON, default=dict)
    message: Mapped[str] = mapped_column(String)
    response: Mapped[str] = mapped_column(String)
    personality: Mapped[dict] = mapped_column(_JSON, default=dict)
    context: Mapped[dict] = mapped_column(_JSON, default=dict)
    meta_data: Mapped[str | None] = mapped_column(String)
    confidence: Mapped[float | None] = mapped_column(Float)
    processing_time: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("idx_user_session_created", "user_id", "session_id", "created_at"),
    )


class UserPreferences(Base):
    __tablename__ = "user_preferences"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    interaction_style: Mapped[dict] = mapped_column(_JSON, default=dict)
    preferred_topics: Mapped[dict] = mapped_column(_JSON, default=dict)


class UserPersonality(Base):
    __tablename__ = "user_personality"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    traits: Mapped[dict] = mapped_column(_JSON, default=dict)
    interaction_style: Mapped[dict] = mapped_column(_JSON, default=dict)
    context_preferences: Mapped[dict] = mapped_column(_JSON, default=dict)
    adaptation_rate: Mapped[float] = mapped_column(Float, default=0.1)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class UserAccount(Base):
    __tablename__ = "user_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(150), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
