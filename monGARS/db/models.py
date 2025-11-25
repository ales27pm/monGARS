"""Canonical SQLAlchemy models shared across runtime and migrations."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import (
    Column,
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

try:  # pragma: no cover - optional dependency in lightweight envs
    from pgvector.sqlalchemy import Vector
except ModuleNotFoundError:  # pragma: no cover - tests run without pgvector
    Vector = None  # type: ignore[assignment]

if Vector is not None:  # pragma: no branch - evaluated once at import
    _VECTOR = JSON().with_variant(Vector(3072), "postgresql")
else:
    _VECTOR = _JSON


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
    vector: Mapped[list[float] | None] = mapped_column(_VECTOR, default=list)

    _vector_index = None
    if Vector is not None:  # pragma: no branch - evaluated at import
        _vector_index = Index(
            "ix_conversation_history_vector_cosine",
            "vector",
            postgresql_using="ivfflat",
            postgresql_with={"lists": "100"},
            postgresql_ops={"vector": "vector_cosine_ops"},
        )

    __table_args__ = tuple(
        filter(
            None,
            (
                Index("idx_user_timestamp", "user_id", "timestamp"),
                _vector_index,
            ),
        )
    )


def _default_memory_ttl() -> datetime:
    """Return the default TTL for memory entries."""

    return datetime.now(timezone.utc) + timedelta(hours=24)


class MemoryEntry(Base):
    __tablename__ = "memory_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    query: Mapped[str] = mapped_column(String, nullable=False)
    response: Mapped[str] = mapped_column(String, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    ttl: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_default_memory_ttl, nullable=False
    )

    __table_args__ = (Index("ix_memory_entries_user_ttl", "user_id", "ttl"),)


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


class OperatorApproval(Base):
    __tablename__ = "operator_approvals"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(64), nullable=False)
    prompt_hash = Column(String(8), nullable=False)
    pii_entities = Column(JSON, nullable=False)
    approval_token = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    approved_at = Column(DateTime(timezone=True), nullable=True)
    approved_by = Column(String(64), nullable=True)
