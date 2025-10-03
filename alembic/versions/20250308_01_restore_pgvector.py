"""Restore pgvector-backed embeddings for conversation history."""

from __future__ import annotations

import json
from typing import Any, Iterable

import sqlalchemy as sa

from alembic import op

try:  # pragma: no cover - optional dependency during lightweight tests
    from pgvector.sqlalchemy import Vector
except ModuleNotFoundError:  # pragma: no cover - downgrade guard
    Vector = None  # type: ignore[assignment]

revision = "20250308_01"
down_revision = "20250304_01"
branch_labels = None
depends_on = None


def _load_rows(bind: sa.Connection, table: sa.Table) -> list[sa.Row[Any]]:
    stmt = sa.select(table.c.id, table.c.vector)
    result: Iterable[sa.Row[Any]] = bind.execute(stmt)
    return list(result)


def _normalise_vector(payload: Any) -> list[float] | None:
    if payload is None:
        return None
    if isinstance(payload, str):
        candidate = payload.strip()
        if not candidate:
            return None
        payload = json.loads(candidate)
    if isinstance(payload, (list, tuple)):
        try:
            return [float(value) for value in payload]
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise TypeError("Vector payload contains non-numeric values") from exc
    if hasattr(payload, "tolist"):
        raw = payload.tolist()
        return _normalise_vector(raw)
    return None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql" or Vector is None:
        # Non-PostgreSQL backends continue using JSON storage.
        return

    op.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
    op.execute(sa.text("DROP INDEX IF EXISTS ix_conversation_history_vector_cosine"))

    with op.batch_alter_table("conversation_history", schema=None) as batch_op:
        batch_op.add_column(sa.Column("vector_new", Vector(3072), nullable=True))

    metadata = sa.MetaData()
    history = sa.Table(
        "conversation_history",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("vector", sa.JSON()),
    )

    rows = _load_rows(bind, history)
    update_stmt = sa.text(
        "UPDATE conversation_history SET vector_new = :vector WHERE id = :id"
    )

    for row in rows:
        vector = _normalise_vector(row.vector)
        if not vector:
            continue
        bind.execute(update_stmt, {"id": row.id, "vector": vector})

    with op.batch_alter_table("conversation_history", schema=None) as batch_op:
        batch_op.drop_column("vector")
        batch_op.alter_column(
            "vector_new",
            new_column_name="vector",
            existing_type=Vector(3072),
            existing_nullable=True,
        )

    op.create_index(
        "ix_conversation_history_vector_cosine",
        "conversation_history",
        ["vector"],
        postgresql_using="ivfflat",
        postgresql_with={"lists": "100"},
        postgresql_ops={"vector": "vector_cosine_ops"},
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql" or Vector is None:
        return

    op.execute(sa.text("DROP INDEX IF EXISTS ix_conversation_history_vector_cosine"))

    metadata = sa.MetaData()
    history = sa.Table(
        "conversation_history",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("vector", Vector(3072)),
    )

    with op.batch_alter_table("conversation_history", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "vector_json",
                sa.dialects.postgresql.JSONB(),
                nullable=True,
            )
        )

    update_stmt = sa.text(
        "UPDATE conversation_history SET vector_json = :vector WHERE id = :id"
    )

    rows = _load_rows(bind, history)
    for row in rows:
        payload = _normalise_vector(row.vector)
        if payload is None:
            continue
        bind.execute(update_stmt, {"id": row.id, "vector": payload})

    with op.batch_alter_table("conversation_history", schema=None) as batch_op:
        batch_op.drop_column("vector")
        batch_op.alter_column(
            "vector_json",
            new_column_name="vector",
            existing_type=sa.dialects.postgresql.JSONB(),
            existing_nullable=True,
        )
