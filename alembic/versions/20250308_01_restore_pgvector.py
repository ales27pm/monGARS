"""Restore pgvector-backed embeddings for conversation history."""

from __future__ import annotations

import json
from typing import Any

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


VECTOR_DIMENSIONS = 3072
ROW_BATCH_SIZE = 500


def _iter_rows(
    bind: sa.Connection, table: sa.Table, *, batch_size: int = ROW_BATCH_SIZE
):
    stmt = sa.select(table.c.id, table.c.vector)
    result = bind.execute(stmt)
    try:
        while True:
            chunk = result.fetchmany(batch_size)
            if not chunk:
                break
            for row in chunk:
                yield row
    finally:
        result.close()


def _normalise_vector(
    payload: Any, *, dimensions: int = VECTOR_DIMENSIONS
) -> list[float] | None:
    if payload is None:
        return None
    if isinstance(payload, str):
        candidate = payload.strip()
        if not candidate:
            return None
        try:
            payload = json.loads(candidate)
        except ValueError:
            # Malformed JSON payloads should be ignored so the migration can continue.
            return None
        payload = json.loads(candidate)
        if not isinstance(payload, (list, tuple)):
        payload = json.loads(candidate)
        if not isinstance(payload, (list, tuple)):
            return None
    if isinstance(payload, (list, tuple)):
        try:
            floats = [float(value) for value in payload]
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise TypeError("Vector payload contains non-numeric values") from exc
        if not floats:
            return None
        if len(floats) > dimensions:
            floats = floats[:dimensions]
        elif len(floats) < dimensions:
            floats.extend(0.0 for _ in range(dimensions - len(floats)))
        return floats
    if hasattr(payload, "tolist"):
        raw = payload.tolist()
        return _normalise_vector(raw, dimensions=dimensions)
    return None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql" or Vector is None:
        # Non-PostgreSQL backends continue using JSON storage.
        return

    op.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
    op.execute(sa.text("DROP INDEX IF EXISTS ix_conversation_history_vector_cosine"))

    with op.batch_alter_table("conversation_history", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("vector_new", Vector(VECTOR_DIMENSIONS), nullable=True)
        )

    metadata = sa.MetaData()
    history = sa.Table(
        "conversation_history",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("vector", sa.JSON()),
    )

    update_stmt = sa.text(
        "UPDATE conversation_history SET vector_new = :vector WHERE id = :id"
    )

    for row in _iter_rows(bind, history):
        vector = _normalise_vector(row.vector)
        if vector is None:
            continue
        bind.execute(update_stmt, {"id": row.id, "vector": vector})

    with op.batch_alter_table("conversation_history", schema=None) as batch_op:
        batch_op.drop_column("vector")
        batch_op.alter_column(
            "vector_new",
            new_column_name="vector",
            existing_type=Vector(VECTOR_DIMENSIONS),
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
        sa.Column("vector", Vector(VECTOR_DIMENSIONS)),
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

    for row in _iter_rows(bind, history):
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
