"""Store conversation vectors as JSON for cross-database compatibility."""

from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

import sqlalchemy as sa

from alembic import op

try:  # pragma: no cover - optional dependency for downgrade
    from pgvector.sqlalchemy import Vector
except ModuleNotFoundError:  # pragma: no cover - fallback for type hints only
    Vector = None  # type: ignore[assignment]

revision = "20250130_01"
down_revision = "20250108_03"
branch_labels = None
depends_on = None


def _json_type(dialect_name: str) -> sa.types.TypeEngine:
    if dialect_name == "postgresql":
        return sa.dialects.postgresql.JSONB()
    return sa.JSON()


def _server_default(dialect_name: str) -> sa.TextClause | None:
    if dialect_name == "postgresql":
        return sa.text("'[]'::jsonb")
    if dialect_name in {"sqlite", "mysql"}:
        return sa.text("'[]'")
    return None


def _coerce_vector(value: object) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError("Cannot coerce binary vector representation to JSON")
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return []
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as exc:  # pragma: no cover - safety belt
            raise TypeError("Cannot coerce non-JSON string to vector") from exc
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [float(component) for component in value]
    if hasattr(value, "tolist"):
        raw = value.tolist()
        return [float(component) for component in raw]
    raise TypeError(f"Unsupported vector payload: {type(value)!r}")


def upgrade() -> None:
    bind = op.get_bind()
    dialect_name = bind.dialect.name
    json_type = _json_type(dialect_name)
    default_clause = _server_default(dialect_name)

    with op.batch_alter_table("conversation_history", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "vector_json", json_type, nullable=True, server_default=default_clause
            )
        )

    metadata = sa.MetaData()
    source_vector_type: sa.types.TypeEngine
    if dialect_name == "postgresql" and Vector is not None:
        source_vector_type = Vector(3072)
    else:
        source_vector_type = sa.JSON()
    history = sa.Table(
        "conversation_history",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("vector", source_vector_type),
    )

    update_stmt = sa.text(
        "UPDATE conversation_history SET vector_json = :vector WHERE id = :id"
    ).bindparams(
        sa.bindparam("vector", type_=json_type),
        sa.bindparam("id", type_=sa.Integer),
    )

    offset = 0
    batch_size = 500
    while True:
        select_stmt = (
            sa.select(history.c.id, history.c.vector).limit(batch_size).offset(offset)
        )
        results: Iterable[sa.Row[Any]] = bind.execute(select_stmt)
        rows = list(results)
        if not rows:
            break
        for row in rows:
            vector_value = row.vector
            if vector_value is None:
                continue
            payload = _coerce_vector(vector_value)
            bind.execute(update_stmt, {"id": row.id, "vector": payload})
        offset += batch_size

    with op.batch_alter_table("conversation_history", schema=None) as batch_op:
        batch_op.drop_column("vector")
        batch_op.alter_column(
            "vector_json",
            new_column_name="vector",
            existing_type=json_type,
            existing_nullable=True,
            server_default=default_clause,
        )


def downgrade() -> None:
    bind = op.get_bind()
    dialect_name = bind.dialect.name
    json_type = _json_type(dialect_name)

    if dialect_name == "postgresql" and Vector is None:
        raise RuntimeError("pgvector is required to downgrade this migration")

    vector_type: sa.types.TypeEngine
    if dialect_name == "postgresql":
        vector_type = Vector(3072)  # type: ignore[misc]
    else:
        vector_type = sa.JSON()

    with op.batch_alter_table("conversation_history", schema=None) as batch_op:
        batch_op.add_column(sa.Column("vector_raw", vector_type, nullable=True))

    metadata = sa.MetaData()
    history = sa.Table(
        "conversation_history",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("vector", json_type),
    )

    select_stmt = sa.select(history.c.id, history.c.vector)
    results: Iterable[sa.Row[Any]] = bind.execute(select_stmt)

    if dialect_name == "postgresql":
        update_stmt = sa.text(
            "UPDATE conversation_history SET vector_raw = :vector WHERE id = :id"
        ).bindparams(
            sa.bindparam("vector", type_=vector_type),
            sa.bindparam("id", type_=sa.Integer),
        )
        for row in results:
            payload = row.vector or []
            if isinstance(payload, str):
                payload = json.loads(payload)
            if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
                raise TypeError(
                    "Unexpected payload type while downgrading vector column"
                )
            bind.execute(update_stmt, {"id": row.id, "vector": list(payload)})
    else:
        update_stmt = sa.text(
            "UPDATE conversation_history SET vector_raw = :vector WHERE id = :id"
        ).bindparams(
            sa.bindparam("vector", type_=json_type),
            sa.bindparam("id", type_=sa.Integer),
        )
        for row in results:
            payload = row.vector
            if payload is None:
                continue
            if isinstance(payload, str):
                payload = json.loads(payload)
            bind.execute(update_stmt, {"id": row.id, "vector": payload})

    with op.batch_alter_table("conversation_history", schema=None) as batch_op:
        batch_op.drop_column("vector")
        batch_op.alter_column(
            "vector_raw",
            new_column_name="vector",
            existing_type=vector_type,
            existing_nullable=True,
            server_default=None,
        )

    if dialect_name == "postgresql":
        with op.batch_alter_table("conversation_history", schema=None) as batch_op:
            batch_op.alter_column("vector", server_default=None)
