"""Align SQLModel tables and legacy artefacts with ORM expectations."""

from __future__ import annotations

import json

import sqlalchemy as sa

from alembic import op

revision = "20250304_01"
down_revision = "20250130_01"
branch_labels = None
depends_on = None


def _json_type(dialect_name: str) -> sa.types.TypeEngine:
    if dialect_name == "postgresql":
        return sa.dialects.postgresql.JSONB()
    return sa.JSON()


def _json_default_clause(
    dialect_name: str, payload: object
) -> sa.sql.elements.TextClause | None:
    text = json.dumps(payload, separators=(",", ":"))
    if dialect_name == "postgresql":
        return sa.text(f"'{text}'::jsonb")
    if dialect_name in {"sqlite", "mysql"}:
        return sa.text(f"'{text}'")
    return None


def _fill_nulls(
    table: str,
    column: str,
    value: object,
    *,
    type_: sa.types.TypeEngine | None = None,
) -> None:
    statement = sa.text(f"UPDATE {table} SET {column} = :value WHERE {column} IS NULL")
    if type_ is not None:
        statement = statement.bindparams(
            sa.bindparam("value", value=value, type_=type_)
        )
    else:
        statement = statement.bindparams(sa.bindparam("value", value=value))
    op.execute(statement)


def _ensure_legacy_tables(
    bind: sa.engine.Connection, json_type: sa.types.TypeEngine
) -> None:
    inspector = sa.inspect(bind)

    if not inspector.has_table("conversation_sessions"):
        op.create_table(
            "conversation_sessions",
            sa.Column("user_id", sa.String(), primary_key=True, nullable=False),
            sa.Column("session_data", json_type, nullable=True),
            sa.Column(
                "last_active",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
        )

    if not inspector.has_table("emotion_trends"):
        op.create_table(
            "emotion_trends",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("user_id", sa.String(), nullable=True),
            sa.Column("emotion", sa.String(), nullable=True),
            sa.Column(
                "timestamp",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
        )


def upgrade() -> None:
    bind = op.get_bind()
    dialect_name = bind.dialect.name
    json_type = _json_type(dialect_name)

    _ensure_legacy_tables(bind, json_type)

    empty_object_default = _json_default_clause(dialect_name, {})

    for column in ("input_data", "output_data", "personality", "context"):
        _fill_nulls("interactions", column, {}, type_=json_type)
        op.alter_column(
            "interactions",
            column,
            existing_type=json_type,
            nullable=False,
            server_default=empty_object_default,
        )

    for column in ("message", "response"):
        _fill_nulls("interactions", column, "")
        op.alter_column(
            "interactions",
            column,
            existing_type=sa.String(),
            nullable=False,
        )

    for column in ("interaction_style", "preferred_topics"):
        _fill_nulls("user_preferences", column, {}, type_=json_type)
        op.alter_column(
            "user_preferences",
            column,
            existing_type=json_type,
            nullable=False,
            server_default=empty_object_default,
        )

    for column in ("traits", "interaction_style", "context_preferences"):
        _fill_nulls("user_personality", column, {}, type_=json_type)
        op.alter_column(
            "user_personality",
            column,
            existing_type=json_type,
            nullable=False,
            server_default=empty_object_default,
        )


def downgrade() -> None:
    bind = op.get_bind()
    dialect_name = bind.dialect.name
    json_type = _json_type(dialect_name)

    for column in ("traits", "interaction_style", "context_preferences"):
        op.alter_column(
            "user_personality",
            column,
            existing_type=json_type,
            nullable=True,
            server_default=None,
        )

    for column in ("interaction_style", "preferred_topics"):
        op.alter_column(
            "user_preferences",
            column,
            existing_type=json_type,
            nullable=True,
            server_default=None,
        )

    for column in ("message", "response"):
        op.alter_column(
            "interactions",
            column,
            existing_type=sa.String(),
            nullable=True,
        )

    for column in ("input_data", "output_data", "personality", "context"):
        op.alter_column(
            "interactions",
            column,
            existing_type=json_type,
            nullable=True,
            server_default=None,
        )
