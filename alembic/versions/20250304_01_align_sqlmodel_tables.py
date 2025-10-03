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
    if dialect_name == "sqlite":
        return sa.JSON()
    if dialect_name == "mysql":
        return sa.dialects.mysql.JSON()
    raise RuntimeError(f"Unsupported dialect: {dialect_name}")


def _json_default_clause(
    dialect_name: str, payload: object
) -> sa.sql.elements.TextClause | None:
    text = json.dumps(payload, separators=(",", ":"))
    if dialect_name == "postgresql":
        return sa.text(f"'{text}'::jsonb")
    if dialect_name == "sqlite":
        return sa.text(f"'{text}'")
    if dialect_name == "mysql":
        return None
    raise RuntimeError(f"Unsupported dialect: {dialect_name}")


def _fill_nulls(
    table: str,
    column: str,
    value: object,
    *,
    type_: sa.types.TypeEngine | None = None,
) -> None:
    table_clause = sa.table(table, sa.column(column))
    bind_param = sa.bindparam("value", value=value, type_=type_)
    statement = (
        sa.update(table_clause)
        .where(table_clause.c[column].is_(None))
        .values({column: bind_param})
    )
    op.execute(statement)


def _apply_column_alterations(
    table: str,
    columns: list[dict[str, object]],
    *,
    use_batch: bool,
) -> None:
    if use_batch:
        with op.batch_alter_table(table, recreate="always") as batch:
            for column in columns:
                batch.alter_column(column["name"], **column["alter"])  # type: ignore[arg-type]
    else:
        for column in columns:
            op.alter_column(table, column["name"], **column["alter"])  # type: ignore[arg-type]


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
    string_default = sa.text("''")
    use_batch = dialect_name == "sqlite"

    json_alter_kwargs = {"existing_type": json_type, "nullable": False}
    if empty_object_default is not None:
        json_alter_kwargs["server_default"] = empty_object_default

    interactions_columns = [
        {
            "name": column,
            "fill": {"value": {}, "type_": json_type},
            "alter": json_alter_kwargs.copy(),
        }
        for column in ("input_data", "output_data", "personality", "context")
    ]
    interactions_columns.extend(
        {
            "name": column,
            "fill": {"value": "", "type_": None},
            "alter": {
                "existing_type": sa.String(),
                "nullable": False,
                "server_default": string_default,
            },
        }
        for column in ("message", "response")
    )

    user_preferences_columns = [
        {
            "name": column,
            "fill": {"value": {}, "type_": json_type},
            "alter": json_alter_kwargs.copy(),
        }
        for column in ("interaction_style", "preferred_topics")
    ]

    user_personality_columns = [
        {
            "name": column,
            "fill": {"value": {}, "type_": json_type},
            "alter": json_alter_kwargs.copy(),
        }
        for column in ("traits", "interaction_style", "context_preferences")
    ]

    operations = [
        {"table": "interactions", "columns": interactions_columns},
        {"table": "user_preferences", "columns": user_preferences_columns},
        {"table": "user_personality", "columns": user_personality_columns},
    ]

    for operation in operations:
        for column in operation["columns"]:
            fill = column["fill"]  # type: ignore[assignment]
            _fill_nulls(
                operation["table"],
                column["name"],
                fill["value"],
                type_=fill["type_"],
            )
        _apply_column_alterations(
            operation["table"],
            [
                {"name": column["name"], "alter": column["alter"]}
                for column in operation["columns"]
            ],
            use_batch=use_batch,
        )


def downgrade() -> None:
    bind = op.get_bind()
    dialect_name = bind.dialect.name
    json_type = _json_type(dialect_name)
    use_batch = dialect_name == "sqlite"

    downgrade_operations = [
        {
            "table": "user_personality",
            "columns": [
                {
                    "name": column,
                    "alter": {
                        "existing_type": json_type,
                        "nullable": True,
                        "server_default": None,
                    },
                }
                for column in ("traits", "interaction_style", "context_preferences")
            ],
        },
        {
            "table": "user_preferences",
            "columns": [
                {
                    "name": column,
                    "alter": {
                        "existing_type": json_type,
                        "nullable": True,
                        "server_default": None,
                    },
                }
                for column in ("interaction_style", "preferred_topics")
            ],
        },
        {
            "table": "interactions",
            "columns": [
                {
                    "name": column,
                    "alter": {
                        "existing_type": sa.String(),
                        "nullable": True,
                        "server_default": None,
                    },
                }
                for column in ("message", "response")
            ],
        },
        {
            "table": "interactions",
            "columns": [
                {
                    "name": column,
                    "alter": {
                        "existing_type": json_type,
                        "nullable": True,
                        "server_default": None,
                    },
                }
                for column in ("input_data", "output_data", "personality", "context")
            ],
        },
    ]

    for operation in downgrade_operations:
        _apply_column_alterations(
            operation["table"],
            operation["columns"],
            use_batch=use_batch,
        )

    inspector = sa.inspect(bind)
    for table_name in ("emotion_trends", "conversation_sessions"):
        if inspector.has_table(table_name):
            op.drop_table(table_name)
