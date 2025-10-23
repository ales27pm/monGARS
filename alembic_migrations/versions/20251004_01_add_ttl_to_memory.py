"""Add TTL column to memory entries"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20251004_01"
down_revision = "20250308_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_names = inspector.get_table_names()

    if "memory_entries" not in table_names:
        op.create_table(
            "memory_entries",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("user_id", sa.String(), nullable=False, index=True),
            sa.Column("query", sa.String(), nullable=False),
            sa.Column("response", sa.String(), nullable=False),
            sa.Column(
                "timestamp",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column("ttl", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index(
            "ix_memory_entries_user_ttl",
            "memory_entries",
            ["user_id", "ttl"],
        )
        return

    columns = {col["name"] for col in inspector.get_columns("memory_entries")}
    if "ttl" in columns:
        return

    op.add_column(
        "memory_entries",
        sa.Column("ttl", sa.DateTime(timezone=True), nullable=True),
    )

    op.execute(
        sa.text(
            "UPDATE memory_entries SET ttl = timestamp + INTERVAL '24 hours' WHERE ttl IS NULL"
        )
    )

    op.alter_column("memory_entries", "ttl", nullable=False)

    if not any(
        index["name"] == "ix_memory_entries_user_ttl"
        for index in inspector.get_indexes("memory_entries")
    ):
        op.create_index(
            "ix_memory_entries_user_ttl",
            "memory_entries",
            ["user_id", "ttl"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "memory_entries" not in inspector.get_table_names():
        return

    columns = {col["name"] for col in inspector.get_columns("memory_entries")}
    if "ttl" in columns:
        op.drop_column("memory_entries", "ttl")
