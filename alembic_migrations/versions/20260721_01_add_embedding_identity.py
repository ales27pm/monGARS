"""Partition persisted vectors by embedding backend identity."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260721_01"
down_revision = "20251004_01"
branch_labels = None
depends_on = None

INDEX_NAME = "idx_user_embedding_identity_timestamp"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "conversation_history" not in inspector.get_table_names():
        return

    columns = {
        column["name"] for column in inspector.get_columns("conversation_history")
    }
    if "embedding_identity" not in columns:
        with op.batch_alter_table("conversation_history", schema=None) as batch_op:
            batch_op.add_column(
                sa.Column("embedding_identity", sa.String(length=64), nullable=True)
            )

    indexes = {
        index["name"] for index in sa.inspect(bind).get_indexes("conversation_history")
    }
    if INDEX_NAME not in indexes:
        op.create_index(
            INDEX_NAME,
            "conversation_history",
            ["user_id", "embedding_identity", "timestamp"],
            unique=False,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "conversation_history" not in inspector.get_table_names():
        return

    indexes = {index["name"] for index in inspector.get_indexes("conversation_history")}
    if INDEX_NAME in indexes:
        op.drop_index(INDEX_NAME, table_name="conversation_history")

    columns = {
        column["name"]
        for column in sa.inspect(bind).get_columns("conversation_history")
    }
    if "embedding_identity" in columns:
        with op.batch_alter_table("conversation_history", schema=None) as batch_op:
            batch_op.drop_column("embedding_identity")
