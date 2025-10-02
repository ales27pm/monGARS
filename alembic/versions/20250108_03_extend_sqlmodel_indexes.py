"""Align indexes with SQLModel persistence tables."""

from __future__ import annotations

from alembic import op

revision = "20250108_03"
down_revision = "20250108_02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create indexes expected by the SQLModel metadata."""

    op.create_index(
        op.f("ix_conversation_history_user_id"),
        "conversation_history",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_conversation_history_timestamp"),
        "conversation_history",
        ["timestamp"],
        unique=False,
    )
    op.create_index(
        op.f("ix_interactions_user_id"),
        "interactions",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_interactions_session_id"),
        "interactions",
        ["session_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_interactions_created_at"),
        "interactions",
        ["created_at"],
        unique=False,
    )


def downgrade() -> None:
    """Drop SQLModel-aligned indexes."""

    op.drop_index(op.f("ix_interactions_created_at"), table_name="interactions")
    op.drop_index(op.f("ix_interactions_session_id"), table_name="interactions")
    op.drop_index(op.f("ix_interactions_user_id"), table_name="interactions")
    op.drop_index(
        op.f("ix_conversation_history_timestamp"),
        table_name="conversation_history",
    )
    op.drop_index(
        op.f("ix_conversation_history_user_id"),
        table_name="conversation_history",
    )
