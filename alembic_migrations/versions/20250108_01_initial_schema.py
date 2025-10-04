"""Create core persistence tables"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "20250108_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "conversation_history",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("query", sa.String(), nullable=True),
        sa.Column("response", sa.String(), nullable=True),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("vector", Vector(3072), nullable=True),
    )
    op.create_index(
        "idx_user_timestamp",
        "conversation_history",
        ["user_id", "timestamp"],
    )

    op.create_table(
        "conversation_sessions",
        sa.Column("user_id", sa.String(), primary_key=True, nullable=False),
        sa.Column("session_data", sa.JSON(), nullable=True),
        sa.Column(
            "last_active",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.create_table(
        "user_preferences",
        sa.Column("user_id", sa.String(), primary_key=True, nullable=False),
        sa.Column("interaction_style", sa.JSON(), nullable=True),
        sa.Column("preferred_topics", sa.JSON(), nullable=True),
    )

    op.create_table(
        "user_personality",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("traits", sa.JSON(), nullable=True),
        sa.Column("interaction_style", sa.JSON(), nullable=True),
        sa.Column("context_preferences", sa.JSON(), nullable=True),
        sa.Column(
            "adaptation_rate", sa.Float(), server_default=sa.text("0.1"), nullable=False
        ),
        sa.Column(
            "confidence", sa.Float(), server_default=sa.text("0.5"), nullable=False
        ),
        sa.Column(
            "last_updated",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("user_id"),
    )
    op.create_index(
        op.f("ix_user_personality_user_id"),
        "user_personality",
        ["user_id"],
        unique=False,
    )

    op.create_table(
        "emotion_trends",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("emotion", sa.String(), nullable=True),
        sa.Column(
            "timestamp",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.create_table(
        "interactions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=True),
        sa.Column("input_data", sa.JSON(), nullable=True),
        sa.Column("output_data", sa.JSON(), nullable=True),
        sa.Column("message", sa.String(), nullable=True),
        sa.Column("response", sa.String(), nullable=True),
        sa.Column("personality", sa.JSON(), nullable=True),
        sa.Column("context", sa.JSON(), nullable=True),
        sa.Column("meta_data", sa.String(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("processing_time", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "idx_user_session_created",
        "interactions",
        ["user_id", "session_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_user_session_created", table_name="interactions")
    op.drop_table("interactions")

    op.drop_table("emotion_trends")

    op.drop_index(op.f("ix_user_personality_user_id"), table_name="user_personality")
    op.drop_table("user_personality")

    op.drop_table("user_preferences")

    op.drop_table("conversation_sessions")

    op.drop_index("idx_user_timestamp", table_name="conversation_history")
    op.drop_table("conversation_history")

    op.execute("DROP EXTENSION IF EXISTS vector")
