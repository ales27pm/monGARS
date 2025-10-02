"""Add user accounts table"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "20250108_02"
down_revision = "20250108_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_accounts",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("username", sa.String(length=150), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column(
            "is_admin",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
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
            nullable=False,
        ),
    )
    op.create_index(
        "ix_user_accounts_username",
        "user_accounts",
        ["username"],
        unique=True,
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION update_user_accounts_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = now();
            RETURN NEW;
        END;
        $$ LANGUAGE 'plpgsql';
        """
    )
    op.execute(
        """
        CREATE TRIGGER update_user_accounts_updated_at
        BEFORE UPDATE ON user_accounts
        FOR EACH ROW
        EXECUTE FUNCTION update_user_accounts_updated_at();
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS update_user_accounts_updated_at ON user_accounts")
    op.execute("DROP FUNCTION IF EXISTS update_user_accounts_updated_at()")
    op.drop_index("ix_user_accounts_username", table_name="user_accounts")
    op.drop_table("user_accounts")
