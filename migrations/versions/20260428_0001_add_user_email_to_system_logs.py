"""add user_email to system_logs

Revision ID: 20260428_0001
Revises:
Create Date: 2026-04-28
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260428_0001"
down_revision = None
branch_labels = None
depends_on = None


def _column_names(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return {col["name"] for col in inspector.get_columns(table_name)}


def upgrade() -> None:
    if "system_logs" not in sa.inspect(op.get_bind()).get_table_names():
        return

    if "user_email" in _column_names("system_logs"):
        return

    with op.batch_alter_table("system_logs") as batch_op:
        batch_op.add_column(sa.Column("user_email", sa.String(length=255), nullable=True))


def downgrade() -> None:
    if "system_logs" not in sa.inspect(op.get_bind()).get_table_names():
        return

    if "user_email" not in _column_names("system_logs"):
        return

    with op.batch_alter_table("system_logs") as batch_op:
        batch_op.drop_column("user_email")
