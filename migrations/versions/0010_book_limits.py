"""Fase 1.3: book_limits_per_user tracking dinámico.

Revision ID: 0010
Revises: 0009
Create Date: 2026-04-21

Trackea por bookmaker:
  - max_accepted_stake_usd: mayor stake que el book ha aceptado
  - last_rejected_stake_usd: último stake rechazado (señal de restricción)
  - limit_status: 'full' | 'restricted' | 'closed'
  - notes: historial de eventos (JSONB)

Útil principalmente en US books (DK/FanDuel/BetMGM) que restringen cuentas
ganadoras en 2-4 semanas. Permite:
  - Dashboard que muestra el status por book.
  - Kelly max cap ajustado por limit reported.
  - Alertas Telegram cuando detecta transición full → restricted.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "book_limits_per_user",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("bookmaker", sa.Text, nullable=False),
        sa.Column("max_accepted_stake_usd", sa.Numeric(10, 2)),
        sa.Column("last_rejected_stake_usd", sa.Numeric(10, 2)),
        sa.Column("last_rejected_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("limit_status", sa.Text, nullable=False, server_default="full"),
        sa.Column("notes", sa.dialects.postgresql.JSONB, server_default=sa.text("'[]'::jsonb")),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("bookmaker", name="uq_book_limits_book"),
        sa.CheckConstraint(
            "limit_status IN ('full', 'restricted', 'closed')",
            name="ck_book_limits_status",
        ),
    )
    op.create_index("idx_book_limits_status", "book_limits_per_user", ["limit_status"])

    # Seed rows iniciales para todos los books del catálogo regional (si existen)
    op.execute(
        """
        INSERT INTO book_limits_per_user (bookmaker, limit_status)
        SELECT DISTINCT bookmaker, 'full' FROM odds_history
        WHERE ts > now() - interval '7 days'
        ON CONFLICT (bookmaker) DO NOTHING
        """
    )


def downgrade() -> None:
    op.drop_index("idx_book_limits_status", table_name="book_limits_per_user")
    op.drop_table("book_limits_per_user")
