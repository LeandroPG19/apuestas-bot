"""Drop tabla arb_opportunities.

Revision ID: 0032
Revises: 0031
Create Date: 2026-04-25

Razón: el módulo arbitrage fue eliminado del proyecto por decisión del
usuario 2026-04-25 (no usa arbitrajes, gasta data API innecesariamente).
La tabla `arb_opportunities` fue creada en migración 0011 para auditoría
del scanner; al eliminar el scanner queda como tabla huérfana.

Estado pre-drop verificado 2026-04-25 (DB local):
  - rows: 0 (sin data histórica relevante)
  - FK incoming: 0 (ninguna otra tabla referencia esta)

DOWN: re-aplica el CREATE TABLE original de migración 0011 (estructura
preservada en docstring).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0032"
down_revision: str | None = "0031"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Idempotente: IF EXISTS evita error si ya fue droppeada manualmente.
    op.execute("DROP INDEX IF EXISTS idx_arb_opportunities_status")
    op.execute("DROP INDEX IF EXISTS idx_arb_opportunities_match")
    op.execute("DROP TABLE IF EXISTS arb_opportunities")


def downgrade() -> None:
    # Re-crea la tabla original (espejo de migración 0011) para reversibilidad.
    op.create_table(
        "arb_opportunities",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("match_id", sa.BigInteger, sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("market", sa.String(50), nullable=False),
        sa.Column("outcome_a", sa.String(50), nullable=False),
        sa.Column("outcome_b", sa.String(50), nullable=False),
        sa.Column("book_a", sa.String(50), nullable=False),
        sa.Column("book_b", sa.String(50), nullable=False),
        sa.Column("odds_a", sa.Numeric(6, 3), nullable=False),
        sa.Column("odds_b", sa.Numeric(6, 3), nullable=False),
        sa.Column("guaranteed_profit_pct", sa.Numeric(5, 2), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="detected"),
        sa.Column(
            "detected_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "status IN ('detected','taken','expired','rejected')",
            name="ck_arb_status",
        ),
        sa.CheckConstraint("guaranteed_profit_pct > 0", name="ck_arb_positive_profit"),
    )
    op.create_index(
        "idx_arb_opportunities_match",
        "arb_opportunities",
        ["match_id", "detected_at"],
    )
    op.create_index(
        "idx_arb_opportunities_status",
        "arb_opportunities",
        ["status", "detected_at"],
    )
