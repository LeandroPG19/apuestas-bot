"""Fase 2.1 + 2.2: public_betting_snapshots + arb_opportunities.

Revision ID: 0011
Revises: 0010
Create Date: 2026-04-21

Dos tablas nuevas para Fase 2:

1. `public_betting_snapshots` — % público vs % money por outcome por book.
   Scrape de Action Network / Vegas Insider / BettingPros.
   Fase 2.2: contrarian signal `|pct_money - pct_bets| > 15pp` + `line_not_moved`
   bump p_model +3pp hacia el lado de sharp.

2. `arb_opportunities` — auditoría de arbitrages detectados por scanner.
   Se persiste todo arb encontrado (tomado o no) para análisis retrospectivo.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ─── public_betting_snapshots ────────────────────────────────────────
    op.create_table(
        "public_betting_snapshots",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("match_id", sa.BigInteger, sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("market", sa.Text, nullable=False),
        sa.Column("outcome", sa.Text, nullable=False),
        sa.Column("line", sa.Numeric(6, 2)),
        sa.Column("book", sa.Text),
        sa.Column("pct_bets", sa.Numeric(5, 4)),  # 0-1: % del número de tickets
        sa.Column("pct_money", sa.Numeric(5, 4)),  # 0-1: % del dinero apostado
        sa.Column(
            "source", sa.Text, nullable=False
        ),  # action_network | vegas_insider | betting_pros
        sa.Column(
            "captured_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint("pct_bets IS NULL OR (pct_bets >= 0 AND pct_bets <= 1)"),
        sa.CheckConstraint("pct_money IS NULL OR (pct_money >= 0 AND pct_money <= 1)"),
    )
    op.create_index(
        "idx_public_betting_match_market",
        "public_betting_snapshots",
        ["match_id", "market", "captured_at"],
        postgresql_ops={"captured_at": "DESC"},
    )

    # ─── arb_opportunities ───────────────────────────────────────────────
    op.create_table(
        "arb_opportunities",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("match_id", sa.BigInteger, sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("market", sa.Text, nullable=False),
        sa.Column(
            "legs",
            sa.dialects.postgresql.JSONB,
            nullable=False,
            comment="[{outcome, book, odds}, ...]",
        ),
        sa.Column("guaranteed_profit_pct", sa.Numeric(6, 5), nullable=False),
        sa.Column(
            "stakes_per_book",
            sa.dialects.postgresql.JSONB,
            nullable=False,
            comment="{book: fraction (0-1)}",
        ),
        sa.Column("total_implied_prob", sa.Numeric(6, 5)),
        sa.Column(
            "status",
            sa.Text,
            nullable=False,
            server_default="detected",
            comment="detected | taken | expired | stale",
        ),
        sa.Column(
            "detected_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("taken_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("actual_profit_usd", sa.Numeric(10, 2)),
        sa.CheckConstraint(
            "status IN ('detected','taken','expired','stale')",
            name="ck_arb_status",
        ),
        sa.CheckConstraint("guaranteed_profit_pct > 0", name="ck_arb_positive_profit"),
    )
    op.create_index("idx_arb_opportunities_match", "arb_opportunities", ["match_id", "detected_at"])
    op.create_index("idx_arb_opportunities_status", "arb_opportunities", ["status", "detected_at"])


def downgrade() -> None:
    op.drop_index("idx_arb_opportunities_status", table_name="arb_opportunities")
    op.drop_index("idx_arb_opportunities_match", table_name="arb_opportunities")
    op.drop_table("arb_opportunities")
    op.drop_index("idx_public_betting_match_market", table_name="public_betting_snapshots")
    op.drop_table("public_betting_snapshots")
