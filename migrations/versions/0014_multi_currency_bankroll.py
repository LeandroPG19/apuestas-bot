"""Multi-currency bankroll (USD + MXN) + FX rates.

Revision ID: 0014
Revises: 0013
Create Date: 2026-04-21

Añade:
  - `bankroll_state`: balance actual por (is_paper, currency) → 4 filas max.
  - `fx_rates`: tipo de cambio histórico diario (base, quote, rate).
  - `bets.currency` + `bets.fx_rate_used`: moneda de la apuesta + tipo usado.

Backfill:
  - bankroll_state seed con USD=default_bankroll_units, MXN=0 para paper y real.
  - bets existentes → currency='USD', fx_rate_used=NULL (legado).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0014"
down_revision: str | None = "0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ─── bankroll_state: balance por moneda ──────────────────────────────
    op.create_table(
        "bankroll_state",
        sa.Column("is_paper", sa.Boolean, nullable=False),
        sa.Column(
            "currency",
            sa.String(3),
            nullable=False,
            comment="ISO 4217: USD, MXN",
        ),
        sa.Column(
            "balance",
            sa.Numeric(14, 2),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("is_paper", "currency", name="pk_bankroll_state"),
        sa.CheckConstraint(
            "currency IN ('USD', 'MXN')",
            name="ck_bankroll_state_currency",
        ),
    )

    # ─── fx_rates: histórico tipos de cambio ─────────────────────────────
    op.create_table(
        "fx_rates",
        sa.Column("captured_at", sa.Date, nullable=False),
        sa.Column("base_currency", sa.String(3), nullable=False),
        sa.Column("quote_currency", sa.String(3), nullable=False),
        sa.Column(
            "rate",
            sa.Numeric(12, 6),
            nullable=False,
            comment="1 base = N quote (ej. 1 USD = 17.25 MXN)",
        ),
        sa.Column("source", sa.Text, nullable=True),
        sa.PrimaryKeyConstraint(
            "captured_at",
            "base_currency",
            "quote_currency",
            name="pk_fx_rates",
        ),
        sa.CheckConstraint("rate > 0", name="ck_fx_rates_positive"),
    )
    op.create_index(
        "idx_fx_rates_pair",
        "fx_rates",
        ["base_currency", "quote_currency", "captured_at"],
    )

    # ─── bets: moneda + tipo de cambio usado ─────────────────────────────
    op.add_column(
        "bets",
        sa.Column(
            "currency",
            sa.String(3),
            nullable=False,
            server_default="USD",
            comment="Moneda del stake/odds",
        ),
    )
    op.add_column(
        "bets",
        sa.Column(
            "fx_rate_used",
            sa.Numeric(12, 6),
            nullable=True,
            comment="Tipo de cambio a USD cuando se emitió el pick",
        ),
    )

    # ─── Seed bankroll_state con USD + MXN para paper y real ─────────────
    op.execute(
        """
        INSERT INTO bankroll_state (is_paper, currency, balance)
        VALUES
            (true,  'USD', 200.00),
            (true,  'MXN', 0.00),
            (false, 'USD', 200.00),
            (false, 'MXN', 0.00)
        ON CONFLICT DO NOTHING
        """
    )

    # ─── bot_state: key-value para config (moneda primaria, etc.) ────────
    op.create_table(
        "bot_state",
        sa.Column("key", sa.Text, primary_key=True),
        sa.Column("value", sa.Text, nullable=False),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.execute(
        """
        INSERT INTO bot_state (key, value)
        VALUES
            ('primary_currency_paper', 'USD'),
            ('primary_currency_real', 'USD')
        ON CONFLICT DO NOTHING
        """
    )


def downgrade() -> None:
    op.drop_table("bot_state")
    op.drop_column("bets", "fx_rate_used")
    op.drop_column("bets", "currency")
    op.drop_index("idx_fx_rates_pair", table_name="fx_rates")
    op.drop_table("fx_rates")
    op.drop_table("bankroll_state")
