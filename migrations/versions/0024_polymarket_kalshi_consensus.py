"""Tablas Polymarket + Kalshi + consensus cols en pick_alerts.

Revision ID: 0024
Revises: 0023
Create Date: 2026-04-24

- `polymarket_prices`: midpoints del CLOB + condition_id (append por captura).
- `kalshi_prices`: midpoints yes/no del Kalshi trading API.
- `pick_alerts`: columnas `p_consensus_sharp` y `market_consensus_delta`
  para persistir el consenso calculado en emit time (Sprint 6c wire).

Retention: 90 días de snapshots; particionamiento se hace en 0025.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0024"
down_revision: str | None = "0023"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS apuestas.polymarket_prices (
            id            bigserial PRIMARY KEY,
            condition_id  text        NOT NULL,
            question      text,
            sport         text        NOT NULL,
            token_id      text        NOT NULL,
            midpoint      numeric(6,4) NOT NULL,
            volume_usd    numeric(14,2),
            end_date      timestamptz,
            captured_at   timestamptz  NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_polymarket_sport_ts "
        "ON apuestas.polymarket_prices (sport, captured_at DESC)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_polymarket_condition "
        "ON apuestas.polymarket_prices (condition_id)"
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS apuestas.kalshi_prices (
            ticker        text         NOT NULL,
            title         text,
            sport         text         NOT NULL,
            yes_midpoint  numeric(6,4) NOT NULL,
            no_midpoint   numeric(6,4) NOT NULL,
            volume        numeric(14,2),
            close_ts      timestamptz,
            captured_at   timestamptz  NOT NULL DEFAULT now(),
            PRIMARY KEY (ticker, captured_at)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_kalshi_sport_ts "
        "ON apuestas.kalshi_prices (sport, captured_at DESC)"
    )

    # Columnas nuevas en pick_alerts (Sprint 6c wire).
    with op.batch_alter_table("pick_alerts", schema="apuestas") as batch:
        batch.add_column(sa.Column("p_consensus_sharp", sa.Numeric(6, 4), nullable=True))
        batch.add_column(sa.Column("market_consensus_delta", sa.Numeric(6, 4), nullable=True))


def downgrade() -> None:
    op.execute("ALTER TABLE apuestas.pick_alerts DROP COLUMN IF EXISTS market_consensus_delta")
    op.execute("ALTER TABLE apuestas.pick_alerts DROP COLUMN IF EXISTS p_consensus_sharp")
    op.execute("DROP TABLE IF EXISTS apuestas.kalshi_prices")
    op.execute("DROP TABLE IF EXISTS apuestas.polymarket_prices")
