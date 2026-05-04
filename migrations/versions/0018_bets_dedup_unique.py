"""B-dedup: unique partial index en bets para garantizar no duplicates.

Revision ID: 0018
Revises: 0017
Create Date: 2026-04-22

Crea unique partial index sobre (match_id, market, outcome, DATE(placed_at), is_paper)
WHERE status = 'pending'. Garantiza a nivel BD que no puedan insertarse dos bets
paper pendientes para el mismo (match, market, outcome) el mismo día.

Antes: dedup era solo en deep_analysis.py (app-level), no survived race conditions
ni ejecuciones concurrentes del flow.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0018"
down_revision: str | None = "0017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Limpia duplicados existentes antes de crear el unique index
    op.execute(
        """
        DELETE FROM bets b
        USING bets b2
        WHERE b.id > b2.id
          AND b.match_id = b2.match_id
          AND b.market = b2.market
          AND b.outcome = b2.outcome
          AND b.status = 'pending'
          AND b2.status = 'pending'
          AND b.is_paper = b2.is_paper
          AND DATE(b.placed_at AT TIME ZONE 'UTC') = DATE(b2.placed_at AT TIME ZONE 'UTC')
        """
    )

    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_bets_pending_daily
        ON bets (match_id, market, outcome, is_paper, DATE(placed_at AT TIME ZONE 'UTC'))
        WHERE status = 'pending'
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_bets_pending_daily")
