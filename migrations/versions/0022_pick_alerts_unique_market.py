"""Unique index: un solo outcome por (match, market, line) vivo.

Revision ID: 0022
Revises: 0021
Create Date: 2026-04-23

Impide que existan simultáneamente alertas de home y away (o cualquier par
opuesto) para el mismo (match_id, market, line). Garantía a nivel BD contra
la clase de bug BOS-NYY / BRE-LEN observada el 2026-04-22/23.

Pre-req: migración 0019 ya marcó como 'void' las contradicciones vivas,
así que el CREATE UNIQUE INDEX no encontrará conflictos.

CONCURRENTLY obligatorio (plan §6 SDD expand-contract).
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0022"
down_revision: str | None = "0021"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_pick_alerts_market
        ON apuestas.pick_alerts
            (match_id, market, (COALESCE(line, -999)::numeric(6,2)))
        WHERE outcome_result IS NULL OR outcome_result = 'pending'
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS apuestas.uq_pick_alerts_market")
