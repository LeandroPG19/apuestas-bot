"""Pre-pivot: void de bets contradictorias vivas.

Revision ID: 0019
Revises: 0018
Create Date: 2026-04-23

Antes de demoler el subsistema de bankroll e instalar los unique indexes
nuevos (0021/0022), marcamos como 'void' cualquier bet pendiente que viva
simultáneamente con otra bet del mismo (match_id, market, line) pero con
outcome distinto — la prueba de que el dedup previo era incompleto.

Sin este paso, el CREATE UNIQUE INDEX de 0022 fallaría por conflicto.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0019"
down_revision: str | None = "0018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        WITH contras AS (
            SELECT match_id, market, COALESCE(line, -999) AS lk
            FROM apuestas.bets
            WHERE status = 'pending'
            GROUP BY match_id, market, COALESCE(line, -999)
            HAVING COUNT(DISTINCT outcome) > 1
        )
        UPDATE apuestas.bets b
        SET status = 'void',
            notes = COALESCE(notes, '')
                    || ' [auto-voided 2026-04-23 pre-pivot: contradictory pair]',
            settled_at = now()
        FROM contras c
        WHERE b.match_id = c.match_id
          AND b.market = c.market
          AND COALESCE(b.line, -999) = c.lk
          AND b.status = 'pending';
        """
    )


def downgrade() -> None:
    # Downgrade es no-op: los bets voided ya quedaron con notes explícita.
    # No los resucitamos porque eran basura original.
    pass
