"""CLV tracking — Sprint 12.

Revision ID: 0027
Revises: 0026
Create Date: 2026-04-24

Añade:
- `pick_closing_lines`: snapshots Pinnacle closing odds capturadas 30-60 min
  pre-kickoff. Clave (pick_alert_id, captured_at).
- `pick_alerts.clv_pct`: CLV calculado post-match.
- `pick_alerts.closing_pinn_odds`: odds de cierre Pinnacle de-vigged.
- `pick_alerts.closing_captured_at`: timestamp snapshot.

Fórmula CLV:
    clv_pct = (p_pick_devigged / p_closing_devigged) - 1
    positivo = tomas odds mejores que el cierre (skill real)
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0027"
down_revision: str | None = "0026"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS apuestas.pick_closing_lines (
            id bigserial PRIMARY KEY,
            pick_alert_id bigint REFERENCES apuestas.pick_alerts(id) ON DELETE CASCADE,
            match_id bigint REFERENCES apuestas.matches(id),
            market text NOT NULL,
            outcome text NOT NULL,
            line numeric(6,2),
            pinnacle_odds numeric(8,3),
            pinnacle_odds_devigged numeric(8,3),
            devig_method text,
            minutes_to_kickoff integer,
            captured_at timestamptz NOT NULL DEFAULT now(),
            UNIQUE (pick_alert_id, captured_at)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_pcl_pick "
        "ON apuestas.pick_closing_lines (pick_alert_id, captured_at DESC)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_pcl_match "
        "ON apuestas.pick_closing_lines (match_id, market, outcome)"
    )

    # Columnas CLV en pick_alerts
    op.execute("ALTER TABLE apuestas.pick_alerts ADD COLUMN IF NOT EXISTS clv_pct numeric(7,4)")
    op.execute(
        "ALTER TABLE apuestas.pick_alerts ADD COLUMN IF NOT EXISTS closing_pinn_odds numeric(8,3)"
    )
    op.execute(
        "ALTER TABLE apuestas.pick_alerts ADD COLUMN IF NOT EXISTS closing_captured_at timestamptz"
    )

    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_pa_clv "
        "ON apuestas.pick_alerts (clv_pct) WHERE clv_pct IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS apuestas.idx_pa_clv")
    op.execute("ALTER TABLE apuestas.pick_alerts DROP COLUMN IF EXISTS closing_captured_at")
    op.execute("ALTER TABLE apuestas.pick_alerts DROP COLUMN IF EXISTS closing_pinn_odds")
    op.execute("ALTER TABLE apuestas.pick_alerts DROP COLUMN IF EXISTS clv_pct")
    op.execute("DROP TABLE IF EXISTS apuestas.pick_closing_lines CASCADE")
