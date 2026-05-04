"""Snapshot inmutable de alertas — CLV retrospectivo sin bankroll.

Revision ID: 0023
Revises: 0022
Create Date: 2026-04-23

Plan §7.4 / Sprint 5 G7. Tras eliminar bankroll/PnL, perdimos la capacidad
de calcular CLV sobre picks emitidos. `pick_alerts_snapshots` es una
tabla append-only que captura un punto-en-el-tiempo por cada emit/upgrade
/close de una alerta, con:
  - odds de todos los books al momento
  - midpoints Polymarket/Kalshi (Sprint 6 los pobla)
  - p_consensus_sharp y p_model
  - EV al momento

Calcular CLV retrospectivo: clv = (p_consensus_emit − p_consensus_close) /
p_consensus_close, sin depender de `closing_line` en pick_alerts.

El trigger `prevent_mutation` bloquea UPDATE/DELETE para garantizar que
los snapshots son evidencia auditable del skill del bot.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0023"
down_revision: str | None = "0022"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS apuestas.pick_alerts_snapshots (
            id                bigserial PRIMARY KEY,
            pick_alert_id     bigint NOT NULL
                              REFERENCES apuestas.pick_alerts(id) ON DELETE CASCADE,
            snapshot_type     text NOT NULL
                              CHECK (snapshot_type IN ('emit','upgrade','close')),
            odds_all_books    jsonb NOT NULL,
            polymarket_mid    numeric(6,4),
            kalshi_mid        numeric(6,4),
            p_consensus_sharp numeric(6,4),
            p_model           numeric(6,4),
            ev_at_snapshot    numeric(7,4),
            snapshot_at       timestamptz NOT NULL DEFAULT now()
        )
        """
    )

    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_pas_alert ON apuestas.pick_alerts_snapshots (pick_alert_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_pas_snapshot_at "
        "ON apuestas.pick_alerts_snapshots (snapshot_at DESC)"
    )

    # Trigger append-only: bloquea UPDATE/DELETE.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION apuestas.prevent_snapshot_mutation()
        RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION
              'pick_alerts_snapshots es append-only — % no permitido',
              TG_OP;
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        """
        DROP TRIGGER IF EXISTS tr_pas_no_update ON apuestas.pick_alerts_snapshots
        """
    )
    op.execute(
        """
        CREATE TRIGGER tr_pas_no_update
        BEFORE UPDATE OR DELETE ON apuestas.pick_alerts_snapshots
        FOR EACH ROW EXECUTE FUNCTION apuestas.prevent_snapshot_mutation()
        """
    )

    # Pylance/mypy placeholder — evitamos import no usado
    _ = sa


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS tr_pas_no_update ON apuestas.pick_alerts_snapshots")
    op.execute("DROP FUNCTION IF EXISTS apuestas.prevent_snapshot_mutation()")
    op.execute("DROP TABLE IF EXISTS apuestas.pick_alerts_snapshots CASCADE")
