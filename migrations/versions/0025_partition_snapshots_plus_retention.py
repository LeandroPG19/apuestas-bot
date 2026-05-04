"""Retention policy para snapshots + índice de partición lógica (Gap 10 / A16).

Revision ID: 0025
Revises: 0024
Create Date: 2026-04-24

Sin pg_partman disponible, simulamos particionamiento vía:
  1. Índice BRIN sobre snapshot_at (barato para series temporales).
  2. Función `purge_old_snapshots(retention_days)` invocable por cron.
  3. Trigger `before insert` innecesario (append-only ya cubierto por 0023).

Retention default: 730 días (2 años). Ajustable via env en el wrapper flow.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0025"
down_revision: str | None = "0024"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_pas_brin_snapshot_at "
        "ON apuestas.pick_alerts_snapshots USING BRIN (snapshot_at)"
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION apuestas.purge_old_snapshots(retention_days int DEFAULT 730)
        RETURNS int AS $$
        DECLARE deleted_count int;
        BEGIN
            -- Desactiva trigger append-only solo para esta limpieza administrativa.
            ALTER TABLE apuestas.pick_alerts_snapshots DISABLE TRIGGER tr_pas_no_update;
            DELETE FROM apuestas.pick_alerts_snapshots
            WHERE snapshot_at < now() - (retention_days || ' days')::interval;
            GET DIAGNOSTICS deleted_count = ROW_COUNT;
            ALTER TABLE apuestas.pick_alerts_snapshots ENABLE TRIGGER tr_pas_no_update;
            RETURN deleted_count;
        END;
        $$ LANGUAGE plpgsql
        """
    )


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS apuestas.purge_old_snapshots(int)")
    op.execute("DROP INDEX IF EXISTS apuestas.idx_pas_brin_snapshot_at")
