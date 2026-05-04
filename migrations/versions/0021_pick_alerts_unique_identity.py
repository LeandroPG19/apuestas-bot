"""Unique index: una alerta por (match, market, line, outcome) viva.

Revision ID: 0021
Revises: 0020
Create Date: 2026-04-23

Impone a nivel BD que sólo exista una alerta "viva" (outcome_result IS NULL
o 'pending') por cada tuple (match_id, market, line, outcome). Upgrades son
modificaciones del row existente; nunca INSERTs nuevos.

CONCURRENTLY obligatorio para zero-downtime (plan §6 SDD expand-contract).
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0021"
down_revision: str | None = "0020"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Dedup exacto previo: si existen dos filas VIVAS con misma identidad
    # (match, market, line, outcome), conservamos la de menor id y marcamos
    # el resto como 'expired' — esto ocurre porque el legacy uq_bets_pending_daily
    # incluía is_paper, permitiendo un PAPER y un REAL idénticos.
    op.execute(
        """
        WITH ranked AS (
            SELECT id,
                   ROW_NUMBER() OVER (
                       PARTITION BY match_id, market,
                           COALESCE(line, -999), outcome
                       ORDER BY id ASC
                   ) AS rn
            FROM apuestas.pick_alerts
            WHERE outcome_result IS NULL OR outcome_result = 'pending'
        )
        UPDATE apuestas.pick_alerts pa
        SET outcome_result = 'expired',
            result_settled_at = now(),
            notes = COALESCE(notes, '')
                    || ' [auto-expired 2026-04-23 pre-unique-identity]'
        FROM ranked r
        WHERE pa.id = r.id AND r.rn > 1
        """
    )

    # En dev la tabla es pequeña; en prod con tabla grande usar CONCURRENTLY
    # desde conexión autocommit (requiere env.py con transaction_per_migration=False).
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_pick_alerts_identity
        ON apuestas.pick_alerts
            (match_id, market, (COALESCE(line, -999)::numeric(6,2)), outcome)
        WHERE outcome_result IS NULL OR outcome_result = 'pending'
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS apuestas.uq_pick_alerts_identity")
