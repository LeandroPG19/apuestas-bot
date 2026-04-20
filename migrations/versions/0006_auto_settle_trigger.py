"""Auto-settle trigger: match finished → NOTIFY settle_bets worker.

Cuando `matches.status` cambia a 'finished' y los scores están populados,
el trigger:
1. Emite NOTIFY en el canal `apuestas_match_finished` con payload JSON
   `{match_id, home_score, away_score}`.
2. Escribe un evento en `settlement_queue` para workers que no usen LISTEN.

Workers (systemd-user o taskiq) hacen LISTEN o polean la queue, disparan
`settle_bets_flow` cuando detectan un match. De esta forma NO se necesita
`apuestas settle` manual tras cada partido.

Revision ID: 0006
Revises: 0005
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ─── Cola de settlement para polling workers ─────────────────────────
    op.create_table(
        "settlement_queue",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("match_id", sa.BigInteger, sa.ForeignKey("matches.id"), nullable=False),
        sa.Column(
            "created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("processed_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("status", sa.Text, nullable=False, server_default="pending"),
        sa.Column("notes", sa.Text),
        sa.CheckConstraint(
            "status IN ('pending','processing','done','failed')",
            name="ck_settlement_queue_status",
        ),
    )
    op.create_index(
        "idx_settlement_queue_pending",
        "settlement_queue",
        ["created_at"],
        postgresql_where=sa.text("status = 'pending'"),
    )

    # ─── Trigger function: se dispara al marcar un match finished ────────
    op.execute(
        """
        CREATE OR REPLACE FUNCTION trg_match_finished_to_settle()
        RETURNS TRIGGER
        LANGUAGE plpgsql AS $$
        DECLARE
            payload jsonb;
        BEGIN
            -- Solo actuar cuando el match pasa a finished con ambos scores
            IF (NEW.status = 'finished' AND OLD.status <> 'finished'
                AND NEW.home_score IS NOT NULL AND NEW.away_score IS NOT NULL)
            THEN
                payload := jsonb_build_object(
                    'match_id', NEW.id,
                    'home_score', NEW.home_score,
                    'away_score', NEW.away_score,
                    'sport_code', NEW.sport_code
                );
                -- 1. NOTIFY para workers con LISTEN activo (baja latencia)
                PERFORM pg_notify('apuestas_match_finished', payload::text);
                -- 2. Insertar en cola para workers polling (fallback resiliente)
                INSERT INTO settlement_queue (match_id, status)
                VALUES (NEW.id, 'pending')
                ON CONFLICT DO NOTHING;
            END IF;
            RETURN NEW;
        END;
        $$;
        """
    )

    op.execute(
        """
        CREATE TRIGGER matches_auto_settle_trigger
        AFTER UPDATE OF status ON matches
        FOR EACH ROW
        EXECUTE FUNCTION trg_match_finished_to_settle();
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS matches_auto_settle_trigger ON matches")
    op.execute("DROP FUNCTION IF EXISTS trg_match_finished_to_settle()")
    op.drop_table("settlement_queue")
