"""Pivot: elimina subsistema bankroll y renombra bets -> pick_alerts.

Revision ID: 0020
Revises: 0019
Create Date: 2026-04-23

Cambios:
- DROP tablas: bankroll_history, bankroll_state, fx_rates, fiscal_events.
- RENAME post_mortems -> pick_analysis y drop de columnas monetarias
  (pnl_units, kelly_predicted, ev_realized, clv).
- RENAME bets -> pick_alerts, drop columnas monetarias/identidad legacy y
  añade campos del nuevo sistema one-alert-per-identity con upgrade.
- RENAME decision_log.bet_id -> decision_log.pick_alert_id.
- Simplifica bot_state (quita primary_currency y primary_currency_locked_at).

Rollback completo: recrea tablas/columnas pero los datos dropeados
quedaron archivados en backups/legacy_bankroll/. El downgrade solo
restaura el schema, no los datos.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0020"
down_revision: str | None = "0019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ---------- DROP tablas bankroll ----------
    op.execute("DROP TABLE IF EXISTS apuestas.bankroll_history CASCADE")
    op.execute("DROP TABLE IF EXISTS apuestas.bankroll_state CASCADE")
    op.execute("DROP TABLE IF EXISTS apuestas.fx_rates CASCADE")
    op.execute("DROP TABLE IF EXISTS apuestas.fiscal_events CASCADE")

    # ---------- post_mortems -> pick_analysis ----------
    op.execute("ALTER TABLE apuestas.post_mortems RENAME TO pick_analysis")
    # Quita columnas monetarias
    for col in (
        "pnl_units",
        "kelly_predicted",
        "ev_realized",
        "ev_realized_vs_predicted",
        "clv",
    ):
        op.execute(f"ALTER TABLE apuestas.pick_analysis DROP COLUMN IF EXISTS {col}")

    # ---------- bets -> pick_alerts ----------
    op.execute("ALTER TABLE apuestas.bets RENAME TO pick_alerts")

    # Drop columnas monetarias/legacy
    drop_cols = [
        "stake_units",
        "pnl_units",
        "closing_line",
        "clv",
        "odds_displayed",
        "currency",
        "fx_rate_used",
        "is_paper",
        "test_data",
        "detection_to_placement_seconds",
        "slippage_bps",
    ]
    for col in drop_cols:
        op.execute(f"ALTER TABLE apuestas.pick_alerts DROP COLUMN IF EXISTS {col}")

    # Añade columnas del nuevo sistema
    with op.batch_alter_table("pick_alerts", schema="apuestas") as batch:
        batch.add_column(sa.Column("best_odds_seen", sa.Numeric(8, 3), nullable=True))
        batch.add_column(sa.Column("best_odds_book", sa.Text(), nullable=True))
        batch.add_column(
            sa.Column(
                "best_odds_updated_at",
                sa.TIMESTAMP(timezone=True),
                nullable=True,
            )
        )
        batch.add_column(
            sa.Column(
                "upgrade_count",
                sa.Integer(),
                nullable=False,
                server_default="0",
            )
        )
        batch.add_column(
            sa.Column(
                "last_alert_at",
                sa.TIMESTAMP(timezone=True),
                nullable=True,
            )
        )
        batch.add_column(sa.Column("outcome_result", sa.Text(), nullable=True))
        batch.add_column(
            sa.Column(
                "result_settled_at",
                sa.TIMESTAMP(timezone=True),
                nullable=True,
            )
        )
        batch.add_column(sa.Column("first_message_id", sa.BigInteger(), nullable=True))
        batch.add_column(
            sa.Column(
                "shap_top5",
                sa.dialects.postgresql.JSONB(astext_type=sa.Text()),
                nullable=True,
            )
        )

    # Backfill: los bets legacy con status en {won, lost, void, halfwon, halflost, cashed}
    # pasan a outcome_result; el resto quedan NULL (pending/unknown).
    op.execute(
        """
        UPDATE apuestas.pick_alerts
        SET outcome_result = CASE
            WHEN status IN ('won','lost','void','cashed','halfwon','halflost')
                THEN status
            ELSE NULL
        END,
            result_settled_at = settled_at
        """
    )

    # Drop el CHECK antiguo (ck_bets_status) y recrea sin 'pending' semantico
    # — pero mantenemos la columna status por compatibilidad (alternate semántica)
    op.execute("ALTER TABLE apuestas.pick_alerts DROP CONSTRAINT IF EXISTS ck_bets_status")
    op.execute("ALTER TABLE apuestas.pick_alerts DROP CONSTRAINT IF EXISTS ck_bets_odds")
    op.execute("ALTER TABLE apuestas.pick_alerts DROP CONSTRAINT IF EXISTS ck_bets_stake")

    # Renombra indexes legacy (siguen activos pero con nombre consistente)
    op.execute("ALTER INDEX IF EXISTS apuestas.bets_pkey RENAME TO pick_alerts_pkey")
    op.execute("ALTER INDEX IF EXISTS apuestas.idx_bets_match RENAME TO idx_pick_alerts_match")
    op.execute("DROP INDEX IF EXISTS apuestas.idx_bets_paper")  # is_paper ya no existe
    op.execute("DROP INDEX IF EXISTS apuestas.idx_bets_pending")
    op.execute("DROP INDEX IF EXISTS apuestas.idx_bets_not_test")
    op.execute("DROP INDEX IF EXISTS apuestas.idx_bets_slippage")
    op.execute("DROP INDEX IF EXISTS apuestas.uq_bets_pending_daily")

    # ---------- decision_log.bet_id -> pick_alert_id ----------
    op.execute("ALTER TABLE apuestas.decision_log RENAME COLUMN bet_id TO pick_alert_id")
    op.execute(
        "ALTER TABLE apuestas.decision_log "
        "RENAME CONSTRAINT decision_log_bet_id_fkey "
        "TO decision_log_pick_alert_id_fkey"
    )

    # ---------- pick_analysis.bet_id -> pick_alert_id ----------
    op.execute("ALTER TABLE apuestas.pick_analysis RENAME COLUMN bet_id TO pick_alert_id")
    # Si tenía FK con nombre antiguo, renombrar
    op.execute(
        """
        DO $$
        DECLARE fk_name text;
        BEGIN
            SELECT conname INTO fk_name FROM pg_constraint
            WHERE conrelid='apuestas.pick_analysis'::regclass AND contype='f'
              AND pg_get_constraintdef(oid) LIKE '%pick_alert_id%';
            IF fk_name IS NOT NULL AND fk_name <> 'pick_analysis_pick_alert_id_fkey' THEN
                EXECUTE format(
                    'ALTER TABLE apuestas.pick_analysis RENAME CONSTRAINT %I TO pick_analysis_pick_alert_id_fkey',
                    fk_name
                );
            END IF;
        END $$;
        """
    )

    # ---------- bot_state: simplifica ----------
    op.execute("ALTER TABLE apuestas.bot_state DROP COLUMN IF EXISTS primary_currency")
    op.execute("ALTER TABLE apuestas.bot_state DROP COLUMN IF EXISTS primary_currency_locked_at")


def downgrade() -> None:
    # Recrea schemas (no los datos, que quedaron en el snapshot legal).
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS apuestas.bankroll_state (
            is_paper boolean NOT NULL,
            currency varchar(3) NOT NULL,
            balance numeric(14,2) NOT NULL DEFAULT 200.00,
            updated_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (is_paper, currency)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS apuestas.bankroll_history (
            ts timestamptz NOT NULL,
            is_paper boolean NOT NULL DEFAULT true,
            bankroll_units numeric(14,4) NOT NULL,
            delta_units numeric(14,4),
            pick_alert_id bigint,
            event text,
            notes text
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS apuestas.fx_rates (
            captured_at timestamptz NOT NULL,
            base_currency varchar(3) NOT NULL,
            quote_currency varchar(3) NOT NULL,
            rate numeric(12,6) NOT NULL,
            PRIMARY KEY (captured_at, base_currency, quote_currency)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS apuestas.fiscal_events (
            id bigserial PRIMARY KEY,
            bet_id bigint,
            event_date date NOT NULL,
            pnl_mxn numeric(14,2) NOT NULL,
            created_at timestamptz DEFAULT now()
        )
        """
    )

    # Revert pick_alerts -> bets
    op.execute("ALTER TABLE apuestas.decision_log RENAME COLUMN pick_alert_id TO bet_id")
    op.execute(
        "ALTER TABLE apuestas.decision_log "
        "RENAME CONSTRAINT decision_log_pick_alert_id_fkey "
        "TO decision_log_bet_id_fkey"
    )

    op.execute("ALTER TABLE apuestas.pick_alerts RENAME TO bets")
    for col, ddl in [
        ("stake_units", "numeric(10,3)"),
        ("pnl_units", "numeric(10,3)"),
        ("closing_line", "numeric(8,3)"),
        ("clv", "numeric(6,4)"),
        ("odds_displayed", "numeric(8,3)"),
        ("currency", "varchar(3) DEFAULT 'USD'"),
        ("fx_rate_used", "numeric(12,6)"),
        ("is_paper", "boolean DEFAULT true NOT NULL"),
        ("test_data", "boolean DEFAULT false NOT NULL"),
        ("detection_to_placement_seconds", "integer"),
        ("slippage_bps", "integer"),
    ]:
        op.execute(f"ALTER TABLE apuestas.bets ADD COLUMN {col} {ddl}")
    for col in (
        "best_odds_seen",
        "best_odds_book",
        "best_odds_updated_at",
        "upgrade_count",
        "last_alert_at",
        "outcome_result",
        "result_settled_at",
        "first_message_id",
        "shap_top5",
    ):
        op.execute(f"ALTER TABLE apuestas.bets DROP COLUMN IF EXISTS {col}")

    # Revert pick_analysis -> post_mortems
    op.execute("ALTER TABLE apuestas.pick_analysis RENAME COLUMN pick_alert_id TO bet_id")
    op.execute("ALTER TABLE apuestas.pick_analysis RENAME TO post_mortems")
    # Las columnas monetarias dropeadas no se recuperan (sin datos que restaurar)

    # bot_state
    op.execute(
        "ALTER TABLE apuestas.bot_state ADD COLUMN IF NOT EXISTS primary_currency varchar(3)"
    )
    op.execute(
        "ALTER TABLE apuestas.bot_state "
        "ADD COLUMN IF NOT EXISTS primary_currency_locked_at timestamptz"
    )
