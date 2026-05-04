"""Fase 0.7: TimescaleDB retention + compression + continuous aggregate.

Revision ID: 0009
Revises: 0008
Create Date: 2026-04-21

Política:
- Raw odds_history > 2 años → compress (ahorro ~80% disk).
- Continuous aggregate `odds_history_hourly_agg`: avg/min/max odds por
  (match_id, market, outcome, bookmaker, bucket=1h) para queries rápidos
  de features (evita escanear 50M rows al generar rolling features).

IMPORTANTE: reversible. En downgrade restaura retention default (forever)
y elimina continuous aggregate.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Activar compression en odds_history (ya hypertable por migración 0001)
    op.execute(
        """
        ALTER TABLE odds_history SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'match_id, bookmaker, market',
            timescaledb.compress_orderby = 'ts DESC, outcome'
        )
        """
    )

    # Política: compress chunks >30 días
    op.execute(
        """
        SELECT add_compression_policy('odds_history', INTERVAL '30 days',
                                      if_not_exists => TRUE)
        """
    )

    # Retention: drop chunks >730 días (2 años) — solo raw, aggregate se mantiene
    op.execute(
        """
        SELECT add_retention_policy('odds_history', INTERVAL '730 days',
                                    if_not_exists => TRUE)
        """
    )

    # Continuous aggregate: bucket 1h con avg/min/max odds
    op.execute(
        """
        CREATE MATERIALIZED VIEW IF NOT EXISTS odds_history_hourly_agg
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 hour', ts) AS bucket,
            match_id,
            bookmaker,
            market,
            outcome,
            line,
            AVG(odds)::numeric(8, 3) AS odds_avg,
            MIN(odds)::numeric(8, 3) AS odds_min,
            MAX(odds)::numeric(8, 3) AS odds_max,
            COUNT(*) AS n_snapshots,
            MAX(CASE WHEN is_closing THEN odds END)::numeric(8, 3) AS odds_closing
        FROM odds_history
        GROUP BY bucket, match_id, bookmaker, market, outcome, line
        WITH NO DATA
        """
    )

    # Refresh policy: cada 30 min, refresca [-4h, -1h]
    op.execute(
        """
        SELECT add_continuous_aggregate_policy('odds_history_hourly_agg',
            start_offset => INTERVAL '4 hours',
            end_offset => INTERVAL '1 hour',
            schedule_interval => INTERVAL '30 minutes',
            if_not_exists => TRUE)
        """
    )


def downgrade() -> None:
    op.execute("DROP MATERIALIZED VIEW IF EXISTS odds_history_hourly_agg CASCADE")
    op.execute("SELECT remove_retention_policy('odds_history', if_exists => TRUE)")
    op.execute("SELECT remove_compression_policy('odds_history', if_exists => TRUE)")
    op.execute("ALTER TABLE odds_history SET (timescaledb.compress = false)")
