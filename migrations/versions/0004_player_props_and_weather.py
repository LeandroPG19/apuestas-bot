"""Player props + weather buckets + player_game_logs + materialized view.

Revision ID: 0004
Revises: 0003
Create Date: 2026-04-19

Añade:
- player_prop_lines  (§23.6): líneas ofrecidas por bookmaker
- player_game_logs   (§23.6): stats individuales por partido
- player_game_logs.weather_bucket (§24.3): bucket climático del game
- player_weather_splits (mat view §24.3): agregados por bucket
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ─── Player prop lines ──────────────────────────────────────────────
    op.create_table(
        "player_prop_lines",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("match_id", sa.BigInteger, sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("player_id", sa.BigInteger, sa.ForeignKey("players.id"), nullable=False),
        sa.Column("prop_type", sa.Text, nullable=False),  # 'nba_points', 'mlb_home_run', etc.
        sa.Column("line", sa.Numeric(6, 2), nullable=False),
        sa.Column("over_odds", sa.Numeric(8, 3)),
        sa.Column("under_odds", sa.Numeric(8, 3)),
        sa.Column("bookmaker", sa.Text, nullable=False),
        sa.Column("captured_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint(
            "match_id",
            "player_id",
            "prop_type",
            "line",
            "bookmaker",
            name="uq_prop_line",
        ),
        sa.CheckConstraint("over_odds IS NULL OR over_odds > 1.0", name="ck_prop_over_odds"),
        sa.CheckConstraint("under_odds IS NULL OR under_odds > 1.0", name="ck_prop_under_odds"),
    )
    op.create_index("idx_prop_lines_match", "player_prop_lines", ["match_id", "prop_type"])
    op.create_index(
        "idx_prop_lines_player",
        "player_prop_lines",
        ["player_id", "captured_at"],
        postgresql_ops={"captured_at": "DESC"},
    )

    # ─── Player game logs ───────────────────────────────────────────────
    op.create_table(
        "player_game_logs",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("player_id", sa.BigInteger, sa.ForeignKey("players.id"), nullable=False),
        sa.Column("match_id", sa.BigInteger, sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("sport_code", sa.Text, sa.ForeignKey("sports.code"), nullable=False),
        sa.Column(
            "stats",
            sa.dialects.postgresql.JSONB,
            nullable=False,
            comment='Ej. {"points":28,"rebounds":7,"assists":5}',
        ),
        sa.Column("starter", sa.Boolean),
        sa.Column("minutes_played", sa.Numeric(5, 2)),
        sa.Column("position", sa.Text),
        sa.Column(
            "weather_bucket",
            sa.dialects.postgresql.JSONB,
            comment='§24.3: {"temp":"cool","wind":"moderate",...}',
        ),
        sa.Column("ingested_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("player_id", "match_id", name="uq_player_game"),
    )
    op.create_index(
        "idx_player_logs_player_match",
        "player_game_logs",
        ["player_id", "match_id"],
    )
    op.create_index(
        "idx_player_logs_sport_pos",
        "player_game_logs",
        ["sport_code", "position"],
    )
    # GIN para queries por weather bucket JSONB
    op.execute(
        "CREATE INDEX idx_player_logs_weather_bucket ON player_game_logs USING gin(weather_bucket)"
    )
    # GIN para queries por stats JSONB (ej. filtrar por stats->>'home_runs' > 0)
    op.execute("CREATE INDEX idx_player_logs_stats_gin ON player_game_logs USING gin(stats)")

    # ─── Materialized view: player weather splits ───────────────────────
    # Incluye sólo buckets con >=3 samples (§24.3 HAVING).
    op.execute(
        """
        CREATE MATERIALIZED VIEW player_weather_splits AS
        SELECT
          player_id,
          sport_code,
          weather_bucket,
          COUNT(*) AS sample_size,

          -- MLB
          AVG(NULLIF(stats->>'home_runs','')::numeric) AS avg_home_runs,
          AVG(NULLIF(stats->>'total_bases','')::numeric) AS avg_total_bases,
          AVG(NULLIF(stats->>'hits','')::numeric) AS avg_hits,
          AVG(NULLIF(stats->>'strikeouts','')::numeric) AS avg_strikeouts,

          -- NBA
          AVG(NULLIF(stats->>'points','')::numeric) AS avg_points,
          AVG(NULLIF(stats->>'rebounds','')::numeric) AS avg_rebounds,
          AVG(NULLIF(stats->>'assists','')::numeric) AS avg_assists,

          -- NFL
          AVG(NULLIF(stats->>'passing_yards','')::numeric) AS avg_passing_yards,
          AVG(NULLIF(stats->>'rushing_yards','')::numeric) AS avg_rushing_yards,
          AVG(NULLIF(stats->>'receiving_yards','')::numeric) AS avg_receiving_yards,

          -- Fútbol
          AVG(NULLIF(stats->>'goals','')::numeric) AS avg_goals,
          AVG(NULLIF(stats->>'shots_on_target','')::numeric) AS avg_shots_on_target,
          AVG(NULLIF(stats->>'yellow_cards','')::numeric) AS avg_yellow_cards,

          MIN(ingested_at) AS first_game,
          MAX(ingested_at) AS last_game
        FROM player_game_logs
        WHERE weather_bucket IS NOT NULL
        GROUP BY player_id, sport_code, weather_bucket
        HAVING COUNT(*) >= 3
        """
    )
    # UNIQUE INDEX requerido para REFRESH CONCURRENTLY
    op.execute(
        """
        CREATE UNIQUE INDEX idx_player_weather_splits_pk
        ON player_weather_splits (player_id, sport_code, weather_bucket)
        """
    )
    op.execute(
        """
        CREATE INDEX idx_player_weather_splits_sport
        ON player_weather_splits (sport_code)
        """
    )


def downgrade() -> None:
    op.execute("DROP MATERIALIZED VIEW IF EXISTS player_weather_splits")
    op.drop_table("player_game_logs")
    op.drop_table("player_prop_lines")
