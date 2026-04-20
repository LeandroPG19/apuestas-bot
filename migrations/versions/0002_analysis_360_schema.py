"""Analysis 360: venues, travel, player_news, injuries, lineups, streaks, transfers, coaching, h2h, weather, officials.

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-19

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ─── Venues + home advantage ────────────────────────────────────────
    op.create_table(
        "venues",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("external_id", sa.Text, unique=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("city", sa.Text),
        sa.Column("country", sa.Text),
        sa.Column("timezone", sa.Text),
        sa.Column("lat", sa.Numeric(9, 6)),
        sa.Column("lon", sa.Numeric(9, 6)),
        sa.Column("altitude_m", sa.Integer),
        sa.Column("capacity", sa.Integer),
        sa.Column("surface", sa.Text),  # turf|grass|hardwood|clay|dirt|ring|cage
        sa.Column("roof", sa.Text),  # open|retractable|dome|indoor
        sa.Column("metadata", sa.JSON),
    )
    op.create_index(
        "idx_venues_name_trgm",
        "venues",
        ["name"],
        postgresql_using="gin",
        postgresql_ops={"name": "gin_trgm_ops"},
    )

    op.create_foreign_key("fk_matches_venue", "matches", "venues", ["venue_id"], ["id"])
    op.create_foreign_key("fk_teams_venue", "teams", "venues", ["venue_id"], ["id"])

    op.create_table(
        "venue_factors",
        sa.Column("venue_id", sa.BigInteger, sa.ForeignKey("venues.id"), primary_key=True),
        sa.Column("sport_code", sa.Text, sa.ForeignKey("sports.code"), primary_key=True),
        sa.Column("home_win_rate", sa.Numeric(5, 4)),
        sa.Column("home_ats_rate", sa.Numeric(5, 4)),
        sa.Column("over_rate", sa.Numeric(5, 4)),
        sa.Column("avg_total", sa.Numeric(6, 2)),
        sa.Column("weather_impact_score", sa.Numeric(4, 2)),
        sa.Column("sample_games", sa.Integer),
        sa.Column("last_computed", sa.TIMESTAMP(timezone=True)),
    )

    # ─── Travel log por equipo-evento ────────────────────────────────────
    op.create_table(
        "travel_log",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("match_id", sa.BigInteger, sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("team_id", sa.BigInteger, sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("is_home", sa.Boolean, nullable=False),
        sa.Column("previous_match_id", sa.BigInteger, sa.ForeignKey("matches.id")),
        sa.Column("rest_days", sa.Integer),
        sa.Column("back_to_back", sa.Boolean),
        sa.Column("distance_km", sa.Numeric(10, 2)),
        sa.Column("timezone_delta_hours", sa.Integer),
        sa.Column("altitude_delta_m", sa.Integer),
        sa.Column("total_games_last_7_days", sa.Integer),
        sa.Column("total_games_last_14_days", sa.Integer),
        sa.Column(
            "computed_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("match_id", "team_id", name="uq_travel_match_team"),
    )
    op.create_index("idx_travel_match", "travel_log", ["match_id"])
    op.create_index("idx_travel_team", "travel_log", ["team_id"])

    # ─── Noticias por jugador ────────────────────────────────────────────
    op.create_table(
        "player_news",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("player_id", sa.BigInteger, sa.ForeignKey("players.id"), nullable=False),
        sa.Column("player_name", sa.Text, nullable=False),
        sa.Column("source", sa.Text, nullable=False),
        sa.Column("url", sa.Text),
        sa.Column("title", sa.Text),
        sa.Column("content", sa.Text),
        sa.Column("published_at", sa.TIMESTAMP(timezone=True)),
        sa.Column(
            "ingested_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("sentiment_score", sa.Numeric(5, 4)),
        sa.Column("impact_rating", sa.Text),  # minor|moderate|major
        sa.Column("embedding", Vector(1024)),
    )
    op.create_index("idx_player_news_player_ts", "player_news", ["player_id", "published_at"])
    op.execute(
        "CREATE INDEX idx_player_news_embedding_hnsw ON player_news "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )

    # ─── Lesiones ────────────────────────────────────────────────────────
    op.create_table(
        "injuries",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("player_id", sa.BigInteger, sa.ForeignKey("players.id"), nullable=False),
        sa.Column("sport_code", sa.Text, sa.ForeignKey("sports.code")),
        sa.Column("status", sa.Text, nullable=False),  # out|doubtful|questionable|probable|active
        sa.Column("body_part", sa.Text),
        sa.Column("expected_return", sa.Date),
        sa.Column("games_missed", sa.Integer, server_default="0"),
        sa.Column("source", sa.Text),
        sa.Column("reported_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("last_verified_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("verification_score", sa.Numeric(3, 2)),
        sa.Column("metadata", sa.JSON),
        sa.CheckConstraint(
            "status IN ('out','doubtful','questionable','probable','active')",
            name="ck_injuries_status",
        ),
    )
    op.create_index("idx_injuries_player", "injuries", ["player_id", "reported_at"])
    op.create_index(
        "idx_injuries_active",
        "injuries",
        ["player_id", "status"],
        postgresql_where=sa.text("status IN ('out','doubtful','questionable')"),
    )

    # ─── Lineups ─────────────────────────────────────────────────────────
    op.create_table(
        "lineups",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("match_id", sa.BigInteger, sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("team_id", sa.BigInteger, sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("starter_ids", sa.dialects.postgresql.ARRAY(sa.BigInteger)),
        sa.Column("bench_ids", sa.dialects.postgresql.ARRAY(sa.BigInteger)),
        sa.Column("formation", sa.Text),
        sa.Column("confirmed", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("source", sa.Text),
        sa.Column(
            "updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("match_id", "team_id", name="uq_lineups_match_team"),
    )
    op.create_index("idx_lineups_match", "lineups", ["match_id"])

    # ─── Streaks ─────────────────────────────────────────────────────────
    op.create_table(
        "team_streaks",
        sa.Column("team_id", sa.BigInteger, sa.ForeignKey("teams.id"), primary_key=True),
        sa.Column("sport_code", sa.Text, sa.ForeignKey("sports.code"), primary_key=True),
        sa.Column("metric", sa.Text, primary_key=True),  # W/L/O/U/ATS+/ATS-/home_W/away_W
        sa.Column("current_length", sa.Integer, nullable=False),
        sa.Column("direction", sa.Text, nullable=False),
        sa.Column("last_n_values", sa.dialects.postgresql.ARRAY(sa.Numeric(10, 3))),
        sa.Column(
            "computed_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("idx_team_streaks_team", "team_streaks", ["team_id"])

    op.create_table(
        "player_streaks",
        sa.Column("player_id", sa.BigInteger, sa.ForeignKey("players.id"), primary_key=True),
        sa.Column("metric", sa.Text, primary_key=True),  # points|rebounds|Ks|goals|yards...
        sa.Column("current_length", sa.Integer, nullable=False),
        sa.Column("direction", sa.Text, nullable=False),
        sa.Column("last_n_values", sa.dialects.postgresql.ARRAY(sa.Numeric(10, 3))),
        sa.Column(
            "computed_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )

    # ─── Transfers ───────────────────────────────────────────────────────
    op.create_table(
        "transfers",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("player_id", sa.BigInteger, sa.ForeignKey("players.id")),
        sa.Column("from_team_id", sa.BigInteger, sa.ForeignKey("teams.id")),
        sa.Column("to_team_id", sa.BigInteger, sa.ForeignKey("teams.id")),
        sa.Column("transfer_date", sa.Date, nullable=False),
        sa.Column("fee_usd", sa.Numeric(14, 2)),
        sa.Column("transfer_type", sa.Text),  # trade|free_agency|loan|waiver|draft
        sa.Column("source", sa.Text),
        sa.Column("impact_rating", sa.Text),  # low|medium|high
        sa.Column("notes", sa.Text),
    )
    op.create_index("idx_transfers_player_date", "transfers", ["player_id", "transfer_date"])
    op.create_index("idx_transfers_to_team_date", "transfers", ["to_team_id", "transfer_date"])

    # ─── Cambios de cuerpo técnico ───────────────────────────────────────
    op.create_table(
        "coaching_changes",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("team_id", sa.BigInteger, sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("old_coach", sa.Text),
        sa.Column("new_coach", sa.Text),
        sa.Column("change_date", sa.Date, nullable=False),
        sa.Column("reason", sa.Text),  # fired|resigned|retired|promoted|interim
        sa.Column("system_change_notes", sa.Text),
        sa.Column("source", sa.Text),
    )
    op.create_index("idx_coaching_team_date", "coaching_changes", ["team_id", "change_date"])

    # ─── H2H histórico ───────────────────────────────────────────────────
    op.create_table(
        "h2h_history",
        sa.Column("team_a_id", sa.BigInteger, sa.ForeignKey("teams.id"), primary_key=True),
        sa.Column("team_b_id", sa.BigInteger, sa.ForeignKey("teams.id"), primary_key=True),
        sa.Column("total_meetings", sa.Integer, server_default="0"),
        sa.Column("a_wins", sa.Integer, server_default="0"),
        sa.Column("b_wins", sa.Integer, server_default="0"),
        sa.Column("draws", sa.Integer, server_default="0"),
        sa.Column("a_cover_ats", sa.Integer, server_default="0"),
        sa.Column("over_hits", sa.Integer, server_default="0"),
        sa.Column("last_10_summary", sa.JSON),
        sa.Column("venue_specific", sa.JSON),
        sa.Column("last_computed", sa.TIMESTAMP(timezone=True)),
        sa.CheckConstraint("team_a_id < team_b_id", name="ck_h2h_order"),
    )

    # ─── Rolling stats home/away split ───────────────────────────────────
    op.create_table(
        "team_stats_rolling_home",
        sa.Column("team_id", sa.BigInteger, sa.ForeignKey("teams.id"), primary_key=True),
        sa.Column("sport_code", sa.Text, sa.ForeignKey("sports.code"), primary_key=True),
        sa.Column("window_size", sa.Integer, primary_key=True),  # 5, 10, 20
        sa.Column("metrics", sa.JSON, nullable=False),
        sa.Column("sample_size", sa.Integer),
        sa.Column("last_computed", sa.TIMESTAMP(timezone=True)),
    )
    op.create_table(
        "team_stats_rolling_away",
        sa.Column("team_id", sa.BigInteger, sa.ForeignKey("teams.id"), primary_key=True),
        sa.Column("sport_code", sa.Text, sa.ForeignKey("sports.code"), primary_key=True),
        sa.Column("window_size", sa.Integer, primary_key=True),
        sa.Column("metrics", sa.JSON, nullable=False),
        sa.Column("sample_size", sa.Integer),
        sa.Column("last_computed", sa.TIMESTAMP(timezone=True)),
    )
    op.create_table(
        "player_stats_rolling",
        sa.Column("player_id", sa.BigInteger, sa.ForeignKey("players.id"), primary_key=True),
        sa.Column("window_size", sa.Integer, primary_key=True),
        sa.Column("split", sa.Text, primary_key=True),  # all|home|away|vs_team_X
        sa.Column("metrics", sa.JSON, nullable=False),
        sa.Column("sample_size", sa.Integer),
        sa.Column("last_computed", sa.TIMESTAMP(timezone=True)),
    )

    # ─── Weather forecast por match ──────────────────────────────────────
    op.create_table(
        "weather_forecast",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("match_id", sa.BigInteger, sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("forecast_ts", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column(
            "captured_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("temp_c", sa.Numeric(5, 2)),
        sa.Column("wind_kph", sa.Numeric(6, 2)),
        sa.Column("wind_direction_deg", sa.Integer),
        sa.Column("precip_mm", sa.Numeric(6, 2)),
        sa.Column("humidity_pct", sa.Integer),
        sa.Column("conditions", sa.Text),
        sa.Column("source", sa.Text),
    )
    op.create_index("idx_weather_match", "weather_forecast", ["match_id", "forecast_ts"])

    # ─── Officials (árbitros, umpires) ───────────────────────────────────
    op.create_table(
        "officials",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("external_id", sa.Text, unique=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("sport_code", sa.Text, sa.ForeignKey("sports.code")),
        sa.Column("role", sa.Text),  # referee|umpire|judge
        sa.Column("stats", sa.JSON),  # métricas agregadas
        sa.Column("last_computed", sa.TIMESTAMP(timezone=True)),
    )
    op.create_index(
        "idx_officials_name_trgm",
        "officials",
        ["name"],
        postgresql_using="gin",
        postgresql_ops={"name": "gin_trgm_ops"},
    )

    op.create_table(
        "match_officials",
        sa.Column("match_id", sa.BigInteger, sa.ForeignKey("matches.id"), primary_key=True),
        sa.Column("official_id", sa.BigInteger, sa.ForeignKey("officials.id"), primary_key=True),
        sa.Column("role", sa.Text, primary_key=True),
    )


def downgrade() -> None:
    op.drop_table("match_officials")
    op.drop_table("officials")
    op.drop_table("weather_forecast")
    op.drop_table("player_stats_rolling")
    op.drop_table("team_stats_rolling_away")
    op.drop_table("team_stats_rolling_home")
    op.drop_table("h2h_history")
    op.drop_table("coaching_changes")
    op.drop_table("transfers")
    op.drop_table("player_streaks")
    op.drop_table("team_streaks")
    op.drop_table("lineups")
    op.drop_table("injuries")
    op.drop_table("player_news")
    op.drop_table("travel_log")
    op.drop_table("venue_factors")
    op.drop_constraint("fk_teams_venue", "teams", type_="foreignkey")
    op.drop_constraint("fk_matches_venue", "matches", type_="foreignkey")
    op.drop_table("venues")
