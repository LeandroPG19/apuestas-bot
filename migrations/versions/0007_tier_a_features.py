"""Tier A features gratuitos — schema completo.

Añade schema para cerrar gap vs sharps pro:
- play_by_play: eventos granulares NBA/NFL
- referees + referee_bias_profile: sesgos por árbitro
- coaching_tendencies: hábitos por coach (timeouts, clutch, hack-a-X)
- line_movement_snapshots: snapshots por libro cada N min para steam detector
- steam_moves: movimientos coordinados detectados cross-books
- tracking_proxies: derivables de PBP (acceleration, usage, distance proxies)
- injury_feed: ingesta estructurada de Rotoworld + ESPN beat writers
- bluesky_posts: sentiment via atproto (reemplaza Twitter/X)
- polymarket_markets: futures MVP, Ballon d'Or, etc.

Revision ID: 0007
Revises: 0006
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ───────────────── Play-by-play granular ─────────────────
    op.create_table(
        "play_by_play",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("match_id", sa.BigInteger, sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("sport_code", sa.Text, sa.ForeignKey("sports.code"), nullable=False),
        sa.Column("period", sa.Integer, nullable=False),  # quarter (NBA), half (soccer)
        sa.Column("clock_seconds_remaining", sa.Integer),
        sa.Column("event_type", sa.Text, nullable=False),
        sa.Column("team_id", sa.BigInteger, sa.ForeignKey("teams.id")),
        sa.Column("player_id", sa.BigInteger, sa.ForeignKey("players.id")),
        sa.Column("description", sa.Text),
        sa.Column("home_score", sa.Integer),
        sa.Column("away_score", sa.Integer),
        sa.Column("metadata", JSONB),
        sa.Column("ingested_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_pbp_match_period", "play_by_play", ["match_id", "period"])
    op.create_index(
        "idx_pbp_clutch",
        "play_by_play",
        ["match_id", "clock_seconds_remaining"],
        postgresql_where=sa.text("clock_seconds_remaining <= 180"),
    )
    op.create_index("idx_pbp_player", "play_by_play", ["player_id"])

    # ───────────────── Referees + profile ─────────────────
    op.create_table(
        "referees",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("external_id", sa.Text, unique=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("sport_code", sa.Text, sa.ForeignKey("sports.code")),
        sa.Column("role", sa.Text),  # main, crew_chief, umpire, var, etc.
    )
    op.create_index("idx_referees_sport", "referees", ["sport_code"])

    op.create_table(
        "referee_bias_profile",
        sa.Column("referee_id", sa.BigInteger, sa.ForeignKey("referees.id"), primary_key=True),
        sa.Column("sport_code", sa.Text, primary_key=True),
        sa.Column("n_games", sa.Integer, nullable=False),
        sa.Column("home_win_rate", sa.Numeric(5, 4)),
        sa.Column("home_ats_rate", sa.Numeric(5, 4)),  # NBA/NFL against-the-spread
        sa.Column("over_rate", sa.Numeric(5, 4)),
        sa.Column("avg_total", sa.Numeric(6, 2)),
        sa.Column("fouls_per_game", sa.Numeric(6, 2)),  # NBA
        sa.Column("cards_per_game", sa.Numeric(5, 2)),  # soccer
        sa.Column("strikezone_size_pct", sa.Numeric(5, 4)),  # MLB umpire
        sa.Column("var_interventions_per_game", sa.Numeric(4, 2)),  # soccer
        sa.Column("last_computed", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "match_referees",
        sa.Column("match_id", sa.BigInteger, sa.ForeignKey("matches.id"), primary_key=True),
        sa.Column("referee_id", sa.BigInteger, sa.ForeignKey("referees.id"), primary_key=True),
        sa.Column("role", sa.Text),
    )

    # ───────────────── Coaching tendencies ─────────────────
    op.create_table(
        "coaches",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("external_id", sa.Text, unique=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("sport_code", sa.Text, sa.ForeignKey("sports.code")),
        sa.Column("current_team_id", sa.BigInteger, sa.ForeignKey("teams.id")),
        sa.Column("hired_at", sa.Date),
    )

    op.create_table(
        "coaching_tendencies",
        sa.Column("coach_id", sa.BigInteger, sa.ForeignKey("coaches.id"), primary_key=True),
        sa.Column("sport_code", sa.Text, primary_key=True),
        sa.Column("timeout_usage_pre_clutch", sa.Numeric(5, 4)),  # NBA/NFL
        sa.Column("clutch_close_out_offense_rate", sa.Numeric(5, 4)),
        sa.Column("lineup_pattern_mins_3_4", JSONB),  # Voulgaris: substituciones por minuto
        sa.Column("hack_a_player_rate", sa.Numeric(5, 4)),  # NBA intentional fouls
        sa.Column("go_for_it_4th_down_rate", sa.Numeric(5, 4)),  # NFL
        sa.Column("bullpen_high_leverage_usage", sa.Numeric(5, 4)),  # MLB
        sa.Column("pitch_count_early_hook_avg", sa.Numeric(5, 2)),  # MLB
        sa.Column("n_games_sample", sa.Integer),
        sa.Column("last_computed", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "match_coaches",
        sa.Column("match_id", sa.BigInteger, sa.ForeignKey("matches.id"), primary_key=True),
        sa.Column("team_id", sa.BigInteger, sa.ForeignKey("teams.id"), primary_key=True),
        sa.Column("coach_id", sa.BigInteger, sa.ForeignKey("coaches.id"), nullable=False),
    )

    # ───────────────── Line movement snapshots (steam detector) ─────────
    op.create_table(
        "line_movement_snapshots",
        sa.Column("ts", sa.TIMESTAMP(timezone=True), primary_key=True),
        sa.Column("match_id", sa.BigInteger, primary_key=True),
        sa.Column("bookmaker", sa.Text, primary_key=True),
        sa.Column("market", sa.Text, primary_key=True),
        sa.Column("outcome", sa.Text, primary_key=True),
        sa.Column("odds", sa.Numeric(8, 3), nullable=False),
        sa.Column("line", sa.Numeric(6, 2)),
        sa.Column("volume_indicator", sa.Numeric(6, 2)),  # % move since last snapshot
    )
    op.execute(
        "SELECT create_hypertable('line_movement_snapshots', 'ts', "
        "chunk_time_interval => INTERVAL '1 day')"
    )
    op.create_index(
        "idx_line_movement_match_market",
        "line_movement_snapshots",
        ["match_id", "market", "ts"],
    )

    op.create_table(
        "steam_moves",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("detected_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("match_id", sa.BigInteger, sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("market", sa.Text, nullable=False),
        sa.Column("outcome", sa.Text, nullable=False),
        sa.Column("direction", sa.Text, nullable=False),  # "up" / "down"
        sa.Column("magnitude_pct", sa.Numeric(5, 4), nullable=False),
        sa.Column("n_books_moved", sa.Integer, nullable=False),
        sa.Column("pinnacle_leading", sa.Boolean, server_default=sa.false()),
        sa.Column("books_involved", JSONB),
        sa.Column("window_minutes", sa.Integer),
    )
    op.create_index("idx_steam_match", "steam_moves", ["match_id", "detected_at"])

    # ───────────────── Tracking proxies (derivables PBP) ─────────
    op.create_table(
        "player_tracking_proxies",
        sa.Column("player_id", sa.BigInteger, sa.ForeignKey("players.id"), primary_key=True),
        sa.Column("match_id", sa.BigInteger, sa.ForeignKey("matches.id"), primary_key=True),
        sa.Column("usage_rate", sa.Numeric(6, 4)),
        sa.Column("touches_per_36", sa.Numeric(6, 2)),
        sa.Column("acceleration_proxy", sa.Numeric(6, 3)),  # transition/halfcourt split
        sa.Column("distance_proxy_km", sa.Numeric(6, 3)),  # minutes × team pace
        sa.Column("defensive_load", sa.Numeric(6, 3)),
        sa.Column("fatigue_index", sa.Numeric(5, 3)),
        sa.Column("computed_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )

    # ───────────────── Injury feed estructurado ─────────────────
    op.create_table(
        "injury_feed",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("player_id", sa.BigInteger, sa.ForeignKey("players.id")),
        sa.Column("player_name_raw", sa.Text),
        sa.Column("team_id", sa.BigInteger, sa.ForeignKey("teams.id")),
        sa.Column("source", sa.Text, nullable=False),  # rotoworld, espn, bluesky, reddit
        sa.Column("reporter", sa.Text),  # beat writer name
        sa.Column("status_reported", sa.Text),  # out, doubtful, questionable, probable, active
        sa.Column("body_part", sa.Text),
        sa.Column("severity_estimate", sa.Text),  # minor, moderate, major
        sa.Column("expected_return_date", sa.Date),
        sa.Column("raw_text", sa.Text),
        sa.Column("sentiment_score", sa.Numeric(5, 4)),
        sa.Column("confidence_score", sa.Numeric(4, 3)),
        sa.Column("reported_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("ingested_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_injury_feed_player", "injury_feed", ["player_id", "reported_at"])
    op.create_index("idx_injury_feed_ts", "injury_feed", ["reported_at"])

    # ───────────────── Bluesky posts (sentiment) ─────────────────
    op.create_table(
        "bluesky_posts",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("post_uri", sa.Text, unique=True, nullable=False),
        sa.Column("author_handle", sa.Text, nullable=False),
        sa.Column("author_followers", sa.Integer),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("published_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("likes", sa.Integer, server_default="0"),
        sa.Column("reposts", sa.Integer, server_default="0"),
        sa.Column("teams_mentioned", sa.ARRAY(sa.BigInteger)),
        sa.Column("players_mentioned", sa.ARRAY(sa.BigInteger)),
        sa.Column("sports_mentioned", sa.ARRAY(sa.Text)),
        sa.Column("sentiment_score", sa.Numeric(5, 4)),
        sa.Column("ingested_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_bluesky_published", "bluesky_posts", ["published_at"])

    # ───────────────── Polymarket markets ─────────────────
    op.create_table(
        "polymarket_markets",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("condition_id", sa.Text, unique=True, nullable=False),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("sport_code", sa.Text, sa.ForeignKey("sports.code")),
        sa.Column("event_type", sa.Text),  # mvp, ballon_dor, champion
        sa.Column("end_date", sa.TIMESTAMP(timezone=True)),
        sa.Column("outcomes", JSONB),
        sa.Column("current_prices", JSONB),
        sa.Column("volume_24h_usd", sa.Numeric(14, 2)),
        sa.Column("last_updated", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_polymarket_sport_endate", "polymarket_markets", ["sport_code", "end_date"])


def downgrade() -> None:
    for tbl in (
        "polymarket_markets",
        "bluesky_posts",
        "injury_feed",
        "player_tracking_proxies",
        "steam_moves",
        "line_movement_snapshots",
        "match_coaches",
        "coaching_tendencies",
        "coaches",
        "match_referees",
        "referee_bias_profile",
        "referees",
        "play_by_play",
    ):
        op.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
