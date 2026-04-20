"""Core schema: sports, teams, players, matches, odds_history, predictions, bets, news_articles.

Revision ID: 0001
Revises:
Create Date: 2026-04-19

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Extensiones (idempotentes, ya instaladas por init SQL pero explicitar)
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute("CREATE EXTENSION IF NOT EXISTS btree_gin")
    op.execute("CREATE EXTENSION IF NOT EXISTS unaccent")

    # ─── Catálogos ───────────────────────────────────────────────────────
    op.create_table(
        "sports",
        sa.Column("code", sa.Text, primary_key=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("has_draws", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("active", sa.Boolean, nullable=False, server_default=sa.true()),
    )
    op.execute(
        """
        INSERT INTO sports (code, name, has_draws) VALUES
            ('nba', 'NBA Basketball', false),
            ('mlb', 'MLB Baseball', false),
            ('nfl', 'NFL Football', false),
            ('soccer', 'Soccer', true),
            ('boxing', 'Boxing', false),
            ('mma', 'MMA', false)
        """
    )

    op.create_table(
        "leagues",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("external_id", sa.Text, unique=True),
        sa.Column("sport_code", sa.Text, sa.ForeignKey("sports.code"), nullable=False),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("country", sa.Text),
        sa.Column("tier", sa.Integer),
        sa.Column("metadata", sa.JSON),
    )
    op.create_index("idx_leagues_sport", "leagues", ["sport_code"])

    op.create_table(
        "teams",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("external_id", sa.Text, unique=True),
        sa.Column("sport_code", sa.Text, sa.ForeignKey("sports.code"), nullable=False),
        sa.Column("league_id", sa.BigInteger, sa.ForeignKey("leagues.id")),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("short_name", sa.Text),
        sa.Column("city", sa.Text),
        sa.Column("country", sa.Text),
        sa.Column("abbreviation", sa.Text),
        sa.Column("venue_id", sa.BigInteger),
        sa.Column("active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("metadata", sa.JSON),
    )
    op.create_index("idx_teams_sport", "teams", ["sport_code"])
    op.create_index(
        "idx_teams_name_trgm",
        "teams",
        ["name"],
        postgresql_using="gin",
        postgresql_ops={"name": "gin_trgm_ops"},
    )

    op.create_table(
        "players",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("external_id", sa.Text, unique=True),
        sa.Column("sport_code", sa.Text, sa.ForeignKey("sports.code"), nullable=False),
        sa.Column("team_id", sa.BigInteger, sa.ForeignKey("teams.id")),
        sa.Column("full_name", sa.Text, nullable=False),
        sa.Column("first_name", sa.Text),
        sa.Column("last_name", sa.Text),
        sa.Column("position", sa.Text),
        sa.Column("jersey_number", sa.Integer),
        sa.Column("birthdate", sa.Date),
        sa.Column("height_cm", sa.Integer),
        sa.Column("weight_kg", sa.Numeric(6, 2)),
        sa.Column("active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("metadata", sa.JSON),
    )
    op.create_index("idx_players_team", "players", ["team_id"])
    op.create_index(
        "idx_players_name_trgm",
        "players",
        ["full_name"],
        postgresql_using="gin",
        postgresql_ops={"full_name": "gin_trgm_ops"},
    )

    # ─── Matches ─────────────────────────────────────────────────────────
    op.create_table(
        "matches",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("external_id", sa.Text, unique=True),
        sa.Column("sport_code", sa.Text, sa.ForeignKey("sports.code"), nullable=False),
        sa.Column("league_id", sa.BigInteger, sa.ForeignKey("leagues.id")),
        sa.Column("season", sa.Text),
        sa.Column("stage", sa.Text),
        sa.Column("home_team_id", sa.BigInteger, sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("away_team_id", sa.BigInteger, sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("venue_id", sa.BigInteger),
        sa.Column("start_time", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("status", sa.Text, nullable=False, server_default="scheduled"),
        sa.Column("home_score", sa.Integer),
        sa.Column("away_score", sa.Integer),
        sa.Column("metadata", sa.JSON),
        sa.Column(
            "created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint("home_team_id <> away_team_id", name="ck_matches_different_teams"),
        sa.CheckConstraint(
            "status IN ('scheduled','live','finished','cancelled','postponed','void')",
            name="ck_matches_status",
        ),
    )
    op.create_index("idx_matches_start", "matches", ["start_time"])
    op.create_index("idx_matches_status", "matches", ["status", "start_time"])
    op.create_index("idx_matches_teams", "matches", ["home_team_id", "away_team_id"])
    op.create_index("idx_matches_sport_league", "matches", ["sport_code", "league_id"])

    # ─── Odds history (hypertable TimescaleDB) ───────────────────────────
    op.create_table(
        "odds_history",
        sa.Column("ts", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("match_id", sa.BigInteger, nullable=False),
        sa.Column("bookmaker", sa.Text, nullable=False),
        sa.Column("market", sa.Text, nullable=False),
        sa.Column("outcome", sa.Text, nullable=False),
        sa.Column("line", sa.Numeric(6, 2)),
        sa.Column("odds", sa.Numeric(8, 3), nullable=False),
        sa.Column("implied_prob", sa.Numeric(6, 5), sa.Computed("1.0/odds", persisted=True)),
        sa.Column("is_closing", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.CheckConstraint("odds > 1.0", name="ck_odds_positive"),
    )
    op.execute(
        "SELECT create_hypertable('odds_history', 'ts', chunk_time_interval => INTERVAL '1 day')"
    )
    op.execute(
        "ALTER TABLE odds_history SET (timescaledb.compress, timescaledb.compress_segmentby = 'match_id,bookmaker,market,outcome')"
    )
    op.execute("SELECT add_compression_policy('odds_history', INTERVAL '3 days')")
    op.execute("SELECT add_retention_policy('odds_history', INTERVAL '2 years')")
    op.create_index(
        "idx_odds_match_market", "odds_history", ["match_id", "market", "outcome", "ts"]
    )
    op.create_index(
        "idx_odds_closing",
        "odds_history",
        ["match_id", "is_closing"],
        postgresql_where=sa.text("is_closing = true"),
    )

    # ─── Predictions ─────────────────────────────────────────────────────
    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("match_id", sa.BigInteger, sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("model_name", sa.Text, nullable=False),
        sa.Column("model_version", sa.Text, nullable=False),
        sa.Column("market", sa.Text, nullable=False),
        sa.Column("outcome", sa.Text, nullable=False),
        sa.Column("line", sa.Numeric(6, 2)),
        sa.Column("probability", sa.Numeric(6, 5), nullable=False),
        sa.Column("p_lower", sa.Numeric(6, 5)),  # conformal CI lower
        sa.Column("p_upper", sa.Numeric(6, 5)),  # conformal CI upper
        sa.Column("fair_odds", sa.Numeric(8, 3), sa.Computed("1.0/probability", persisted=True)),
        sa.Column("best_odds", sa.Numeric(8, 3)),
        sa.Column("best_bookmaker", sa.Text),
        sa.Column("ev", sa.Numeric(6, 4)),
        sa.Column("kelly_fraction", sa.Numeric(6, 4)),
        sa.Column("features_snapshot", sa.JSON),
        sa.Column("shap_top5", sa.JSON),
        sa.Column("llm_analysis", sa.JSON),
        sa.Column("explanation", sa.JSON),
        sa.Column("decision", sa.Text, nullable=False, server_default="skip"),
        sa.Column("skip_reason", sa.Text),
        sa.Column("analysis_complete", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column(
            "created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint("probability > 0 AND probability < 1", name="ck_predictions_prob"),
        sa.CheckConstraint("decision IN ('bet','skip','flagged')", name="ck_predictions_decision"),
    )
    op.create_index("idx_predictions_match", "predictions", ["match_id"])
    op.create_index(
        "idx_predictions_ev_positive", "predictions", ["ev"], postgresql_where=sa.text("ev > 0.02")
    )
    op.create_index("idx_predictions_model", "predictions", ["model_name", "model_version"])
    op.create_index("idx_predictions_created", "predictions", ["created_at"])

    # ─── Bets (paper + real) ─────────────────────────────────────────────
    op.create_table(
        "bets",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("match_id", sa.BigInteger, sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("prediction_id", sa.BigInteger, sa.ForeignKey("predictions.id")),
        sa.Column("bookmaker", sa.Text, nullable=False),
        sa.Column("market", sa.Text, nullable=False),
        sa.Column("outcome", sa.Text, nullable=False),
        sa.Column("line", sa.Numeric(6, 2)),
        sa.Column("stake_units", sa.Numeric(10, 3), nullable=False),
        sa.Column("odds_placed", sa.Numeric(8, 3), nullable=False),
        sa.Column(
            "placed_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("status", sa.Text, nullable=False, server_default="pending"),
        sa.Column("is_paper", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("pnl_units", sa.Numeric(10, 3)),
        sa.Column("closing_line", sa.Numeric(8, 3)),
        sa.Column("clv", sa.Numeric(6, 4)),
        sa.Column("settled_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("notes", sa.Text),
        sa.CheckConstraint("odds_placed > 1.0", name="ck_bets_odds"),
        sa.CheckConstraint("stake_units > 0", name="ck_bets_stake"),
        sa.CheckConstraint(
            "status IN ('pending','won','lost','void','cashed','halfwon','halflost')",
            name="ck_bets_status",
        ),
    )
    op.create_index("idx_bets_match", "bets", ["match_id"])
    op.create_index(
        "idx_bets_pending",
        "bets",
        ["status", "placed_at"],
        postgresql_where=sa.text("status = 'pending'"),
    )
    op.create_index("idx_bets_paper", "bets", ["is_paper", "placed_at"])

    # ─── Bankroll history (hypertable) ───────────────────────────────────
    op.create_table(
        "bankroll_history",
        sa.Column("ts", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("is_paper", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("bankroll_units", sa.Numeric(14, 4), nullable=False),
        sa.Column("delta_units", sa.Numeric(14, 4)),
        sa.Column("bet_id", sa.BigInteger),
        sa.Column("event", sa.Text),  # deposit|withdraw|bet_settled|adjustment
        sa.Column("notes", sa.Text),
    )
    op.execute(
        "SELECT create_hypertable('bankroll_history', 'ts', chunk_time_interval => INTERVAL '7 days')"
    )
    op.create_index("idx_bankroll_paper_ts", "bankroll_history", ["is_paper", "ts"])

    # ─── News articles (pgvector HNSW) ───────────────────────────────────
    op.create_table(
        "news_articles",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("source", sa.Text, nullable=False),
        sa.Column("url", sa.Text, unique=True),
        sa.Column("title", sa.Text),
        sa.Column("content", sa.Text),
        sa.Column("lang", sa.Text),
        sa.Column("published_at", sa.TIMESTAMP(timezone=True)),
        sa.Column(
            "ingested_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("sports", sa.dialects.postgresql.ARRAY(sa.Text)),
        sa.Column("teams_mentioned", sa.dialects.postgresql.ARRAY(sa.BigInteger)),
        sa.Column("players_mentioned", sa.dialects.postgresql.ARRAY(sa.BigInteger)),
        sa.Column("sentiment_score", sa.Numeric(5, 4)),
        sa.Column("embedding", Vector(1024)),
    )
    op.create_index("idx_news_published", "news_articles", ["published_at"])
    op.create_index("idx_news_source_pub", "news_articles", ["source", "published_at"])
    op.create_index("idx_news_teams", "news_articles", ["teams_mentioned"], postgresql_using="gin")
    op.create_index("idx_news_sports", "news_articles", ["sports"], postgresql_using="gin")
    op.execute(
        "CREATE INDEX idx_news_embedding_hnsw ON news_articles "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )
    # unaccent NO es IMMUTABLE por default → wrapper IMMUTABLE para índice funcional
    op.execute(
        """
        CREATE OR REPLACE FUNCTION immutable_unaccent(text)
        RETURNS text
        LANGUAGE sql IMMUTABLE PARALLEL SAFE STRICT
        AS $$ SELECT public.unaccent('public.unaccent', $1) $$
        """
    )
    op.execute(
        "CREATE INDEX idx_news_fts ON news_articles USING gin("
        "to_tsvector('spanish', immutable_unaccent(coalesce(title,'') || ' ' || coalesce(content,''))))"
    )

    # ─── Ingest checkpoints ──────────────────────────────────────────────
    op.create_table(
        "ingest_checkpoints",
        sa.Column("source", sa.Text, primary_key=True),
        sa.Column("resource", sa.Text, primary_key=True),
        sa.Column("last_ts", sa.TIMESTAMP(timezone=True)),
        sa.Column("last_external_id", sa.Text),
        sa.Column("items_processed", sa.BigInteger, nullable=False, server_default="0"),
        sa.Column(
            "updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )

    # ─── Bot state ───────────────────────────────────────────────────────
    op.create_table(
        "bot_state",
        sa.Column("key", sa.Text, primary_key=True),
        sa.Column("value", sa.JSON, nullable=False),
        sa.Column(
            "updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.execute(
        """
        INSERT INTO bot_state (key, value) VALUES
            ('paused', '{"paused": false, "reason": null, "paused_at": null}'),
            ('kelly_fraction_override', '{"fraction": 0.25, "reason": "default"}'),
            ('market_catalog_version', '"v1"')
        """
    )


def downgrade() -> None:
    # TimescaleDB drop correcto
    op.execute("DROP TABLE IF EXISTS bot_state CASCADE")
    op.execute("DROP TABLE IF EXISTS ingest_checkpoints CASCADE")
    op.execute("DROP TABLE IF EXISTS news_articles CASCADE")
    op.execute("DROP TABLE IF EXISTS bankroll_history CASCADE")
    op.execute("DROP TABLE IF EXISTS bets CASCADE")
    op.execute("DROP TABLE IF EXISTS predictions CASCADE")
    op.execute("DROP TABLE IF EXISTS odds_history CASCADE")
    op.execute("DROP TABLE IF EXISTS matches CASCADE")
    op.execute("DROP TABLE IF EXISTS players CASCADE")
    op.execute("DROP TABLE IF EXISTS teams CASCADE")
    op.execute("DROP TABLE IF EXISTS leagues CASCADE")
    op.execute("DROP TABLE IF EXISTS sports CASCADE")
