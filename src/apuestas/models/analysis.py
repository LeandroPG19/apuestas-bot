"""Modelos ORM para análisis 360° (§16) + post-mortems (§21)."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    TIMESTAMP,
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    ForeignKey,
    Integer,
    Numeric,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from apuestas.db import Base

# ═══════════════════════ §16 capa 1-2: noticias ═══════════════════════════


class NewsArticle(Base):
    """Noticias generales (capa 1). Embeddings BGE-M3 en pgvector HNSW."""

    __tablename__ = "news_articles"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, init=False)
    source: Mapped[str] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(Text, unique=True, default=None)
    title: Mapped[str | None] = mapped_column(Text, default=None)
    content: Mapped[str | None] = mapped_column(Text, default=None)
    lang: Mapped[str | None] = mapped_column(Text, default=None)
    published_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), default=None)
    ingested_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), init=False
    )
    sports: Mapped[list[str] | None] = mapped_column(ARRAY(Text), default=None)
    teams_mentioned: Mapped[list[int] | None] = mapped_column(ARRAY(BigInteger), default=None)
    players_mentioned: Mapped[list[int] | None] = mapped_column(ARRAY(BigInteger), default=None)
    sentiment_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), default=None)
    # `embedding vector(1024)` se gestiona vía raw SQL — no hay type SQLAlchemy


class PlayerNews(Base):
    """Noticias por jugador (capa 2)."""

    __tablename__ = "player_news"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, init=False)
    player_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("players.id"))
    player_name: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(Text, default=None)
    title: Mapped[str | None] = mapped_column(Text, default=None)
    content: Mapped[str | None] = mapped_column(Text, default=None)
    published_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), default=None)
    ingested_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), init=False
    )
    sentiment_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), default=None)
    impact_rating: Mapped[str | None] = mapped_column(Text, default=None)


# ═══════════════════════ §16 capa 3-4: injuries + lineups ═════════════════


class Injury(Base):
    __tablename__ = "injuries"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, init=False)
    player_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("players.id"))
    status: Mapped[str] = mapped_column(Text)  # out|doubtful|questionable|probable|active
    reported_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True))
    sport_code: Mapped[str | None] = mapped_column(Text, ForeignKey("sports.code"), default=None)
    body_part: Mapped[str | None] = mapped_column(Text, default=None)
    expected_return: Mapped[date | None] = mapped_column(Date, default=None)
    games_missed: Mapped[int] = mapped_column(Integer, default=0, init=False)
    source: Mapped[str | None] = mapped_column(Text, default=None)
    last_verified_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), default=None
    )
    verification_score: Mapped[Decimal | None] = mapped_column(Numeric(3, 2), default=None)
    meta: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB, default=None)

    __table_args__ = (
        CheckConstraint(
            "status IN ('out','doubtful','questionable','probable','active')",
            name="ck_injuries_status",
        ),
    )


class Lineup(Base):
    __tablename__ = "lineups"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, init=False)
    match_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("matches.id"))
    team_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("teams.id"))
    starter_ids: Mapped[list[int] | None] = mapped_column(ARRAY(BigInteger), default=None)
    bench_ids: Mapped[list[int] | None] = mapped_column(ARRAY(BigInteger), default=None)
    formation: Mapped[str | None] = mapped_column(Text, default=None)
    confirmed: Mapped[bool] = mapped_column(Boolean, default=False, init=False)
    source: Mapped[str | None] = mapped_column(Text, default=None)
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), init=False
    )


# ═══════════════════════ §16 capa 5: streaks ═════════════════════════════


class TeamStreak(Base):
    __tablename__ = "team_streaks"

    team_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("teams.id"), primary_key=True)
    sport_code: Mapped[str] = mapped_column(Text, ForeignKey("sports.code"), primary_key=True)
    metric: Mapped[str] = mapped_column(Text, primary_key=True)
    current_length: Mapped[int] = mapped_column(Integer)
    direction: Mapped[str] = mapped_column(Text)
    last_n_values: Mapped[list[Decimal] | None] = mapped_column(ARRAY(Numeric(10, 3)), default=None)
    computed_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), init=False
    )


class PlayerStreak(Base):
    __tablename__ = "player_streaks"

    player_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("players.id"), primary_key=True)
    metric: Mapped[str] = mapped_column(Text, primary_key=True)
    current_length: Mapped[int] = mapped_column(Integer)
    direction: Mapped[str] = mapped_column(Text)
    last_n_values: Mapped[list[Decimal] | None] = mapped_column(ARRAY(Numeric(10, 3)), default=None)
    computed_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), init=False
    )


# ═══════════════════════ §16 capa 6-7: transfers + coaching ══════════════


class Transfer(Base):
    __tablename__ = "transfers"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, init=False)
    transfer_date: Mapped[date] = mapped_column(Date)
    player_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("players.id"), default=None
    )
    from_team_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("teams.id"), default=None
    )
    to_team_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("teams.id"), default=None)
    fee_usd: Mapped[Decimal | None] = mapped_column(Numeric(14, 2), default=None)
    transfer_type: Mapped[str | None] = mapped_column(Text, default=None)
    source: Mapped[str | None] = mapped_column(Text, default=None)
    impact_rating: Mapped[str | None] = mapped_column(Text, default=None)
    notes: Mapped[str | None] = mapped_column(Text, default=None)


class CoachingChange(Base):
    __tablename__ = "coaching_changes"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, init=False)
    team_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("teams.id"))
    change_date: Mapped[date] = mapped_column(Date)
    old_coach: Mapped[str | None] = mapped_column(Text, default=None)
    new_coach: Mapped[str | None] = mapped_column(Text, default=None)
    reason: Mapped[str | None] = mapped_column(Text, default=None)
    system_change_notes: Mapped[str | None] = mapped_column(Text, default=None)
    source: Mapped[str | None] = mapped_column(Text, default=None)


# ═══════════════════════ §16 capa 8: H2H + rolling splits ════════════════


class H2HHistory(Base):
    __tablename__ = "h2h_history"

    team_a_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("teams.id"), primary_key=True)
    team_b_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("teams.id"), primary_key=True)
    total_meetings: Mapped[int] = mapped_column(Integer, default=0, init=False)
    a_wins: Mapped[int] = mapped_column(Integer, default=0, init=False)
    b_wins: Mapped[int] = mapped_column(Integer, default=0, init=False)
    draws: Mapped[int] = mapped_column(Integer, default=0, init=False)
    a_cover_ats: Mapped[int] = mapped_column(Integer, default=0, init=False)
    over_hits: Mapped[int] = mapped_column(Integer, default=0, init=False)
    last_10_summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    venue_specific: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    last_computed: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), default=None)

    __table_args__ = (CheckConstraint("team_a_id < team_b_id", name="ck_h2h_order"),)


class TeamStatsRollingHome(Base):
    __tablename__ = "team_stats_rolling_home"

    team_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("teams.id"), primary_key=True)
    sport_code: Mapped[str] = mapped_column(Text, ForeignKey("sports.code"), primary_key=True)
    window_size: Mapped[int] = mapped_column(Integer, primary_key=True)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB)
    sample_size: Mapped[int | None] = mapped_column(Integer, default=None)
    last_computed: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), default=None)


class TeamStatsRollingAway(Base):
    __tablename__ = "team_stats_rolling_away"

    team_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("teams.id"), primary_key=True)
    sport_code: Mapped[str] = mapped_column(Text, ForeignKey("sports.code"), primary_key=True)
    window_size: Mapped[int] = mapped_column(Integer, primary_key=True)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB)
    sample_size: Mapped[int | None] = mapped_column(Integer, default=None)
    last_computed: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), default=None)


class PlayerStatsRolling(Base):
    __tablename__ = "player_stats_rolling"

    player_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("players.id"), primary_key=True)
    window_size: Mapped[int] = mapped_column(Integer, primary_key=True)
    split: Mapped[str] = mapped_column(Text, primary_key=True)  # all|home|away|vs_team_X
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB)
    sample_size: Mapped[int | None] = mapped_column(Integer, default=None)
    last_computed: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), default=None)


# ═══════════════════════ §16 capa 9: venue + travel + weather + officials ═


class VenueFactor(Base):
    __tablename__ = "venue_factors"

    venue_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("venues.id"), primary_key=True)
    sport_code: Mapped[str] = mapped_column(Text, ForeignKey("sports.code"), primary_key=True)
    home_win_rate: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), default=None)
    home_ats_rate: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), default=None)
    over_rate: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), default=None)
    avg_total: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), default=None)
    weather_impact_score: Mapped[Decimal | None] = mapped_column(Numeric(4, 2), default=None)
    sample_games: Mapped[int | None] = mapped_column(Integer, default=None)
    last_computed: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), default=None)


class TravelLog(Base):
    __tablename__ = "travel_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, init=False)
    match_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("matches.id"))
    team_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("teams.id"))
    is_home: Mapped[bool] = mapped_column(Boolean)
    previous_match_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("matches.id"), default=None
    )
    rest_days: Mapped[int | None] = mapped_column(Integer, default=None)
    back_to_back: Mapped[bool | None] = mapped_column(Boolean, default=None)
    distance_km: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), default=None)
    timezone_delta_hours: Mapped[int | None] = mapped_column(Integer, default=None)
    altitude_delta_m: Mapped[int | None] = mapped_column(Integer, default=None)
    total_games_last_7_days: Mapped[int | None] = mapped_column(Integer, default=None)
    total_games_last_14_days: Mapped[int | None] = mapped_column(Integer, default=None)
    computed_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), init=False
    )


class WeatherForecast(Base):
    __tablename__ = "weather_forecast"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, init=False)
    match_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("matches.id"))
    forecast_ts: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True))
    captured_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), init=False
    )
    temp_c: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), default=None)
    wind_kph: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), default=None)
    wind_direction_deg: Mapped[int | None] = mapped_column(Integer, default=None)
    precip_mm: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), default=None)
    humidity_pct: Mapped[int | None] = mapped_column(Integer, default=None)
    conditions: Mapped[str | None] = mapped_column(Text, default=None)
    source: Mapped[str | None] = mapped_column(Text, default=None)


class Official(Base):
    __tablename__ = "officials"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, init=False)
    name: Mapped[str] = mapped_column(Text)
    external_id: Mapped[str | None] = mapped_column(Text, unique=True, default=None)
    sport_code: Mapped[str | None] = mapped_column(Text, ForeignKey("sports.code"), default=None)
    role: Mapped[str | None] = mapped_column(Text, default=None)
    stats: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    last_computed: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), default=None)


class MatchOfficial(Base):
    __tablename__ = "match_officials"

    match_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("matches.id"), primary_key=True)
    official_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("officials.id"), primary_key=True
    )
    role: Mapped[str] = mapped_column(Text, primary_key=True)


# ═══════════════════════ §21 post-mortems + calibration ══════════════════


class PostMortem(Base):
    __tablename__ = "post_mortems"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, init=False)
    bet_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("bets.id"), unique=True)
    event_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("matches.id"))
    prediction_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB)
    features_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB)
    shap_top5: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    llm_analysis_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB)
    ev_predicted: Mapped[Decimal] = mapped_column(Numeric(6, 4))
    kelly_predicted: Mapped[Decimal] = mapped_column(Numeric(6, 4))
    outcome: Mapped[str] = mapped_column(Text)
    pnl_units: Mapped[Decimal] = mapped_column(Numeric(10, 3))
    narrative: Mapped[dict[str, Any]] = mapped_column(JSONB)
    actual_final_score: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    actual_lineups: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    actual_key_events: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB, default=None)
    clv: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), default=None)
    prediction_error: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), default=None)
    calibration_miss: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), default=None)
    ev_realized: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), default=None)
    ev_realized_vs_predicted: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), default=None)
    llm_alignment_score: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), default=None)
    shap_attribution_check: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), default=None)
    line_movement_assessment_correct: Mapped[bool | None] = mapped_column(Boolean, default=None)
    discrepancy_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 3), default=None)
    post_mortem_generated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), init=False
    )
    review_status: Mapped[str] = mapped_column(Text, default="auto")
    human_notes: Mapped[str | None] = mapped_column(Text, default=None)
    pattern_tags: Mapped[list[str] | None] = mapped_column(ARRAY(Text), default=None)


class CalibrationRolling(Base):
    __tablename__ = "calibration_rolling"

    sport_code: Mapped[str] = mapped_column(Text, ForeignKey("sports.code"), primary_key=True)
    market: Mapped[str] = mapped_column(Text, primary_key=True)
    confidence_bucket: Mapped[str] = mapped_column(Text, primary_key=True)
    window_days: Mapped[int] = mapped_column(Integer, primary_key=True)
    n_predictions: Mapped[int] = mapped_column(Integer)
    mean_predicted: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), default=None)
    mean_actual: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), default=None)
    calibration_gap: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), default=None)
    brier_realized: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), default=None)
    ece_realized: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), default=None)
    last_computed: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), init=False
    )


class BotState(Base):
    __tablename__ = "bot_state"

    key: Mapped[str] = mapped_column(Text, primary_key=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSONB)
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), init=False
    )


class PatternBlacklist(Base):
    __tablename__ = "pattern_blacklist"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, init=False)
    tag: Mapped[str] = mapped_column(Text, unique=True)
    sport_code: Mapped[str | None] = mapped_column(Text, ForeignKey("sports.code"), default=None)
    reason: Mapped[str | None] = mapped_column(Text, default=None)
    active: Mapped[bool] = mapped_column(Boolean, default=True, init=False)
    added_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), init=False
    )
    expires_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), default=None)
    confidence_penalty: Mapped[Decimal] = mapped_column(Numeric(4, 3), default=Decimal("0.1"))
