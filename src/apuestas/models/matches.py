"""Modelos ORM para matches + odds + predictions + bets + bankroll."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    TIMESTAMP,
    BigInteger,
    Boolean,
    CheckConstraint,
    ForeignKey,
    Integer,
    Numeric,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from apuestas.db import Base


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, init=False)
    sport_code: Mapped[str] = mapped_column(Text, ForeignKey("sports.code"))
    home_team_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("teams.id"))
    away_team_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("teams.id"))
    start_time: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    external_id: Mapped[str | None] = mapped_column(Text, unique=True, default=None)
    league_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("leagues.id"), default=None
    )
    season: Mapped[str | None] = mapped_column(Text, default=None)
    stage: Mapped[str | None] = mapped_column(Text, default=None)
    venue_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("venues.id"), default=None)
    status: Mapped[str] = mapped_column(Text, default="scheduled")
    home_score: Mapped[int | None] = mapped_column(Integer, default=None)
    away_score: Mapped[int | None] = mapped_column(Integer, default=None)
    meta: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB, default=None)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), init=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), init=False
    )

    __table_args__ = (
        CheckConstraint("home_team_id <> away_team_id", name="ck_matches_different_teams"),
    )


class OddsHistory(Base):
    """Hypertable TimescaleDB. No tiene PK single-column por diseño."""

    __tablename__ = "odds_history"

    ts: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), primary_key=True)
    match_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    bookmaker: Mapped[str] = mapped_column(Text, primary_key=True)
    market: Mapped[str] = mapped_column(Text, primary_key=True)
    outcome: Mapped[str] = mapped_column(Text, primary_key=True)
    odds: Mapped[Decimal] = mapped_column(Numeric(8, 3))
    line: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), default=None)
    is_closing: Mapped[bool] = mapped_column(Boolean, default=False, init=False)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, init=False)
    match_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("matches.id"))
    model_name: Mapped[str] = mapped_column(Text)
    model_version: Mapped[str] = mapped_column(Text)
    market: Mapped[str] = mapped_column(Text)
    outcome: Mapped[str] = mapped_column(Text)
    probability: Mapped[Decimal] = mapped_column(Numeric(6, 5))
    line: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), default=None)
    p_lower: Mapped[Decimal | None] = mapped_column(Numeric(6, 5), default=None)
    p_upper: Mapped[Decimal | None] = mapped_column(Numeric(6, 5), default=None)
    best_odds: Mapped[Decimal | None] = mapped_column(Numeric(8, 3), default=None)
    best_bookmaker: Mapped[str | None] = mapped_column(Text, default=None)
    ev: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), default=None)
    kelly_fraction: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), default=None)
    features_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    shap_top5: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB, default=None)
    llm_analysis: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    explanation: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    decision: Mapped[str] = mapped_column(Text, default="skip")
    skip_reason: Mapped[str | None] = mapped_column(Text, default=None)
    analysis_complete: Mapped[bool] = mapped_column(Boolean, default=True, init=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), init=False
    )


class Bet(Base):
    __tablename__ = "bets"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, init=False)
    match_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("matches.id"))
    bookmaker: Mapped[str] = mapped_column(Text)
    market: Mapped[str] = mapped_column(Text)
    outcome: Mapped[str] = mapped_column(Text)
    stake_units: Mapped[Decimal] = mapped_column(Numeric(10, 3))
    odds_placed: Mapped[Decimal] = mapped_column(Numeric(8, 3))
    prediction_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("predictions.id"), default=None
    )
    line: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), default=None)
    placed_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), init=False
    )
    status: Mapped[str] = mapped_column(Text, default="pending")
    is_paper: Mapped[bool] = mapped_column(Boolean, default=True, init=False)
    pnl_units: Mapped[Decimal | None] = mapped_column(Numeric(10, 3), default=None)
    closing_line: Mapped[Decimal | None] = mapped_column(Numeric(8, 3), default=None)
    clv: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), default=None)
    settled_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), default=None)
    notes: Mapped[str | None] = mapped_column(Text, default=None)


class IngestCheckpoint(Base):
    """Última ingesta por (source, resource) para reanudar tras reinicio."""

    __tablename__ = "ingest_checkpoints"

    source: Mapped[str] = mapped_column(Text, primary_key=True)
    resource: Mapped[str] = mapped_column(Text, primary_key=True)
    last_ts: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), default=None)
    last_external_id: Mapped[str | None] = mapped_column(Text, default=None)
    items_processed: Mapped[int] = mapped_column(BigInteger, default=0, init=False)
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), init=False
    )
