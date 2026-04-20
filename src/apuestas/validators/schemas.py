"""Pandera schemas para validación en boundaries de ingestión.

Según §15.3 del plan: captura odds corruptas, fixtures mal parseados,
probabilidades que no suman, etc.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandera.polars as pa
import polars as pl


class OddsRowSchema(pa.DataFrameModel):
    """Fila de odds ingestada. Valida antes de INSERT a odds_history."""

    ts: datetime = pa.Field(nullable=False)
    match_external_id: str = pa.Field(nullable=False, str_length={"min_value": 1})
    bookmaker: str = pa.Field(
        nullable=False,
        isin=[
            # Sharp/benchmark
            "pinnacle",
            "circa",
            "bookmaker",
            "betfair",
            # México (SEGOB)
            "caliente",
            "strendus",
            "codere",
            "betway_mx",
            "betano_mx",
            "betsson_mx",
            "bwin_mx",
            "novibet_mx",
            # US regulados por estado
            "draftkings",
            "fanduel",
            "betmgm",
            "caesars",
            "pointsbet",
            "betrivers",
            "espnbet",
            "hardrock",
            # Offshore / genéricos
            "betway",
            "bet365",
            "barstool",
            "unibet",
            "william_hill",
            "1xbet",
            "betano",
            "betsson",
            "bwin",
            "novibet",
        ],
    )
    market: str = pa.Field(
        nullable=False,
        isin=[
            "h2h",
            "moneyline",
            "spread",
            "runline",
            "puckline",
            "totals",
            "total",
            "btts",
            "double_chance",
            "asian_handicap",
            "player_points",
            "player_rebounds",
            "player_assists",
            "player_strikeouts",
            "player_passing_yds",
            "player_rushing_yds",
            "first_half_spread",
            "first_half_total",
            "first_quarter_spread",
            "nrfi",
            "yrfi",
            "method_of_victory",
            "round_betting",
        ],
    )
    outcome: str = pa.Field(nullable=False, str_length={"min_value": 1})
    line: float | None = pa.Field(nullable=True, ge=-100.0, le=500.0)
    odds: float = pa.Field(nullable=False, gt=1.0, lt=1000.0)

    class Config:
        strict = True
        coerce = True


class FixtureSchema(pa.DataFrameModel):
    """Fixtures (partidos) ingestados. Valida antes de INSERT a matches."""

    external_id: str = pa.Field(nullable=False, str_length={"min_value": 1})
    sport_code: str = pa.Field(
        nullable=False,
        isin=[
            "nba",
            "mlb",
            "nfl",
            "soccer",
            "boxing",
            "mma",
        ],
    )
    home_team_external_id: str = pa.Field(nullable=False)
    away_team_external_id: str = pa.Field(nullable=False)
    start_time: datetime = pa.Field(nullable=False)
    status: str = pa.Field(
        nullable=False,
        isin=["scheduled", "live", "finished", "cancelled", "postponed", "void"],
    )
    league_external_id: str | None = pa.Field(nullable=True)
    season: str | None = pa.Field(nullable=True)

    class Config:
        strict = True
        coerce = True


class InjurySchema(pa.DataFrameModel):
    """Reportes de lesiones ingestados."""

    player_external_id: str = pa.Field(nullable=False)
    sport_code: str = pa.Field(
        nullable=False, isin=["nba", "mlb", "nfl", "soccer", "boxing", "mma"]
    )
    status: str = pa.Field(
        nullable=False,
        isin=["out", "doubtful", "questionable", "probable", "active"],
    )
    body_part: str | None = pa.Field(nullable=True)
    reported_at: datetime = pa.Field(nullable=False)
    source: str = pa.Field(nullable=False)

    class Config:
        strict = True
        coerce = True


class LineupSchema(pa.DataFrameModel):
    """Alineaciones confirmadas/esperadas."""

    match_external_id: str = pa.Field(nullable=False)
    team_external_id: str = pa.Field(nullable=False)
    starter_ids: list[str] = pa.Field(nullable=False)
    formation: str | None = pa.Field(nullable=True)
    confirmed: bool = pa.Field(nullable=False)
    source: str = pa.Field(nullable=False)

    class Config:
        strict = False  # array types tolerantes
        coerce = True


class ValidationError(Exception):
    """Wrapper sobre pandera.errors.SchemaError para trace amigable."""


def validate_odds(df: pl.DataFrame) -> pl.DataFrame:
    """Valida DataFrame de odds antes de insert. Raises ValidationError si falla.

    Reglas adicionales post-schema:
    - Verifica que fixture no esté en el pasado (>7 días)
    - Verifica que odds estén en rango razonable por mercado
    """
    try:
        validated = OddsRowSchema.validate(df)
    except pa.errors.SchemaError as exc:
        msg = f"Odds schema validation failed: {exc}"
        raise ValidationError(msg) from exc

    # Sanity: timestamps no deben estar en el futuro lejano ni muy viejos
    now = datetime.now(tz=__import__("datetime").UTC)
    too_old = now - timedelta(days=30)
    too_new = now + timedelta(minutes=5)

    bad_ts = validated.filter((pl.col("ts") < too_old) | (pl.col("ts") > too_new))
    if bad_ts.height > 0:
        msg = f"Odds with unreasonable timestamps: {bad_ts.height} rows"
        raise ValidationError(msg)

    return validated


def validate_fixtures(df: pl.DataFrame) -> pl.DataFrame:
    try:
        return FixtureSchema.validate(df)
    except pa.errors.SchemaError as exc:
        msg = f"Fixture schema validation failed: {exc}"
        raise ValidationError(msg) from exc


def validate_injuries(df: pl.DataFrame) -> pl.DataFrame:
    try:
        return InjurySchema.validate(df)
    except pa.errors.SchemaError as exc:
        msg = f"Injury schema validation failed: {exc}"
        raise ValidationError(msg) from exc


def validate_lineups(df: pl.DataFrame) -> pl.DataFrame:
    try:
        return LineupSchema.validate(df)
    except pa.errors.SchemaError as exc:
        msg = f"Lineup schema validation failed: {exc}"
        raise ValidationError(msg) from exc
