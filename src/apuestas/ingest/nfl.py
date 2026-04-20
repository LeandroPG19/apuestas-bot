"""Ingesta NFL vía nflreadpy (port Python de nflfastR).

Features clave disponibles:
- PBP (play-by-play) desde 1999 con EPA, WPA, CPOE precomputados
- Schedules + scoreboards
- Team stats agregadas
- FTN charting data (2022+) — pressure rate, routes, coverages
- Snap counts, injury reports, depth charts

nflreadpy es nativamente polars-friendly (usa DuckDB + parquet cacheable).
Algunas funciones son síncronas → to_thread.
"""

from __future__ import annotations

import asyncio
from typing import Any

import polars as pl

from apuestas.obs.logging import get_logger
from apuestas.validators.schemas import validate_fixtures

logger = get_logger(__name__)


async def _run_sync(fn: Any, *args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(fn, *args, **kwargs)


async def fetch_schedules(seasons: list[int]) -> pl.DataFrame:
    """Schedules por temporada(s). Formato nflverse."""
    import nflreadpy as nfl  # type: ignore[import-untyped]

    def _fetch() -> pl.DataFrame:
        df = nfl.load_schedules(seasons=seasons)
        if not isinstance(df, pl.DataFrame):
            df = pl.from_pandas(df) if hasattr(df, "to_pandas") else pl.DataFrame(df)
        return df

    return await _run_sync(_fetch)


async def fetch_pbp(seasons: list[int]) -> pl.DataFrame:
    """Play-by-play con EPA/WPA/CPOE.

    Cuidado: cada temporada ~47k plays ~100-200 MB en memoria.
    """
    import nflreadpy as nfl

    def _fetch() -> pl.DataFrame:
        df = nfl.load_pbp(seasons=seasons)
        if not isinstance(df, pl.DataFrame):
            df = pl.from_pandas(df) if hasattr(df, "to_pandas") else pl.DataFrame(df)
        return df

    return await _run_sync(_fetch)


async def fetch_injuries(seasons: list[int]) -> pl.DataFrame:
    import nflreadpy as nfl

    def _fetch() -> pl.DataFrame:
        df = nfl.load_injuries(seasons=seasons)
        if not isinstance(df, pl.DataFrame):
            df = pl.from_pandas(df) if hasattr(df, "to_pandas") else pl.DataFrame(df)
        return df

    return await _run_sync(_fetch)


async def fetch_rosters(seasons: list[int]) -> pl.DataFrame:
    import nflreadpy as nfl

    def _fetch() -> pl.DataFrame:
        df = nfl.load_rosters(seasons=seasons)
        if not isinstance(df, pl.DataFrame):
            df = pl.from_pandas(df) if hasattr(df, "to_pandas") else pl.DataFrame(df)
        return df

    return await _run_sync(_fetch)


async def fetch_snap_counts(seasons: list[int]) -> pl.DataFrame:
    import nflreadpy as nfl

    def _fetch() -> pl.DataFrame:
        df = nfl.load_snap_counts(seasons=seasons)
        if not isinstance(df, pl.DataFrame):
            df = pl.from_pandas(df) if hasattr(df, "to_pandas") else pl.DataFrame(df)
        return df

    return await _run_sync(_fetch)


def schedules_to_fixtures(df: pl.DataFrame) -> pl.DataFrame:
    """Transforma nflreadpy schedules al schema universal de fixtures."""
    if df.height == 0:
        return pl.DataFrame(
            schema={
                "external_id": pl.Utf8,
                "sport_code": pl.Utf8,
                "home_team_external_id": pl.Utf8,
                "away_team_external_id": pl.Utf8,
                "start_time": pl.Datetime(time_zone="UTC"),
                "status": pl.Utf8,
                "league_external_id": pl.Utf8,
                "season": pl.Utf8,
            }
        )

    # nflreadpy schedule cols: game_id, season, game_type, week, gameday, gametime,
    # away_team, home_team, result, total, away_score, home_score, ...
    return df.select(
        pl.col("game_id").cast(pl.Utf8).alias("external_id"),
        pl.lit("nfl").alias("sport_code"),
        pl.col("home_team").cast(pl.Utf8).alias("home_team_external_id"),
        pl.col("away_team").cast(pl.Utf8).alias("away_team_external_id"),
        (
            pl.col("gameday").cast(pl.Utf8)
            + pl.lit(" ")
            + pl.col("gametime").cast(pl.Utf8).fill_null("00:00")
        )
        .str.to_datetime(format="%Y-%m-%d %H:%M", time_zone="UTC", strict=False)
        .alias("start_time"),
        pl.when(pl.col("result").is_not_null())
        .then(pl.lit("finished"))
        .otherwise(pl.lit("scheduled"))
        .alias("status"),
        pl.lit("nfl").alias("league_external_id"),
        pl.col("season").cast(pl.Utf8).alias("season"),
    ).filter(pl.col("external_id").is_not_null())


async def ingest_nfl_seasons(seasons: list[int]) -> pl.DataFrame:
    raw_df = await fetch_schedules(seasons)
    df = schedules_to_fixtures(raw_df)
    if df.height == 0:
        return df
    validated = validate_fixtures(df)
    logger.info("nfl.seasons_ingested", seasons=seasons, rows=validated.height)
    return validated


def compute_epa_team_rolling(pbp: pl.DataFrame, *, window_games: int = 5) -> pl.DataFrame:
    """Feature engineering: EPA ofensivo/defensivo rolling por equipo.

    Base para modelos NFL. Retorna DF con cols:
    [team, season, week, off_epa_per_play, def_epa_per_play, pass_epa, rush_epa,
     off_success_rate, def_success_rate, cpoe_avg, ...]
    """
    if pbp.height == 0:
        return pl.DataFrame()

    # Filtrar plays válidos
    plays = pbp.filter(
        pl.col("play_type").is_in(["pass", "run"])
        & pl.col("epa").is_not_null()
        & pl.col("posteam").is_not_null()
    )

    team_week = plays.group_by(["posteam", "season", "week"]).agg(
        pl.col("epa").mean().alias("off_epa_per_play"),
        (pl.col("epa") > 0).mean().alias("off_success_rate"),
        pl.col("epa").filter(pl.col("play_type") == "pass").mean().alias("off_pass_epa"),
        pl.col("epa").filter(pl.col("play_type") == "run").mean().alias("off_rush_epa"),
        pl.col("cpoe").mean().alias("cpoe_avg"),
        pl.len().alias("plays"),
    )

    return team_week.sort(["posteam", "season", "week"])
