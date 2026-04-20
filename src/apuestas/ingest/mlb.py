"""Ingesta MLB vía pybaseball (Statcast pitch-level 2008+).

Funcionalidad clave:
- Schedule y boxscores (MLB Stats API directo vía pybaseball)
- Statcast pitch-level por día (~100-200k pitches/día temporada activa)
- xwOBA, barrels, exit velocity, launch angle, spin rate
- Park factors (rolling 3-5 años)
- Pitcher matchups + handedness

pybaseball es SÍNCRONO → envolver con asyncio.to_thread.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from typing import Any

import polars as pl

from apuestas.obs.logging import get_logger
from apuestas.validators.schemas import validate_fixtures

logger = get_logger(__name__)


async def _run_sync(fn: Any, *args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(fn, *args, **kwargs)


async def fetch_schedule(year: int) -> list[dict[str, Any]]:
    """Schedule completo de temporada vía MLB Stats API (pybaseball helper)."""

    def _fetch() -> list[dict[str, Any]]:
        # schedule_and_record es per-team; hay que iterar
        # Para MVP usamos MLB Stats API directo via statsapi
        import statsapi  # type: ignore[import-untyped]

        games = statsapi.schedule(start_date=f"{year}-03-01", end_date=f"{year}-11-15")
        return games  # type: ignore[no-any-return]

    return await _run_sync(_fetch)


async def fetch_statcast_daily(target_date: date) -> Any:
    """Statcast pitch-level del día. Tarda ~10-30s por día."""
    from pybaseball import statcast

    def _fetch() -> Any:
        df = statcast(start_dt=target_date.isoformat(), end_dt=target_date.isoformat())
        return df

    return await _run_sync(_fetch)


async def fetch_team_batting(year: int) -> Any:
    """Batting stats agregado por equipo."""
    from pybaseball import team_batting

    def _fetch() -> Any:
        return team_batting(year)

    return await _run_sync(_fetch)


async def fetch_team_pitching(year: int) -> Any:
    from pybaseball import team_pitching

    def _fetch() -> Any:
        return team_pitching(year)

    return await _run_sync(_fetch)


async def fetch_pitcher_stats(year: int, stat_type: str = "pitching") -> Any:
    """FanGraphs pitching leaderboard (FIP, xFIP, SIERA, K%, BB%)."""
    from pybaseball import pitching_stats

    def _fetch() -> Any:
        return pitching_stats(year)

    return await _run_sync(_fetch)


async def fetch_batter_stats(year: int) -> Any:
    """FanGraphs batting leaderboard (xwOBA, ISO, hard-hit%, barrels)."""
    from pybaseball import batting_stats

    def _fetch() -> Any:
        return batting_stats(year)

    return await _run_sync(_fetch)


def schedule_to_fixtures(raw: list[dict[str, Any]]) -> pl.DataFrame:
    """Transforma statsapi.schedule output al schema de fixtures."""
    status_map = {
        "Scheduled": "scheduled",
        "Pre-Game": "scheduled",
        "Warmup": "scheduled",
        "In Progress": "live",
        "Final": "finished",
        "Game Over": "finished",
        "Postponed": "postponed",
        "Cancelled": "cancelled",
    }
    rows: list[dict[str, Any]] = []
    for g in raw:
        game_id = str(g.get("game_id", ""))
        if not game_id:
            continue
        rows.append(
            {
                "external_id": game_id,
                "sport_code": "mlb",
                "home_team_external_id": str(g.get("home_id", "")),
                "away_team_external_id": str(g.get("away_id", "")),
                "start_time": g.get("game_datetime"),
                "status": status_map.get(g.get("status", ""), "scheduled"),
                "league_external_id": "mlb",
                "season": str(g.get("season", "")),
            }
        )
    if not rows:
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
    return pl.DataFrame(rows).with_columns(
        pl.col("start_time").str.to_datetime(time_zone="UTC", strict=False).alias("start_time")
    )


async def ingest_mlb_season(year: int) -> pl.DataFrame:
    raw = await fetch_schedule(year)
    df = schedule_to_fixtures(raw)
    if df.height == 0:
        return df
    validated = validate_fixtures(df)
    logger.info("mlb.season_ingested", year=year, rows=validated.height)
    return validated


async def ingest_mlb_today() -> list[dict[str, Any]]:
    today = datetime.now(tz=UTC).date()
    import statsapi  # type: ignore[import-untyped]

    def _fetch() -> list[dict[str, Any]]:
        return statsapi.schedule(date=today.isoformat())  # type: ignore[no-any-return]

    return await _run_sync(_fetch)
