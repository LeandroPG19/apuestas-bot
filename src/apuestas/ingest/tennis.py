"""Ingesta Tenis ATP/WTA — Jeff Sackmann GitHub CSVs + API-Tennis.

Sackmann: gold standard histórico con match-level ATP/WTA 1968+ (MIT license).
API-Tennis: schedule + live odds (api-sports.io tier Pro ya cubre).

Endpoints:
- ATP matches: https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv
- WTA matches: https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv
- Rankings: atp_rankings_{decade}s.csv
"""

from __future__ import annotations

from datetime import datetime
from io import StringIO
from typing import Any, Literal

import httpx
import polars as pl

from apuestas.ingest.http_base import BaseAPIClient
from apuestas.obs.logging import get_logger
from apuestas.validators.schemas import validate_fixtures

logger = get_logger(__name__)


Tour = Literal["atp", "wta"]


SACKMANN_BASE = {
    "atp": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master",
    "wta": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master",
}


async def fetch_sackmann_matches(tour: Tour, year: int) -> pl.DataFrame:
    """Descarga CSV de matches de una temporada ATP/WTA."""
    url = f"{SACKMANN_BASE[tour]}/{tour}_matches_{year}.csv"
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("tennis.sackmann_fetch_fail", tour=tour, year=year, error=str(exc))
            return pl.DataFrame()

    try:
        df = pl.read_csv(StringIO(resp.text), infer_schema_length=10000)
    except Exception as exc:
        logger.warning("tennis.sackmann_parse_fail", tour=tour, year=year, error=str(exc))
        return pl.DataFrame()

    logger.info("tennis.sackmann_loaded", tour=tour, year=year, rows=df.height)
    return df


def sackmann_to_fixtures(df: pl.DataFrame, tour: Tour) -> pl.DataFrame:
    """Normaliza al schema universal `fixtures`."""
    if df.height == 0:
        return _empty_fixtures_df()
    required = {"winner_id", "loser_id", "tourney_date", "tourney_id"}
    if not required.issubset(df.columns):
        logger.warning("tennis.sackmann_missing_cols", missing=list(required - set(df.columns)))
        return _empty_fixtures_df()

    # tourney_date típicamente int YYYYMMDD
    df = df.with_columns(
        pl.col("tourney_date")
        .cast(pl.Utf8)
        .str.to_datetime(format="%Y%m%d", time_zone="UTC", strict=False)
        .alias("start_time")
    )

    return df.select(
        (
            pl.col("tourney_id").cast(pl.Utf8) + pl.lit("_") + pl.col("match_num").cast(pl.Utf8)
        ).alias("external_id"),
        pl.lit("tennis").alias("sport_code"),
        pl.col("winner_id").cast(pl.Utf8).alias("home_team_external_id"),
        pl.col("loser_id").cast(pl.Utf8).alias("away_team_external_id"),
        pl.col("start_time"),
        pl.lit("finished").alias("status"),
        pl.col("tourney_id").cast(pl.Utf8).alias("league_external_id"),
        pl.col("tourney_date").cast(pl.Utf8).str.slice(0, 4).alias("season"),
    ).filter(pl.col("external_id").is_not_null())


def _empty_fixtures_df() -> pl.DataFrame:
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


async def ingest_tour_season(tour: Tour, year: int) -> pl.DataFrame:
    """Flujo completo: fetch → normalize → validate."""
    raw = await fetch_sackmann_matches(tour, year)
    df = sackmann_to_fixtures(raw, tour)
    if df.height == 0:
        return df
    return validate_fixtures(df)


async def ingest_multi_seasons(
    tours: tuple[Tour, ...] = ("atp", "wta"),
    years: tuple[int, ...] = (2023, 2024, 2025, 2026),
) -> dict[str, int]:
    """Backfill multi-season. Idempotente vía ON CONFLICT en INSERT."""
    stats: dict[str, int] = {}
    for tour in tours:
        for year in years:
            try:
                df = await ingest_tour_season(tour, year)
                stats[f"{tour}_{year}"] = df.height
            except Exception as exc:
                logger.warning("tennis.ingest_fail", tour=tour, year=year, error=str(exc))
                stats[f"{tour}_{year}"] = -1
    return stats


# ═══════════════════════ API-Tennis (live) ═══════════════════════════════


class APITennisClient(BaseAPIClient):
    """api-sports.io Tennis — comparte suscripción con API-Football."""

    base_url = "https://v1.tennis.api-sports.io"
    source_name = "api_tennis"
    rate_limit = (60, 60.0)

    def __init__(self, *, api_key: str | None = None) -> None:
        from apuestas.config import get_settings

        settings = get_settings()
        key = api_key or (
            settings.apis.api_football_key.get_secret_value()
            if settings.apis.api_football_key
            else None
        )
        if not key:
            msg = "API key requerida (API_FOOTBALL_KEY)"
            raise ValueError(msg)
        super().__init__(api_key=key)
        self._key = key

    def _default_headers(self) -> dict[str, str]:
        return {
            "x-apisports-key": self._key,
            "User-Agent": "apuestas-bot/0.1 (+api_tennis)",
        }

    async def fetch_fixtures_today(self) -> list[dict[str, Any]]:
        today = datetime.now(tz=__import__("datetime").UTC).strftime("%Y-%m-%d")
        data = await self.get("/games", params={"date": today})
        return data.get("response", [])

    async def fetch_odds(self, *, fixture_id: int) -> list[dict[str, Any]]:
        data = await self.get("/odds", params={"game": fixture_id})
        return data.get("response", [])
