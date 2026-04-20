"""Ingesta NBA vía swar/nba_api.

- Schedule + boxscores + advanced stats + shot charts + injuries.
- stats.nba.com aplica rate limit agresivo; nba_api lo maneja con ~1 req/2s.
- Las llamadas nba_api son SÍNCRONAS; envolvemos con asyncio.to_thread().
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import polars as pl

from apuestas.obs.logging import get_logger
from apuestas.validators.schemas import validate_fixtures

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


async def _run_sync(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Wrapper para ejecutar calls nba_api síncronas en pool de threads."""
    return await asyncio.to_thread(fn, *args, **kwargs)


async def fetch_season_games(season: str = "2024-25") -> list[dict[str, Any]]:
    """Descarga LeagueGameLog con todos los partidos de la temporada.

    Args:
        season: formato 'YYYY-YY' ej '2024-25'.

    Returns:
        Lista de dicts con game_id, game_date, matchup, team_id, etc.
    """
    from nba_api.stats.endpoints import leaguegamelog

    def _fetch() -> list[dict[str, Any]]:
        log = leaguegamelog.LeagueGameLog(season=season, season_type_all_star="Regular Season")
        df = log.get_data_frames()[0]
        return df.to_dict(orient="records")

    games = await _run_sync(_fetch)
    logger.info("nba.season_games", season=season, count=len(games))
    return games


async def fetch_scoreboard(date: datetime) -> list[dict[str, Any]]:
    """Scoreboard del día (schedule + live scores)."""
    from nba_api.live.nba.endpoints import scoreboard

    def _fetch() -> list[dict[str, Any]]:
        sb = scoreboard.ScoreBoard()
        games = sb.games.get_dict()
        return games if isinstance(games, list) else []

    return await _run_sync(_fetch)


async def fetch_boxscore_advanced(game_id: str) -> dict[str, Any]:
    """Boxscore avanzado (ORtg/DRtg/Pace/eFG%/TS% por jugador y equipo)."""
    from nba_api.stats.endpoints import boxscoreadvancedv3

    def _fetch() -> dict[str, Any]:
        bs = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=game_id)
        return bs.get_dict()

    return await _run_sync(_fetch)


async def fetch_team_stats(season: str = "2024-25") -> list[dict[str, Any]]:
    """Stats agregadas por equipo en la temporada."""
    from nba_api.stats.endpoints import leaguedashteamstats

    def _fetch() -> list[dict[str, Any]]:
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced",
        )
        df = stats.get_data_frames()[0]
        return df.to_dict(orient="records")

    return await _run_sync(_fetch)


async def fetch_player_stats(season: str = "2024-25") -> list[dict[str, Any]]:
    from nba_api.stats.endpoints import leaguedashplayerstats

    def _fetch() -> list[dict[str, Any]]:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced",
        )
        df = stats.get_data_frames()[0]
        return df.to_dict(orient="records")

    return await _run_sync(_fetch)


async def fetch_injury_report() -> list[dict[str, Any]]:
    """NBA Official Injury Report (scraping oficial).

    Nota: nba_api no tiene endpoint oficial de injuries; scraping desde
    nba.com/players/reports con playwright o fetch manual del PDF.
    Placeholder para Fase 3-4.10 (scraping).
    """
    logger.warning("nba.injuries.not_implemented", msg="Usa scraping nba.com/injury-report o ESPN")
    return []


# ═══════════════════════ Transformadores a Polars ═══════════════════════


def games_to_fixtures(raw: list[dict[str, Any]]) -> pl.DataFrame:
    """Convierte LeagueGameLog de nba_api al schema de fixtures."""
    if not raw:
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

    # nba_api produce 2 rows por partido (home + away) — deduplicamos por game_id
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for g in raw:
        game_id = str(g.get("GAME_ID", ""))
        if game_id in seen:
            continue
        seen.add(game_id)
        matchup: str = g.get("MATCHUP", "")
        # "LAL vs. BOS" (home) o "LAL @ BOS" (away) — detectar
        if " vs. " in matchup:
            is_home = True
            home_abbr = matchup.split(" vs. ", maxsplit=1)[0].strip()
            away_abbr = matchup.split(" vs. ")[1].strip()
        elif " @ " in matchup:
            is_home = False
            away_abbr = matchup.split(" @ ", maxsplit=1)[0].strip()
            home_abbr = matchup.split(" @ ")[1].strip()
        else:
            continue

        # Este row puede ser del visitante; saltar si no es del local
        if not is_home:
            continue

        rows.append(
            {
                "external_id": game_id,
                "sport_code": "nba",
                "home_team_external_id": home_abbr,
                "away_team_external_id": away_abbr,
                "start_time": g.get("GAME_DATE"),
                "status": "finished" if g.get("WL") in {"W", "L"} else "scheduled",
                "league_external_id": "nba",
                "season": str(g.get("SEASON_ID", "")),
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

    df = pl.DataFrame(rows)
    # GAME_DATE viene como 'YYYY-MM-DD' → parse a datetime UTC a las 00:00
    df = df.with_columns(
        pl.col("start_time")
        .str.to_datetime(format="%Y-%m-%d", time_zone="UTC", strict=False)
        .alias("start_time")
    )
    return df


# ═══════════════════════ Flujos orquestados ═════════════════════════════


async def ingest_nba_season(season: str = "2024-25") -> pl.DataFrame:
    """Descarga temporada completa (~1,230 juegos/regular)."""
    raw = await fetch_season_games(season)
    df = games_to_fixtures(raw)
    if df.height == 0:
        return df
    try:
        return validate_fixtures(df)
    except Exception:
        logger.exception("nba.fixtures.validation_failed", season=season)
        raise


async def ingest_nba_today() -> list[dict[str, Any]]:
    """Partidos del día actual."""
    return await fetch_scoreboard(datetime.now(tz=UTC))
