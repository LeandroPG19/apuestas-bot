"""Seed histórico para entrenamiento ML (Gap #8).

Pobla `teams`, `matches` y (cuando aplica) `team_stats_rolling_*` con N temporadas
de NBA, MLB y fútbol (Liga MX + EPL). Idempotente: usa `ON CONFLICT DO NOTHING`
sobre `external_id`. Llamable con:

    python -m apuestas.scripts.seed_historical --sport nba --seasons 2023,2024,2025
    python -m apuestas.scripts.seed_historical --sport mlb --seasons 2024,2025
    python -m apuestas.scripts.seed_historical --sport soccer --league liga_mx --seasons 2024,2025

Dependencias (ya pineadas en pyproject): nba_api, pybaseball, soccerdata.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.ingest._match_resolver import resolve_or_create_match
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def _seed_nba(seasons: list[int]) -> int:
    """NBA via nba_api.LeagueGameLog. 1 game por fila, Home/Away resolvables."""
    try:
        from nba_api.stats.endpoints import leaguegamelog  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("seed.nba_api_missing")
        return 0

    total = 0
    for season in seasons:
        season_str = f"{season}-{str(season + 1)[-2:]}"
        try:
            df = await asyncio.to_thread(
                lambda s=season_str: leaguegamelog.LeagueGameLog(
                    season=s, season_type_all_star="Regular Season"
                ).get_data_frames()[0]
            )
        except Exception as exc:
            logger.warning("seed.nba_fetch_fail", season=season, error=str(exc)[:120])
            continue

        # nba_api devuelve 2 rows por juego (home + away perspective). Agrupar por GAME_ID.
        grouped = df.groupby("GAME_ID")
        async with session_scope() as session:
            for game_id, g in grouped:
                if len(g) < 2:
                    continue
                home_row = g[g["MATCHUP"].str.contains(" vs. ", regex=False)]
                away_row = g[g["MATCHUP"].str.contains(" @ ", regex=False)]
                if home_row.empty or away_row.empty:
                    continue
                home_name = home_row.iloc[0]["TEAM_NAME"]
                away_name = away_row.iloc[0]["TEAM_NAME"]
                game_date_raw = home_row.iloc[0]["GAME_DATE"]
                try:
                    start = datetime.fromisoformat(str(game_date_raw)).replace(tzinfo=UTC)
                except ValueError:
                    continue

                match_id = await resolve_or_create_match(
                    session,
                    sport_code="nba",
                    home_name=home_name,
                    away_name=away_name,
                    start_time=start,
                    source="nba_api",
                )
                if match_id is None:
                    continue

                home_pts = int(home_row.iloc[0].get("PTS", 0) or 0)
                away_pts = int(away_row.iloc[0].get("PTS", 0) or 0)
                await session.execute(
                    text(
                        """
                        UPDATE matches
                           SET status = 'finished',
                               home_score = :hs,
                               away_score = :as_,
                               external_id = COALESCE(external_id, :ext),
                               season = :season
                         WHERE id = :id AND status != 'finished'
                        """
                    ),
                    {
                        "id": match_id,
                        "hs": home_pts,
                        "as_": away_pts,
                        "ext": f"nba_api:{game_id}",
                        "season": season_str,
                    },
                )
                total += 1
        logger.info("seed.nba_done", season=season_str, games=total)

    return total


async def _seed_mlb(seasons: list[int]) -> int:
    """MLB via pybaseball.schedule_and_record (requires team iteration)."""
    try:
        import pybaseball  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("seed.pybaseball_missing")
        return 0

    # MLB teams estables (30). pybaseball.standings() devuelve [] en 2024+.
    MLB_TEAMS = (
        "ARI",
        "ATL",
        "BAL",
        "BOS",
        "CHC",
        "CHW",
        "CIN",
        "CLE",
        "COL",
        "DET",
        "HOU",
        "KCR",
        "LAA",
        "LAD",
        "MIA",
        "MIL",
        "MIN",
        "NYM",
        "NYY",
        "OAK",
        "PHI",
        "PIT",
        "SDP",
        "SEA",
        "SFG",
        "STL",
        "TBR",
        "TEX",
        "TOR",
        "WSN",
    )
    total = 0
    for season in seasons:
        for team_abbr in MLB_TEAMS:
            try:
                sched = await asyncio.to_thread(pybaseball.schedule_and_record, season, team_abbr)
            except Exception:
                continue
            async with session_scope() as session:
                for _, row in sched.iterrows():
                    opp = row.get("Opp")
                    date_raw = row.get("Date")
                    if not opp or not date_raw:
                        continue
                    # pybaseball Home_Away: "@" = away game, "" = home game
                    home_away = str(row.get("Home_Away", "") or "").strip()
                    is_home = home_away != "@"
                    home_name = team_abbr if is_home else str(opp)
                    away_name = str(opp) if is_home else team_abbr
                    # Skip cuando team es visitante — ya lo sembraremos desde POV del home
                    if not is_home:
                        continue
                    try:
                        start = datetime.strptime(f"{date_raw}, {season}", "%A, %b %d, %Y").replace(
                            tzinfo=UTC, hour=23
                        )
                    except (ValueError, TypeError):
                        continue

                    match_id = await resolve_or_create_match(
                        session,
                        sport_code="mlb",
                        home_name=home_name,
                        away_name=away_name,
                        start_time=start,
                        source="pybaseball",
                    )
                    if match_id is None:
                        continue
                    # Runs scored/against = home_score/away_score
                    try:
                        r_val = row.get("R")
                        ra_val = row.get("RA")
                        if r_val is not None and ra_val is not None:
                            hs = int(float(r_val))
                            as_ = int(float(ra_val))
                            await session.execute(
                                text(
                                    "UPDATE matches SET home_score=:hs, away_score=:as_, "
                                    "status='finished' WHERE id=:id AND status != 'finished'"
                                ),
                                {"id": match_id, "hs": hs, "as_": as_},
                            )
                    except (ValueError, TypeError):
                        pass
                    total += 1
        logger.info("seed.mlb_done", season=season, rows=total)

    return total


async def _seed_soccer(league: str, seasons: list[int]) -> int:
    """Soccer via soccerdata.FBref (scraping público fbref.com)."""
    try:
        import soccerdata as sd  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("seed.soccerdata_missing")
        return 0

    # fbref via soccerdata solo expone ligas BIG-5 + internacionales.
    # Para ligas individuales hay que usar football-data.co.uk CSV (gap #0.1a).
    # Liga MX NO está disponible en fuentes gratis estándar → fallback vía
    # Pinnacle guest live (se acumula orgánicamente).
    league_map = {
        "epl": "Big 5 European Leagues Combined",
        "la_liga": "Big 5 European Leagues Combined",
        "bundesliga": "Big 5 European Leagues Combined",
        "serie_a": "Big 5 European Leagues Combined",
        "ligue_1": "Big 5 European Leagues Combined",
        "big5": "Big 5 European Leagues Combined",
    }
    sd_league = league_map.get(league)
    if sd_league is None:
        if league == "liga_mx":
            logger.info(
                "seed.liga_mx_no_free_source",
                hint="Liga MX fbref histórico no disponible gratis; confiamos en Pinnacle live + Caliente scraper catchup",
            )
        else:
            logger.warning("seed.soccer_unknown_league", league=league)
        return 0

    total = 0
    for season in seasons:
        season_str = f"{season}-{season + 1}"
        try:
            fb = sd.FBref(leagues=sd_league, seasons=season_str)
            schedule = await asyncio.to_thread(fb.read_schedule)
        except Exception as exc:
            logger.warning("seed.soccer_fail", season=season_str, error=str(exc)[:120])
            continue

        async with session_scope() as session:
            for _, row in schedule.iterrows():
                home_name = row.get("home_team") or row.get("Home")
                away_name = row.get("away_team") or row.get("Away")
                date_raw = row.get("date") or row.get("Date")
                if not home_name or not away_name or not date_raw:
                    continue
                try:
                    start = datetime.fromisoformat(str(date_raw)).replace(tzinfo=UTC)
                except ValueError:
                    try:
                        start = datetime.strptime(str(date_raw), "%Y-%m-%d").replace(tzinfo=UTC)
                    except ValueError:
                        continue
                match_id = await resolve_or_create_match(
                    session,
                    sport_code="soccer",
                    home_name=str(home_name),
                    away_name=str(away_name),
                    start_time=start,
                    source=f"fbref:{league}",
                )
                if match_id is None:
                    continue
                total += 1
        logger.info("seed.soccer_done", league=league, season=season_str, rows=total)

    return total


async def _seed_soccer_with_odds(seasons: list[int], leagues: list[str]) -> dict[str, int]:
    """Fase 0.1a: seed matches + odds históricas soccer via football-data.co.uk.

    Pobla matches + odds_history + scores para ligas configuradas y temporadas dadas.
    Usa el validator `historical_odds_integrity` para filtrar rows corruptos.

    Returns dict con contadores: matches_created, odds_rows_inserted, invalid_odds_skipped.
    """
    from apuestas.ingest.football_data_csv import (
        LEAGUE_CODES,
        fetch_league_season,
        match_to_historical_odds_rows,
    )
    from apuestas.validators.historical_odds_integrity import batch_validate

    counters = {"matches_created": 0, "odds_rows_inserted": 0, "invalid_odds_skipped": 0}
    for league in leagues:
        if league not in LEAGUE_CODES:
            logger.warning("seed.unknown_soccer_league", league=league)
            continue
        for season in seasons:
            try:
                parsed_matches = await fetch_league_season(league, season)
            except Exception as exc:
                logger.warning(
                    "seed.soccer_odds_fetch_fail",
                    league=league,
                    season=season,
                    error=str(exc)[:120],
                )
                continue
            if not parsed_matches:
                continue

            async with session_scope() as session:
                for parsed in parsed_matches:
                    match_id = await resolve_or_create_match(
                        session,
                        sport_code="soccer",
                        home_name=parsed["home_name"],
                        away_name=parsed["away_name"],
                        start_time=parsed["start_time"],
                        source=f"football_data:{league}",
                    )
                    if match_id is None:
                        continue
                    counters["matches_created"] += 1

                    # Update score si está disponible
                    if parsed["home_score"] is not None:
                        await session.execute(
                            text(
                                """
                                UPDATE matches
                                SET home_score = :hs, away_score = :as_,
                                    status = 'finished'
                                WHERE id = :id AND status != 'finished'
                                """
                            ),
                            {
                                "id": match_id,
                                "hs": parsed["home_score"],
                                "as_": parsed["away_score"],
                            },
                        )

                    odds_rows = match_to_historical_odds_rows(parsed, match_id=match_id)
                    valid, cnt = batch_validate(odds_rows)
                    counters["invalid_odds_skipped"] += sum(v for k, v in cnt.items() if k != "ok")
                    for row in valid:
                        for outcome, odds in row.outcomes_odds.items():
                            await session.execute(
                                text(
                                    """
                                    INSERT INTO odds_history
                                      (ts, match_id, bookmaker, market, outcome, odds, is_closing)
                                    VALUES
                                      (:ts, :mid, :bk, :mk, :oc, :od, :cl)
                                    ON CONFLICT DO NOTHING
                                    """
                                ),
                                {
                                    "ts": row.ts,
                                    "mid": match_id,
                                    "bk": row.bookmaker,
                                    "mk": row.market,
                                    "oc": outcome,
                                    "od": odds,
                                    "cl": row.is_closing,
                                },
                            )
                            counters["odds_rows_inserted"] += 1

            logger.info(
                "seed.soccer_odds_done",
                league=league,
                season=season,
                counters=counters,
            )
    return counters


async def _seed_tennis_with_odds(seasons: list[int], tours: list[str]) -> dict[str, int]:
    """Fase 0.1b: seed tenis ATP/WTA + odds via tennis-data.co.uk."""
    from apuestas.ingest.tennis_data_csv import (
        fetch_tour_season,
        match_to_historical_odds_rows,
    )
    from apuestas.validators.historical_odds_integrity import batch_validate

    counters = {"matches_created": 0, "odds_rows_inserted": 0, "invalid_odds_skipped": 0}
    for tour in tours:
        for season in seasons:
            try:
                parsed_matches = await fetch_tour_season(tour, season)
            except Exception as exc:
                logger.warning(
                    "seed.tennis_fetch_fail", tour=tour, season=season, error=str(exc)[:120]
                )
                continue
            if not parsed_matches:
                continue

            async with session_scope() as session:
                for parsed in parsed_matches:
                    match_id = await resolve_or_create_match(
                        session,
                        sport_code="tennis",
                        home_name=parsed["winner_name"],
                        away_name=parsed["loser_name"],
                        start_time=parsed["start_time"],
                        source=f"tennis_data:{tour}",
                    )
                    if match_id is None:
                        continue
                    counters["matches_created"] += 1

                    # Update con sets finales (tennis usa "home_score" como winner_sets)
                    if parsed["winner_sets"] is not None:
                        await session.execute(
                            text(
                                """
                                UPDATE matches
                                SET home_score = :ws, away_score = :ls,
                                    status = 'finished'
                                WHERE id = :id AND status != 'finished'
                                """
                            ),
                            {
                                "id": match_id,
                                "ws": parsed["winner_sets"],
                                "ls": parsed["loser_sets"],
                            },
                        )

                    odds_rows = match_to_historical_odds_rows(parsed, match_id=match_id)
                    valid, cnt = batch_validate(odds_rows)
                    counters["invalid_odds_skipped"] += sum(v for k, v in cnt.items() if k != "ok")
                    for row in valid:
                        for outcome, odds in row.outcomes_odds.items():
                            # tennis: winner→home, loser→away (convención del bot)
                            outcome_normalized = "home" if outcome == "winner" else "away"
                            await session.execute(
                                text(
                                    """
                                    INSERT INTO odds_history
                                      (ts, match_id, bookmaker, market, outcome, odds, is_closing)
                                    VALUES
                                      (:ts, :mid, :bk, :mk, :oc, :od, :cl)
                                    ON CONFLICT DO NOTHING
                                    """
                                ),
                                {
                                    "ts": row.ts,
                                    "mid": match_id,
                                    "bk": row.bookmaker,
                                    "mk": row.market,
                                    "oc": outcome_normalized,
                                    "od": odds,
                                    "cl": row.is_closing,
                                },
                            )
                            counters["odds_rows_inserted"] += 1

            logger.info("seed.tennis_done", tour=tour, season=season, counters=counters)
    return counters


async def _seed_us_sports_with_odds(sports: list[str]) -> dict[str, int]:
    """Fase 0.1c: seed NBA/NFL/NHL historic odds vía SBR community datasets."""
    from apuestas.ingest.sbr_archive_scraper import (
        fetch_sport,
        match_to_historical_odds_rows,
    )
    from apuestas.validators.historical_odds_integrity import batch_validate

    counters = {"matches_created": 0, "odds_rows_inserted": 0, "invalid_odds_skipped": 0}
    for sport in sports:
        try:
            parsed_matches = await fetch_sport(sport)
        except Exception as exc:
            logger.warning("seed.sbr_fetch_fail", sport=sport, error=str(exc)[:120])
            continue
        if not parsed_matches:
            logger.info("seed.sbr_empty_dataset", sport=sport)
            continue

        async with session_scope() as session:
            for parsed in parsed_matches:
                match_id = await resolve_or_create_match(
                    session,
                    sport_code=sport,
                    home_name=parsed["home_name"],
                    away_name=parsed["away_name"],
                    start_time=parsed["start_time"],
                    source=f"sbr:{sport}",
                )
                if match_id is None:
                    continue
                counters["matches_created"] += 1

                if parsed["home_score"] is not None:
                    await session.execute(
                        text(
                            """
                            UPDATE matches
                            SET home_score = :hs, away_score = :as_,
                                status = 'finished'
                            WHERE id = :id AND status != 'finished'
                            """
                        ),
                        {
                            "id": match_id,
                            "hs": parsed["home_score"],
                            "as_": parsed["away_score"],
                        },
                    )

                odds_rows = match_to_historical_odds_rows(parsed, match_id=match_id)
                valid, cnt = batch_validate(odds_rows)
                counters["invalid_odds_skipped"] += sum(v for k, v in cnt.items() if k != "ok")
                for row in valid:
                    for outcome, odds in row.outcomes_odds.items():
                        await session.execute(
                            text(
                                """
                                INSERT INTO odds_history
                                  (ts, match_id, bookmaker, market, outcome, odds, is_closing)
                                VALUES
                                  (:ts, :mid, :bk, :mk, :oc, :od, :cl)
                                ON CONFLICT DO NOTHING
                                """
                            ),
                            {
                                "ts": row.ts,
                                "mid": match_id,
                                "bk": row.bookmaker,
                                "mk": row.market,
                                "oc": outcome,
                                "od": odds,
                                "cl": row.is_closing,
                            },
                        )
                        counters["odds_rows_inserted"] += 1

        logger.info("seed.sbr_sport_done", sport=sport, counters=counters)
    return counters


async def main(args: argparse.Namespace) -> None:
    # Sports que no necesitan seasons (us-sports-odds usa dataset completo)
    if args.sport == "us-sports-odds" and not args.seasons:
        seasons = []
    elif not args.seasons:
        from datetime import datetime as _dt

        current = _dt.now(UTC).year
        seasons = list(range(current - 4, current + 1))
    else:
        seasons = [int(s) for s in args.seasons.split(",")]
    total: int | dict[str, int]
    if args.sport == "nba":
        total = await _seed_nba(seasons)
    elif args.sport == "mlb":
        total = await _seed_mlb(seasons)
    elif args.sport == "soccer":
        total = await _seed_soccer(args.league, seasons)
    elif args.sport == "soccer-odds":
        # Fase 0.1a: soccer con odds históricas multi-liga
        leagues = args.league.split(",") if args.league else ["epl", "la_liga", "bundesliga"]
        total = await _seed_soccer_with_odds(seasons, leagues)
    elif args.sport == "tennis":
        tours = args.league.split(",") if args.league else ["atp"]
        total = await _seed_tennis_with_odds(seasons, tours)
    elif args.sport == "us-sports-odds":
        # Fase 0.1c: NBA/NFL/NHL odds históricas
        sports = args.league.split(",") if args.league else ["nba", "nfl", "nhl"]
        total = await _seed_us_sports_with_odds(sports)
    elif args.sport == "liga-mx":
        # Fase 5.11 — Liga MX + Expansion vía fbref.com scraping directo
        from apuestas.ingest.fbref_liga_mx import ingest_liga_mx_multi_seasons

        leagues = args.league.split(",") if args.league else ["liga_mx", "liga_expansion"]
        combined: dict[int, int] = {}
        for lg in leagues:
            r = await ingest_liga_mx_multi_seasons(league_slug=lg, seasons=seasons)
            for k, v in r.items():
                combined[k] = combined.get(k, 0) + v
        total = combined
    else:
        raise SystemExit(f"Unknown sport: {args.sport}")
    logger.info("seed.complete", sport=args.sport, total=total)
    print(f"Seed complete: sport={args.sport} total={total}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed histórico multi-sport")
    p.add_argument(
        "--sport",
        required=True,
        choices=[
            "nba",
            "mlb",
            "soccer",
            "soccer-odds",
            "tennis",
            "us-sports-odds",
            "liga-mx",
        ],
    )
    p.add_argument("--seasons", default="", help="CSV ej: 2023,2024,2025")
    p.add_argument("--league", default="", help="CSV ligas si aplica")
    return p.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
