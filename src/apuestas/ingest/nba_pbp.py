"""NBA Play-by-Play ingest — el edge principal de Voulgaris.

`nba_api` expone PlayByPlayV3 (gratis). Capturamos TODOS los eventos de
un partido (puntos, faltas, timeouts, cambios, tiempos muertos) con
timestamp (quarter + clock) para derivar features granulares:

- Clutch tendencies (T ≤ 3 min Q4)
- Foul patterns por ref
- Lineup rotation analysis
- Hack-a-X detection
- Transition vs halfcourt split

Uso:
    from apuestas.ingest.nba_pbp import ingest_pbp_for_game
    n = await ingest_pbp_for_game(game_id="0022300500")
"""

from __future__ import annotations

import asyncio
from datetime import UTC
from typing import Any

import polars as pl
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


EVENT_TYPE_MAP: dict[int, str] = {
    1: "made_shot",
    2: "missed_shot",
    3: "free_throw",
    4: "rebound",
    5: "turnover",
    6: "foul",
    7: "violation",
    8: "substitution",
    9: "timeout",
    10: "jump_ball",
    11: "ejection",
    12: "period_start",
    13: "period_end",
    14: "instant_replay",
    18: "end_of_game",
}


def _parse_clock(clock_str: str) -> int | None:
    """'PT11M23.50S' → 683."""
    if not clock_str or not isinstance(clock_str, str):
        return None
    try:
        s = clock_str.replace("PT", "").replace("S", "")
        mins_part, _, secs_part = s.partition("M")
        mins = int(mins_part or 0)
        secs = float(secs_part or 0)
        return int(mins * 60 + secs)
    except (ValueError, AttributeError):  # fmt: skip
        return None


async def _fetch_pbp_blocking(game_id: str) -> list[dict[str, Any]]:
    """nba_api es sync; lo envolvemos en thread."""

    def _do() -> list[dict[str, Any]]:
        try:
            from nba_api.stats.endpoints import playbyplayv3
        except ImportError:
            logger.warning("nba_pbp.nba_api_missing")
            return []
        try:
            pbp = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=30)
            df = pbp.get_data_frames()[0]
            if df is None or len(df) == 0:
                return []
            return df.to_dict(orient="records")
        except Exception as exc:
            logger.warning("nba_pbp.fetch_fail", game_id=game_id, error=str(exc))
            return []

    return await asyncio.to_thread(_do)


async def ingest_pbp_for_game(*, game_id: str, match_id: int | None = None) -> int:
    """Descarga PBP y persiste en play_by_play. Retorna n eventos insertados."""
    events = await _fetch_pbp_blocking(game_id)
    if not events:
        return 0

    # Si no se dio match_id, intentar resolverlo por external_id
    if match_id is None:
        async with session_scope() as s:
            r = await s.execute(
                text("SELECT id FROM matches WHERE external_id = :ext"),
                {"ext": game_id},
            )
            row = r.first()
            if not row:
                logger.warning("nba_pbp.match_not_found", game_id=game_id)
                return 0
            match_id = int(row[0])

    inserted = 0
    async with session_scope() as s:
        for ev in events:
            # PBP v3 devuelve actionType como string ("period", "Jump Ball", etc.)
            action_raw = ev.get("actionType") or ev.get("actionId") or ""
            if isinstance(action_raw, (int, float)):
                event_type = EVENT_TYPE_MAP.get(int(action_raw), "other")
            else:
                event_type = str(action_raw)[:64] or "other"
            period = int(ev.get("period") or 0)
            clock = _parse_clock(str(ev.get("clock", "")))
            desc = str(ev.get("description") or ev.get("actionType", ""))[:500]
            home_score = ev.get("scoreHome")
            away_score = ev.get("scoreAway")
            team_ext = ev.get("teamId")
            player_ext = ev.get("personId") or ev.get("playerId")

            # Resolver team_id / player_id via external_id
            team_id = None
            player_id = None
            if team_ext:
                tr = await s.execute(
                    text("SELECT id FROM teams WHERE external_id = :e LIMIT 1"),
                    {"e": f"nba_team_{team_ext}"},
                )
                trow = tr.first()
                team_id = int(trow[0]) if trow else None
            if player_ext:
                pr = await s.execute(
                    text("SELECT id FROM players WHERE external_id = :e LIMIT 1"),
                    {"e": f"nba_player_{player_ext}"},
                )
                prow = pr.first()
                player_id = int(prow[0]) if prow else None

            await s.execute(
                text(
                    """
                    INSERT INTO play_by_play
                        (match_id, sport_code, period, clock_seconds_remaining,
                         event_type, team_id, player_id, description,
                         home_score, away_score, metadata)
                    VALUES
                        (:m, 'nba', :p, :c, :e, :t, :pl, :d, :hs, :as_, CAST(:meta AS jsonb))
                    """
                ),
                {
                    "m": match_id,
                    "p": period,
                    "c": clock,
                    "e": event_type,
                    "t": team_id,
                    "pl": player_id,
                    "d": desc,
                    "hs": int(home_score) if home_score not in (None, "", "NaN") else None,
                    "as_": int(away_score) if away_score not in (None, "", "NaN") else None,
                    "meta": _safe_json(ev),
                },
            )
            inserted += 1

    logger.info("nba_pbp.ingested", game_id=game_id, match_id=match_id, events=inserted)
    return inserted


def _safe_json(obj: Any) -> str:
    import json as _json

    try:
        # Filtrar keys no serializables
        clean = {
            k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
            for k, v in obj.items()
        }
        return _json.dumps(clean, ensure_ascii=False)
    except Exception:
        return "{}"


async def _games_on_date_nba(game_date: str) -> list[dict]:
    """Lista de games NBA en una fecha vía ScoreboardV3 (PlayByPlayV2 está deprecated)."""

    def _sync():  # type: ignore[no-untyped-def]
        from nba_api.stats.endpoints import scoreboardv3

        sb = scoreboardv3.ScoreboardV3(game_date=game_date)
        return sb.get_data_frames()

    try:
        dfs = await asyncio.to_thread(_sync)
    except Exception as exc:
        logger.warning("nba_pbp.scoreboard_fail", date=game_date, error=str(exc)[:100])
        return []
    if not dfs or len(dfs) == 0:
        return []
    # ScoreboardV3 devuelve: df0=header, df1=games (gameId, gameStatus...),
    # df2=teams por gameId (home+away interleaved).
    if len(dfs) < 3:
        return []
    games_df = dfs[1]  # gameId + status
    teams_df = dfs[2]  # gameId × team (2 rows per game)
    if games_df is None or len(games_df) == 0:
        return []

    # Agrupar teams por gameId
    teams_by_game: dict[str, list[dict]] = {}
    for tr in teams_df.to_dict(orient="records"):
        gid = str(tr.get("gameId") or "")
        teams_by_game.setdefault(gid, []).append(tr)

    out: list[dict] = []
    for r in games_df.to_dict(orient="records"):
        gid = str(r.get("gameId") or "")
        if not gid:
            continue
        team_rows = teams_by_game.get(gid, [])
        if len(team_rows) < 2:
            continue
        # gameCode format: "YYYYMMDD/AWYHOM" (6 chars sin @, 3+3 tricodes)
        game_code = str(r.get("gameCode") or "")
        home_name = ""
        away_name = ""
        tricode_part = game_code.rsplit("/", maxsplit=1)[-1].upper()
        if len(tricode_part) == 6:
            away_slug = tricode_part[:3]
            home_slug = tricode_part[3:]
            for tr in team_rows:
                trc = str(tr.get("teamTricode") or "").upper()
                name = f"{tr.get('teamCity') or ''} {tr.get('teamName') or ''}".strip()
                if trc == away_slug:
                    away_name = name
                elif trc == home_slug:
                    home_name = name
        if not home_name or not away_name:
            # Fallback: primer = away, segundo = home (asumido)
            away_name = (
                f"{team_rows[0].get('teamCity') or ''} {team_rows[0].get('teamName') or ''}".strip()
            )
            home_name = (
                f"{team_rows[1].get('teamCity') or ''} {team_rows[1].get('teamName') or ''}".strip()
            )
        out.append(
            {
                "game_id_nba": gid,
                "home_team_name": home_name,
                "away_team_name": away_name,
            }
        )
    return out


async def _match_nba_game_to_db(session, game_date: str, home_name: str, away_name: str):  # type: ignore[no-untyped-def]
    """Matchea game → match.id por fecha + fuzzy team names."""
    from datetime import datetime as _dt
    from datetime import timedelta as _td

    try:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT m.id,
                           ht.name AS home_name,
                           at.name AS away_name
                    FROM matches m
                    JOIN teams ht ON ht.id = m.home_team_id
                    JOIN teams at ON at.id = m.away_team_id
                    WHERE m.sport_code = 'nba'
                      AND DATE(m.start_time AT TIME ZONE 'UTC') BETWEEN :d1 AND :d2
                    """
                ),
                {
                    "d1": _dt.strptime(game_date, "%Y-%m-%d").replace(tzinfo=UTC).date(),
                    "d2": (
                        _dt.strptime(game_date, "%Y-%m-%d").replace(tzinfo=UTC) + _td(days=1)
                    ).date(),
                },
            )
        ).fetchall()
    except Exception as exc:
        logger.warning("nba_pbp.match_lookup_fail", error=str(exc)[:100])
        return None
    if not rows:
        return None
    home_low = home_name.lower().strip()
    away_low = away_name.lower().strip()
    # Exact match first
    for r in rows:
        rh = (r.home_name or "").lower().strip()
        ra = (r.away_name or "").lower().strip()
        if rh == home_low and ra == away_low:
            return int(r.id)
    # Fuzzy: shared suffix token (team name like "Knicks", "Lakers")
    home_tokens = [t for t in home_low.split() if len(t) > 3]
    away_tokens = [t for t in away_low.split() if len(t) > 3]
    for r in rows:
        rh = (r.home_name or "").lower()
        ra = (r.away_name or "").lower()
        if any(t in rh for t in home_tokens) and any(t in ra for t in away_tokens):
            return int(r.id)
    return None


async def ingest_nba_pbp_for_date(game_date: str, *, rate_limit_sec: float = 1.2) -> int:
    """Descarga PBP para todos los games NBA de una fecha y los persiste.

    Returns:
        Número total de eventos insertados.
    """
    games = await _games_on_date_nba(game_date)
    if not games:
        logger.info("nba_pbp.no_games", date=game_date)
        return 0
    total = 0
    async with session_scope() as session:
        for g in games:
            match_id = await _match_nba_game_to_db(
                session, game_date, g["home_team_name"], g["away_team_name"]
            )
            if match_id is None:
                logger.info(
                    "nba_pbp.no_match_in_db",
                    game=g["game_id_nba"],
                    home=g["home_team_name"],
                    away=g["away_team_name"],
                )
                continue
            await asyncio.sleep(rate_limit_sec)
            n = await ingest_pbp_for_game(game_id=g["game_id_nba"], match_id=match_id)
            total += n
    logger.info("nba_pbp.date_done", date=game_date, total_events=total)
    return total


async def ingest_nba_pbp_range(start_date, end_date) -> int:  # type: ignore[no-untyped-def]
    """Ingesta PBP para rango [start_date, end_date] (date objects).

    Args:
        start_date: date object.
        end_date: date object (inclusive).
    """
    from datetime import timedelta as _td

    total = 0
    current = start_date
    while current <= end_date:
        n = await ingest_nba_pbp_for_date(current.strftime("%Y-%m-%d"))
        total += n
        current += _td(days=1)
    logger.info(
        "nba_pbp.range_done",
        start=str(start_date),
        end=str(end_date),
        total_events=total,
    )
    return total


async def derive_clutch_events(match_id: int, last_seconds: int = 180) -> pl.DataFrame:
    """Extrae eventos del clutch (Q4 ≤ 3 min). Base para coaching tendencies."""
    async with session_scope() as s:
        r = await s.execute(
            text(
                """
                SELECT period, clock_seconds_remaining, event_type, team_id,
                       player_id, description, home_score, away_score
                FROM play_by_play
                WHERE match_id = :m
                  AND period >= 4
                  AND clock_seconds_remaining IS NOT NULL
                  AND clock_seconds_remaining <= :cs
                ORDER BY period ASC, clock_seconds_remaining DESC
                """
            ),
            {"m": match_id, "cs": last_seconds},
        )
        rows = [dict(row._mapping) for row in r.all()]
    return pl.DataFrame(rows) if rows else pl.DataFrame()
