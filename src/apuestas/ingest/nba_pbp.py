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
            data = pbp.get_normalized_dict().get("PlayByPlay", [])
            return list(data)
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
            action_type_id = ev.get("actionType") or ev.get("actionId") or 0
            event_type = EVENT_TYPE_MAP.get(int(action_type_id), "other")
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
                    "hs": int(home_score) if home_score is not None else None,
                    "as_": int(away_score) if away_score is not None else None,
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
