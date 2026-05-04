"""NBA injury NLP ingester — Fase 1 wire (#150 support).

Scrape NBA.com injury report PDF ~3x/día + Sofascore fallback.
Populate `injury_reports_normalized` tabla.

Uso: python -m apuestas.ingest.injury_nlp_ingest
"""

from __future__ import annotations

import asyncio

import httpx
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

STATUS_MAP = {
    "out": "out",
    "doubtful": "doubtful",
    "questionable": "questionable",
    "probable": "probable",
    "day-to-day": "questionable",
    "gtd": "questionable",
    "available": "available",
}


async def ensure_injury_table() -> None:
    async with session_scope() as s:
        await s.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS injury_reports_normalized (
                  id bigserial PRIMARY KEY,
                  sport_code text NOT NULL,
                  team_id bigint,
                  player_name text NOT NULL,
                  status text NOT NULL,
                  reason text,
                  reported_at timestamptz NOT NULL DEFAULT now(),
                  source text
                )
                """
            )
        )
        await s.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_injury_team_status "
                "ON injury_reports_normalized (team_id, status, reported_at DESC)"
            )
        )


async def fetch_espn_nba_injuries() -> list[dict]:
    """ESPN NBA injury JSON público sin auth."""
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url, headers={"User-Agent": "apuestas/0.1"})
        if r.status_code != 200:
            return []
        data = r.json()
    except Exception as exc:
        logger.warning("injury_nlp.espn_fail", error=str(exc)[:80])
        return []

    out = []
    for team in data.get("injuries", []):
        team_name = team.get("displayName") or team.get("team", {}).get("displayName", "")
        for injury in team.get("injuries", []):
            athlete = injury.get("athlete") or {}
            if isinstance(athlete, str):
                try:
                    import json as _json

                    athlete = _json.loads(athlete.replace("'", '"'))
                except Exception:
                    athlete = {}
            player = athlete.get("displayName", "") if isinstance(athlete, dict) else ""
            status_raw = str(injury.get("status", "")).lower()
            status = STATUS_MAP.get(status_raw, "questionable")
            reason_raw = injury.get("type") or injury.get("shortComment") or ""
            if isinstance(reason_raw, dict):
                reason = str(reason_raw.get("description", ""))[:200]
            else:
                reason = str(reason_raw)[:200]
            if player:
                out.append(
                    {
                        "team_name": team_name,
                        "player_name": player,
                        "status": status,
                        "reason": reason,
                    }
                )
    return out


async def persist_injuries(injuries: list[dict]) -> int:
    if not injuries:
        return 0
    await ensure_injury_table()
    n = 0
    for inj in injuries:
        # Session aislada por row para evitar abort en caso de error
        try:
            async with session_scope() as s:
                team_name = inj.get("team_name") or ""
                team_id = None
                if team_name:
                    # Fuzzy match: exact + partial (last word) + trigram
                    last_word = team_name.split()[-1] if team_name else ""
                    team_row = (
                        await s.execute(
                            text(
                                "SELECT id FROM teams WHERE sport_code='nba' AND ("
                                "  name = :tn "
                                "  OR name ILIKE :lk "
                                "  OR :tn ILIKE '%' || name || '%' "
                                "  OR (short_name IS NOT NULL AND short_name = :tn) "
                                "  OR (abbreviation IS NOT NULL AND abbreviation ILIKE :tn) "
                                "  OR (:lw <> '' AND name ILIKE :lwk)"
                                ") ORDER BY similarity(name, :tn) DESC NULLS LAST LIMIT 1"
                            ),
                            {
                                "tn": team_name,
                                "lk": f"%{team_name}%",
                                "lw": last_word,
                                "lwk": f"%{last_word}%",
                            },
                        )
                    ).first()
                    team_id = int(team_row.id) if team_row else None
                await s.execute(
                    text(
                        "INSERT INTO injury_reports_normalized "
                        "(sport_code, team_id, player_name, player_name_raw, status, reason, reported_at, source) "
                        "VALUES ('nba', :tid, :p, :p, :st, :rsn, NOW(), 'espn') "
                        "ON CONFLICT (team_id, player_name) WHERE team_id IS NOT NULL "
                        "DO UPDATE SET status = EXCLUDED.status, reason = EXCLUDED.reason, reported_at = EXCLUDED.reported_at, source = EXCLUDED.source"
                    ),
                    {
                        "tid": team_id,
                        "p": inj["player_name"],
                        "st": inj["status"],
                        "rsn": inj["reason"],
                    },
                )
                n += 1
        except Exception as exc:
            logger.debug("injury_nlp.persist_fail", error=str(exc)[:80])
            continue
    return n


async def fetch_espn_mlb_injuries() -> list[dict]:
    """ESPN MLB injury JSON público sin auth (mismo formato que NBA)."""
    url = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/injuries"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url, headers={"User-Agent": "apuestas/0.1"})
        if r.status_code != 200:
            return []
        data = r.json()
    except Exception as exc:
        logger.warning("injury_nlp.espn_mlb_fail", error=str(exc)[:80])
        return []

    out = []
    for team in data.get("injuries", []):
        team_name = team.get("displayName") or team.get("team", {}).get("displayName", "")
        for injury in team.get("injuries", []):
            athlete = injury.get("athlete") or {}
            if isinstance(athlete, str):
                try:
                    import json as _json

                    athlete = _json.loads(athlete.replace("'", '"'))
                except Exception:
                    athlete = {}
            player = athlete.get("displayName", "") if isinstance(athlete, dict) else ""
            status_raw = str(injury.get("status", "")).lower()
            status = STATUS_MAP.get(status_raw, "questionable")
            reason_raw = injury.get("type") or injury.get("shortComment") or ""
            if isinstance(reason_raw, dict):
                reason = str(reason_raw.get("description", ""))[:200]
            else:
                reason = str(reason_raw)[:200]
            if player:
                out.append(
                    {
                        "team_name": team_name,
                        "player_name": player,
                        "status": status,
                        "reason": reason,
                        "sport_code": "mlb",
                    }
                )
    return out


async def persist_injuries_mlb(injuries: list[dict]) -> int:
    """Variante MLB: busca team con sport_code='mlb' (no 'nba') + sport_code en INSERT."""
    if not injuries:
        return 0
    await ensure_injury_table()
    n = 0
    for inj in injuries:
        try:
            async with session_scope() as s:
                team_name = inj.get("team_name") or ""
                team_id = None
                if team_name:
                    last_word = team_name.split()[-1] if team_name else ""
                    team_row = (
                        await s.execute(
                            text(
                                "SELECT id FROM teams WHERE sport_code='mlb' AND ("
                                "  name = :tn "
                                "  OR name ILIKE :lk "
                                "  OR :tn ILIKE '%' || name || '%' "
                                "  OR (short_name IS NOT NULL AND short_name = :tn) "
                                "  OR (abbreviation IS NOT NULL AND abbreviation ILIKE :tn) "
                                "  OR (:lw <> '' AND name ILIKE :lwk)"
                                ") ORDER BY similarity(name, :tn) DESC NULLS LAST LIMIT 1"
                            ),
                            {
                                "tn": team_name,
                                "lk": f"%{team_name}%",
                                "lw": last_word,
                                "lwk": f"%{last_word}%",
                            },
                        )
                    ).first()
                    team_id = int(team_row.id) if team_row else None
                await s.execute(
                    text(
                        "INSERT INTO injury_reports_normalized "
                        "(sport_code, team_id, player_name, player_name_raw, status, reason, reported_at, source) "
                        "VALUES ('mlb', :tid, :p, :p, :st, :rsn, NOW(), 'espn') "
                        "ON CONFLICT (team_id, player_name) WHERE team_id IS NOT NULL "
                        "DO UPDATE SET status = EXCLUDED.status, reason = EXCLUDED.reason, reported_at = EXCLUDED.reported_at, source = EXCLUDED.source"
                    ),
                    {
                        "tid": team_id,
                        "p": inj["player_name"],
                        "st": inj["status"],
                        "rsn": inj["reason"],
                    },
                )
                n += 1
        except Exception as exc:
            logger.debug("injury_nlp.persist_mlb_fail", error=str(exc)[:80])
            continue
    return n


# Ligas soccer cubiertas por ESPN injury feed (slug oficial ESPN).
# Cada llamada es un GET HTTP único — NO consume créditos OddsAPI.
ESPN_SOCCER_LEAGUES: tuple[str, ...] = (
    "eng.1",  # EPL
    "esp.1",  # LaLiga
    "ita.1",  # Serie A
    "ger.1",  # Bundesliga
    "fra.1",  # Ligue 1
    "mex.1",  # Liga MX
    "usa.1",  # MLS
    "uefa.champions",
    "uefa.europa",
    "ned.1",  # Eredivisie
    "por.1",  # Primeira Liga
    "bra.1",  # Brasileirão
    "arg.1",  # Primera Argentina
)


async def fetch_espn_soccer_injuries() -> list[dict]:
    """ESPN soccer injury JSON por liga.

    Cubre el gap reportado en CLAUDE.md (~0 records soccer vs 1k NBA / 327 MLB).
    Estructura JSON idéntica a NBA/MLB; solo cambia path por league slug.
    """
    out: list[dict] = []
    async with httpx.AsyncClient(timeout=12.0) as client:
        for league in ESPN_SOCCER_LEAGUES:
            url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/injuries"
            try:
                r = await client.get(url, headers={"User-Agent": "apuestas/0.1"})
                if r.status_code != 200:
                    continue
                data = r.json()
            except Exception as exc:
                logger.debug("injury_nlp.espn_soccer_fail", league=league, error=str(exc)[:80])
                continue

            for team in data.get("injuries", []):
                team_name = team.get("displayName") or team.get("team", {}).get("displayName", "")
                for injury in team.get("injuries", []):
                    athlete = injury.get("athlete") or {}
                    if isinstance(athlete, str):
                        try:
                            import json as _json

                            athlete = _json.loads(athlete.replace("'", '"'))
                        except Exception:
                            athlete = {}
                    player = athlete.get("displayName", "") if isinstance(athlete, dict) else ""
                    status_raw = str(injury.get("status", "")).lower()
                    status = STATUS_MAP.get(status_raw, "questionable")
                    reason_raw = injury.get("type") or injury.get("shortComment") or ""
                    if isinstance(reason_raw, dict):
                        reason = str(reason_raw.get("description", ""))[:200]
                    else:
                        reason = str(reason_raw)[:200]
                    if player:
                        out.append(
                            {
                                "team_name": team_name,
                                "player_name": player,
                                "status": status,
                                "reason": reason,
                                "sport_code": "soccer",
                                "league": league,
                            }
                        )
    return out


async def persist_injuries_soccer(injuries: list[dict]) -> int:
    """Variante soccer: busca team con sport_code='soccer' (multi-liga) + insert con sport_code='soccer'."""
    if not injuries:
        return 0
    await ensure_injury_table()
    n = 0
    for inj in injuries:
        try:
            async with session_scope() as s:
                team_name = inj.get("team_name") or ""
                team_id = None
                if team_name:
                    last_word = team_name.split()[-1] if team_name else ""
                    # Soccer multi-liga: relajamos sport_code (algunos teams están como 'soccer_epl')
                    team_row = (
                        await s.execute(
                            text(
                                "SELECT id FROM teams WHERE ("
                                "  sport_code = 'soccer' OR sport_code LIKE 'soccer_%' "
                                "  OR sport_code IS NULL"
                                ") AND ("
                                "  name = :tn "
                                "  OR name ILIKE :lk "
                                "  OR :tn ILIKE '%' || name || '%' "
                                "  OR (short_name IS NOT NULL AND short_name = :tn) "
                                "  OR (abbreviation IS NOT NULL AND abbreviation ILIKE :tn) "
                                "  OR (:lw <> '' AND name ILIKE :lwk)"
                                ") ORDER BY similarity(name, :tn) DESC NULLS LAST LIMIT 1"
                            ),
                            {
                                "tn": team_name,
                                "lk": f"%{team_name}%",
                                "lw": last_word,
                                "lwk": f"%{last_word}%",
                            },
                        )
                    ).first()
                    team_id = int(team_row.id) if team_row else None
                await s.execute(
                    text(
                        "INSERT INTO injury_reports_normalized "
                        "(sport_code, team_id, player_name, player_name_raw, status, reason, reported_at, source) "
                        "VALUES ('soccer', :tid, :p, :p, :st, :rsn, NOW(), 'espn') "
                        "ON CONFLICT (team_id, player_name) WHERE team_id IS NOT NULL "
                        "DO UPDATE SET status = EXCLUDED.status, reason = EXCLUDED.reason, reported_at = EXCLUDED.reported_at, source = EXCLUDED.source"
                    ),
                    {
                        "tid": team_id,
                        "p": inj["player_name"],
                        "st": inj["status"],
                        "rsn": inj["reason"],
                    },
                )
                n += 1
        except Exception as exc:
            logger.debug("injury_nlp.persist_soccer_fail", error=str(exc)[:80])
            continue
    return n


# API-Football fallback: ESPN no popula injuries para soccer (HTTP 200 pero
# array vacío en TODAS las ligas testeadas 2026-04-25). API-Football paid tier
# SÍ tiene data real. Mapping league_slug → API-Football league_id desde
# apuestas.ingest.api_football.LEAGUE_IDS.
APIFOOT_INJURY_LEAGUES: tuple[tuple[str, int], ...] = (
    ("epl", 39),
    ("la_liga", 140),
    ("serie_a", 135),
    ("bundesliga", 78),
    ("ligue_1", 61),
    ("liga_mx", 262),
    ("mls", 253),
    ("champions", 2),
    ("europa", 3),
    ("brasileirao", 71),
)

# Mapping API-Football "type" → status canonical (alineado con STATUS_MAP de ESPN)
APIFOOT_TYPE_TO_STATUS = {
    "Missing Fixture": "out",
    "Questionable": "questionable",
    "Probable": "probable",
    "Doubtful": "doubtful",
}


async def fetch_api_football_soccer_injuries(*, season: int | None = None) -> list[dict]:
    """Fetch injuries via API-Football paid tier (soccer multi-liga).

    ESPN tiene data injuries=[] vacío para soccer, así que API-Football es la
    fuente real. Costo: ~10 requests/run × 1 crédito API-Football (NO OddsAPI).
    Plan Pro $19/mes = 7,500 req/día → margen 99%.
    """
    from datetime import UTC, datetime

    try:
        from apuestas.ingest.api_football import (
            APIFootballClient,
            _api_football_key_available,
        )
    except Exception as exc:
        logger.debug("injury_nlp.apifoot_import_fail", error=str(exc)[:80])
        return []

    if not _api_football_key_available():
        logger.info("injury_nlp.apifoot_no_key")
        return []

    if season is None:
        # API-Football season convention: "2025" cubre 2025-2026 europeo.
        # Para liga_mx/MLS calendario calendario natural usamos current year.
        season = datetime.now(tz=UTC).year

    out: list[dict] = []
    try:
        client = APIFootballClient()
    except ValueError:
        return []
    async with client.session():
        for slug, league_id in APIFOOT_INJURY_LEAGUES:
            try:
                raw = await client.fetch_injuries(league=league_id, season=season)
            except Exception as exc:
                logger.debug(
                    "injury_nlp.apifoot_league_fail",
                    league=slug,
                    error=str(exc)[:80],
                )
                continue
            for e in raw:
                player = e.get("player", {}) or {}
                team = e.get("team", {}) or {}
                player_name = player.get("name") or ""
                team_name = team.get("name") or ""
                status_raw = player.get("type") or ""
                status = APIFOOT_TYPE_TO_STATUS.get(status_raw, "out")
                reason = (player.get("reason") or "")[:200]
                if player_name and team_name:
                    out.append(
                        {
                            "team_name": team_name,
                            "player_name": player_name,
                            "status": status,
                            "reason": reason,
                            "sport_code": "soccer",
                            "league": slug,
                        }
                    )
    return out


async def main():
    nba_injuries = await fetch_espn_nba_injuries()
    n_nba = await persist_injuries(nba_injuries)
    print(f"NBA injuries persisted: {n_nba} (fetched: {len(nba_injuries)})")

    mlb_injuries = await fetch_espn_mlb_injuries()
    n_mlb = await persist_injuries_mlb(mlb_injuries)
    print(f"MLB injuries persisted: {n_mlb} (fetched: {len(mlb_injuries)})")

    soccer_espn = await fetch_espn_soccer_injuries()
    n_soccer_espn = await persist_injuries_soccer(soccer_espn)
    print(f"Soccer injuries (ESPN) persisted: {n_soccer_espn} (fetched: {len(soccer_espn)})")

    soccer_apifoot = await fetch_api_football_soccer_injuries()
    n_soccer_af = await persist_injuries_soccer(soccer_apifoot)
    print(
        f"Soccer injuries (API-Football) persisted: {n_soccer_af} "
        f"(fetched: {len(soccer_apifoot)} across {len(APIFOOT_INJURY_LEAGUES)} leagues)"
    )


if __name__ == "__main__":
    asyncio.run(main())
