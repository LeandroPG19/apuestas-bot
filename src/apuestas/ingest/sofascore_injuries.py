"""Soccer injuries via Sofascore (Cloudflare-protected) usando Camoufox.

Endpoint público: `/api/v1/team/{sofascore_team_id}/sidelined`. Devuelve JSON
con jugadores actualmente lesionados/suspendidos por team.

Estrategia:
  1. Para cada team soccer top con `external_id_sofascore` poblado, query
     `/sidelined` via Camoufox (bypass Cloudflare 403).
  2. Parse JSON: `sidelined[].player`, `sidelined[].reason`, `sidelined[].type`
     (injury|suspension), `sidelined[].startTimestamp`, `sidelined[].endTimestamp`.
  3. INSERT a `injury_reports_normalized` con sport_code='soccer'.

Throughput: ~3-5s por team × 200 teams = ~15 min total. Schedule nightly o
cada 6h para mantener fresh sin saturar.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

SOFASCORE_API = "https://api.sofascore.com/api/v1"
# Map status sofascore → apuestas
_STATUS_MAP = {
    "injury": "out",
    "suspension": "out",
    "national_team": "out",
    "international duty": "out",
    "minor injury": "questionable",
    "doubtful": "doubtful",
    "knee injury": "out",
    "muscle injury": "doubtful",
    "fitness": "questionable",
}


async def _fetch_sidelined(team_sofascore_id: int) -> list[dict[str, Any]]:
    """Fetch /sidelined endpoint via Camoufox. Devuelve lista de injuries."""
    from camoufox.async_api import AsyncCamoufox  # type: ignore[import-untyped]

    url = f"{SOFASCORE_API}/team/{team_sofascore_id}/sidelined"
    try:
        async with AsyncCamoufox(headless=True, humanize=False) as browser:
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle", timeout=20000)
            content = await page.content()
            # Sofascore devuelve JSON envuelto en <pre>. Extraer.
            try:
                # Si content empieza con {, es raw JSON
                start = content.find("{")
                end = content.rfind("}")
                if start >= 0 and end > start:
                    raw = content[start : end + 1]
                    data = json.loads(raw)
                else:
                    return []
            except json.JSONDecodeError:
                return []
            return list(data.get("sidelined") or [])
    except Exception as exc:
        logger.warning(
            "sofascore_injuries.fetch_fail",
            team_id=team_sofascore_id,
            error=str(exc)[:120],
        )
        return []


def _normalize_status(reason_type: str, reason: str) -> str:
    """Mapea status raw de Sofascore al enum del bot."""
    rl = (reason or "").lower()
    if reason_type == "injury":
        if any(kw in rl for kw in ("rupture", "torn", "fracture", "surgery", "out")):
            return "out"
        if any(kw in rl for kw in ("strain", "knock", "minor")):
            return "doubtful"
        return "questionable"
    if reason_type == "suspension":
        return "out"
    return _STATUS_MAP.get(rl, "questionable")


async def _persist_team_injuries(
    team_id: int,
    sofascore_id: int,
    injuries_raw: list[dict[str, Any]],
) -> int:
    """Persiste injuries de un team a injury_reports_normalized."""
    if not injuries_raw:
        return 0
    n = 0
    async with session_scope() as session:
        for inj in injuries_raw:
            try:
                player = (inj.get("player") or {}).get("name") or ""
                if not player:
                    continue
                reason_type = (inj.get("type") or "").lower()
                reason = (inj.get("reason") or "")[:200]
                status = _normalize_status(reason_type, reason)
                # Skip si ya pasó endTimestamp
                end_ts = inj.get("endTimestamp")
                if end_ts:
                    try:
                        end_dt = datetime.fromtimestamp(int(end_ts), tz=UTC)
                        if end_dt < datetime.now(tz=UTC):
                            continue
                    except (ValueError, TypeError):
                        pass

                await session.execute(
                    text(
                        """
                        INSERT INTO injury_reports_normalized
                          (sport_code, team_id, player_name, status, reason,
                           reported_at, source)
                        VALUES ('soccer', :tid, :p, :st, :rsn, NOW(), 'sofascore')
                        ON CONFLICT (team_id, player_name) WHERE team_id IS NOT NULL
                        DO UPDATE SET status = EXCLUDED.status,
                                      reason = EXCLUDED.reason,
                                      reported_at = EXCLUDED.reported_at,
                                      source = EXCLUDED.source
                        """
                    ),
                    {"tid": team_id, "p": player[:100], "st": status, "rsn": reason},
                )
                n += 1
            except Exception as exc:
                logger.debug(
                    "sofascore_injuries.persist_fail",
                    team_id=team_id,
                    player=player[:50],
                    error=str(exc)[:80],
                )
    return n


async def _get_top_soccer_teams_to_scan() -> list[tuple[int, int]]:
    """Top teams soccer con external_id_sofascore + matches próximas 7 días.

    Limit defensivo: 100 teams (top mais activos). Cada uno cuesta ~5s con
    camoufox; 100 × 5s = ~8 min total per run.
    """
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT DISTINCT t.id, tei.external_id::bigint AS sofascore_id
                FROM teams t
                JOIN team_external_id tei ON tei.team_id = t.id AND tei.source = 'sofascore'
                WHERE t.sport_code = 'soccer'
                  AND tei.external_id ~ '^\\d+$'
                  AND EXISTS (
                      SELECT 1 FROM matches m
                      WHERE (m.home_team_id = t.id OR m.away_team_id = t.id)
                        AND m.status = 'scheduled'
                        AND m.start_time > NOW()
                        AND m.start_time < NOW() + INTERVAL '7 days'
                  )
                LIMIT 100
                """
            )
        )
        return [(int(r.id), int(r.sofascore_id)) for r in result.all()]


async def ingest_soccer_injuries() -> dict[str, int]:
    """Entry point: ingesta injuries para top teams soccer con matches próximas."""
    teams = await _get_top_soccer_teams_to_scan()
    if not teams:
        logger.info("sofascore_injuries.no_teams_with_external_id")
        return {"teams_scanned": 0, "injuries_persisted": 0}

    total_persisted = 0
    teams_with_data = 0

    # Sequential (camoufox no soporta bien concurrent browser instances)
    for team_id, sofascore_id in teams:
        injuries = await _fetch_sidelined(sofascore_id)
        if injuries:
            n = await _persist_team_injuries(team_id, sofascore_id, injuries)
            total_persisted += n
            if n > 0:
                teams_with_data += 1
        # Pequeña pausa entre teams para no saturar Cloudflare
        await asyncio.sleep(1.0)

    logger.info(
        "sofascore_injuries.done",
        teams_scanned=len(teams),
        teams_with_data=teams_with_data,
        injuries_persisted=total_persisted,
    )
    return {
        "teams_scanned": len(teams),
        "teams_with_data": teams_with_data,
        "injuries_persisted": total_persisted,
    }


if __name__ == "__main__":
    asyncio.run(ingest_soccer_injuries())
