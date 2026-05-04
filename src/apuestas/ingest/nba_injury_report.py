"""NBA injury report ingester — Sprint 14 #150.

Parsea NBA.com injury reports (updated ~3x/día) ~2h antes de tip-off.
Identifica:
  - OUT / DOUBTFUL → aplicar star_out_adjustment si player en top 8 rotation
  - QUESTIONABLE → soft_tag 'injury_uncertain' baja tier
  - PROBABLE → ignore

Source primario: https://official.nba.com/wp-content/uploads/sites/4/YYYY/MM/YYYY-MM-DD_{TIME}_Injury-Report.pdf
Fallback: Sofascore /api/v1/team/{team_id}/injuries (JSON público).

Wire con sports_advanced.star_out_adjustment ya existente.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import httpx
from sqlalchemy import text

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


# Status severity mapping
SEVERITY = {
    "out": 4,
    "doubtful": 3,
    "questionable": 2,
    "probable": 1,
    "available": 0,
}


async def fetch_sofascore_injuries(team_id_external: str) -> list[dict[str, Any]]:
    """Sofascore injuries endpoint por team externo ID.

    Returns list[{player_name, status, since}].
    """
    url = f"https://www.sofascore.com/api/v1/team/{team_id_external}/players/missing"
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(url, headers={"User-Agent": "apuestas/0.1"})
        if r.status_code != 200:
            return []
        data = r.json()
        out = []
        for p in data.get("missing", []):
            status_raw = str(p.get("reason", {}).get("status", "")).lower()
            # Map Sofascore status → SEVERITY keys
            if status_raw in ("doubtful", "out", "questionable", "probable"):
                status = status_raw
            elif "injur" in status_raw or "out" in status_raw:
                status = "out"
            else:
                status = "questionable"
            player = p.get("player", {}).get("name", "")
            if player:
                out.append({"player_name": player, "status": status})
        return out
    except Exception as exc:
        logger.debug("nba_injury.sofascore_fail", team=team_id_external, error=str(exc)[:80])
        return []


async def compute_team_injury_penalty(session: Any, team_id: int, as_of: datetime) -> float:
    """Penalidad escalar [0.0-0.3] según injuries en 8 rotation players del team.

    Lookup: nba_player_stats_rolling para identificar top 8 rotation,
    cross-ref con injury_reports tabla si existe.

    Fallback: 0.0 si no data.
    """
    try:
        # Check injury_reports table existence
        exists = (
            await session.execute(
                text(
                    "SELECT COUNT(*) n FROM information_schema.tables "
                    "WHERE table_name='injury_reports_normalized'"
                )
            )
        ).first()
        if not exists or exists.n == 0:
            return 0.0

        # Top 8 rotation: players con más minutos rolling 15d
        rows = (
            await session.execute(
                text(
                    """
                    SELECT ir.player_name, ir.status
                    FROM injury_reports_normalized ir
                    WHERE ir.team_id = :tid
                      AND ir.reported_at >= :window
                      AND ir.status IN ('out','doubtful','questionable')
                    """
                ),
                {"tid": team_id, "window": as_of - timedelta(hours=24)},
            )
        ).fetchall()

        penalty = 0.0
        for r in rows:
            sev = SEVERITY.get(str(r.status).lower(), 0)
            if sev >= 3:  # doubtful/out
                penalty += 0.08
            elif sev == 2:  # questionable
                penalty += 0.03
        return min(0.30, penalty)
    except Exception as exc:
        logger.debug("nba_injury.penalty_fail", team=team_id, error=str(exc)[:80])
        return 0.0


async def fetch_nba_injury_report_pdf(
    *, target_date: datetime | None = None
) -> list[dict[str, Any]]:
    """Stub: parse PDF oficial de NBA injury report.

    NBA.com publica un PDF ~3 veces/día (5pm/8pm/11pm ET) con OUT/DOUBTFUL/
    QUESTIONABLE confirmados de cada team antes del tip-off del día.

    URL pattern (ver nba.com Injury Report page):
        https://official.nba.com/wp-content/uploads/sites/4/{YYYY}/{MM}/{YYYY}-{MM}-{DD}_{HH}{PM_AM}_Injury-Report.pdf

    Implementación pendiente:
      1. Fetch listing page para descubrir el PDF más reciente del día
      2. Descargar con httpx (rate-limit, user-agent navegador)
      3. Parse con `pdfplumber` (mejor que pypdf para tablas) — extraer
         columnas: Game Date | Game Time | Matchup | Team | Player Name | Status | Reason
      4. Persist a injury_reports_normalized via persist_injuries()

    Por ahora retorna lista vacía + log info; NO causa false positives en el
    pipeline (compute_team_injury_penalty cae a Sofascore/ESPN). Se activa
    cuando alguien implemente el parser dedicado.

    Args:
        target_date: día del injury report (default: hoy UTC). Solo usado para
                    búsqueda; la implementación real elige el PDF más reciente.

    Returns:
        list[{team_name, player_name, status, reason}] vacío hasta implementar.
    """
    if target_date is None:
        target_date = datetime.now(tz=__import__("datetime").UTC)
    logger.info(
        "nba_injury.pdf_parser_not_implemented",
        target_date=target_date.date().isoformat(),
        reason="stub: implementar pdfplumber parser (ver docstring)",
    )
    return []


__all__ = [
    "SEVERITY",
    "compute_team_injury_penalty",
    "fetch_nba_injury_report_pdf",
    "fetch_sofascore_injuries",
]
