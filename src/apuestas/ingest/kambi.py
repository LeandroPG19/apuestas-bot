"""Kambi multi-operador CDN scraper — odds gratis sin auth.

API pública descubierta 2026-04-21. Responde desde MX sin VPN.
Sprint B abr-2026: generalizado a multi-operador. Cada operador cuenta como
un sharp/soft adicional en `line_shopping` y `market_consensus`.

Endpoint: https://eu-offering-api.kambicdn.com/offering/v2018/{operator}/listView/{sport}/{league}.json

Formato: odds europeas multiplicadas por 1000 (ej. 1540 = 1.540).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.ingest._match_resolver import resolve_or_create_match
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

# Operadores Kambi B2B que comparten la misma CDN. Cada uno produce odds
# distintas (mismo provider, distintos margins/ofertas) → sharp/soft adicionales
# para line_shopping. Validar HTTP 200 por operador antes de persistir.
OPERATORS: dict[str, str] = {
    "ub": "unibet",  # bookmaker label canónico
    "betsson": "betsson",
    "888sport": "888sport",
    "nordicbet": "nordicbet",
    "comeon": "comeon",
    "expekt": "expekt",
    "mariacasino": "mariacasino",
}

# Subset validado HTTP 200 desde MX (2026-04-25). El resto puede devolver
# 429/400 según rate-limit/región — el orquestador los tolera (rows=0 + warn).
DEFAULT_OPERATORS: tuple[str, ...] = ("ub", "comeon")

BASE_TEMPLATE = "https://eu-offering-api.kambicdn.com/offering/v2018/{op}/listView"

# Map interno sport_code → (sport_path, league_path) para Kambi
SPORT_MAP: dict[str, tuple[str, str]] = {
    "nba": ("basketball", "nba"),
    "nfl": ("american_football", "nfl"),
    "nhl": ("ice_hockey", "nhl"),
    "mlb": ("baseball", "mlb"),
    "soccer_epl": ("football", "england/premier_league"),
    "soccer_laliga": ("football", "spain/la_liga"),
    "soccer_seriea": ("football", "italy/serie_a"),
    "soccer_bundesliga": ("football", "germany/bundesliga"),
    "soccer_ligue1": ("football", "france/ligue_1"),
}


async def fetch_kambi_sport(
    sport_code: str, *, operator: str = "ub", timeout: float = 10.0
) -> list[dict[str, Any]]:
    """Descarga events JSON desde Kambi para un operador. Retorna lista raw."""
    mapping = SPORT_MAP.get(sport_code)
    if mapping is None:
        return []
    sport_path, league_path = mapping
    base = BASE_TEMPLATE.format(op=operator)
    url = f"{base}/{sport_path}/{league_path}.json?lang=en_US&market=US"
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                logger.info("kambi.non_200", sport=sport_code, op=operator, status=resp.status_code)
                return []
            data = resp.json()
    except Exception as exc:
        logger.info("kambi.fetch_fail", sport=sport_code, op=operator, error=str(exc)[:120])
        return []
    return data.get("events") or []


def parse_moneyline(event_raw: dict[str, Any]) -> tuple[str, str, datetime, float, float] | None:
    """Extrae (home, away, start, home_odds, away_odds) de un event Kambi."""
    ev = event_raw.get("event") or {}
    home = ev.get("homeName") or ev.get("englishName", "").split(" - ")[0]
    away = ev.get("awayName") or ev.get("englishName", "").split(" - ")[-1]
    start_raw = ev.get("start")
    if not (home and away and start_raw):
        return None
    try:
        start = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
    except ValueError:
        return None

    # Buscar betOffer con criterion "Moneyline" (o ENG equivalent)
    for bo in event_raw.get("betOffers") or []:
        criterion = (bo.get("criterion") or {}).get("label", "").lower()
        if "moneyline" not in criterion and "match" not in criterion:
            continue
        outcomes = bo.get("outcomes") or []
        if len(outcomes) != 2:
            continue
        home_odds: float | None = None
        away_odds: float | None = None
        for o in outcomes:
            # Kambi: odds están × 1000 (1540 = 1.540)
            decimal = float(o.get("odds", 0)) / 1000.0
            if decimal < 1.01:
                continue
            label = (o.get("label") or o.get("englishLabel") or "").lower()
            if home.lower() in label:
                home_odds = decimal
            elif away.lower() in label:
                away_odds = decimal
        if home_odds and away_odds:
            return home, away, start, home_odds, away_odds
    return None


async def ingest_kambi_sport(sport_code: str, *, operator: str = "ub", persist: bool = True) -> int:
    """Fetch + persist odds Kambi para un sport+operador. Retorna rows insertadas.

    El bookmaker en `odds_history` se etiqueta con el label canónico del
    operador (ub→unibet, betsson→betsson, ...) para que cada operador cuente
    como una source distinta en line_shopping/market_consensus.
    """
    events_raw = await fetch_kambi_sport(sport_code, operator=operator)
    if not events_raw:
        return 0

    bookmaker_label = OPERATORS.get(operator, operator)
    inserted = 0
    ts = datetime.now(tz=UTC)

    async with session_scope() as session:
        for ev_raw in events_raw:
            parsed = parse_moneyline(ev_raw)
            if parsed is None:
                continue
            home, away, start, home_odds, away_odds = parsed

            # Sanity: overround ∈ [0.97, 1.15]
            overround = 1.0 / home_odds + 1.0 / away_odds
            if overround < 0.97 or overround > 1.15:
                logger.info(
                    "kambi.rejected_anomalous",
                    home=home[:20],
                    away=away[:20],
                    op=operator,
                    overround=round(overround, 3),
                )
                continue

            match_id = await resolve_or_create_match(
                session,
                sport_code=_canonical_sport(sport_code),
                home_name=home,
                away_name=away,
                start_time=start,
                source=f"kambi:{operator}",
            )
            if match_id is None:
                continue

            if persist:
                for outcome, odds_val in (("home", home_odds), ("away", away_odds)):
                    await session.execute(
                        text(
                            """
                            INSERT INTO odds_history
                              (ts, match_id, bookmaker, market, outcome, line, odds)
                            VALUES (:ts, :mid, :bm, 'h2h', :oc, NULL, :od)
                            """
                        ),
                        {
                            "ts": ts,
                            "mid": match_id,
                            "bm": bookmaker_label,
                            "oc": outcome,
                            "od": odds_val,
                        },
                    )
                    inserted += 1

    logger.info(
        "kambi.persisted",
        sport=sport_code,
        op=operator,
        events=len(events_raw),
        rows=inserted,
    )
    return inserted


async def run_kambi_multi_operator(
    *,
    operators: list[str] | None = None,
    sports: list[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Itera todos los operadores × sports y agrega counts.

    Devuelve dict {operator: {sport: rows}}. Tolerante a fallos por operador.
    """
    ops = operators if operators is not None else list(OPERATORS.keys())
    sps = sports if sports is not None else list(SPORT_MAP.keys())
    out: dict[str, dict[str, int]] = {}
    for op in ops:
        op_results: dict[str, int] = {}
        for sp in sps:
            try:
                op_results[sp] = await ingest_kambi_sport(sp, operator=op)
            except Exception as exc:
                logger.warning("kambi.op_sport_fail", op=op, sport=sp, error=str(exc)[:120])
                op_results[sp] = 0
        out[op] = op_results
    return out


def _canonical_sport(sport_code: str) -> str:
    """sport_code Kambi → sport_code canónico del bot."""
    if sport_code.startswith("soccer_"):
        return "soccer"
    return sport_code
