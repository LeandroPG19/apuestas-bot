"""Pinnacle Sports — guest API scraping (§7 del plan, cierre sharp gratis).

La API oficial de Pinnacle para el público general se cerró el 23-jul-2025,
pero el endpoint `guest.api.arcadia.pinnacle.com` que sirve al sitio web sigue
disponible con un token guest fijo. Cubre 100% de deportes principales y los
prices son LOS MISMOS que el book real (closing line sharp), por lo que es la
fuente gold standard para de-vigging Shin (§7) y cálculo de CLV.

Legalidad: el token es público y el scraping de data pública se considera OK
en la mayoría de jurisdicciones (robots.txt no lo bloquea). No compra/apuesta
automatizada — solo lectura.

Endpoints:
    GET /0.1/sports                                → lista de deportes
    GET /0.1/leagues/{league_id}/matchups          → eventos por liga
    GET /0.1/leagues/{league_id}/markets/straight  → mercados (moneyline, total, spread)

Riesgos:
- El token puede rotar en cualquier momento (~cada 6-12 meses históricamente).
  Si empieza a dar 401, chequear network tab del sitio y actualizar `_GUEST_KEY`.
- Rate-limit IP: conservative 30 req/min para no triggear bloqueo.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.ingest.http_base import BaseAPIClient
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

# Token guest público extraído del bundle JavaScript del sitio (pinnacle.com).
# Si Pinnacle cambia esto, hay que actualizarlo. Ver:
#   https://www.pinnacle.com/_next/static/chunks/ → buscar "X-API-Key"
_GUEST_KEY = "CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R"

# Pinnacle sport_ids son estables (nombres oficiales). Los league_ids bajo
# cada sport se auto-discoveran en runtime desde la API — NO hardcoded, así
# cuando Pinnacle cambia una liga de ID (por temporada o re-org), el bot se
# adapta automáticamente sin rebuild.
PINNACLE_SPORT_IDS: dict[str, int] = {
    "baseball": 3,
    "basketball": 4,
    "boxing": 6,
    "football": 15,  # American football (NFL, NCAA)
    "hockey": 19,
    "soccer": 29,
    "tennis": 33,
    "mma": 22,
    "cricket": 8,
    "esports": 12,
}

# Filtros opcionales: sport_code → patrón regex a matchear contra league.name
# Si un sport_code no está aquí, se devuelven TODOS los leagues con matchupCount>0.
_LEAGUE_NAME_FILTERS: dict[str, str] = {
    "nba": r"^NBA$",
    "mlb": r"^MLB$",
    "nfl": r"^NFL$",
    "nhl": r"^NHL$",
    "soccer_epl": r"England - Premier League",
    "soccer_laliga": r"Spain - La Liga",
    "soccer_bundesliga": r"Germany - Bundesliga",
    "soccer_seriea": r"Italy - Serie A",
    "soccer_ligue1": r"France - Ligue 1",
    "soccer_ucl": r"UEFA[\s\-]+Champions League(?! Women)",
    "soccer_uel": r"UEFA[\s\-]+Europa League",
    "soccer_uecl": r"UEFA[\s\-]+Conference League",
    "soccer_liga_mx": r"Mexico - Liga MX$",
    "soccer_mls": r"USA - Major League Soccer",
    "soccer_world_cup": r"FIFA - World Cup",
    "soccer_eredivisie": r"Netherlands - Eredivisie",
    "soccer_brasil": r"Brazil - Serie A",
    "soccer_argentina": r"Argentina - Liga Pro",
    "soccer_concacaf_cl": r"CONCACAF - Champions",
    "soccer_libertadores": r"CONMEBOL - Copa Libertadores",
    "soccer_sudamericana": r"CONMEBOL - Copa Sudamericana",
}

# sport_code → sport_id para saber en qué /sports/{id}/leagues buscar.
_SPORT_CODE_TO_PINNACLE_ID: dict[str, int] = {
    "nba": 4,
    "mlb": 3,
    "nfl": 15,
    "nhl": 19,
    "soccer_epl": 29,
    "soccer_laliga": 29,
    "soccer_bundesliga": 29,
    "soccer_seriea": 29,
    "soccer_ligue1": 29,
    "soccer_ucl": 29,
    "soccer_uel": 29,
    "soccer_uecl": 29,
    "soccer_liga_mx": 29,
    "soccer_mls": 29,
    "soccer_world_cup": 29,
    "soccer_eredivisie": 29,
    "soccer_brasil": 29,
    "soccer_argentina": 29,
    "soccer_concacaf_cl": 29,
    "soccer_libertadores": 29,
    "soccer_sudamericana": 29,
    "soccer": 29,  # genérico — todas las leagues de soccer activas
    "basketball": 4,
    "baseball": 3,
    "hockey": 19,
    "football": 15,
    "boxing": 6,
    "tennis": 33,
    "mma": 22,
}


async def discover_leagues(sport_code: str, max_leagues: int = 15) -> list[int]:
    """Auto-descubre league_ids activos en Pinnacle para un sport_code.

    - Consulta `/sports/{pinnacle_sport_id}/leagues` en vivo.
    - Filtra leagues con `matchupCount > 0` (activos).
    - Si `sport_code` tiene entry en `_LEAGUE_NAME_FILTERS`, aplica regex para
      quedarse con la(s) liga(s) específica(s); si no, devuelve todas activas.
    - Ordena por matchupCount desc para priorizar las más pobladas.

    Returns: lista de league_ids ordenados por actividad.
    """
    import re

    pinnacle_sport_id = _SPORT_CODE_TO_PINNACLE_ID.get(sport_code)
    if pinnacle_sport_id is None:
        logger.warning("pinnacle.unknown_sport_code", sport_code=sport_code)
        return []

    client = PinnacleClient()
    async with client.session():
        try:
            leagues = await client.get(f"/sports/{pinnacle_sport_id}/leagues")
        except Exception as exc:
            logger.warning("pinnacle.discover_fail", sport=sport_code, error=str(exc)[:80])
            return []

    # Filtro de actividad
    active = [lg for lg in leagues if (lg.get("matchupCount") or 0) > 0]

    # Filtro por nombre si el sport_code es específico
    pattern = _LEAGUE_NAME_FILTERS.get(sport_code)
    if pattern:
        rx = re.compile(pattern, re.IGNORECASE)
        active = [lg for lg in active if rx.search(lg.get("name") or "")]

    # Orden por actividad (más matchups primero)
    active.sort(key=lambda lg: -(lg.get("matchupCount") or 0))
    return [int(lg["id"]) for lg in active[:max_leagues]]


@dataclass(slots=True, frozen=True)
class PinnacleMatchup:
    """Matchup pre-match de Pinnacle."""

    matchup_id: int
    league_id: int
    league_name: str
    home: str
    away: str
    start_time: datetime
    status: str
    has_markets: bool


@dataclass(slots=True, frozen=True)
class PinnacleOdds:
    """Una línea de mercado (moneyline, total o spread)."""

    matchup_id: int
    market_type: str  # moneyline | total | spread
    period: int  # 0 = full game · 1 = first half etc.
    outcome: str  # home | away | over | under
    price_american: int
    points: float | None = None


def american_to_decimal(price: int) -> float:
    """Convierte odds americanas (-108, +142) a decimal (1.926, 2.42)."""
    if price > 0:
        return round(price / 100 + 1, 4)
    return round(100 / abs(price) + 1, 4)


class PinnacleClient(BaseAPIClient):
    """Cliente guest Pinnacle. Sin credenciales, solo header X-API-Key."""

    base_url = "https://guest.api.arcadia.pinnacle.com/0.1"
    source_name = "pinnacle_guest"
    rate_limit = (30, 60.0)  # 30 req/min conservative

    def _default_headers(self) -> dict[str, str]:
        return {
            "User-Agent": "Mozilla/5.0 (apuestas-bot research)",
            "X-API-Key": _GUEST_KEY,
            "Accept": "application/json",
            "Referer": "https://www.pinnacle.com/",
        }

    async def fetch_sports(self) -> list[dict[str, Any]]:
        return await self.get("/sports")

    async def fetch_matchups(self, league_id: int) -> list[dict[str, Any]]:
        """Todos los matchups pending + live de una liga."""
        return await self.get(f"/leagues/{league_id}/matchups")

    async def fetch_markets_straight(self, league_id: int) -> list[dict[str, Any]]:
        """Mercados straight (moneyline, total, spread) de una liga."""
        return await self.get(f"/leagues/{league_id}/markets/straight")


_PROP_NAME_PREFIXES = (
    "Over ",
    "Under ",
    "Home Goals",
    "Away Goals",
    "Home Corners",
    "Away Corners",
    "Home Cards",
    "Away Cards",
    "Total Goals",
    "Total Corners",
    "Total Cards",
    "1st Half",
    "2nd Half",
    "Half ",
    "1st Quarter",
    "2nd Quarter",
    "3rd Quarter",
    "4th Quarter",
    "Quarter ",
    "Period ",
    "Inning ",
    "1st Inning",
    "5 Innings",
    "F5",
    "Set ",
    "1st Set",
    "2nd Set",
    "3rd Set",
    "Race to",
)
_PROP_EXACT_NAMES = frozenset(
    {
        "Over",
        "Under",
        "Yes",
        "No",
        "Odd",
        "Even",
        "Draw",
        "Tie",
        "None",
        "Corner",
        "Card",
    }
)


def parse_matchup(raw: dict[str, Any]) -> PinnacleMatchup | None:
    """Parsea un matchup raw. Filtra props y mercados que NO son matches reales.

    Pinnacle retorna en `/matchups` tanto partidos 1v1 como:
    - Totals ('Over X.5' / 'Under X.5')
    - Props stat ('Home Goals (N Games)', 'Home Corners', etc.)
    - Parity bets ('Odd vs Even')
    - Yes/No futures
    Estos NO son match odds y NO deben entrar al pipeline de picks core.
    """
    participants = raw.get("participants") or []
    names = [str(p.get("name", "")) for p in participants if isinstance(p, dict)]

    if len(names) != 2:
        return None

    # Filtro 1: prefijos de prop markets
    if any(n.startswith(_PROP_NAME_PREFIXES) for n in names):
        return None

    # Filtro 2: nombres exactos de props/parity/futures
    if any(n in _PROP_EXACT_NAMES for n in names):
        return None

    # Filtro 3: patrón "X Team vs Y Team" pero con spread explícito ("Flyers +1.5 Games")
    # Son series bets de playoffs, válidos pero el parser core los maneja aparte
    if any(" +" in n or " -" in n for n in names if " Games" in n):
        return None

    league = raw.get("league") or {}
    start_raw = raw.get("startTime")
    if not start_raw:
        return None
    try:
        start_time = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
    except (TypeError, ValueError):  # fmt: skip
        return None

    return PinnacleMatchup(
        matchup_id=int(raw["id"]),
        league_id=int(league.get("id", 0)),
        league_name=str(league.get("name", "")),
        home=str(names[0]),
        away=str(names[1]),
        start_time=start_time,
        status=str(raw.get("status", "pending")),
        has_markets=bool(raw.get("hasMarkets", False)),
    )


def parse_market(
    raw: dict[str, Any],
    participants_by_id: dict[int, str] | None = None,
) -> list[PinnacleOdds]:
    """Un market raw puede tener 2-3 prices. Los convertimos a PinnacleOdds.

    - moneyline: solo aceptamos exactamente 2 prices (match 1v1).
      Si tiene N>2, es futures/MVP/championship → skip.
    - total: 2 prices over/under
    - spread: 2 prices home/away
    """
    m_type = raw.get("type")
    if m_type not in ("moneyline", "total", "spread"):
        return []
    # Pinnacle retorna multiples markets del mismo type+period cuando hay
    # "alternate lines" (ej. NHL: 3-way moneyline + 2-way OT moneyline ambos
    # marcados como type=moneyline). Solo queremos el "standard" (no alternate).
    # Detectamos alternate via `isAlternate=true` o `cutoffAt` distinto al match start.
    if raw.get("isAlternate") is True:
        return []
    matchup_id = int(raw.get("matchupId", 0))
    period = int(raw.get("period", 0))
    prices = raw.get("prices") or []
    pbi = participants_by_id or {}

    # Moneyline con >2 prices = mercado de futures, no interesa para match odds
    if m_type == "moneyline" and len(prices) != 2:
        return []
    if m_type in ("total", "spread") and len(prices) != 2:
        return []

    out: list[PinnacleOdds] = []
    for i, price in enumerate(prices):
        pid = price.get("participantId")
        pts = price.get("points")
        american = price.get("price")
        if american is None:
            continue
        if m_type == "moneyline":
            # Intentar mapear via participants_by_id (home=primero en matchups),
            # fallback a orden del array
            mapped = pbi.get(pid) if pid is not None else None
            outcome = mapped if mapped in ("home", "away") else ("home" if i == 0 else "away")
        elif m_type == "total":
            outcome = "over" if i == 0 else "under"
        else:  # spread
            outcome = "home" if i == 0 else "away"
        out.append(
            PinnacleOdds(
                matchup_id=matchup_id,
                market_type=m_type,
                period=period,
                outcome=outcome,
                price_american=int(american),
                points=float(pts) if pts is not None else None,
            )
        )
    return out


async def ingest_league(
    sport_code: str,
    *,
    persist: bool = False,
    max_leagues: int = 15,
) -> tuple[list[PinnacleMatchup], list[PinnacleOdds]]:
    """Descarga matchups + mercados de UN sport_code (puede cubrir varias leagues).

    Auto-discover en runtime con `discover_leagues`: consulta API Pinnacle,
    filtra leagues activas (matchupCount>0), aplica filtro de nombre si el
    sport_code es específico. Cuando cambia la temporada / Pinnacle reasigna
    league_ids, el bot se adapta sin rebuild.

    Returns: (matchups, odds) agregados de todas las leagues del sport_code.
    """
    league_ids = await discover_leagues(sport_code, max_leagues=max_leagues)
    if not league_ids:
        logger.info("pinnacle.no_active_leagues", sport_code=sport_code)
        return [], []

    all_matchups: list[PinnacleMatchup] = []
    all_odds: list[PinnacleOdds] = []
    participants_by_id: dict[int, str] = {}

    client = PinnacleClient()
    async with client.session():
        for league_id in league_ids:
            try:
                matchups_raw = await client.fetch_matchups(league_id)
                markets_raw = await client.fetch_markets_straight(league_id)
            except Exception as exc:
                logger.warning(
                    "pinnacle.fetch_failed",
                    sport=sport_code,
                    league_id=league_id,
                    error=str(exc)[:120],
                )
                continue

            for raw in matchups_raw:
                parsed = parse_matchup(raw)
                if parsed is not None:
                    all_matchups.append(parsed)
                for i, p in enumerate(raw.get("participants") or []):
                    pid = p.get("id")
                    if pid is not None:
                        participants_by_id[pid] = "home" if i == 0 else "away"

            for raw in markets_raw:
                all_odds.extend(parse_market(raw, participants_by_id))

    logger.info(
        "pinnacle.ingest.done",
        sport=sport_code,
        leagues_scanned=len(league_ids),
        matchups=len(all_matchups),
        odds=len(all_odds),
    )

    if persist and all_odds:
        await _persist_odds(sport_code, all_matchups, all_odds)

    return all_matchups, all_odds


_SPORT_CANONICAL_MAP: dict[str, str] = {
    "soccer_epl": "soccer",
    "soccer_laliga": "soccer",
    "soccer_bundesliga": "soccer",
    "soccer_seriea": "soccer",
    "soccer_ligue1": "soccer",
    "soccer_ucl": "soccer",
    "soccer_liga_mx": "soccer",
    "soccer_mls": "soccer",
    "soccer_world_cup": "soccer",
    "nba": "nba",
    "mlb": "mlb",
    "nfl": "nfl",
    "nhl": "nhl",
    "tennis": "tennis",
    "boxing": "boxing",
    "mma": "mma",
}


def _canonical_sport_code(sport_code: str) -> str:
    """Mapea sport_codes específicos de Pinnacle al catálogo `sports.code` de la DB.

    Evita FK violation cuando `sport_code='soccer_epl'` pero la tabla `sports`
    solo conoce `'soccer'`. Si el código ya es canónico se devuelve tal cual.

    Fallback automático: cualquier `soccer_*` no listado mapea a `soccer`. Sin
    esto, los sport_codes nuevos (`soccer_uecl`, `soccer_brasil`,
    `soccer_libertadores`, `soccer_sudamericana`, `soccer_eredivisie`, etc.)
    abortaban el INSERT por FK violation contra `sports.code`.
    """
    mapped = _SPORT_CANONICAL_MAP.get(sport_code)
    if mapped is not None:
        return mapped
    if sport_code.startswith("soccer_"):
        return "soccer"
    return sport_code


async def _persist_odds(
    sport_code: str,
    matchups: list[PinnacleMatchup],
    odds: list[PinnacleOdds],
) -> None:
    """Inserta snapshots en odds_history con bookmaker='pinnacle'.

    Resuelve cada matchup a un match_id: primero busca por external_id exacto,
    si no existe usa el resolver compartido (fuzzy pg_trgm + crea match/teams
    faltantes). Así el scraper no requiere ingesta previa para persistir.
    """
    from apuestas.ingest._match_resolver import resolve_or_create_match

    # Normaliza el sport_code a los codes canónicos del enum sports(code):
    # soccer_*, tennis, boxing, mma → los genéricos sin prefijo.
    canonical_sport = _canonical_sport_code(sport_code)

    matchup_to_match: dict[int, int] = {}
    async with session_scope() as session:
        for m in matchups:
            # 1) Exact match by external_id (rápido, evita fuzzy en runs repetidos)
            r = await session.execute(
                text("SELECT id FROM matches WHERE external_id = :ext LIMIT 1"),
                {"ext": f"pinnacle:{m.matchup_id}"},
            )
            row = r.first()
            if row is not None:
                matchup_to_match[m.matchup_id] = int(row[0])
                continue
            # 2) Fuzzy resolve (crea match + teams si faltan) + marca external_id
            match_id = await resolve_or_create_match(
                session,
                sport_code=canonical_sport,
                home_name=m.home,
                away_name=m.away,
                start_time=m.start_time,
                source="pinnacle",
            )
            if match_id is None:
                continue
            # Marca el external_id canonical pinnacle para acelerar próximas llamadas
            await session.execute(
                text(
                    "UPDATE matches SET external_id = :ext "
                    "WHERE id = :id AND (external_id IS NULL OR external_id NOT LIKE 'pinnacle:%')"
                ),
                {"ext": f"pinnacle:{m.matchup_id}", "id": match_id},
            )
            matchup_to_match[m.matchup_id] = match_id

    # Sanity check: agrupa odds por (matchup_id, market_type, period, line) y valida
    # overround ∈ [0.97, 1.15]. Descarta pairs con overround anómalo (odds corruptas).
    from collections import defaultdict as _dd

    pairs: dict[tuple[int, str, int, float | None], list[PinnacleOdds]] = _dd(list)
    for o in odds:
        key = (o.matchup_id, o.market_type, o.period, o.points)
        pairs[key].append(o)

    valid_odds: list[PinnacleOdds] = []
    rejected_anomalous = 0
    for key, entries in pairs.items():
        if len(entries) < 2:
            valid_odds.extend(entries)  # single (raro, pero deja pasar)
            continue
        overround = sum(1.0 / american_to_decimal(e.price_american) for e in entries)
        if 0.97 <= overround <= 1.15:
            valid_odds.extend(entries)
        else:
            rejected_anomalous += len(entries)
            logger.info(
                "pinnacle.rejected_anomalous_overround",
                sport=sport_code,
                matchup=key[0],
                market=key[1],
                overround=round(overround, 3),
            )

    ts = datetime.now(tz=UTC)
    inserted = 0
    fk_violations = 0
    # Pre-filter: solo conservar match_ids que efectivamente existen en `matches`.
    # Sin esto, identity_repair (que purga orphan duplicates) puede haber borrado
    # un match entre construcción de `matchup_to_match` y el INSERT → FK violation
    # aborta la TRANSACCIÓN ENTERA y todos los demás odds del sport se pierden.
    candidate_ids = {
        matchup_to_match[o.matchup_id]
        for o in valid_odds
        if o.matchup_id in matchup_to_match and o.period == 0
    }
    async with session_scope() as session:
        if candidate_ids:
            existing_rows = await session.execute(
                text("SELECT id FROM matches WHERE id = ANY(:ids)"),
                {"ids": list(candidate_ids)},
            )
            existing_ids = {int(r[0]) for r in existing_rows}
        else:
            existing_ids = set()

        for o in valid_odds:
            match_id = matchup_to_match.get(o.matchup_id)
            if match_id is None or o.period != 0:
                continue
            if match_id not in existing_ids:
                fk_violations += 1
                continue
            # Solo full-game (period=0)
            market_map = {"moneyline": "h2h", "total": "totals", "spread": "spreads"}
            market = market_map[o.market_type]
            decimal = american_to_decimal(o.price_american)
            await session.execute(
                text(
                    """
                    INSERT INTO odds_history
                      (ts, match_id, bookmaker, market, outcome, line, odds)
                    VALUES (:ts, :mid, 'pinnacle', :mk, :oc, :ln, :od)
                    """
                ),
                {
                    "ts": ts,
                    "mid": match_id,
                    "mk": market,
                    "oc": o.outcome,
                    "ln": o.points,
                    "od": decimal,
                },
            )
            inserted += 1
    logger.info(
        "pinnacle.persisted",
        sport=sport_code,
        rows=inserted,
        rejected_anomalous=rejected_anomalous,
        fk_violations=fk_violations,
    )
