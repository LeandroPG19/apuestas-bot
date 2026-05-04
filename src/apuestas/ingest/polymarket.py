"""Polymarket ingest — mercados de predicción sobre eventos deportivos.

Polymarket tiene 2 categorías de markets útiles:
- **Futures** (NBA MVP, Champions League winner): persistidos en
  `polymarket_markets` como benchmark de fair value.
- **Game-by-game** (h2h sobre matches específicos): persistidos en
  `odds_history` con `bookmaker='polymarket'` para que `line_shopping` y
  `market_consensus` los traten como una fuente sharp más.

CLOB: precios = midpoint(yes_bid, yes_ask) ∈ [0,1] = prob implícita pura
(no bookmaker margin). Sprint B abr-2026.

APIs gratis sin auth:
  https://gamma-api.polymarket.com/markets   ← discovery
  https://clob.polymarket.com/midpoint        ← precio actual
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


BASE_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"


SPORT_TAG_MAP: dict[str, list[str]] = {
    "nba": ["nba", "basketball"],
    "nfl": ["nfl", "football", "super-bowl"],
    "mlb": ["mlb", "baseball", "world-series"],
    "soccer": ["soccer", "world-cup", "champions-league", "premier-league"],
    "boxing": ["boxing"],
    "tennis": ["tennis", "grand-slam"],
    "nhl": ["nhl", "hockey"],
}


async def fetch_polymarket_active(sport_code: str | None = None) -> list[dict[str, Any]]:
    """Trae mercados activos. Si sport_code, filtra por tags sport."""
    tags = SPORT_TAG_MAP.get(sport_code or "", [])
    params: dict[str, Any] = {"active": "true", "closed": "false", "limit": "200"}
    if tags:
        params["tag_slug"] = tags[0]

    async with httpx.AsyncClient(
        timeout=15,
        headers={"User-Agent": "apuestas-bot/0.1", "Accept": "application/json"},
    ) as c:
        try:
            r = await c.get(f"{BASE_URL}/markets", params=params)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            logger.warning("polymarket.fetch_fail", error=str(exc))
            return []

    markets = data if isinstance(data, list) else data.get("data", [])
    logger.info("polymarket.fetched", sport=sport_code, count=len(markets))
    return markets


async def persist_markets(markets: list[dict[str, Any]], sport_code: str) -> int:
    """Persiste markets con outcomes + current_prices."""
    if not markets:
        return 0
    inserted = 0
    async with session_scope() as s:
        for m in markets:
            import json as _json

            cond_id = m.get("conditionId") or m.get("id")
            if not cond_id:
                continue
            question = m.get("question", "")[:500]
            event_type = _infer_event_type(question)
            end_str = m.get("endDate") or m.get("end_date_iso")
            end_dt = _parse_ts(end_str)

            outcomes_list = m.get("outcomes", [])
            prices_list = m.get("outcomePrices", m.get("clobTokenIds", []))
            outcomes_json = _json.dumps(outcomes_list if isinstance(outcomes_list, list) else [])
            current_json = _json.dumps(
                dict(zip(outcomes_list, prices_list, strict=False))
                if isinstance(outcomes_list, list) and isinstance(prices_list, list)
                else {}
            )
            volume = float(m.get("volume24hr", m.get("volume", 0)) or 0)

            try:
                await s.execute(
                    text(
                        """
                        INSERT INTO polymarket_markets
                            (condition_id, question, sport_code, event_type,
                             end_date, outcomes, current_prices, volume_24h_usd,
                             last_updated)
                        VALUES (:c, :q, :s, :et, :ed, CAST(:o AS jsonb),
                                CAST(:p AS jsonb), :v, NOW())
                        ON CONFLICT (condition_id) DO UPDATE SET
                            current_prices = EXCLUDED.current_prices,
                            volume_24h_usd = EXCLUDED.volume_24h_usd,
                            last_updated = NOW()
                        """
                    ),
                    {
                        "c": str(cond_id)[:100],
                        "q": question,
                        "s": sport_code,
                        "et": event_type,
                        "ed": end_dt,
                        "o": outcomes_json,
                        "p": current_json,
                        "v": volume,
                    },
                )
                inserted += 1
            except Exception as exc:
                logger.debug("polymarket.persist_fail", error=str(exc))
    return inserted


def _infer_event_type(question: str) -> str:
    low = question.lower()
    if "mvp" in low:
        return "mvp"
    if "ballon" in low:
        return "ballon_dor"
    if "champion" in low or "win the" in low:
        return "champion"
    if "cy young" in low:
        return "cy_young"
    if "rookie" in low:
        return "roy"
    return "other"


def _parse_ts(s: Any) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None


async def run_ingest() -> dict[str, int]:
    """Orquestador: trae mercados para todos los sports soportados."""
    results: dict[str, int] = {}
    for sport in ("nba", "nfl", "mlb", "soccer", "boxing", "tennis", "nhl"):
        markets = await fetch_polymarket_active(sport)
        results[sport] = await persist_markets(markets, sport)
    return results


# ─────────────────── Sprint B abr-2026: game-by-game markets ───────────────────


import re as _re
from datetime import UTC as _UTC
from datetime import timedelta as _timedelta

_VS_RE = _re.compile(r"\b(?:vs\.?|@|versus)\b", flags=_re.IGNORECASE)


def _normalize_team_name(name: str) -> str:
    """Normaliza para fuzzy match: lowercase + alphanumeric + colapsa whitespace."""
    s = _re.sub(r"[^a-z0-9 ]", "", name.lower())
    return _re.sub(r"\s+", " ", s).strip()


def _question_team_tokens(question: str) -> list[str]:
    """Extrae tokens 'TeamA' y 'TeamB' de una pregunta tipo 'Lakers vs Warriors?'.

    Heurística simple — Polymarket usa templates consistentes para game markets
    de NBA/NHL/MLB ("Will the Lakers beat the Warriors?", "Lakers @ Warriors",
    "Lakers vs Warriors"). Devuelve [] si no se detecta el patrón.
    """
    parts = _VS_RE.split(question)
    if len(parts) != 2:
        return []
    return [_normalize_team_name(parts[0]), _normalize_team_name(parts[1])]


_WIN_RE = _re.compile(r"^will\s+(.+?)\s+win\b", flags=_re.IGNORECASE)
_DRAW_RE = _re.compile(r"\bend\s+in\s+a\s+draw\b|\btie\b", flags=_re.IGNORECASE)
_NON_H2H_KEYWORDS = ("over", "under", "total", "prop", "first", "highest", "lowest", "more than")


def _is_h2h_win_question(question: str) -> tuple[bool, str | None]:
    """Detecta si question es del tipo 'Will TeamX win [vs/over TeamY]?'.

    Returns (is_h2h, winning_team_norm). winning_team_norm es el equipo cuyo
    "Yes" significa victoria. Rechaza draws, totals, props.
    """
    if _DRAW_RE.search(question):
        return False, None
    low = question.lower()
    if any(kw in low for kw in _NON_H2H_KEYWORDS):
        return False, None
    m = _WIN_RE.match(question)
    if not m:
        return False, None
    # winning_team puede ser "the Lakers" o "Lakers"; limpiar artículos comunes
    raw = m.group(1).strip()
    raw = _re.sub(r"^(the|los|las|el|la)\s+", "", raw, flags=_re.IGNORECASE)
    return True, _normalize_team_name(raw)


async def _fetch_clob_midpoint(client: httpx.AsyncClient, *, token_id: str) -> float | None:
    """Trae midpoint actual del CLOB para un token Yes (probabilidad implícita)."""
    try:
        r = await client.get(f"{CLOB_URL}/midpoint", params={"token_id": token_id}, timeout=8.0)
        r.raise_for_status()
        data = r.json()
        mid = data.get("mid")
        if mid is None:
            return None
        return float(mid)
    except Exception as exc:
        logger.debug("polymarket.clob_fail", token=token_id[:16], error=str(exc)[:80])
        return None


async def _load_upcoming_matches(
    sport_codes: list[str], hours_ahead: int = 48
) -> list[dict[str, Any]]:
    """Trae upcoming matches con nombres normalizados para fuzzy match."""
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT m.id, m.sport_code, m.start_time,
                       t1.name AS home_name, t2.name AS away_name
                FROM matches m
                JOIN teams t1 ON t1.id = m.home_team_id
                JOIN teams t2 ON t2.id = m.away_team_id
                WHERE m.sport_code = ANY(:sports)
                  AND m.status IN ('scheduled', 'live')
                  AND m.start_time BETWEEN NOW() AND NOW() + (:h || ' hours')::interval
                """
            ),
            {"sports": sport_codes, "h": str(hours_ahead)},
        )
        rows = []
        for r in result.all():
            d = dict(r._mapping)
            d["home_norm"] = _normalize_team_name(d["home_name"] or "")
            d["away_norm"] = _normalize_team_name(d["away_name"] or "")
            rows.append(d)
        return rows


def _match_question_to_event(
    question: str, upcoming: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Busca match cuya pareja (home, away) coincida con tokens del question.

    Acepta cualquier orden (Polymarket no garantiza home-first) y match parcial
    (el token de Polymarket puede ser "Lakers" mientras que el match es
    "Los Angeles Lakers"). Retorna el match dict o None.
    """
    tokens = _question_team_tokens(question)
    if not tokens:
        return None
    t1, t2 = tokens
    for m in upcoming:
        h = m["home_norm"]
        a = m["away_norm"]
        # Containment bidireccional (cubre 'Lakers' ↔ 'Los Angeles Lakers')
        if ((t1 in h or h in t1) and (t2 in a or a in t2)) or (
            (t1 in a or a in t1) and (t2 in h or h in t2)
        ):
            return m
    return None


def _odds_from_prob(prob: float, *, min_prob: float = 0.01) -> float | None:
    """Convierte probabilidad ∈ [0,1] a odds decimales (1/p). Filtra casos
    extremos para no violar el CHECK ck_odds_positive (odds > 1.0)."""
    if prob is None or prob <= min_prob or prob >= 1.0:
        return None
    return round(1.0 / prob, 4)


async def fetch_game_markets(
    *, sport_code: str, hours_ahead: int = 48, max_markets: int = 200
) -> list[dict[str, Any]]:
    """Trae markets game-by-game para `sport_code` con endDate ∈ ventana próxima.

    Estrategia validada (2026-04-26): el endpoint `/markets` con `tag_slug`
    NO filtra correctamente. El filtro real está en `/events` cuyos items
    incluyen `tags: [{slug, label}]` confiables. Para cada event que matchea
    el sport, devolvemos sus markets internos.
    """
    now = datetime.now(tz=_UTC)
    end_max = now + _timedelta(hours=hours_ahead)
    params: dict[str, Any] = {
        "active": "true",
        "closed": "false",
        "limit": str(max_markets),
        "end_date_min": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_date_max": end_max.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    sport_slugs = set(SPORT_TAG_MAP.get(sport_code, []))
    if not sport_slugs:
        return []

    async with httpx.AsyncClient(
        timeout=15,
        headers={"User-Agent": "apuestas-bot/0.1", "Accept": "application/json"},
    ) as c:
        try:
            r = await c.get(f"{BASE_URL}/events", params=params)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            logger.warning("polymarket.fetch_games_fail", sport=sport_code, error=str(exc)[:120])
            return []

    events = data if isinstance(data, list) else (data.get("data", []) or [])

    # Filtro client-side por tags['slug']. Cada event puede tener varios
    # markets (h2h, spreads, props); aplanamos a lista de markets con
    # heredamos el campo `event_tags` para downstream filters.
    flat_markets: list[dict[str, Any]] = []
    for ev in events:
        ev_tags = ev.get("tags") or []
        ev_slugs = {t.get("slug", "") for t in ev_tags if isinstance(t, dict)}
        if not (ev_slugs & sport_slugs):
            continue
        ev_markets = ev.get("markets") or []
        for m in ev_markets:
            if not isinstance(m, dict):
                continue
            m["_event_title"] = ev.get("title", "")
            m["_event_slugs"] = list(ev_slugs)
            flat_markets.append(m)
    return flat_markets[:max_markets]


async def _persist_game_odds_history(
    *,
    match_id: int,
    yes_prob: float,
    no_prob: float | None,
    home_is_yes: bool,
    sport_code: str = "",
    condition_id: str = "",
    question: str = "",
    yes_token_id: str = "",
    volume_usd: float | None = None,
    end_date: datetime | None = None,
) -> int:
    """Persiste snapshot Polymarket en DOS sinks:
    - `odds_history` (h2h) → consumido por `line_shopping` / `best_odds`
    - `polymarket_prices` (midpoint) → consumido por `consensus_fetch.fetch_polymarket_midpoint`
      → `market_consensus` (3-source sharp).

    Sin esto, el game ingester solo alimentaba line_shopping pero
    `market_consensus` no veía Polymarket games (solo futures via `polymarket_markets`).
    """
    home_prob = yes_prob if home_is_yes else (no_prob if no_prob is not None else 1 - yes_prob)
    away_prob = (1 - home_prob) if no_prob is None else (no_prob if home_is_yes else yes_prob)

    odds_home = _odds_from_prob(home_prob)
    odds_away = _odds_from_prob(away_prob)
    if odds_home is None or odds_away is None:
        return 0
    now = datetime.now(tz=_UTC)
    rows_inserted = 0
    async with session_scope() as session:
        # 1) odds_history (h2h) para line_shopping
        for outcome, odds in (("home", odds_home), ("away", odds_away)):
            try:
                await session.execute(
                    text(
                        """
                        INSERT INTO odds_history
                          (ts, match_id, bookmaker, market, outcome, line, odds, is_closing)
                        VALUES
                          (:ts, :mid, 'polymarket', 'h2h', :oc, NULL, :odds, false)
                        """
                    ),
                    {"ts": now, "mid": match_id, "oc": outcome, "odds": odds},
                )
                rows_inserted += 1
            except Exception as exc:
                logger.debug("polymarket.persist_odds_fail", error=str(exc)[:120])

        # 2) polymarket_prices (midpoint Yes=home_win) para market_consensus
        if condition_id and yes_token_id and sport_code:
            try:
                await session.execute(
                    text(
                        """
                        INSERT INTO polymarket_prices
                          (condition_id, question, sport, token_id,
                           midpoint, volume_usd, end_date, captured_at)
                        VALUES
                          (:cid, :q, :sp, :tid, :mid, :vol, :ed, :ts)
                        """
                    ),
                    {
                        "cid": str(condition_id)[:100],
                        "q": str(question)[:500],
                        "sp": sport_code.lower(),
                        "tid": str(yes_token_id),
                        "mid": float(home_prob),  # home_win midpoint
                        "vol": volume_usd,
                        "ed": end_date,
                        "ts": now,
                    },
                )
            except Exception as exc:
                logger.debug("polymarket.persist_prices_fail", error=str(exc)[:120])
    return rows_inserted


async def ingest_game_markets(*, hours_ahead: int = 48) -> dict[str, int]:
    """Pipeline completo: fetch Gamma → match a upcoming → CLOB midpoint → odds_history.

    Itera por sport (nba, nhl, mlb, nfl, soccer). Por cada market que matcha
    a un upcoming match, snapshota la prob actual desde el CLOB y persiste
    en `odds_history` con bookmaker='polymarket'.
    """
    sports = ["nba", "nhl", "mlb", "nfl", "soccer"]
    upcoming = await _load_upcoming_matches(sports, hours_ahead=hours_ahead)
    if not upcoming:
        logger.info("polymarket.games.no_upcoming")
        return dict.fromkeys(sports, 0)

    by_sport_count: dict[str, int] = dict.fromkeys(sports, 0)
    async with httpx.AsyncClient(
        timeout=15,
        headers={"User-Agent": "apuestas-bot/0.1", "Accept": "application/json"},
    ) as clob_client:
        for sport in sports:
            markets = await fetch_game_markets(sport_code=sport, hours_ahead=hours_ahead)
            sport_upcoming = [m for m in upcoming if m["sport_code"] == sport]
            for m in markets:
                question = m.get("question", "") or ""
                # 1) Filtro estricto: solo h2h "Will TeamX win" (rechaza
                #    draws/totals/props para no contaminar odds_history h2h)
                is_h2h, winning_team_norm = _is_h2h_win_question(question)
                if not is_h2h or not winning_team_norm:
                    continue
                # 2) Match a un upcoming match (puede usar título del event como fallback)
                event_title = m.get("_event_title", "") or ""
                matched = _match_question_to_event(event_title or question, sport_upcoming)
                if matched is None:
                    continue
                # CLOB token ids: outcomePrices[0]=Yes, [1]=No (convención Polymarket)
                token_ids = m.get("clobTokenIds")
                if isinstance(token_ids, str):
                    try:
                        import json as _json

                        token_ids = _json.loads(token_ids)
                    except ValueError:
                        token_ids = []
                if not isinstance(token_ids, list) or len(token_ids) < 1:
                    continue
                yes_token = str(token_ids[0])
                yes_mid = await _fetch_clob_midpoint(clob_client, token_id=yes_token)
                if yes_mid is None:
                    continue
                no_mid: float | None = None
                if len(token_ids) >= 2:
                    no_mid = await _fetch_clob_midpoint(clob_client, token_id=str(token_ids[1]))
                # Match robusto: el winning_team del question matchea home o away
                home_norm = matched["home_norm"]
                away_norm = matched["away_norm"]
                wt = winning_team_norm
                if wt in home_norm or home_norm in wt:
                    home_is_yes = True
                elif wt in away_norm or away_norm in wt:
                    home_is_yes = False
                else:
                    # winning_team no matchea ni home ni away → skip (evita odds erróneas)
                    continue
                # Metadatos para polymarket_prices (consumido por consensus_fetch)
                cond_id = m.get("conditionId") or m.get("id") or ""
                vol_24h = m.get("volume24hr") or m.get("volume")
                try:
                    vol_usd = float(vol_24h) if vol_24h is not None else None
                except (TypeError, ValueError):
                    vol_usd = None
                end_dt = _parse_ts(m.get("endDate") or m.get("end_date_iso"))
                inserted = await _persist_game_odds_history(
                    match_id=int(matched["id"]),
                    yes_prob=yes_mid,
                    no_prob=no_mid,
                    home_is_yes=home_is_yes,
                    sport_code=sport,
                    condition_id=str(cond_id),
                    question=question,
                    yes_token_id=yes_token,
                    volume_usd=vol_usd,
                    end_date=end_dt,
                )
                by_sport_count[sport] += inserted
    logger.info("polymarket.games.done", **{k: str(v) for k, v in by_sport_count.items()})
    return by_sport_count
