"""The Odds API — ingest optimizado para plan Basic ($30/mes, 20k créditos).

Estrategia calidad/precio máxima:
1. **Regions**: `us,eu` — US para DK/FD/MGM/Caesars (apostables con VPN US),
   EU para Pinnacle/Unibet/Bet365/Betfair (sharp benchmark + line shopping).
   2 regiones × 1 market = 2 créditos por request.
2. **Markets**: solo `h2h` por default. El 80% del EV+ está en moneylines.
   Spreads/totals pueden añadirse puntualmente si usuario lo pide.
3. **Frecuencia adaptativa** por proximidad del partido más cercano del sport:
   - <2h:   cada 15 min  (captura steam moves críticos)
   - 2-6h:  cada 30 min
   - 6-24h: cada 1h
   - >24h:  cada 3h      (odds estables, no vale polling agresivo)
4. **Budget tracker** en Valkey: stop automático si queda <10% del mes.
5. **Skip sports sin partidos próximos 48h** (no quemamos créditos si no hay
   qué analizar).

Budget esperado:
- Promedio 7 sports activos × cada ~45 min = 224 req/día
- 224 × 2 créditos = 448 créditos/día = 13,440/mes
- 33% margen sobre 20k para picos ad-hoc.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
from sqlalchemy import text

from apuestas.config import get_settings
from apuestas.db import session_scope
from apuestas.ingest._match_resolver import resolve_or_create_match
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

API_BASE = "https://api.the-odds-api.com/v4"
DEFAULT_REGIONS = "us,eu"
# Expandimos a 4 markets (antes solo h2h). Cada request = regions × markets créditos:
# 2 regiones × 4 markets = 8 créditos/poll (vs 2 antes).
# Budget ajustado: 40 polls × 8 = 320/ciclo × 4 ciclos/día = 1280/día × 30 = 38,400/mes.
# OVERFLOW del plan 20k. Por eso limitamos a `h2h,totals` por default (4 créditos)
# y team_totals se pide solo en sports con props activos (soccer, nba, mlb).
DEFAULT_MARKETS = "h2h,totals"
EXTENDED_MARKETS = "h2h,spreads,totals,team_totals,alternate_totals"  # soccer/NBA/MLB

# Budget configurable vía env. Sprint B abr-2026 — paid plan 20k/mes ya no es
# necesario tras cablear Polymarket games + Kambi multi-operador + Pinnacle guest;
# ahora the-odds-api queda como **safety net** para books US (BetMGM/Caesars/
# BetRivers) que las otras fuentes no cubren. Default tier free 500/mes.
import os as _os

try:
    BUDGET_MONTHLY = int(_os.environ.get("THE_ODDS_API_BUDGET_MONTHLY", "500"))
except ValueError:
    BUDGET_MONTHLY = 500
# Stop polls cuando queden <20% del plan
BUDGET_SAFETY_MARGIN = 0.20

# Sports soportados por The Odds API que nos interesan
SPORT_KEY_MAP: dict[str, str] = {
    # ═══ US mayores (alta prioridad) ═══
    "nba": "basketball_nba",
    "wnba": "basketball_wnba",
    "mlb": "baseball_mlb",
    "nfl": "americanfootball_nfl",
    "ncaaf": "americanfootball_ncaaf",
    "ncaab": "basketball_ncaab",
    "cfl": "americanfootball_cfl",
    "ufl": "americanfootball_ufl",
    "nhl": "icehockey_nhl",
    "ahl": "icehockey_ahl",
    # ═══ Baseball internacional ═══
    "kbo": "baseball_kbo",
    "npb": "baseball_npb",
    # ═══ Basketball europeo ═══
    "euroleague": "basketball_euroleague",
    # ═══ Soccer — Europa top 5 ═══
    "soccer_epl": "soccer_epl",
    "soccer_laliga": "soccer_spain_la_liga",
    "soccer_bundesliga": "soccer_germany_bundesliga",
    "soccer_seriea": "soccer_italy_serie_a",
    "soccer_ligue1": "soccer_france_ligue_one",
    # ═══ Soccer — Europa segunda división ═══
    "soccer_efl_champ": "soccer_efl_champ",
    "soccer_england_league1": "soccer_england_league1",
    "soccer_england_league2": "soccer_england_league2",
    "soccer_laliga2": "soccer_spain_segunda_division",
    "soccer_seriea_b": "soccer_italy_serie_b",
    "soccer_ligue2": "soccer_france_ligue_two",
    "soccer_bundesliga2": "soccer_germany_bundesliga2",
    # ═══ Soccer — Europa otras ═══
    "soccer_netherlands_eredivisie": "soccer_netherlands_eredivisie",
    "soccer_portugal_primeira_liga": "soccer_portugal_primeira_liga",
    "soccer_belgium_first_div": "soccer_belgium_first_div",
    "soccer_austria_bundesliga": "soccer_austria_bundesliga",
    "soccer_denmark_superliga": "soccer_denmark_superliga",
    "soccer_sweden_allsvenskan": "soccer_sweden_allsvenskan",
    "soccer_norway_eliteserien": "soccer_norway_eliteserien",
    "soccer_greece_super_league": "soccer_greece_super_league",
    "soccer_turkey_super_league": "soccer_turkey_super_league",
    "soccer_russia_premier": "soccer_russia_premier_league",
    "soccer_poland_ekstraklasa": "soccer_poland_ekstraklasa",
    "soccer_switzerland_super": "soccer_switzerland_superleague",
    # ═══ Soccer — América ═══
    "soccer_liga_mx": "soccer_mexico_ligamx",
    "soccer_mls": "soccer_usa_mls",
    "soccer_brazil": "soccer_brazil_campeonato",
    "soccer_brazil_b": "soccer_brazil_serie_b",
    "soccer_argentina": "soccer_argentina_primera_division",
    "soccer_chile": "soccer_chile_campeonato",
    "soccer_copa_libertadores": "soccer_conmebol_copa_libertadores",
    "soccer_copa_sudamericana": "soccer_conmebol_copa_sudamericana",
    # ═══ Soccer — Asia/Oceanía ═══
    "soccer_australia_aleague": "soccer_australia_aleague",
    "soccer_j_league": "soccer_japan_j_league",
    "soccer_k_league": "soccer_korea_kleague1",
    "soccer_china_superleague": "soccer_china_superleague",
    "soccer_saudi": "soccer_saudi_arabia_pro_league",
    # ═══ Soccer — Internacional ═══
    "soccer_ucl": "soccer_uefa_champs_league",
    "soccer_uel": "soccer_uefa_europa_league",
    "soccer_ufa_conference": "soccer_uefa_europa_conference_league",
    "soccer_fa_cup": "soccer_fa_cup",
    # ═══ Hockey europeo ═══
    "icehockey_shl": "icehockey_sweden_hockey_league",
    "icehockey_allsvenskan": "icehockey_sweden_allsvenskan",
    "icehockey_liiga": "icehockey_liiga",
    # ═══ Rugby/AFL ═══
    "rugbyleague_nrl": "rugbyleague_nrl",
    "aussierules_afl": "aussierules_afl",
    # ═══ Cricket ═══
    "cricket_ipl": "cricket_ipl",
    "cricket_psl": "cricket_psl",
    "cricket_odi": "cricket_odi",
    # ═══ Individuales ═══
    "mma": "mma_mixed_martial_arts",
    "boxing": "boxing_boxing",
    # ═══ Handball ═══
    "handball_germany": "handball_germany_bundesliga",
}


def _adaptive_interval_min(hours_to_nearest: float) -> int:
    """Intervalo de polling (min) según proximidad del partido más cercano."""
    if hours_to_nearest < 2:
        return 15
    if hours_to_nearest < 6:
        return 30
    if hours_to_nearest < 24:
        return 60
    return 180


async def _hours_to_nearest_match(sport_code: str) -> float | None:
    """Horas hasta el próximo partido scheduled del sport. None = sin partidos.

    Para soccer sub-sports (soccer_laliga, soccer_epl, etc.) también busca en
    sport_code='soccer' genérico (donde Pinnacle/Sofascore almacena muchos partidos).
    """
    # Códigos alternativos para buscar en DB (ej. soccer_laliga → también busca soccer)
    alt_codes = [sport_code]
    if sport_code.startswith("soccer_"):
        alt_codes.append("soccer")
    if sport_code == "nba":
        alt_codes.append("basketball")

    async with session_scope() as s:
        r = (
            await s.execute(
                text(
                    """
                    SELECT EXTRACT(EPOCH FROM (MIN(start_time) - NOW()))::float / 3600 AS hrs
                    FROM matches
                    WHERE sport_code = ANY(:codes) AND status = 'scheduled'
                      AND start_time > NOW() AND start_time < NOW() + INTERVAL '48 hours'
                    """
                ),
                {"codes": alt_codes},
            )
        ).first()
    if r is None or r.hrs is None:
        return None
    return float(r.hrs)


async def _should_poll(sport_code: str) -> tuple[bool, str]:
    """Decide si polling ahora: respeta intervalo adaptativo por sport.

    Guarda último timestamp por sport en Valkey o DB.
    """
    hrs = await _hours_to_nearest_match(sport_code)
    if hrs is None:
        return False, f"sin partidos 48h [{sport_code}]"

    interval_min = _adaptive_interval_min(hrs)
    now = datetime.now(tz=UTC)
    async with session_scope() as s:
        r = (
            await s.execute(
                text("SELECT value, updated_at FROM bot_state WHERE key = :k"),
                {"k": f"odds_api_last_poll_{sport_code}"},
            )
        ).first()

    if r is None:
        return True, f"primera vez [{sport_code}, match en {hrs:.1f}h]"

    elapsed_min = (now - r.updated_at).total_seconds() / 60
    if elapsed_min >= interval_min:
        return True, (
            f"ready [{sport_code}, últ={elapsed_min:.0f}min, "
            f"intervalo={interval_min}min, match en {hrs:.1f}h]"
        )
    return False, (f"skip [{sport_code}, últ={elapsed_min:.0f}min < {interval_min}min]")


async def _mark_polled(sport_code: str, credits_used: int, credits_remaining: int) -> None:
    """Registra timestamp + créditos tras poll exitoso."""
    async with session_scope() as s:
        await s.execute(
            text(
                """
                INSERT INTO bot_state (key, value, updated_at)
                VALUES (:k, :v, NOW())
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                """
            ),
            {
                "k": f"odds_api_last_poll_{sport_code}",
                "v": f"credits_used={credits_used},remaining={credits_remaining}",
            },
        )


async def get_budget_status() -> dict[str, Any]:
    """Estado actual del presupuesto mensual.

    Si no hay fila en bot_state, auto-init consultando /sports (cuesta 0 créditos
    según The Odds API docs) para obtener el header x-requests-remaining real.
    """
    async with session_scope() as s:
        r = (
            await s.execute(
                text("SELECT value, updated_at FROM bot_state WHERE key = 'odds_api_budget'")
            )
        ).first()
    if r is None:
        # Auto-init: consulta /sports (endpoint gratuito) para obtener créditos reales
        settings = get_settings()
        key_obj = settings.apis.the_odds_api_key
        if key_obj is not None:
            api_key = key_obj.get_secret_value().strip()
            if api_key and len(api_key) >= 20:
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        resp = await client.get(f"{API_BASE}/sports", params={"apiKey": api_key})
                        if resp.status_code == 200:
                            real_remaining = int(
                                resp.headers.get("x-requests-remaining", BUDGET_MONTHLY)
                            )
                            await _update_budget(real_remaining)
                            logger.info(
                                "odds_api.budget_auto_init",
                                remaining=real_remaining,
                            )
                            return {
                                "used": BUDGET_MONTHLY - real_remaining,
                                "remaining": real_remaining,
                                "remaining_pct": (real_remaining / BUDGET_MONTHLY) * 100,
                                "safe_to_poll": (real_remaining / BUDGET_MONTHLY)
                                > BUDGET_SAFETY_MARGIN,
                            }
                except Exception as exc:
                    logger.warning("odds_api.budget_auto_init_fail", error=str(exc)[:100])
        return {
            "used": 0,
            "remaining": BUDGET_MONTHLY,
            "remaining_pct": 100.0,
            "safe_to_poll": True,
        }
    parts = dict(item.split("=") for item in str(r.value).split(",") if "=" in item)
    used = int(parts.get("used", 0))
    remaining = int(parts.get("remaining", BUDGET_MONTHLY))
    pct = (remaining / BUDGET_MONTHLY) * 100
    return {
        "used": used,
        "remaining": remaining,
        "remaining_pct": pct,
        "safe_to_poll": pct > (BUDGET_SAFETY_MARGIN * 100),
        "last_update": r.updated_at,
    }


async def _update_budget(credits_remaining: int) -> None:
    """Actualiza budget tracker con lo que reporta The Odds API en headers."""
    used = BUDGET_MONTHLY - credits_remaining if credits_remaining >= 0 else 0
    async with session_scope() as s:
        await s.execute(
            text(
                """
                INSERT INTO bot_state (key, value, updated_at)
                VALUES ('odds_api_budget', :v, NOW())
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                """
            ),
            {"v": f"used={used},remaining={credits_remaining}"},
        )


_MIN_POLL_INTERVAL_SECONDS = 300  # hardcoded 5 min mínimo entre polls de mismo sport
# Cap dinámico: con plan free 500/mes el budget mensual real son ~16/día.
# Plan paid 20k → 666/día → margen amplio. Heurística: cap = budget_monthly / 100.
# Override por env APUESTAS_ODDS_API_MAX_POLLS para casos especiales.
try:
    _MAX_POLLS_PER_RUN = int(_os.environ.get("APUESTAS_ODDS_API_MAX_POLLS", "0")) or max(
        4, BUDGET_MONTHLY // 100
    )
except (TypeError, ValueError):
    _MAX_POLLS_PER_RUN = 8

# Module-level state para throttle de logs `odds_api.budget_critical`.
# Sin esto el warning se dispara una vez por cada poll de cada sport, lo que
# satura logs/telegram.log con 50+ warnings idénticos por ciclo.
_LAST_BUDGET_CRITICAL_LOG: datetime | None = None


# Sports core con player props habilitables. Limpiado: removidos wnba/nhl
# (sport_focus off) y soccer secundarios (blocklist). Solo se usa cuando
# APUESTAS_ENABLE_PROPS=true (opt-in explícito; default off).
_PROPS_ENABLED_SPORTS = {
    "nba",
    "mlb",
    "soccer_epl",
    "soccer_laliga",
    "soccer_bundesliga",
    "soccer_seriea",
    "soccer_ligue1",
    "soccer_liga_mx",
    "soccer_mls",
    "soccer_ucl",
    "soccer_brazil",
    "soccer_argentina",
}

# Top props markets por sport (top volumen / edge histórico). Cada mercado
# cuesta 1 crédito × regions. Protegemos presupuesto: máximo 2 markets props.
_PROPS_MARKETS_BY_SPORT: dict[str, tuple[str, ...]] = {
    "nba": ("player_points", "player_rebounds"),
    "mlb": ("batter_home_runs",),
    # Soccer: anytime goalscorer es top mercado props fútbol
    "soccer_epl": ("player_goal_scorer_anytime",),
    "soccer_laliga": ("player_goal_scorer_anytime",),
    "soccer_bundesliga": ("player_goal_scorer_anytime",),
    "soccer_seriea": ("player_goal_scorer_anytime",),
    "soccer_ligue1": ("player_goal_scorer_anytime",),
    "soccer_ucl": ("player_goal_scorer_anytime",),
}


def _markets_for_sport(sport_code: str) -> str:
    """Mercados a solicitar según sport.

    Default global: `h2h,totals` (2 markets × 2 regions = 4 créditos/poll).
    Antes _PROPS_ENABLED_SPORTS forzaba upgrade a `h2h,spreads,totals` (3
    markets = 6 créditos = +50% costo) sin que esos 'spreads' se usaran en
    el detector — gasto silencioso. Eliminado: spreads se piden solo si el
    usuario activa `APUESTAS_ENABLE_PROPS=true` y vienen junto a player props.

    Con APUESTAS_ENABLE_PROPS=true y sport ∈ _PROPS_ENABLED_SPORTS:
    agrega top 1-2 markets de props (+1 crédito × regions cada uno).
    """
    import os

    base = DEFAULT_MARKETS

    if os.environ.get("APUESTAS_ENABLE_PROPS", "false").lower() == "true":
        extras = _PROPS_MARKETS_BY_SPORT.get(sport_code, ())
        if extras:
            base = f"{base},{','.join(extras)}"
    return base


async def fetch_odds_for_sport(
    sport_code: str,
    *,
    regions: str = DEFAULT_REGIONS,
    markets: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Descarga odds + persiste en odds_history. Respeta budget + intervalo adaptativo.

    CRÍTICO (dinero real): protecciones anti-gasto:
    - Siempre valida API key antes de HTTP (evita 401 que sí gasta).
    - Budget check STOPPING si <10% restante.
    - Hardcoded min 5 min entre polls del MISMO sport (incluso con force=True).
    - Sin retries automáticos (cada error = 1 crédito potencial perdido).

    Returns:
        dict con: polled (bool), skip_reason (si no polled), events, rows_persisted,
        credits_used, credits_remaining, budget_pct
    """
    settings = get_settings()
    key_obj = settings.apis.the_odds_api_key
    if key_obj is None:
        return {"polled": False, "skip_reason": "no_api_key"}
    api_key = key_obj.get_secret_value().strip()
    if not api_key or api_key.startswith(("your-", "change-", "paste-")) or len(api_key) < 20:
        return {"polled": False, "skip_reason": "invalid_api_key"}

    sport_key = SPORT_KEY_MAP.get(sport_code)
    if sport_key is None:
        return {"polled": False, "skip_reason": f"unsupported_sport:{sport_code}"}

    # HARD RATE LIMIT: incluso con force=True, nunca < 5 min entre polls del mismo sport
    async with session_scope() as s:
        r = (
            await s.execute(
                text("SELECT updated_at FROM bot_state WHERE key = :k"),
                {"k": f"odds_api_last_poll_{sport_code}"},
            )
        ).first()
    if r is not None:
        elapsed = (datetime.now(tz=UTC) - r.updated_at).total_seconds()
        if elapsed < _MIN_POLL_INTERVAL_SECONDS:
            return {
                "polled": False,
                "skip_reason": f"hard_rate_limit:{int(elapsed)}s<{_MIN_POLL_INTERVAL_SECONDS}s",
            }

    # Check budget (aun con force=True, NUNCA gastar si crítico)
    budget = await get_budget_status()
    if not budget["safe_to_poll"]:
        # Rate-limited log: solo loggear 1 vez por minuto a nivel WARNING.
        # Antes: cada poll de cada sport disparaba warning idéntico → spam de
        # 50+ warnings por ciclo. Ahora: 1 warning/min agregado, los demás a
        # debug para mantener visibilidad sin saturar logs.
        global _LAST_BUDGET_CRITICAL_LOG
        now_ts = datetime.now(tz=UTC)
        elapsed_log = (
            (now_ts - _LAST_BUDGET_CRITICAL_LOG).total_seconds()
            if _LAST_BUDGET_CRITICAL_LOG is not None
            else 999.0
        )
        if elapsed_log > 60.0:
            logger.warning(
                "odds_api.budget_critical",
                remaining_pct=budget["remaining_pct"],
                remaining=budget["remaining"],
                sport=sport_code,
            )
            _LAST_BUDGET_CRITICAL_LOG = now_ts
        else:
            logger.debug(
                "odds_api.budget_critical_throttled",
                sport=sport_code,
                remaining=budget["remaining"],
            )
        return {
            "polled": False,
            "skip_reason": f"budget_critical:{budget['remaining']}_remaining",
        }

    # Check adaptive schedule (si no es force)
    if not force:
        should, reason = await _should_poll(sport_code)
        if not should:
            return {"polled": False, "skip_reason": reason}

    # Resolver markets: si no se pasó explícito, usar adaptativo por sport
    if markets is None:
        markets = _markets_for_sport(sport_code)

    # Fetch
    url = f"{API_BASE}/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            # The Odds API pone créditos en headers
            credits_used = int(resp.headers.get("x-requests-used", 0))
            credits_remaining = int(resp.headers.get("x-requests-remaining", BUDGET_MONTHLY))
            credits_last = int(resp.headers.get("x-requests-last", 0))
    except Exception as exc:
        logger.warning("odds_api.fetch_fail", sport=sport_code, error=str(exc)[:100])
        return {"polled": False, "skip_reason": f"fetch_error:{type(exc).__name__}"}

    await _mark_polled(sport_code, credits_last, credits_remaining)
    await _update_budget(credits_remaining)

    # Persist odds en odds_history
    rows = await _persist_odds(data, sport_code)
    logger.info(
        "odds_api.polled",
        sport=sport_code,
        events=len(data),
        rows_persisted=rows,
        credits_used=credits_last,
        credits_remaining=credits_remaining,
    )
    return {
        "polled": True,
        "events": len(data),
        "rows_persisted": rows,
        "credits_used": credits_last,
        "credits_remaining": credits_remaining,
        "budget_pct": (credits_remaining / BUDGET_MONTHLY) * 100,
    }


async def _persist_odds(events: list[dict[str, Any]], sport_code: str) -> int:
    """Inserta en odds_history. Resuelve/crea matches via _match_resolver.

    Cada match en su propia transacción — si falla uno no aborta el batch.
    """
    if not events:
        return 0
    now = datetime.now(tz=UTC)
    rows_inserted = 0
    for event in events:
        try:
            async with session_scope() as s:
                rows_inserted += await _persist_single_event(s, event, sport_code, now)
        except Exception as exc:
            logger.warning(
                "odds_api.persist_event_fail",
                sport=sport_code,
                event_id=event.get("id", ""),
                error=str(exc)[:120],
            )
    return rows_inserted


async def _persist_single_event(
    s: Any, event: dict[str, Any], sport_code: str, now: datetime
) -> int:
    """Persiste un evento en su propia transacción. Retorna rows insertadas."""
    try:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        start_str = event.get("commence_time", "")
        start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
    except (TypeError, ValueError, KeyError):  # fmt: skip
        return 0
    if not home or not away:
        return 0

    # Mapear sport_codes de OddsAPI a los códigos válidos en la DB.
    # La tabla `sports` solo tiene: soccer, nba, mlb, nfl, nhl, tennis, boxing, mma,
    # laliga, epl, liga_mx. Todos los soccer sub-sports van a 'soccer'.
    _SPORT_DB_MAP = {
        "laliga": "laliga",
        "epl": "epl",
        "liga_mx": "liga_mx",
    }
    if sport_code.startswith("soccer_"):
        # soccer_laliga → laliga, soccer_epl → epl, soccer_liga_mx → liga_mx
        # el resto → soccer genérico
        sub = sport_code.replace("soccer_", "")
        db_sport = _SPORT_DB_MAP.get(sub, "soccer")
    elif sport_code in ("soccer",):
        db_sport = "soccer"
    else:
        db_sport = sport_code

    odds_api_event_id = str(event.get("id") or "").strip() or None
    external_ids: dict[str, str] | None = (
        {"odds_api": odds_api_event_id} if odds_api_event_id else None
    )

    match_id = await resolve_or_create_match(
        session=s,
        sport_code=db_sport,
        home_name=home,
        away_name=away,
        start_time=start,
        source="theoddsapi",
        external_ids=external_ids,
    )
    # Si no matcheó y era un sub-sport, intentar con soccer genérico
    if match_id is None and db_sport != "soccer" and "soccer" in sport_code:
        match_id = await resolve_or_create_match(
            session=s,
            sport_code="soccer",
            home_name=home,
            away_name=away,
            start_time=start,
            source="theoddsapi",
            external_ids=external_ids,
        )
    if match_id is None:
        return 0

    # Asignación inmediata de league_id desde sport_key del feed cuando esté
    # disponible y el match aún no tenga liga. Antes: matches Conmebol/Brasil/
    # Argentina llegaban con `league_id=NULL` y caían al fallback wide
    # `soccer_liga_mx` (que predice prior promedio Liga MX para todo, disparando
    # draw guard del 26.1% que bloqueaba TODOS los picks Sudamericanos).
    # Ahora: el sport_key del feed (`soccer_brazil_campeonato`,
    # `soccer_uefa_champs_league`, etc.) determina la liga real desde la
    # ingesta misma, no en post-processing.
    _SPORT_KEY_TO_LEAGUE_ID: dict[str, int] = {
        "soccer_uefa_champs_league": 24,
        "soccer_uefa_europa_league": 36,
        "soccer_brazil_campeonato": 28,
        "soccer_argentina_primera_division": 29,
        "soccer_conmebol_copa_libertadores": 26,
        "soccer_conmebol_copa_sudamericana": 27,
        "soccer_mexico_ligamx": 20,
        "soccer_usa_mls": 22,
    }
    target_league = _SPORT_KEY_TO_LEAGUE_ID.get(sport_code)
    if target_league is not None:
        # Sport_keys que SOBRESCRIBEN el league_id existente: el feed nos da
        # información unívoca y no debe ser sobreescrita por inferencia
        # incorrecta de otro ingester. Ej: PSG-Bayern viene con sport_key
        # `soccer_uefa_champs_league` → league_id=24 (UCL), incluso si otro
        # path lo había asignado a Bundesliga (league=8) por error.
        _AUTHORITATIVE_KEYS = {
            "soccer_uefa_champs_league",
            "soccer_uefa_europa_league",
            "soccer_conmebol_copa_libertadores",
            "soccer_conmebol_copa_sudamericana",
        }
        clause = (
            "UPDATE matches SET league_id = :lid WHERE id = :mid"
            if sport_code in _AUTHORITATIVE_KEYS
            else "UPDATE matches SET league_id = :lid WHERE id = :mid AND league_id IS NULL"
        )
        try:
            await s.execute(text(clause), {"lid": target_league, "mid": match_id})
        except Exception as exc:
            logger.debug(
                "odds_api.league_assign_fail",
                match_id=match_id,
                sport_key=sport_code,
                error=str(exc)[:120],
            )

    rows_inserted = 0
    for bm in event.get("bookmakers", []):
        book_key = (bm.get("key") or "").lower()
        if not book_key:
            continue
        for mkt in bm.get("markets", []):
            market = (mkt.get("key") or "").lower()
            # Props markets (player_*/pitcher_*/batter_*) se persisten en
            # player_prop_lines en vez de odds_history (schema distinto).
            if market.startswith(("player_", "pitcher_", "batter_")):
                rows_inserted += await _persist_props_market(
                    s=s,
                    match_id=match_id,
                    sport_code=sport_code,
                    bookmaker=book_key,
                    market=market,
                    outcomes=mkt.get("outcomes", []) or [],
                )
                continue
            # Soporta h2h + spreads + totals + team_totals + alternate_totals
            if market not in (
                "h2h",
                "spreads",
                "totals",
                "team_totals",
                "alternate_totals",
                "alternate_spreads",
            ):
                continue
            for outcome in mkt.get("outcomes", []):
                name = (outcome.get("name") or "").strip()
                price = outcome.get("price")
                if not name or price is None:
                    continue
                # Normalizar outcome a home/away/draw/over/under
                if market == "h2h":
                    if name.lower() == home.lower():
                        oc = "home"
                    elif name.lower() == away.lower():
                        oc = "away"
                    elif name.lower() == "draw":
                        oc = "draw"
                    else:
                        continue
                elif market == "totals":
                    oc = "over" if "over" in name.lower() else "under"
                elif name.lower() == home.lower():
                    oc = "home"
                elif name.lower() == away.lower():
                    oc = "away"
                else:
                    continue
                try:
                    odds_dec = float(price)
                except (TypeError, ValueError):  # fmt: skip
                    continue
                if odds_dec <= 1.01 or odds_dec > 1000:
                    continue
                line = outcome.get("point")
                try:
                    await s.execute(
                        text(
                            """
                            INSERT INTO odds_history
                                (ts, match_id, bookmaker, market, outcome, line, odds)
                            VALUES (:ts, :mid, :bk, :mkt, :oc, :line, :odds)
                            ON CONFLICT DO NOTHING
                            """
                        ),
                        {
                            "ts": now,
                            "mid": match_id,
                            "bk": book_key,
                            "mkt": market,
                            "oc": oc,
                            "line": (float(line) if line is not None else None),
                            "odds": odds_dec,
                        },
                    )
                    rows_inserted += 1
                except Exception as exc:  # fmt: skip
                    logger.warning(
                        "odds_api.insert_fail",
                        book=book_key,
                        market=market,
                        error=str(exc)[:100],
                    )
    return rows_inserted


async def _resolve_player_id(
    s: Any, *, player_name: str, sport_code: str, match_id: int
) -> int | None:
    """Resuelve o crea un player por nombre. Matching fuzzy simple (ILIKE).

    Si no existe, inserta en `players` con flag mínimo. El player_id se
    persiste para que detect_value_props pueda hacer joins.
    """
    from sqlalchemy import text as _t

    name_clean = player_name.strip()
    if not name_clean:
        return None
    row = (
        await s.execute(
            _t(
                "SELECT id FROM players "
                "WHERE sport_code = :sc "
                "  AND (LOWER(full_name) = LOWER(:n) OR full_name ILIKE :like) "
                "LIMIT 1"
            ),
            {"sc": sport_code, "n": name_clean, "like": f"%{name_clean}%"},
        )
    ).first()
    if row:
        return int(row.id)

    # Crear player mínimo si no existe (mejor granularidad después via scrapers
    # oficiales). Usamos external_id único por sport+nombre para idempotencia.
    try:
        result = await s.execute(
            _t(
                """
                INSERT INTO players (external_id, sport_code, full_name, created_at)
                VALUES (:ext, :sc, :name, NOW())
                ON CONFLICT (external_id) DO UPDATE SET full_name = EXCLUDED.full_name
                RETURNING id
                """
            ),
            {
                "ext": f"oddsapi:{sport_code}:{name_clean.lower()}",
                "sc": sport_code,
                "name": name_clean,
            },
        )
        pid_row = result.first()
        return int(pid_row.id) if pid_row else None
    except Exception as exc:
        logger.debug("odds_api.player_create_fail", name=name_clean, error=str(exc)[:80])
        return None


async def _persist_props_market(
    *,
    s: Any,
    match_id: int,
    sport_code: str,
    bookmaker: str,
    market: str,
    outcomes: list[dict[str, Any]],
) -> int:
    """Persiste outcomes de un market de props en player_prop_lines.

    Outcome format OddsAPI: {"name": "Over"/"Under", "description": "Player Name", "price", "point"}
    Algunas variantes usan name=player, description=Over/Under. Probamos ambos.
    """
    from sqlalchemy import text as _t

    # Agrupar por (player, line) → reunir over_odds + under_odds
    agg: dict[tuple[str, float], dict[str, float | None]] = {}
    for outcome in outcomes:
        side = None
        player = None
        point = outcome.get("point")
        if point is None:
            continue
        try:
            line = float(point)
        except (TypeError, ValueError):  # fmt: skip
            continue
        raw_name = (outcome.get("name") or "").strip()
        raw_desc = (outcome.get("description") or "").strip()
        if raw_name.lower() in {"over", "under"}:
            side = raw_name.lower()
            player = raw_desc
        elif raw_desc.lower() in {"over", "under"}:
            side = raw_desc.lower()
            player = raw_name
        if not side or not player:
            continue
        try:
            odds = float(outcome.get("price"))
        except (TypeError, ValueError):  # fmt: skip
            continue
        if odds <= 1.01 or odds > 1000:
            continue
        key = (player, line)
        slot = agg.setdefault(key, {"over_odds": None, "under_odds": None})
        slot[f"{side}_odds"] = odds

    if not agg:
        return 0

    rows_inserted = 0
    for (player, line), sides in agg.items():
        player_id = await _resolve_player_id(
            s, player_name=player, sport_code=sport_code, match_id=match_id
        )
        if player_id is None:
            continue
        try:
            await s.execute(
                _t(
                    """
                    INSERT INTO player_prop_lines
                        (match_id, player_id, prop_type, line, over_odds, under_odds,
                         bookmaker, captured_at)
                    VALUES (:mid, :pid, :pt, :ln, :oo, :uo, :bk, NOW())
                    ON CONFLICT (match_id, player_id, prop_type, line, bookmaker)
                    DO UPDATE SET
                        over_odds = EXCLUDED.over_odds,
                        under_odds = EXCLUDED.under_odds,
                        captured_at = NOW()
                    """
                ),
                {
                    "mid": match_id,
                    "pid": player_id,
                    "pt": market,
                    "ln": line,
                    "oo": sides["over_odds"],
                    "uo": sides["under_odds"],
                    "bk": bookmaker,
                },
            )
            rows_inserted += 1
        except Exception as exc:
            logger.debug(
                "odds_api.props_insert_fail",
                match=match_id,
                player=player,
                market=market,
                error=str(exc)[:80],
            )
    return rows_inserted


def _sport_focus_key(sport_code: str) -> str:
    """Mapea sport_code de The Odds API → canonical sport para sport_focus.

    Ejemplo: 'soccer_epl' → 'soccer', 'basketball_nba' → 'nba',
    'baseball_mlb' → 'mlb', 'icehockey_nhl' → 'nhl'.
    Todos los soccer_* colapsan a 'soccer' (gate único multi-liga).
    """
    if sport_code.startswith("soccer_") or sport_code == "soccer":
        return "soccer"
    if sport_code in {"nba", "wnba", "ncaab", "euroleague"}:
        return "nba"
    if sport_code in {"mlb", "kbo", "npb"}:
        return "mlb"
    if sport_code in {"nfl", "ncaaf", "cfl", "ufl"}:
        return "nfl"
    if sport_code in {"nhl", "ahl"} or sport_code.startswith("icehockey_"):
        return "nhl"
    if sport_code == "boxing":
        return "boxing"
    if sport_code == "mma":
        return "mma"
    if sport_code.startswith("tennis"):
        return "tennis"
    return sport_code


async def poll_all_active_sports() -> dict[str, Any]:
    """Poll en batch: solo sports habilitados en `config/enabled_sports.yaml`.

    CRÍTICO: límite hardcoded _MAX_POLLS_PER_RUN para nunca gastar más de
    _MAX_POLLS_PER_RUN * 4-6 créditos por invocación (protección absoluta).
    Pre-filtro por `sport_focus.is_emit_enabled()` evita gastar créditos en
    deportes desactivados (boxing, mma, nhl, tennis, nfl off-season, etc.).
    """
    from apuestas.betting.sport_focus import is_emit_enabled, is_odds_api_key_disabled

    results: dict[str, Any] = {"polled": [], "skipped": [], "total_rows": 0}
    polls_done = 0
    for sport_code in SPORT_KEY_MAP:
        focus_key = _sport_focus_key(sport_code)
        if not is_emit_enabled(focus_key):
            results["skipped"].append(
                {"sport": sport_code, "reason": f"sport_focus_disabled:{focus_key}"}
            )
            continue
        if is_odds_api_key_disabled(sport_code):
            results["skipped"].append({"sport": sport_code, "reason": "odds_api_key_blocklisted"})
            continue
        if polls_done >= _MAX_POLLS_PER_RUN:
            results["skipped"].append(
                {
                    "sport": sport_code,
                    "reason": f"max_polls_per_run_reached:{_MAX_POLLS_PER_RUN}",
                }
            )
            continue
        r = await fetch_odds_for_sport(sport_code)
        if r.get("polled"):
            polls_done += 1
            results["polled"].append(
                {
                    "sport": sport_code,
                    "events": r["events"],
                    "rows": r["rows_persisted"],
                    "credits_remaining": r["credits_remaining"],
                }
            )
            results["total_rows"] += r["rows_persisted"]
        else:
            results["skipped"].append({"sport": sport_code, "reason": r.get("skip_reason", "")})
    results["polls_in_run"] = polls_done
    results["max_credits_used_in_run"] = polls_done * 2  # regions=us,eu × markets=h2h
    return results


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        budget = await get_budget_status()
        print(f"Budget: {budget}")
        r = await poll_all_active_sports()
        print(f"Resultado: {r}")

    asyncio.run(main())
