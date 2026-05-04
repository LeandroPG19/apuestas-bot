"""Catchup flow — ingesta incremental al arrancar `make up`.

§11.5: cuando el bot estuvo apagado, al arrancar se ejecuta este flow
para poner al día:
- Fixtures (schedules próximos 14 días)
- Odds recientes (últimas 2 horas)
- Injuries + lineups
- News RSS + Reddit + Bluesky (consolidate)
- Weather forecast para eventos outdoor próximos

Reanuda desde `ingest_checkpoints` para no re-ingestar ya visto.
Orquestado con Prefect 3 para observabilidad.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from prefect import flow, task
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.ingest.api_football import LEAGUE_IDS, ingest_league_fixtures, ingest_league_odds
from apuestas.ingest.nba import ingest_nba_today
from apuestas.ingest.news_pipeline import run_news_ingest_pipeline
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@task(retries=2, retry_delay_seconds=30)
async def update_checkpoint(source: str, resource: str) -> None:
    async with session_scope() as session:
        await session.execute(
            text(
                """
                INSERT INTO ingest_checkpoints (source, resource, last_ts, items_processed)
                VALUES (:src, :res, NOW(), 0)
                ON CONFLICT (source, resource) DO UPDATE
                SET last_ts = NOW()
                """
            ),
            {"src": source, "res": resource},
        )


def _api_football_available() -> bool:
    """Skip API-Football si no hay key válida configurada."""
    import os as _os

    key = _os.environ.get("API_FOOTBALL_KEY", "").strip()
    return bool(key) and not key.startswith(("your-", "change-"))


@task(retries=1, retry_delay_seconds=10)
async def catchup_soccer_fixtures(seasons: list[int] | None = None) -> dict[str, int]:
    """Ingesta fixtures. Si API-Football no disponible, usa football-data.org
    como fallback para Big-5 (gratis)."""
    seasons = seasons or [2025, 2026]
    results: dict[str, int] = {}

    if _api_football_available():
        for league_slug in ("liga_mx", "liga_expansion_mx", "mls", "epl", "la_liga"):
            if league_slug not in LEAGUE_IDS:
                continue
            for season in seasons:
                try:
                    df = await ingest_league_fixtures(league_slug, season)
                    results[f"{league_slug}_{season}"] = df.height
                except Exception as exc:
                    logger.warning(
                        "catchup.fixtures_fail",
                        league=league_slug,
                        season=season,
                        error=str(exc)[:100],
                    )
    else:
        logger.info("catchup.api_football_skipped", reason="no_valid_key")

    # Fallback gratis: football-data.org para ligas europeas
    try:
        from apuestas.ingest.free_sources import FootballDataOrgClient

        client = FootballDataOrgClient()
        async with client.session():
            for comp in ("epl", "la_liga", "champions"):
                try:
                    matches = await client.fetch_matches(competition=comp, status="SCHEDULED")
                    results[f"fd_{comp}"] = len(matches)
                except Exception as exc:
                    logger.debug("catchup.fd_fail", comp=comp, error=str(exc)[:80])
    except Exception as exc:
        logger.debug("catchup.fd_unavailable", error=str(exc)[:80])

    return results


@task(retries=1, retry_delay_seconds=10)
async def catchup_soccer_odds() -> dict[str, int]:
    results: dict[str, int] = {}
    if not _api_football_available():
        logger.info("catchup.api_football_odds_skipped")
        return results
    for league_slug in ("liga_mx", "epl", "la_liga"):
        try:
            df = await ingest_league_odds(league_slug, 2026)
            results[league_slug] = df.height
        except Exception as exc:
            logger.warning("catchup.soccer_odds_fail", league=league_slug, error=str(exc)[:100])
    return results


_ODDS_API_LAST_POLL = None  # datetime | None — actualizado en cada poll exitoso


@task(retries=0, retry_delay_seconds=0)  # ¡0 retries! cada retry gastaría créditos
async def catchup_odds_api() -> dict[str, int]:
    """Ingesta The Odds API optimizada — budget + frecuencia adaptativa.

    CRÍTICO (dinero real):
    - retries=0 porque cada retry gasta créditos del plan
    - Errores HTTP NO se re-intentan (budget respetado)
    - Si falla, se retoma en el siguiente ciclo (15-180 min después)

    Sprint B abr-2026: tras cablear Polymarket/Kambi/Pinnacle guest,
    the-odds-api queda como SAFETY NET para books US (BetMGM/Caesars/
    BetRivers) no cubiertos por otras fuentes. Polling throttled a 30min
    vía `APUESTAS_ODDS_API_MIN_INTERVAL_MIN` (default 30) → ahorra ~83%
    créditos vs el ciclo de 5min.
    """
    import os

    global _ODDS_API_LAST_POLL

    try:
        min_interval_min = int(os.environ.get("APUESTAS_ODDS_API_MIN_INTERVAL_MIN", "30"))
    except ValueError:
        min_interval_min = 30

    now = datetime.now(tz=UTC)
    if _ODDS_API_LAST_POLL is not None:
        elapsed = (now - _ODDS_API_LAST_POLL).total_seconds() / 60.0
        if elapsed < min_interval_min:
            logger.info(
                "catchup.odds_api_throttled",
                elapsed_min=round(elapsed, 1),
                min_interval_min=min_interval_min,
            )
            return {"throttled": 1}

    from apuestas.ingest.odds_api_optimized import (
        get_budget_status,
        poll_all_active_sports,
    )

    budget = await get_budget_status()
    logger.info(
        "catchup.odds_api_budget",
        remaining=budget["remaining"],
        remaining_pct=f"{budget['remaining_pct']:.1f}%",
    )
    r = await poll_all_active_sports()
    _ODDS_API_LAST_POLL = now
    results: dict[str, int] = {
        "polled_sports": len(r["polled"]),
        "skipped_sports": len(r["skipped"]),
        "total_rows": r["total_rows"],
        "credits_remaining": budget["remaining"],
    }
    return results


@task(retries=1, retry_delay_seconds=15)
async def catchup_pinnacle_guest() -> dict[str, int]:
    """Ingesta Pinnacle guest API — cobertura 100% multi-deporte con
    auto-discover de leagues en runtime (nada hardcoded).

    Sports escaneados:
    - Fútbol: EPL, LaLiga, Bundesliga, Serie A, Ligue 1, UCL, Liga MX, MLS,
      World Cup + ligas genéricas (todas las activas con matchupCount>0).
    - NBA, MLB, NFL, NHL
    - Tenis: auto-discover TODOS los torneos ATP/WTA activos (cambian por semana)
    - Boxing: todos los combates upcoming
    - MMA: todos los eventos upcoming

    Cada sport_code resuelve sus league_ids vía `discover_leagues()` que
    consulta Pinnacle live y filtra por actividad. Al cambiar la temporada o
    al salir nuevos torneos, el bot se adapta sin redeploy.
    """
    from apuestas.ingest.pinnacle_scraper import ingest_league

    # Cobertura completa de los 6 deportes que pidió el usuario + extras.
    # Estos sport_codes NO son league_ids hardcoded — son "contratos" que
    # resuelve discover_leagues en runtime.
    sports_to_scan = (
        # Específicos (filtro regex por nombre en discover_leagues)
        "nba",
        "mlb",
        "nfl",
        "nhl",
        "soccer_epl",
        "soccer_laliga",
        "soccer_bundesliga",
        "soccer_seriea",
        "soccer_ligue1",
        "soccer_ucl",
        "soccer_uel",
        "soccer_uecl",
        "soccer_liga_mx",
        "soccer_mls",
        "soccer_world_cup",
        "soccer_eredivisie",
        "soccer_brasil",
        "soccer_argentina",
        "soccer_libertadores",
        "soccer_sudamericana",
        "soccer_concacaf_cl",
        # Genéricos (todas las leagues activas)
        "tennis",  # auto-descubre TODOS los torneos ATP/WTA/Challenger activos
        "boxing",  # todos los combates
        "mma",  # todos los eventos
    )

    results: dict[str, int] = {}
    for sport_code in sports_to_scan:
        try:
            _matchups, odds = await ingest_league(sport_code, persist=True)
            results[sport_code] = len(odds)
        except Exception as exc:
            logger.warning(
                "catchup.pinnacle_fail",
                sport=sport_code,
                error=str(exc)[:300],
                error_type=type(exc).__name__,
            )

    return results


@task(retries=1, retry_delay_seconds=30)
async def catchup_betfair_exchange() -> dict[str, int]:
    """Betfair Exchange. Fail-soft si no hay credenciales configuradas."""
    from apuestas.ingest.betfair_exchange import _credentials_available, ingest

    if not _credentials_available():
        logger.info("catchup.betfair_skipped", reason="no_credentials")
        return {}
    try:
        odds = await ingest(["soccer", "tennis", "nba", "mlb", "nhl"], hours_ahead=48)
        by_sport: dict[str, int] = {}
        for o in odds:
            by_sport[o.sport_code] = by_sport.get(o.sport_code, 0) + 1
        return by_sport
    except Exception as exc:
        logger.warning("catchup.betfair_fail", error=str(exc)[:100])
        return {}


@task(retries=1, retry_delay_seconds=30)
async def catchup_offshore_books() -> dict[str, int]:
    """Fase 5.10 — offshore sportsbooks accesibles MX+US (BetUS, BetWhale, Everygame,
    SportsBetting.ag, BC.GAME). Opt-in via APUESTAS_ENABLE_<BOOK>=true."""
    import os as _os

    from apuestas.ingest.offshore_sportsbook_generic import OffshoreSportsbookScraper

    offshore_books = ("betus", "betwhale", "everygame", "sportsbetting_ag", "bc_game")
    results: dict[str, int] = {}
    for slug in offshore_books:
        flag = f"APUESTAS_ENABLE_{slug.upper()}"
        if _os.environ.get(flag, "").lower() not in ("1", "true", "yes"):
            continue
        try:
            scraper = OffshoreSportsbookScraper.from_yaml(slug)
            book_results = await scraper.scrape_all_enabled_sports()
            results[slug] = sum(book_results.values())
        except Exception as exc:
            logger.warning(f"catchup.{slug}_fail", error=str(exc)[:100])
    return results


@task(retries=1, retry_delay_seconds=30)
async def catchup_mx_regulated_new() -> dict[str, int]:
    """Fase 5.11 — MX regulados adicionales (Winpot, CampoBet, JugaBet)."""
    import os as _os

    from apuestas.ingest.offshore_sportsbook_generic import OffshoreSportsbookScraper

    mx_new_books = ("winpot", "campobet", "jugabet")
    results: dict[str, int] = {}
    for slug in mx_new_books:
        flag = f"APUESTAS_ENABLE_{slug.upper()}"
        if _os.environ.get(flag, "").lower() not in ("1", "true", "yes"):
            continue
        try:
            scraper = OffshoreSportsbookScraper.from_yaml(slug)
            book_results = await scraper.scrape_all_enabled_sports()
            results[slug] = sum(book_results.values())
        except Exception as exc:
            logger.warning(f"catchup.{slug}_fail", error=str(exc)[:100])
    return results


@task(retries=1, retry_delay_seconds=30)
async def catchup_codere() -> dict[str, int]:
    """Scrapea Codere.mx (HTTP plano + selectolax, sin camoufox). Gap #12."""
    from apuestas.ingest.codere import SPORT_SLUG_TO_CODE as _CODERE_SLUGS
    from apuestas.ingest.codere import ingest_codere_sport

    results: dict[str, int] = {}
    for sport_slug in _CODERE_SLUGS:
        try:
            rows = await ingest_codere_sport(sport_slug, persist=True)
            results[sport_slug] = rows
        except Exception as exc:
            logger.warning("catchup.codere_fail", sport=sport_slug, error=str(exc)[:100])
    return results


@task(retries=1, retry_delay_seconds=60)
async def catchup_caliente() -> dict[str, int]:
    """Scrapea Caliente.mx (Liga MX, Expansión, NBA, MLB, NFL, Boxing) vía camoufox.

    Fail-soft: si camoufox no instalado o Cloudflare bloquea, skip silencioso.
    Persist real en `odds_history` con fuzzy match contra teams existentes.
    """
    from apuestas.ingest.caliente import (
        SPORT_SLUG_TO_CODE,
        CalienteBannedError,
        ingest_caliente_sport,
    )

    results: dict[str, int] = {}
    for sport_slug in SPORT_SLUG_TO_CODE:
        try:
            df = await ingest_caliente_sport(sport_slug, persist=True)
            results[sport_slug] = df.height
        except CalienteBannedError:
            logger.warning("catchup.caliente_banned", sport=sport_slug)
            break
        except Exception as exc:
            logger.warning("catchup.caliente_fail", sport=sport_slug, error=str(exc)[:100])
    return results


@task(retries=0, retry_delay_seconds=0)  # retries=0: no reintentar scrapers rotos
async def catchup_us_books() -> dict[str, int]:
    """DraftKings/FanDuel/BetMGM via camoufox.

    DESACTIVADO 2026-04-22: los scrapers directos retornan 0 (Akamai/Cloudflare
    bloquea incluso con camoufox). Ahora obtenemos DK/FD/MGM vía The Odds API
    ($30/mes plan Basic). Si en el futuro se arreglan los scrapers, volver a
    activar poniendo APUESTAS_SCRAPE_US_BOOKS=true en .env.

    Se mantiene como no-op para compatibilidad con quien importe este task.
    """
    import os

    if os.environ.get("APUESTAS_SCRAPE_US_BOOKS", "false").lower() != "true":
        return {"skipped": 0, "reason": "via_the_odds_api"}  # type: ignore[dict-item]
    from apuestas.ingest.us_books_scraper import fetch_all

    try:
        results = await fetch_all(["nba", "mlb", "nfl", "nhl", "soccer_epl"])
        return {book: len(odds_list) for book, odds_list in results.items()}
    except Exception as exc:
        logger.warning("catchup.us_books_fail", error=str(exc)[:100])
        return {}


@task(retries=1, retry_delay_seconds=20)
async def catchup_polymarket() -> dict[str, int]:
    """Polymarket Gamma API — sharp reference #3 gratis (Sprint 6a).

    60 req/min sin auth. Persiste en `polymarket_markets` vía
    `polymarket.run_ingest()`.
    """
    import os

    if os.environ.get("ENABLE_POLYMARKET", "true").lower() != "true":
        return {"skipped": 0}  # type: ignore[dict-item]
    try:
        from apuestas.ingest.polymarket import run_ingest

        return await run_ingest()
    except Exception as exc:
        logger.warning("catchup.polymarket_fail", error=str(exc)[:120])
        return {}


@task(retries=1, retry_delay_seconds=20)
async def catchup_kambi_multi() -> dict[str, dict[str, int]]:
    """Kambi multi-operador: ub (Unibet) + comeon validados (HTTP 200 desde MX).

    Sprint B abr-2026: cada operador cuenta como sharp/soft adicional en
    line_shopping. Operadores extras (betsson, nordicbet, ...) activables via
    APUESTAS_KAMBI_OPERATORS=ub,comeon,betsson — tolerantes a 429/400.

    Activar con ENABLE_KAMBI_MULTI=true (default true).
    """
    import os

    if os.environ.get("ENABLE_KAMBI_MULTI", "true").lower() != "true":
        return {}
    try:
        from apuestas.ingest.kambi import DEFAULT_OPERATORS, run_kambi_multi_operator

        ops_env = os.environ.get("APUESTAS_KAMBI_OPERATORS", "")
        ops = (
            [o.strip() for o in ops_env.split(",") if o.strip()]
            if ops_env
            else list(DEFAULT_OPERATORS)
        )
        return await run_kambi_multi_operator(operators=ops)
    except Exception as exc:
        logger.warning("catchup.kambi_multi_fail", error=str(exc)[:120])
        return {}


@task(retries=1, retry_delay_seconds=20)
async def catchup_polymarket_games() -> dict[str, int]:
    """Polymarket game-by-game markets → odds_history bookmaker='polymarket'.

    Sprint B abr-2026: matches game markets a upcoming events del bot via
    fuzzy team name match + CLOB midpoint = prob implícita pura. Recorta
    dependencia de the-odds-api en NBA/NHL/MLB/NFL/soccer top.

    Activar con ENABLE_POLYMARKET_GAMES=true (default true: endpoint público
    estable validado HTTP 200 desde MX).
    """
    import os

    if os.environ.get("ENABLE_POLYMARKET_GAMES", "true").lower() != "true":
        return {"skipped": 0}  # type: ignore[dict-item]
    try:
        from apuestas.ingest.polymarket import ingest_game_markets

        return await ingest_game_markets(hours_ahead=48)
    except Exception as exc:
        logger.warning("catchup.polymarket_games_fail", error=str(exc)[:120])
        return {}


@task(retries=1, retry_delay_seconds=20)
async def catchup_kalshi() -> dict[str, int]:
    """Kalshi sports contracts (Sprint 6b).

    Default OFF por riesgo VPN MX. Activar con ENABLE_KALSHI=true.
    """
    import os

    if os.environ.get("ENABLE_KALSHI", "false").lower() != "true":
        return {"skipped": 0}  # type: ignore[dict-item]
    try:
        from apuestas.ingest.kalshi import run_kalshi_ingest

        return await run_kalshi_ingest()
    except Exception as exc:
        logger.warning("catchup.kalshi_fail", error=str(exc)[:120])
        return {}


_ODDSJAM_EMPTY_CYCLES = 0  # contador in-memory de ciclos vacíos consecutivos
_ODDSJAM_AUTO_DISABLE_THRESHOLD = 3


@task(retries=1, retry_delay_seconds=20)
async def catchup_oddsjam() -> dict[str, int]:
    """OddsJam backend — 85+ books soft/sharp sin auth (Sprint 4a).

    Gating por `ENABLE_ODDSJAM` (default true). Itera sobre los deportes
    canónicos usados por el bot; cada uno retorna rows_inserted.
    Fail-safe: un error por deporte no aborta los demás.

    Sprint B abr-2026: AUTO-DISABLE on 3 ciclos vacíos consecutivos
    (endpoint backend OddsJam es scraping interno, propenso a romper sin
    aviso). El auto-disable persiste hasta que el operador reactive vía
    APUESTAS_ODDSJAM_RESET=1 + reinicio.
    """
    import os

    global _ODDSJAM_EMPTY_CYCLES

    if os.environ.get("APUESTAS_ODDSJAM_RESET", "0") == "1":
        _ODDSJAM_EMPTY_CYCLES = 0

    if os.environ.get("ENABLE_ODDSJAM", "true").lower() != "true":
        return {"skipped": 0}  # type: ignore[dict-item]

    if _ODDSJAM_EMPTY_CYCLES >= _ODDSJAM_AUTO_DISABLE_THRESHOLD:
        logger.warning(
            "catchup.oddsjam_auto_disabled",
            reason="3_empty_cycles",
            hint="set APUESTAS_ODDSJAM_RESET=1 + restart to retry",
        )
        return {"auto_disabled": 0}  # type: ignore[dict-item]

    from apuestas.ingest.oddsjam import ingest_oddsjam_sport

    sports = ["nba", "mlb", "nfl", "nhl", "soccer_epl", "soccer_laliga"]
    totals: dict[str, int] = {}
    for sp in sports:
        try:
            n = await ingest_oddsjam_sport(sp)
            totals[sp] = int(n)
        except Exception as exc:
            logger.warning("catchup.oddsjam_sport_fail", sport=sp, error=str(exc)[:120])
            totals[sp] = 0

    total_rows = sum(totals.values())
    if total_rows == 0:
        _ODDSJAM_EMPTY_CYCLES += 1
        logger.warning(
            "catchup.oddsjam_empty_cycle",
            consecutive=_ODDSJAM_EMPTY_CYCLES,
            threshold=_ODDSJAM_AUTO_DISABLE_THRESHOLD,
        )
    else:
        _ODDSJAM_EMPTY_CYCLES = 0
    return totals


@task(retries=2, retry_delay_seconds=30)
async def catchup_nba_scoreboard() -> int:
    try:
        games = await ingest_nba_today()
        return len(games)
    except Exception as exc:
        logger.warning("catchup.nba_scoreboard_fail", error=str(exc))
        return 0


@task(retries=1, retry_delay_seconds=60)
async def catchup_news() -> dict[str, int]:
    try:
        return await run_news_ingest_pipeline()
    except Exception as exc:
        logger.warning("catchup.news_fail", error=str(exc))
        return {"total": 0, "processed": 0, "skipped": 0}


@flow(name="apuestas-catchup", log_prints=True)
async def catchup_flow() -> dict[str, object]:
    """Ejecutable desde `make analyze` o `make up`. Paralelo donde posible.

    En Prefect 3.x `.result()` es sync (bloquea en hilo). Para código async
    puro, invocamos las funciones internas directamente (`.fn`) y usamos
    `asyncio.gather` para paralelismo real. No perdemos observabilidad
    porque cada función interna ya loggea.
    """
    logger.info("catchup.start")

    # Orden crítico: pinnacle primero (siembra teams); luego caliente fuzzy-matchea.
    pinnacle_result: Any = await catchup_pinnacle_guest.fn()

    # Sofascore sync descubre matches en sports que Pinnacle/Kambi no cubren
    # (tennis Challenger, liga mx expansion, segunda europea, KBO, etc.)
    from apuestas.flows.sofascore_sync import sofascore_sync_flow

    gathered: list[Any] = await asyncio.gather(
        catchup_soccer_fixtures.fn(),
        catchup_soccer_odds.fn(),
        catchup_odds_api.fn(),
        catchup_nba_scoreboard.fn(),
        catchup_news.fn(),
        catchup_betfair_exchange.fn(),
        catchup_us_books.fn(),
        catchup_caliente.fn(),
        catchup_codere.fn(),
        sofascore_sync_flow(days_ahead=2),
        catchup_oddsjam.fn(),  # Sprint 4a: +85 books
        catchup_polymarket.fn(),  # Sprint 6a: sharp reference CLOB (futures)
        catchup_kalshi.fn(),  # Sprint 6b: Kalshi (off by default)
        catchup_polymarket_games.fn(),  # Sprint B abr-2026: game-by-game → odds_history
        catchup_kambi_multi.fn(),  # Sprint B abr-2026: Kambi multi-operador
        return_exceptions=True,
    )
    fixtures = gathered[0] if not isinstance(gathered[0], BaseException) else {}
    odds_soccer = gathered[1] if not isinstance(gathered[1], BaseException) else {}
    odds_api = gathered[2] if not isinstance(gathered[2], BaseException) else {}
    nba = gathered[3] if not isinstance(gathered[3], BaseException) else 0
    news = gathered[4] if not isinstance(gathered[4], BaseException) else {}
    pinnacle = pinnacle_result if not isinstance(pinnacle_result, BaseException) else {}
    betfair = gathered[5] if not isinstance(gathered[5], BaseException) else {}
    us_books = gathered[6] if not isinstance(gathered[6], BaseException) else {}
    caliente = gathered[7] if not isinstance(gathered[7], BaseException) else {}
    codere = gathered[8] if not isinstance(gathered[8], BaseException) else {}
    sofascore = gathered[9] if not isinstance(gathered[9], BaseException) else {}
    oddsjam = gathered[10] if not isinstance(gathered[10], BaseException) else {}
    polymarket_res = gathered[11] if not isinstance(gathered[11], BaseException) else {}
    kalshi_res = gathered[12] if not isinstance(gathered[12], BaseException) else {}
    polymarket_games_res = (
        gathered[13] if len(gathered) > 13 and not isinstance(gathered[13], BaseException) else {}
    )
    kambi_multi_res = (
        gathered[14] if len(gathered) > 14 and not isinstance(gathered[14], BaseException) else {}
    )

    await update_checkpoint("catchup", "full_sweep")

    # Actualizar scores de partidos terminados y disparar settlement automático.
    # Se corre DESPUÉS de la ingesta de odds para que los matches ya estén en BD.
    from apuestas.flows.live_scores import live_scores_flow

    # Identity repair: corrige league_id mal-asignados ANTES de live_scores
    # para que el filtrado por league.external_id encuentre las claves correctas.
    # Idempotente — no hace nada si todo ya está bien resuelto.
    try:
        from apuestas.maintenance.identity_repair import repair_league_assignments

        await repair_league_assignments()
    except Exception as exc:
        logger.debug("catchup.identity_repair_fail", error=str(exc)[:120])

    live_scores_result: dict[str, int] = {}
    try:
        live_scores_result = await live_scores_flow(window_hours=72)
    except Exception as exc:
        logger.warning("catchup.live_scores_fail", error=str(exc))

    # Si quedan picks atascados tras live_scores 72h, hace una segunda pasada
    # con ventana extendida 14 días — captura backlog histórico (picks de >3
    # días sin liquidar tras outages prolongados o ligas sin cobertura previa).
    try:
        from apuestas.monitors.watchdog import check_stuck_picks

        stuck = await check_stuck_picks(hours_after_kickoff=6)
        if stuck:
            logger.info("catchup.stuck_picks_detected_running_extended", count=len(stuck))
            try:
                await live_scores_flow(window_hours=24 * 14)
            except Exception as exc:
                logger.warning("catchup.live_scores_extended_fail", error=str(exc))

        from apuestas.monitors.watchdog import run_watchdog

        await run_watchdog()
    except Exception as exc:
        logger.debug("catchup.watchdog_fail", error=str(exc)[:120])

    # Reclasificar picks históricos contra la lógica actual de _classify_alert.
    # Cubre el caso donde se corrige un bug de clasificación (ej: spreads/away
    # signo invertido pre-2026-04-24) y los outcome_result existentes están mal.
    # Idempotente: solo escribe cuando hay diferencia.
    try:
        from apuestas.maintenance.reclassify_alerts import reclassify_resolved_alerts

        await reclassify_resolved_alerts(since_days=30)
    except Exception as exc:
        logger.debug("catchup.reclassify_fail", error=str(exc)[:120])

    summary = {
        "fixtures": fixtures,
        "odds_soccer": odds_soccer,
        "odds_api": odds_api,
        "nba_today": nba,
        "news": news,
        "pinnacle_guest": pinnacle,
        "betfair_exchange": betfair,
        "us_books": us_books,
        "caliente": caliente,
        "codere": codere,
        "sofascore": sofascore,
        "oddsjam": oddsjam,
        "polymarket": polymarket_res,
        "kalshi": kalshi_res,
        "polymarket_games": polymarket_games_res,
        "kambi_multi": kambi_multi_res,
        "live_scores": live_scores_result,
    }
    logger.info("catchup.done", **{k: str(v) for k, v in summary.items()})
    return summary


if __name__ == "__main__":
    asyncio.run(catchup_flow())
