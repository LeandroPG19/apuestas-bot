"""Flow live_scores — ingesta resultados post-match (§19.7).

Marca matches con `status='finished'` y persiste home_score/away_score.
Se ejecuta tras partidos terminados (T + 2h) dentro del flow `deep_analysis`
o via timer systemd user.

Post-pivote 2026-04-23: ya no dispara settle_bets (subsistema demolido).
Sprint 3 añadirá `mark_alert_results(match_id)` para escribir
`pick_alerts.outcome_result` (won/lost/void) sin PnL.

Fuentes: API-Football fixtures (status=FT/AET/PEN) para fútbol;
nba_api boxscore para NBA; NHL Stats API para NHL; The Odds API
scores endpoint para NFL/MLB; Jeff Sackmann/API-Tennis para tenis.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

from prefect import flow, task
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.ingest.api_football import LEAGUE_IDS, APIFootballClient
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

FINAL_STATUSES = {"FT", "AET", "PEN", "AWD", "WO"}


@task(retries=2, retry_delay_seconds=20)
async def pending_finished_matches(window_hours: int = 48) -> list[dict[str, Any]]:
    """Matches que empezaron hace >=2h y aún status != 'finished'.

    Prioriza matches con pick_alerts vivos (prevenir que picks queden atascados
    eternamente cuando hay backlog grande de matches scheduled-pero-finalizados
    sin score guardado — bug detectado fin de semana 25-26 abr 2026).
    """
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT m.id, m.sport_code, m.league_id, m.external_id,
                       m.external_id_nba, m.external_id_nhl, m.external_id_odds_api,
                       m.start_time,
                       EXISTS (
                           SELECT 1 FROM pick_alerts pa
                           WHERE pa.match_id = m.id
                             AND (pa.outcome_result IS NULL OR pa.outcome_result = 'pending')
                       ) AS has_pending_alerts
                FROM matches m
                WHERE m.status <> 'finished'
                  AND m.start_time <= NOW() - INTERVAL '2 hours'
                  AND m.start_time >= NOW() - INTERVAL ':w hours'
                ORDER BY has_pending_alerts DESC, m.start_time DESC
                LIMIT 5000
                """.replace(":w", str(window_hours))
            )
        )
        return [dict(r._mapping) for r in result.all()]


async def _finalize_match(
    *, match_id: int, home_score: int, away_score: int, final_status: str
) -> None:
    """Finaliza el match + actualiza outcome_result de sus pick_alerts.

    Post-pivote 2026-04-23: tras marcar el match como 'finished', invoca
    `mark_alert_results` para clasificar cada alerta viva como
    won/lost/void/halfwon/halflost según score + market + outcome. Sin PnL.
    """
    async with session_scope() as session:
        await session.execute(
            text(
                """
                UPDATE matches
                SET home_score = :hs,
                    away_score = :as_,
                    status = 'finished'
                WHERE id = :mid
                """
            ),
            {"mid": match_id, "hs": home_score, "as_": away_score},
        )
        await mark_alert_results(session, match_id)
        _ = final_status  # reservado para futura columna final_status_raw


async def mark_alert_results(session: Any, match_id: int) -> int:
    """Asigna outcome_result a las alertas vivas de un match recién finalizado.

    Lógica por market:
      - h2h: home/away ganador; draw si empate (soccer). Un outcome
        ausente del ganador → 'lost'.
      - spreads: home - away + line >= 0 (home cubre); si 0 exacto → 'void'.
      - totals: total vs line (over/under); push → 'void'.

    No calcula PnL; solo escribe `outcome_result` y `result_settled_at`.
    Retorna el número de alertas actualizadas.
    """
    match = (
        await session.execute(
            text(
                """
                SELECT id, home_score, away_score, sport_code
                FROM matches WHERE id = :id
                """
            ),
            {"id": match_id},
        )
    ).first()
    if match is None or match.home_score is None or match.away_score is None:
        return 0

    hs = int(match.home_score)
    as_ = int(match.away_score)
    sport = match.sport_code or ""

    alerts = (
        await session.execute(
            text(
                """
                SELECT pa.id, pa.market, pa.outcome, pa.line,
                       p.probability AS p_model
                FROM pick_alerts pa
                LEFT JOIN predictions p ON p.id = pa.prediction_id
                WHERE pa.match_id = :mid
                  AND (pa.outcome_result IS NULL OR pa.outcome_result = 'pending')
                """
            ),
            {"mid": match_id},
        )
    ).all()

    # Sprint 7: feed ADWIN/Page-Hinkley drift monitor
    from apuestas.monitors.concept_drift import (
        BrierDriftMonitor,
        trigger_retrain_event,
    )

    drift_monitor = BrierDriftMonitor.get()

    updated = 0
    for a in alerts:
        result = _classify_alert(
            market=str(a.market).lower(),
            outcome=str(a.outcome).lower(),
            line=float(a.line) if a.line is not None else None,
            home_score=hs,
            away_score=as_,
            sport=sport,
        )
        if result is None:
            # Props u otros markets sin regla sencilla → quedan pending
            # hasta que Sprint futuro añada su classifier.
            continue
        await session.execute(
            text(
                """
                UPDATE pick_alerts
                SET outcome_result = :r,
                    result_settled_at = now()
                WHERE id = :id
                """
            ),
            {"r": result, "id": int(a.id)},
        )
        updated += 1

        # Drift feed — sólo para mercados 2-way (won/lost binario).
        if result in ("won", "lost") and a.p_model is not None:
            try:
                y = 1 if result == "won" else 0
                if drift_monitor.update(
                    sport, str(a.market).lower(), pred=float(a.p_model), actual=y
                ):
                    await trigger_retrain_event(sport, str(a.market).lower())
            except Exception as exc:
                logger.debug("drift.feed_fail", error=str(exc)[:80])
    logger.info(
        "mark_alert_results.done",
        match_id=match_id,
        updated=updated,
        n_alerts=len(alerts),
    )
    return updated


def _classify_alert(
    *,
    market: str,
    outcome: str,
    line: float | None,
    home_score: int,
    away_score: int,
    sport: str,
) -> str | None:
    """Clasifica una alerta según market + outcome + score.

    Retorna 'won' / 'lost' / 'void' / 'halfwon' / 'halflost' / None.
    None = market no soportado (props, etc.) → la alerta queda pending.
    """
    # ── h2h / moneyline ──
    if market in ("h2h", "moneyline", "1x2"):
        if home_score > away_score:
            winner = "home"
        elif away_score > home_score:
            winner = "away"
        else:
            winner = "draw"
        if outcome == winner:
            return "won"
        # Deportes sin empate: un empate (imposible en OT forzada) se maneja
        # como void por si el bot emitió un pick antes de OT.
        if winner == "draw" and sport not in ("soccer", "boxing", "mma"):
            return "void"
        return "lost"

    # ── spreads / handicap ──
    if market in ("spreads", "handicap", "ah"):
        if line is None:
            return None
        diff = home_score - away_score + line
        if outcome == "home":
            if diff > 0:
                return "won"
            if diff == 0:
                return "void"
            return "lost"
        if outcome == "away":
            inv = away_score - home_score + line
            if inv > 0:
                return "won"
            if inv == 0:
                return "void"
            return "lost"
        return None

    # ── totals / over-under ──
    if market in ("totals", "total", "over_under"):
        if line is None:
            return None
        total = home_score + away_score
        if outcome == "over":
            if total > line:
                return "won"
            if total == line:
                return "void"
            return "lost"
        if outcome == "under":
            if total < line:
                return "won"
            if total == line:
                return "void"
            return "lost"
        return None

    return None  # props / mercados no cubiertos por Sprint 3


@task(retries=2, retry_delay_seconds=30)
async def sync_soccer_scores_fdo(matches: list[dict[str, Any]], *, days_back: int = 3) -> int:
    """Scores soccer via football-data.org (free tier, 10 req/min, cobertura top-5+UCL).

    Matching por nombre de equipo + fecha exacta (FDO team IDs son estables pero
    no coinciden con Sofascore/Pinnacle). Usa fuzzy match con RapidFuzz.
    """
    from rapidfuzz import fuzz

    from apuestas.ingest.football_data_org import (
        COMPETITION_CODE_MAP,
        FootballDataOrgClient,
    )
    from apuestas.ingest.football_data_org import (
        FINAL_STATUSES as FDO_FINAL,
    )

    soccer_matches = [m for m in matches if m["sport_code"] == "soccer"]
    if not soccer_matches:
        return 0

    try:
        client = FootballDataOrgClient()
    except ValueError:
        logger.info("live_scores.fdo_key_missing")
        return 0

    # Lookup (home_norm, away_norm, date_str) → match_id
    match_lookup: dict[tuple[str, str, str], int] = {}
    async with session_scope() as session:
        for m in soccer_matches:
            row = (
                await session.execute(
                    text(
                        """
                        SELECT ht.name AS home, at.name AS away
                        FROM matches m
                        JOIN teams ht ON ht.id = m.home_team_id
                        JOIN teams at ON at.id = m.away_team_id
                        WHERE m.id = :mid
                        """
                    ),
                    {"mid": m["id"]},
                )
            ).first()
            if row:
                date_str = m["start_time"].strftime("%Y-%m-%d")
                key = (_normalize_team(row.home), _normalize_team(row.away), date_str)
                match_lookup[key] = int(m["id"])
    if not match_lookup:
        return 0

    updated = 0
    now = datetime.now(tz=UTC)
    date_from = now - timedelta(days=int(days_back))
    async with client.session():
        for _internal_sport, fdo_code in COMPETITION_CODE_MAP.items():
            try:
                fixtures = await client.fetch_matches(
                    fdo_code,
                    date_from=date_from,
                    date_to=now,
                    status="FINISHED",
                )
            except Exception as exc:
                logger.debug(
                    "live_scores.fdo_fetch_fail", competition=fdo_code, error=str(exc)[:120]
                )
                continue

            for fx in fixtures:
                if fx.get("status") not in FDO_FINAL:
                    continue
                score = fx.get("score", {}).get("fullTime", {})
                hs = score.get("home")
                as_ = score.get("away")
                if hs is None or as_ is None:
                    continue
                home_name = _normalize_team((fx.get("homeTeam") or {}).get("name", ""))
                away_name = _normalize_team((fx.get("awayTeam") or {}).get("name", ""))
                date_str = (fx.get("utcDate") or "")[:10]
                mid = match_lookup.get((home_name, away_name, date_str))
                if mid is None:
                    # Fuzzy fallback: mismo día + similaridad ≥ 0.82 en ambos teams
                    for (lh, la, ld), candidate_id in match_lookup.items():
                        if ld != date_str:
                            continue
                        if fuzz.WRatio(home_name, lh) >= 82 and fuzz.WRatio(away_name, la) >= 82:
                            mid = candidate_id
                            break
                if mid is None:
                    continue
                await _finalize_match(
                    match_id=mid,
                    home_score=int(hs),
                    away_score=int(as_),
                    final_status="FT",
                )
                updated += 1

    return updated


@task(retries=2, retry_delay_seconds=30)
async def sync_soccer_scores(matches: list[dict[str, Any]]) -> int:
    updated = 0
    if not matches:
        return 0
    league_to_matches: dict[int, list[dict[str, Any]]] = {}
    for m in matches:
        if m["sport_code"] != "soccer" or m.get("league_id") is None:
            continue
        league_to_matches.setdefault(int(m["league_id"]), []).append(m)
    if not league_to_matches:
        return 0

    # Fix 2026-04-24: skip if api_football key is placeholder → evita 403 spam
    from apuestas.ingest.api_football import _api_football_key_available

    if not _api_football_key_available():
        logger.info("live_scores.api_football_skipped_no_key")
        return 0

    reverse_leagues = {v: k for k, v in LEAGUE_IDS.items()}
    client = APIFootballClient()
    async with client.session():
        for league_id, group in league_to_matches.items():
            slug = reverse_leagues.get(league_id, f"league_{league_id}")
            season = datetime.now(tz=UTC).year
            try:
                fixtures = await client.fetch_fixtures(
                    league=league_id,
                    season=season,
                    date_from=datetime.now(tz=UTC) - timedelta(days=3),
                    date_to=datetime.now(tz=UTC),
                )
            except Exception as exc:
                logger.warning("live_scores.soccer_fetch_fail", league=slug, error=str(exc))
                continue

            by_external = {str(f["fixture"]["id"]): f for f in fixtures if "fixture" in f}
            for match in group:
                ext = str(match.get("external_id") or "")
                fx = by_external.get(ext)
                if not fx:
                    continue
                status = fx.get("fixture", {}).get("status", {}).get("short", "")
                if status not in FINAL_STATUSES:
                    continue
                goals = fx.get("goals") or {}
                hs = goals.get("home")
                as_ = goals.get("away")
                if hs is None or as_ is None:
                    continue
                await _finalize_match(
                    match_id=int(match["id"]),
                    home_score=int(hs),
                    away_score=int(as_),
                    final_status=status,
                )
                updated += 1
    return updated


@task(retries=2, retry_delay_seconds=30)
async def sync_nba_scores_native(matches: list[dict[str, Any]]) -> int:
    """NBA via nba_api boxscore — REQUIERE `external_id_nba` poblado.

    Usa el game_id oficial de nba_api. Si no hay IDs nativos guardados,
    retorna 0 y el fallback (sync_odds_api_scores con team-name matching)
    se encarga desde live_scores_flow.

    Los IDs nativos se pueblan en el flow de ingesta cuando se llama a
    nba_api para obtener schedule/matchups.
    """
    nba_matches = [m for m in matches if m["sport_code"] == "nba" and m.get("external_id_nba")]
    if not nba_matches:
        return 0

    def _fetch_boxscore(game_id: str) -> dict[str, Any] | None:
        try:
            # boxscoresummaryv3 preferido (v2 deprecado desde 4/10/2025).
            from nba_api.stats.endpoints import boxscoresummaryv2

            bs = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id, timeout=20)
            summary = bs.get_normalized_dict()
            line_score = summary.get("LineScore", [])
            if len(line_score) < 2:
                return None
            # Verificar orden por TEAM_ID en lugar de asumir [away, home]
            # nba_api convención: line_score[0]=visitor, line_score[1]=home.
            return {
                "home_pts": int(line_score[1].get("PTS") or 0),
                "away_pts": int(line_score[0].get("PTS") or 0),
                "status": "Final",
            }
        except Exception as exc:
            logger.debug("live_scores.nba_boxscore_fail", game_id=game_id, error=str(exc))
            return None

    updated = 0
    for match in nba_matches:
        ext = str(match["external_id_nba"])
        bs = await asyncio.to_thread(_fetch_boxscore, ext)
        if not bs or bs.get("status") != "Final":
            continue
        await _finalize_match(
            match_id=int(match["id"]),
            home_score=bs["home_pts"],
            away_score=bs["away_pts"],
            final_status="FT",
        )
        updated += 1
    return updated


@task(retries=2, retry_delay_seconds=30)
async def sync_nhl_scores_native(matches: list[dict[str, Any]]) -> int:
    """NHL via api-web.nhle.com — REQUIERE `external_id_nhl` poblado.

    Si no hay IDs nativos guardados, retorna 0 y el fallback
    (sync_odds_api_scores con team-name matching) se encarga.
    """
    from apuestas.ingest.http_base import BaseAPIClient

    nhl_matches = [m for m in matches if m["sport_code"] == "nhl" and m.get("external_id_nhl")]
    if not nhl_matches:
        return 0

    class _NHLClient(BaseAPIClient):
        base_url = "https://api-web.nhle.com/v1"
        source_name = "nhl"
        rate_limit = (60, 60.0)

        def _default_headers(self) -> dict[str, str]:
            return {"Accept": "application/json"}

    updated = 0
    client = _NHLClient(api_key="")
    async with client.session():
        for match in nhl_matches:
            ext = str(match["external_id_nhl"])
            try:
                data = await client.get(f"/gamecenter/{ext}/boxscore", params=None)
            except Exception as exc:
                logger.debug("live_scores.nhl_fail", game_id=ext, error=str(exc))
                continue
            if data.get("gameState") not in {"OFF", "FINAL"}:
                continue
            home = data.get("homeTeam", {})
            away = data.get("awayTeam", {})
            hs = home.get("score")
            as_ = away.get("score")
            if hs is None or as_ is None:
                continue
            await _finalize_match(
                match_id=int(match["id"]),
                home_score=int(hs),
                away_score=int(as_),
                final_status="FT",
            )
            updated += 1
    return updated


def _normalize_team(name: str) -> str:
    """Normaliza nombre de equipo para matching fuzzy.

    - Minúsculas, despunctuación, despunto.
    - Quita acentos / caracteres unicode (Leganés → leganes, Bodø → bodo).
    - Elimina prefijos/sufijos comunes que varían entre fuentes:
      FC, SC, AS, AC, CF, SD, AD, CD, BSC, RC, RB, VfB, VfL, FSV, TSG, BK, SK, FK,
      "1.", "1. FC", "Club", "City", "United" (este último NO se borra).
    """
    import re
    import unicodedata

    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Tokens-prefijo de club que aparecen/desaparecen entre proveedores.
    # United/City/Town/Athletic NO se quitan (parte del nombre identitario).
    _STRIP_TOKENS = {
        "fc",
        "sc",
        "ac",
        "as",
        "cf",
        "sd",
        "ad",
        "cd",
        "bsc",
        "rc",
        "rb",
        "vfb",
        "vfl",
        "fsv",
        "tsg",
        "bk",
        "sk",
        "fk",
        "1",
        "club",
        "real",  # ambiguo: Real Oviedo y Real Madrid se pierden, pero es estable entre fuentes
    }
    tokens = [t for t in s.split() if t and t not in _STRIP_TOKENS]
    return " ".join(tokens)


def _fuzzy_key_match(
    target_home: str,
    target_away: str,
    date_str: str,
    lookup: dict[tuple[str, str, str], int],
    *,
    threshold: float = 0.82,
) -> int | None:
    """Busca match en lookup priorizando fecha exacta + fuzzy teams.

    Fallback escalonado:
    1. Exact (home, away, date) — ya hecho antes, retorna None si llega aquí.
    2. Fuzzy home+away con fecha exacta (threshold 0.82).
    3. Retorna None si ninguno supera threshold.
    """
    from difflib import SequenceMatcher

    best_score = threshold
    best_id: int | None = None
    for (lh, la, ld), mid in lookup.items():
        if ld != date_str:
            continue
        s_home = SequenceMatcher(None, target_home, lh).ratio()
        s_away = SequenceMatcher(None, target_away, la).ratio()
        # Requiere que AMBOS superen threshold para evitar false positives
        min_score = min(s_home, s_away)
        if min_score > best_score:
            best_score = min_score
            best_id = mid
    return best_id


@task(retries=2, retry_delay_seconds=30)
async def sync_soccer_scores_odds_paid(matches: list[dict[str, Any]], *, days_from: int = 3) -> int:
    """Scores soccer vía The Odds API PAID tier (cubre Liga MX + MLS + top 5 EU + UCL).

    Ventaja vs football-data.org free: cubre Liga MX y MLS (que free tier no tiene).
    Ventaja vs API-Football $19/mes: ya pagado ($30/mes 20k créditos, usando <10%).

    Por cada key paid activa, llama /scores con daysFrom=3 y hace fuzzy match
    por (home, away, date) contra los matches soccer en ventana.
    """
    from rapidfuzz import fuzz

    from apuestas.ingest.odds_api import INTERNAL_SPORT_TO_ODDS_KEY, OddsAPIClient

    soccer_matches = [
        m
        for m in matches
        if m["sport_code"]
        in ("soccer", "epl", "laliga", "bundesliga", "seriea", "ligue1", "liga_mx")
    ]
    if not soccer_matches:
        return 0

    try:
        client = OddsAPIClient()
    except ValueError:
        return 0

    # Budget guard: skip si quedan <20% del plan o <=80 créditos absolutos.
    # Sin esto, una sola corrida quema 40 créditos en /scores y el plan free
    # 500/mes se agota en horas.
    try:
        from apuestas.ingest.odds_api_optimized import get_budget_status

        _budget = await get_budget_status()
        if not _budget.get("safe_to_poll", True) or _budget.get("remaining", 1000) < 80:
            logger.warning(
                "live_scores.soccer_odds_paid_skip_budget",
                remaining=_budget.get("remaining"),
                remaining_pct=_budget.get("remaining_pct"),
            )
            return 0
    except Exception as _bexc:
        logger.debug("live_scores.budget_check_fail", error=str(_bexc)[:100])

    # Lookup (home_norm, away_norm, date_str) → match_id
    match_lookup: dict[tuple[str, str, str], int] = {}
    async with session_scope() as session:
        for m in soccer_matches:
            row = (
                await session.execute(
                    text(
                        """
                        SELECT ht.name AS home, at.name AS away
                        FROM matches m
                        JOIN teams ht ON ht.id = m.home_team_id
                        JOIN teams at ON at.id = m.away_team_id
                        WHERE m.id = :mid
                        """
                    ),
                    {"mid": m["id"]},
                )
            ).first()
            if row:
                date_str = m["start_time"].strftime("%Y-%m-%d")
                key = (_normalize_team(row.home), _normalize_team(row.away), date_str)
                match_lookup[key] = int(m["id"])
    if not match_lookup:
        return 0

    # Itera SOLO las ligas con matches en la ventana (no las 39 default).
    # Cada call /scores cuesta 1 crédito; sin filtrar gastábamos 39 créditos
    # por invocación aunque solo hubiera matches en 2-3 ligas.
    sport_keys_all = INTERNAL_SPORT_TO_ODDS_KEY.get("soccer", [])
    leagues_in_window: set[str] = set()
    matches_with_league = 0
    async with session_scope() as session:
        rows = await session.execute(
            text(
                """
                SELECT DISTINCT l.external_id, COUNT(m.id) AS n_matches
                FROM matches m
                JOIN leagues l ON l.id = m.league_id
                WHERE m.id = ANY(:mids)
                  AND l.external_id IS NOT NULL
                GROUP BY l.external_id
                """
            ),
            {"mids": [m["id"] for m in soccer_matches]},
        )
        for r in rows:
            if r.external_id:
                leagues_in_window.add(str(r.external_id))
                matches_with_league += int(r.n_matches)

    # Si hay matches sin league_id resolvible, expandir a un set acotado.
    # Antes: fallback a TODAS las 39 keys → 39 créditos/call → en 5min agotaba
    # plan free 500/mes. Ahora: solo top-10 ligas con más volumen histórico.
    # Trade-off: matches en ligas exóticas (Suiza, Noruega) pueden quedar
    # pending un ciclo extra hasta que identity_repair les asigne league_id.
    sport_keys = [k for k in sport_keys_all if k in leagues_in_window]
    n_unmapped = len(soccer_matches) - matches_with_league
    _SOCCER_TOP_KEYS = (
        "soccer_epl",
        "soccer_spain_la_liga",
        "soccer_germany_bundesliga",
        "soccer_italy_serie_a",
        "soccer_france_ligue_one",
        "soccer_uefa_champs_league",
        "soccer_uefa_europa_league",
        "soccer_mexico_ligamx",
        "soccer_usa_mls",
        "soccer_brazil_campeonato",
    )
    if n_unmapped > 0 or not sport_keys:
        budget_aware_keys = [k for k in _SOCCER_TOP_KEYS if k in sport_keys_all]
        existing = set(sport_keys)
        for k in budget_aware_keys:
            if k not in existing:
                sport_keys.append(k)
        logger.info(
            "live_scores.soccer_expand_top10",
            in_window=list(leagues_in_window)[:10],
            total_keys_capped=len(sport_keys),
            unmapped_matches=n_unmapped,
        )
    updated = 0
    async with client.session():
        for sport_key in sport_keys:
            try:
                raw: list[dict[str, Any]] = await client.get(
                    f"/sports/{sport_key}/scores",
                    params={
                        "apiKey": client._key,
                        "daysFrom": int(days_from),
                        "dateFormat": "iso",
                    },
                )
            except Exception as exc:
                logger.debug(
                    "live_scores.odds_soccer_fail", sport_key=sport_key, error=str(exc)[:120]
                )
                continue

            for game in raw:
                if not game.get("completed"):
                    continue
                scores_raw = game.get("scores") or []
                if len(scores_raw) < 2:
                    continue
                home_name = _normalize_team(game.get("home_team", ""))
                away_name = _normalize_team(game.get("away_team", ""))
                date_str = (game.get("commence_time") or "")[:10]
                mid = match_lookup.get((home_name, away_name, date_str))
                if mid is None:
                    # fuzzy fallback
                    for (lh, la, ld), cid in match_lookup.items():
                        if ld != date_str:
                            continue
                        if fuzz.WRatio(home_name, lh) >= 82 and fuzz.WRatio(away_name, la) >= 82:
                            mid = cid
                            break
                if mid is None:
                    continue
                score_by_team = {_normalize_team(s["name"]): int(s["score"]) for s in scores_raw}
                hs = score_by_team.get(home_name)
                as_ = score_by_team.get(away_name)
                if hs is None or as_ is None:
                    for sname, sscore in score_by_team.items():
                        if hs is None and fuzz.WRatio(sname, home_name) >= 82:
                            hs = sscore
                        if as_ is None and fuzz.WRatio(sname, away_name) >= 82:
                            as_ = sscore
                    if hs is None or as_ is None:
                        continue
                await _finalize_match(
                    match_id=mid, home_score=hs, away_score=as_, final_status="FT"
                )
                updated += 1

    return updated


@task(retries=2, retry_delay_seconds=30)
async def sync_odds_api_scores(
    matches: list[dict[str, Any]], sport_code: str, *, days_from: int = 3
) -> int:
    """Carga scores via The Odds API /scores endpoint.

    Usado para MLB y otros deportes sin handler dedicado.
    Matching por nombre de equipo normalizado + fecha, ya que el external_id
    está en formato Pinnacle y no coincide con los IDs de The Odds API.
    """
    from apuestas.ingest.odds_api import SPORT_KEY_MAP, OddsAPIClient

    target = [m for m in matches if m["sport_code"] == sport_code]
    if not target:
        return 0

    sport_key = SPORT_KEY_MAP.get(sport_code)
    if not sport_key:
        logger.warning("live_scores.odds_api_no_sport_key", sport_code=sport_code)
        return 0

    try:
        client = OddsAPIClient()
    except ValueError:
        logger.info("live_scores.odds_api_key_missing", sport_code=sport_code)
        return 0

    # Construir lookup por (home_normalized, away_normalized, date) desde la BD
    match_lookup: dict[tuple[str, str, str], int] = {}
    async with session_scope() as session:
        for m in target:
            row = (
                await session.execute(
                    text(
                        """
                        SELECT ht.name AS home, at.name AS away
                        FROM matches m
                        JOIN teams ht ON ht.id = m.home_team_id
                        JOIN teams at ON at.id = m.away_team_id
                        WHERE m.id = :mid
                        """
                    ),
                    {"mid": m["id"]},
                )
            ).first()
            if row:
                date_str = m["start_time"].strftime("%Y-%m-%d")
                key = (_normalize_team(row.home), _normalize_team(row.away), date_str)
                match_lookup[key] = int(m["id"])

    if not match_lookup:
        return 0

    try:
        async with client.session():
            raw: list[dict[str, Any]] = await client.get(
                f"/sports/{sport_key}/scores",
                params={
                    "apiKey": client._key,
                    "daysFrom": int(days_from),
                    "dateFormat": "iso",
                },
            )
    except Exception as exc:
        logger.warning("live_scores.odds_api_scores_fail", sport=sport_code, error=str(exc))
        return 0

    updated = 0
    for game in raw:
        if not game.get("completed"):
            continue
        scores_raw = game.get("scores") or []
        if len(scores_raw) < 2:
            continue
        home_name = _normalize_team(game.get("home_team", ""))
        away_name = _normalize_team(game.get("away_team", ""))
        date_str = (game.get("commence_time") or "")[:10]
        match_id = match_lookup.get((home_name, away_name, date_str))
        if match_id is None:
            # Fuzzy fallback: matching por similaridad de nombres + fecha exacta.
            # Resuelve variantes ("NY Yankees" vs "New York Yankees", acentos, etc).
            match_id = _fuzzy_key_match(home_name, away_name, date_str, match_lookup)
            if match_id is None:
                continue
            logger.debug(
                "live_scores.fuzzy_match",
                sport=sport_code,
                api_home=home_name,
                api_away=away_name,
                match_id=match_id,
            )
        # scores_raw: [{name, score}, ...] — home primero según The Odds API docs
        score_by_team = {_normalize_team(s["name"]): int(s["score"]) for s in scores_raw}
        hs = score_by_team.get(home_name)
        as_ = score_by_team.get(away_name)
        if hs is None or as_ is None:
            # The Odds API devuelve team names como los usa en su catálogo;
            # matcheamos por similaridad con los nombres de la lookup entry.
            from difflib import SequenceMatcher

            for sname, sscore in score_by_team.items():
                if hs is None and SequenceMatcher(None, sname, home_name).ratio() > 0.82:
                    hs = sscore
                if as_ is None and SequenceMatcher(None, sname, away_name).ratio() > 0.82:
                    as_ = sscore
            if hs is None or as_ is None:
                continue
        await _finalize_match(match_id=match_id, home_score=hs, away_score=as_, final_status="FT")
        updated += 1

    return updated


@flow(name="apuestas-live-scores", log_prints=True)
async def live_scores_flow(*, window_hours: int = 48) -> dict[str, int]:
    matches = await pending_finished_matches(window_hours=window_hours)
    logger.info("live_scores.start", candidates=len(matches))
    if not matches:
        return {"candidates": 0, "updated_total": 0}

    # daysFrom de TheOddsAPI scores está capado a 3 días por el provider.
    # Para resolver picks ancianos (4-14d) la ruta es FDO (cobertura top-5+UCL)
    # con date_from extendido — controlado por window_hours.
    days_from = 3

    # Estrategia por deporte (The Odds API paid $30/mes 20k créditos — primary):
    # Pre-filtro sport_focus: si nhl/nfl están off, ni siquiera llamamos su /scores
    # endpoint (cada llamada cuesta 1 crédito × N sports × 96 ciclos/día = ~200/día
    # desperdiciados en sports que no se procesan).
    from apuestas.betting.sport_focus import is_emit_enabled

    async def _noop() -> int:
        return 0

    nba_on = is_emit_enabled("nba")
    nhl_on = is_emit_enabled("nhl")
    mlb_on = is_emit_enabled("mlb")
    nfl_on = is_emit_enabled("nfl")
    soccer_on = is_emit_enabled("soccer")

    results: list[Any] = await asyncio.gather(
        sync_soccer_scores_odds_paid.fn(matches, days_from=days_from) if soccer_on else _noop(),
        sync_soccer_scores_fdo.fn(matches, days_back=max(3, window_hours // 24))
        if soccer_on
        else _noop(),
        sync_soccer_scores.fn(matches) if soccer_on else _noop(),
        sync_nba_scores_native.fn(matches) if nba_on else _noop(),
        sync_nhl_scores_native.fn(matches) if nhl_on else _noop(),
        sync_odds_api_scores.fn(matches, "nba", days_from=days_from) if nba_on else _noop(),
        sync_odds_api_scores.fn(matches, "nhl", days_from=days_from) if nhl_on else _noop(),
        sync_odds_api_scores.fn(matches, "mlb", days_from=days_from) if mlb_on else _noop(),
        sync_odds_api_scores.fn(matches, "nfl", days_from=days_from) if nfl_on else _noop(),
        return_exceptions=True,
    )
    soccer_odds: int = results[0] if not isinstance(results[0], BaseException) else 0
    soccer_fdo: int = results[1] if not isinstance(results[1], BaseException) else 0
    soccer_af: int = results[2] if not isinstance(results[2], BaseException) else 0
    nba_native: int = results[3] if not isinstance(results[3], BaseException) else 0
    nhl_native: int = results[4] if not isinstance(results[4], BaseException) else 0
    nba_odds: int = results[5] if not isinstance(results[5], BaseException) else 0
    nhl_odds: int = results[6] if not isinstance(results[6], BaseException) else 0
    mlb_n: int = results[7] if not isinstance(results[7], BaseException) else 0
    nfl_n: int = results[8] if not isinstance(results[8], BaseException) else 0

    soccer_n = soccer_odds + soccer_fdo + soccer_af
    nba_n = nba_native + nba_odds
    nhl_n = nhl_native + nhl_odds

    total = soccer_n + nba_n + nhl_n + mlb_n + nfl_n
    # Prometheus counters (Deuda 2 wire)
    try:
        from apuestas.obs.metrics import inc_live_scores_updated

        for sport, n in (
            ("soccer", soccer_n),
            ("nba", nba_n),
            ("nhl", nhl_n),
            ("mlb", mlb_n),
            ("nfl", nfl_n),
        ):
            if n > 0:
                inc_live_scores_updated(sport, int(n))
    except Exception:
        pass
    logger.info(
        "live_scores.done",
        candidates=len(matches),
        soccer=soccer_n,
        nba=nba_n,
        nhl=nhl_n,
        mlb=mlb_n,
        nfl=nfl_n,
        total=total,
    )
    return {
        "candidates": len(matches),
        "soccer_updated": soccer_n,
        "nba_updated": nba_n,
        "nhl_updated": nhl_n,
        "mlb_updated": mlb_n,
        "nfl_updated": nfl_n,
        "updated_total": total,
    }


async def backfill_scores(
    since: str, *, window_hours_override: int | None = None
) -> dict[str, int]:
    """CLI helper: `python -m apuestas.flows.live_scores --backfill-since YYYY-MM-DD`.

    Amplía la ventana temporal del flow estándar para cubrir partidos
    atrasados (matches.start_time ≥ :since). El flow interno busca matches
    cuyo start_time esté dentro de `window_hours`; calculamos el override
    en horas desde 'since' hasta ahora + 12h.
    """
    from datetime import date

    try:
        target = datetime.fromisoformat(since).replace(tzinfo=UTC)
    except ValueError:
        target = datetime.combine(date.fromisoformat(since), datetime.min.time(), tzinfo=UTC)
    hours = int((datetime.now(tz=UTC) - target).total_seconds() // 3600) + 12
    hours = max(hours, 12)
    if window_hours_override is not None:
        hours = window_hours_override
    logger.info("live_scores.backfill.start", since=since, window_hours=hours)
    return await live_scores_flow(window_hours=hours)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Live scores + pick_alerts settlement")
    parser.add_argument(
        "--backfill-since",
        help="Fecha ISO (YYYY-MM-DD) — procesa todos los matches desde esa fecha",
    )
    parser.add_argument(
        "--window-hours",
        type=int,
        default=None,
        help="Override manual de la ventana (ignora --backfill-since)",
    )
    args = parser.parse_args()

    if args.backfill_since:
        asyncio.run(backfill_scores(args.backfill_since, window_hours_override=args.window_hours))
    else:
        asyncio.run(live_scores_flow(window_hours=args.window_hours or 48))
