"""Flow principal `deep_analysis` (§16 + §22 + §23).

Ejecuta al correr `make analyze`:
- Catchup data reciente (news, odds, lineups, weather).
- Para cada evento próximo 48 h:
  1. Colectar 9 capas × 2 equipos (§16.1).
  2. Validar mirror_check (§16.6).
  3. Build features + predict ML (LightGBM calibrado + MAPIE).
  4. LLM Qwen análisis estructurado espejo.
  5. Detector value bets + line shopping + regional MX/US compare.
  6. Portfolio allocation correlation-aware.
  7. Persistir en predictions + decision_log + bets_paper.
  8. Registrar en cuba-memorys.
  9. Notificar Telegram.

Objetivo: <30 s por evento end-to-end.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from prefect import flow, task
from sqlalchemy import text

from apuestas.betting.detector import (
    DetectorConfig,
    EventOdds,
    detect_all_player_props_for_match,
    detect_value_bets_for_event,
)
from apuestas.db import session_scope
from apuestas.flows.catchup import catchup_flow
from apuestas.llm.client import LlamaClient
from apuestas.llm.embed import EmbedClient
from apuestas.llm.rag import RAGRetriever
from apuestas.mcp import memory as mcp_memory
from apuestas.mcp.client import MCPClient
from apuestas.obs.logging import get_logger
from apuestas.validators.mirror_check import run_mirror_check

logger = get_logger(__name__)


def _should_filter_low_sharp(detail: dict[str, Any]) -> bool:
    """True si el filtro p_consensus_sharp está activo y el detail no lo pasa.

    Backtest 7d 2026-04-26: picks con p_consensus_sharp<0.40 dieron ROI -56%
    (4 won / 21 lost), mientras los ≥0.40 dieron +12.5% (17 won / 11 lost).
    Activación: APUESTAS_FILTER_LOW_SHARP=true (default false).
    Umbral configurable: APUESTAS_MIN_P_CONSENSUS_SHARP (default 0.40).
    Por sport: APUESTAS_FILTER_LOW_SHARP_SPORTS=soccer,nba (default todos).
    """
    import os as _os

    if _os.environ.get("APUESTAS_FILTER_LOW_SHARP", "false").lower() != "true":
        return False
    p_sharp = detail.get("p_consensus_sharp")
    n_sources = detail.get("consensus_sources") or 0
    # Sin consensus disponible (0 fuentes) → no se puede filtrar; deja pasar.
    if p_sharp is None or n_sources == 0:
        return False

    # Filtro selectivo por sport (opt-in para validar gradualmente).
    sports_env = _os.environ.get("APUESTAS_FILTER_LOW_SHARP_SPORTS", "").strip().lower()
    if sports_env:
        allowed = {s.strip() for s in sports_env.split(",") if s.strip()}
        # `_build_notify_detail` usa key "sport" (no "sport_code")
        sport_val = (detail.get("sport") or detail.get("sport_code") or "").lower()
        if sport_val not in allowed:
            return False

    try:
        threshold = float(_os.environ.get("APUESTAS_MIN_P_CONSENSUS_SHARP", "0.40"))
    except ValueError:
        threshold = 0.40
    return float(p_sharp) < threshold


async def _cancel_alert_low_sharp(
    session: Any,
    *,
    alert_id: int,
    vb: Any,
    p_sharp: float,
    correlation_id: str | None,
) -> None:
    """Cancela una pick_alert recién creada por filtro low_consensus_sharp.

    DELETE de la alert (preferible a UPDATE status='cancelled' porque la
    constraint UNIQUE uq_pick_alerts_identity es WHERE outcome_result IS NULL,
    y si dejamos el row, futuros picks duplicados con mejor p_sharp se
    bloquearán por el unique index).
    Persiste en decision_log para trazabilidad.
    """
    try:
        await session.execute(
            text("DELETE FROM pick_alerts WHERE id = :id"),
            {"id": alert_id},
        )
        await session.execute(
            text(
                """
                INSERT INTO decision_log
                  (event_id, market, outcome, line,
                   p_model, fair_odds, best_offer, best_bookmaker,
                   edge, decision, skip_reason, correlation_id)
                VALUES
                  (:event_id, :market, :outcome, :line,
                   :p_model, :fair_odds, :best_offer, :bookmaker,
                   :edge, 'skip', :reason, :cid)
                """
            ),
            {
                "event_id": int(vb.event_id),
                "market": str(vb.market),
                "outcome": str(vb.outcome),
                "line": vb.line,
                "p_model": vb.p_model,
                "fair_odds": (1.0 / vb.p_blended) if vb.p_blended else None,
                "best_offer": vb.odds,
                "bookmaker": vb.bookmaker,
                "edge": vb.edge,
                "reason": f"low_consensus_sharp_{int(p_sharp * 100)}pct",
                "cid": correlation_id,
            },
        )
        logger.info(
            "emit_alerts.low_sharp_filtered",
            alert_id=alert_id,
            p_sharp=round(p_sharp, 3),
            sport=getattr(vb, "sport_code", None),
        )
    except Exception as exc:
        logger.debug("emit_alerts.cancel_low_sharp_fail", error=str(exc)[:120])


async def _check_clv_anti_stale(
    session: Any, *, match_id: int, market: str, outcome: str, line: float | None
) -> tuple[bool, float | None]:
    """F5 — anti-stale guard. Si Pinnacle ya está moviendo en contra del pick
    en últimos 30min más allá del threshold, el "edge" del modelo es stale
    (info pública ya incorporada por el mercado sharp). Retorna
    (False, drift_pct) → cancelar.

    Drift = (odds_now − odds_30min_ago) / odds_30min_ago.
    Negativo = Pinnacle bajó el precio (el outcome se cree MÁS probable ahora,
    pero tu pick estaba apostando que estaba sub-priced — ya no).
    Para outcomes 'home/over', positivo en odds = bueno; negativo = malo.

    Threshold default 3% (subido de 2% el 2026-04-27): el 2% es ruido normal
    de Pinnacle pre-game; cancelaba picks con edge real (ej: Texas-Yankees
    27-abr drift -2.36% canceló pick @2.57 que aún tenía EV +2.7%). Buchdahl
    2023 sugiere drift >3% como señal de movement sharp real.

    Returns: (ok, drift_pct). ok=False → cancelar pick. Fail-open en errores.
    """
    import os as _os

    if _os.environ.get("APUESTAS_CLV_ANTISTALE", "true").lower() != "true":
        return True, None
    try:
        threshold = float(_os.environ.get("APUESTAS_CLV_DRIFT_TOLERANCE", "0.03"))
    except ValueError:
        threshold = 0.03
    try:
        result = await session.execute(
            text(
                """
                WITH latest AS (
                  SELECT odds, ts FROM odds_history
                  WHERE match_id=:mid AND bookmaker='pinnacle' AND market=:mk
                    AND outcome=:oc
                    AND (line IS NOT DISTINCT FROM :ln OR :ln IS NULL)
                  ORDER BY ts DESC LIMIT 1
                ),
                old AS (
                  SELECT odds FROM odds_history
                  WHERE match_id=:mid AND bookmaker='pinnacle' AND market=:mk
                    AND outcome=:oc
                    AND (line IS NOT DISTINCT FROM :ln OR :ln IS NULL)
                    AND ts <= NOW() - INTERVAL '30 minutes'
                  ORDER BY ts DESC LIMIT 1
                )
                SELECT (SELECT odds FROM latest) AS now_odds,
                       (SELECT odds FROM old) AS old_odds
                """
            ),
            {"mid": match_id, "mk": market, "oc": outcome, "ln": line},
        )
        row = result.first()
        if row is None or row.now_odds is None or row.old_odds is None:
            return True, None  # fail-open: sin data Pinnacle, no aplicar
        now_odds = float(row.now_odds)
        old_odds = float(row.old_odds)
        if old_odds <= 0:
            return True, None
        drift = (now_odds - old_odds) / old_odds
        # Drift negativo > threshold → Pinnacle bajó el precio → outcome más
        # probable ahora → tu edge se evaporó → cancelar.
        if drift < -threshold:
            return False, drift
        return True, drift
    except Exception as exc:
        logger.debug("emit_alerts.clv_check_fail", error=str(exc)[:120])
        return True, None


async def _cancel_alert_clv(session: Any, *, alert_id: int, vb: Any, drift: float) -> None:
    """F5 — cancela pick por CLV anti-stale (Pinnacle moviendo en contra)."""
    try:
        await session.execute(
            text("DELETE FROM pick_alerts WHERE id = :id"),
            {"id": alert_id},
        )
        await session.execute(
            text(
                """
                INSERT INTO decision_log
                  (event_id, market, outcome, line,
                   p_model, fair_odds, best_offer, best_bookmaker,
                   edge, decision, skip_reason, correlation_id)
                VALUES
                  (:event_id, :market, :outcome, :line,
                   :p_model, :fair_odds, :best_offer, :bookmaker,
                   :edge, 'skip', :reason, NULL)
                """
            ),
            {
                "event_id": int(vb.event_id),
                "market": str(vb.market),
                "outcome": str(vb.outcome),
                "line": vb.line,
                "p_model": vb.p_model,
                "fair_odds": (1.0 / vb.p_blended) if vb.p_blended else None,
                "best_offer": vb.odds,
                "bookmaker": vb.bookmaker,
                "edge": vb.edge,
                "reason": f"clv_stale_pinn_drift_{int(drift * 100)}pct",
            },
        )
        logger.info(
            "emit_alerts.clv_stale_cancelled",
            alert_id=alert_id,
            pinn_drift=round(drift, 4),
        )
    except Exception as exc:
        logger.debug("emit_alerts.cancel_clv_fail", error=str(exc)[:120])


async def _cancel_alert_slippage(
    session: Any, *, alert_id: int, vb: Any, odds_emitted: float, odds_current: float
) -> None:
    """F4 — cancela pick_alert por slippage detectado. DELETE + decision_log."""
    try:
        await session.execute(
            text("DELETE FROM pick_alerts WHERE id = :id"),
            {"id": alert_id},
        )
        await session.execute(
            text(
                """
                INSERT INTO decision_log
                  (event_id, market, outcome, line,
                   p_model, fair_odds, best_offer, best_bookmaker,
                   edge, decision, skip_reason, correlation_id)
                VALUES
                  (:event_id, :market, :outcome, :line,
                   :p_model, :fair_odds, :best_offer, :bookmaker,
                   :edge, 'skip', :reason, NULL)
                """
            ),
            {
                "event_id": int(vb.event_id),
                "market": str(vb.market),
                "outcome": str(vb.outcome),
                "line": vb.line,
                "p_model": vb.p_model,
                "fair_odds": (1.0 / vb.p_blended) if vb.p_blended else None,
                "best_offer": odds_emitted,
                "bookmaker": vb.bookmaker,
                "edge": vb.edge,
                "reason": (
                    f"slippage_{int((1 - odds_current / odds_emitted) * 100)}pct"
                    if odds_emitted
                    else "slippage_unknown"
                ),
            },
        )
        logger.info(
            "emit_alerts.slippage_cancelled",
            alert_id=alert_id,
            odds_emitted=odds_emitted,
            odds_current=odds_current,
        )
    except Exception as exc:
        logger.debug("emit_alerts.cancel_slippage_fail", error=str(exc)[:120])


async def _check_slippage(
    session: Any,
    *,
    match_id: int,
    bookmaker: str,
    market: str,
    outcome: str,
    line: float | None,
    odds_emitted: float,
) -> tuple[bool, float | None]:
    """F4 — re-check odds antes de notify Telegram para evitar slippage.

    Trae la quote MÁS RECIENTE del mismo (book, market, outcome, line) en los
    últimos 15 min y compara con la odds emitted. Si la actual es <95% de la
    emitted (book ya movió la línea en tu contra ≥5%), retorna (False, current).
    Tolera reducción ≤5% como ruido normal del book.

    Returns: (ok, current_odds). ok=False → cancelar pick.
    Si no hay quote reciente, ok=True (pasa con la emitida) — fail-open.
    """
    import os as _os

    if _os.environ.get("APUESTAS_SLIPPAGE_GUARD", "true").lower() != "true":
        return True, None
    try:
        threshold = float(_os.environ.get("APUESTAS_SLIPPAGE_TOLERANCE", "0.05"))
    except ValueError:
        threshold = 0.05
    try:
        result = await session.execute(
            text(
                """
                SELECT odds FROM odds_history
                WHERE match_id = :mid AND bookmaker = :bm AND market = :mk
                  AND outcome = :oc
                  AND (line IS NOT DISTINCT FROM :ln OR :ln IS NULL)
                  AND ts > NOW() - INTERVAL '15 minutes'
                ORDER BY ts DESC LIMIT 1
                """
            ),
            {"mid": match_id, "bm": bookmaker, "mk": market, "oc": outcome, "ln": line},
        )
        row = result.first()
        if row is None or row.odds is None:
            return True, None  # fail-open: sin re-check, permite notificar
        current = float(row.odds)
        # Slippage: si current < emitted × (1 - threshold), es movimiento adverso
        if current < odds_emitted * (1.0 - threshold):
            return False, current
        return True, current
    except Exception as exc:
        logger.debug("emit_alerts.slippage_check_fail", error=str(exc)[:120])
        return True, None


async def _persist_insufficient_history(
    *, event_id: int, market: str, correlation_id: str | None
) -> None:
    """Marca el evento en `decision_log` con `skip_reason='insufficient_history'`.

    Visibilidad operacional: cuando el modelo está cargado pero ni el path raw
    (runtime_features) ni el legacy (feature_store JSON) lograron construir el
    vector, deja huella explícita en lugar de skip silencioso.
    Nunca re-raises (best-effort logging).
    """
    try:
        async with session_scope() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO decision_log
                      (event_id, market, outcome, line,
                       p_model, p_lower, p_upper,
                       fair_odds, best_offer, best_bookmaker,
                       edge, decision, skip_reason, correlation_id)
                    VALUES
                      (:event_id, :market, '_skip', NULL,
                       NULL, NULL, NULL,
                       NULL, NULL, NULL,
                       NULL, 'skip', 'insufficient_history', :cid)
                    """
                ),
                {
                    "event_id": event_id,
                    "market": market,
                    "cid": correlation_id,
                },
            )
    except Exception as exc:
        logger.debug("deep_analysis.persist_skip_fail", error=str(exc)[:120])


@task(retries=1)
async def get_upcoming_events(hours_ahead: int = 48) -> list[dict[str, Any]]:
    """Eventos próximos dentro de ventana.

    Prioriza sports con modelo ML propio (NBA/NFL) + sports principales (laliga,
    epl, soccer, mlb) por encima de tennis Challenger/ITF. Filtra matches que
    son prop-markets (nombres con "(Corners)", "(Bookings)", "(Games)") que son
    duplicados del match principal ingestados desde Pinnacle.

    Filtro de cobertura mínima: N books distintos en últimas 6h (default 1,
    override `APUESTAS_MIN_BOOKS_FOR_ANALYSIS`). Default 1 = analizar TODO
    match con AL MENOS 1 book — el detector internamente skipea si no hay
    soft book apostable (Pinnacle excluido), pero al menos evalúa cada partido.
    Cleanup de matches huérfanos sin odds está en `cancel_orphan_matches()`.
    """
    import os

    until = datetime.now(tz=UTC) + timedelta(hours=hours_ahead)
    # since=NOW()-30min: captura partidos que están empezando/recién empezaron.
    # Útil para late_line picks (ventana 90 min pre-kickoff aún tiene EV+).
    # El detector tiene late_line soft_tag para baja tier en estos casos.
    since = datetime.now(tz=UTC) - timedelta(minutes=30)
    try:
        min_books = int(os.environ.get("APUESTAS_MIN_BOOKS_FOR_ANALYSIS", "1"))
    except ValueError:
        min_books = 1

    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT m.id, m.external_id, m.sport_code, m.league_id,
                       m.home_team_id, m.away_team_id, m.venue_id,
                       m.start_time, m.status,
                       t1.name AS home_name, t2.name AS away_name,
                       CASE m.sport_code
                         WHEN 'nba' THEN 1
                         WHEN 'nfl' THEN 2
                         WHEN 'nhl' THEN 3
                         WHEN 'mlb' THEN 4
                         WHEN 'soccer' THEN 5
                         WHEN 'laliga' THEN 5
                         WHEN 'epl' THEN 5
                         WHEN 'liga_mx' THEN 5
                         WHEN 'boxing' THEN 6
                         WHEN 'mma' THEN 6
                         WHEN 'tennis' THEN 9
                         ELSE 8
                       END AS priority
                FROM matches m
                JOIN teams t1 ON t1.id = m.home_team_id
                JOIN teams t2 ON t2.id = m.away_team_id
                WHERE m.status = 'scheduled'
                  AND m.start_time BETWEEN :since AND :until
                  -- Filter prop-market duplicates / fake teams.
                  -- Patrón único POSIX regex (~) para mantenibilidad:
                  --   - sufijo derivativo "(Xxx)" al final
                  --   - nombres genéricos Sofascore "Team N"
                  --   - placeholders "Odd"/"Even"/"Home"/"Away" exactos
                  --   - half/period markers "1st Half ...", "Inning N"
                  AND t1.name !~ '\\(Corners|Bookings|Games|Cards|Sets|Points|Goals|Shots\\)'
                  AND t2.name !~ '\\(Corners|Bookings|Games|Cards|Sets|Points|Goals|Shots\\)'
                  AND t1.name NOT IN ('Odd', 'Even', 'Home', 'Away')
                  AND t2.name NOT IN ('Odd', 'Even', 'Home', 'Away')
                  AND t1.name !~ '^(1st Half|2nd Half|Period |Inning |Team )'
                  AND t2.name !~ '^(1st Half|2nd Half|Period |Inning |Team )'
                  -- Requiere al menos 2 books distintos (no solo Pinnacle obligatorio).
                  -- Pinnacle sigue siendo preferido pero Kambi/OddsJam/DraftKings son
                  -- suficientes como referencia sharp si Pinnacle no cotiza ese mercado.
                  -- Análisis multivector requiere REAL line shopping: no basta
                  -- 1 book (Pinnacle es sharp, no apostable) ni 2 books (sin
                  -- variedad para encontrar EV+ por book). Mínimo N books soft
                  -- distintos en últimas 6h (default 5). Esto filtra 1300+
                  -- matches fantasma de ligas menores (Cyprus B-League,
                  -- Marroquí, U21, etc.) que ingestamos pero no tienen
                  -- cobertura de books reales.
                  AND (
                    SELECT COUNT(DISTINCT oh.bookmaker) FROM odds_history oh
                    WHERE oh.match_id = m.id
                      AND oh.ts > NOW() - INTERVAL '6 hours'
                  ) >= :min_books
                ORDER BY priority ASC, m.start_time ASC
                """
            ),
            {"since": since, "until": until, "min_books": min_books},
        )
        rows = [dict(r._mapping) for r in result.all()]

        # Visibilidad de matches descartados por filtro: cuántos quedaron
        # excluidos por nombre derivativo / coverage / status. Permite al
        # operador detectar regresiones de identity resolution sin grep manual.
        skip_breakdown = await session.execute(
            text(
                """
                SELECT
                  COUNT(*) FILTER (
                    WHERE t1.name ~ '\\(Corners|Bookings|Games|Cards|Sets|Points|Goals|Shots\\)'
                       OR t2.name ~ '\\(Corners|Bookings|Games|Cards|Sets|Points|Goals|Shots\\)'
                  ) AS derivative_team,
                  COUNT(*) FILTER (
                    WHERE t1.name LIKE 'Team %' OR t2.name LIKE 'Team %'
                  ) AS generic_team,
                  COUNT(*) FILTER (
                    WHERE m.status != 'scheduled'
                  ) AS not_scheduled,
                  COUNT(*) FILTER (
                    WHERE (
                      SELECT COUNT(DISTINCT oh.bookmaker) FROM odds_history oh
                      WHERE oh.match_id = m.id
                        AND oh.ts > NOW() - INTERVAL '6 hours'
                    ) < :min_books
                  ) AS no_book_coverage
                FROM matches m
                JOIN teams t1 ON t1.id = m.home_team_id
                JOIN teams t2 ON t2.id = m.away_team_id
                WHERE m.start_time BETWEEN :since AND :until
                """
            ),
            {"since": since, "until": until, "min_books": min_books},
        )
        skip_row = skip_breakdown.first()
        if skip_row is not None and any(
            getattr(skip_row, c, 0) > 0
            for c in ("derivative_team", "generic_team", "not_scheduled", "no_book_coverage")
        ):
            logger.info(
                "deep_analysis.events_filter_breakdown",
                accepted=len(rows),
                skipped_derivative_team=int(skip_row.derivative_team or 0),
                skipped_generic_team=int(skip_row.generic_team or 0),
                skipped_not_scheduled=int(skip_row.not_scheduled or 0),
                skipped_no_book_coverage=int(skip_row.no_book_coverage or 0),
            )

    # Dedup matches duplicados (mismo partido distintos team_ids por sport_code
    # fragmentado, ej. Barcelona id=319 'laliga' vs id=820 'soccer'). Bug
    # observado 2026-04-25 con Getafe-Barcelona: match_id=201 (29 books) vs
    # match_id=455 (1 book) — el detector procesaba el segundo y skipeaba todo
    # con `no_qualifying_offer`. Preferimos el que tiene MÁS cobertura de books
    # en las últimas 6 horas.
    if not rows:
        return rows

    match_ids = [int(r["id"]) for r in rows]
    book_counts: dict[int, int] = dict.fromkeys(match_ids, 0)
    async with session_scope() as session:
        cnt_result = await session.execute(
            text(
                """
                SELECT match_id, COUNT(DISTINCT bookmaker) AS n
                FROM odds_history
                WHERE match_id = ANY(:ids) AND ts > NOW() - INTERVAL '6 hours'
                GROUP BY match_id
                """
            ),
            {"ids": match_ids},
        )
        for r in cnt_result.all():
            book_counts[int(r.match_id)] = int(r.n)

    # Bucket por (start_time redondeado a minuto, home_norm, away_norm)
    import re

    def _norm(name: str) -> str:
        s = name.lower().strip()
        s = re.sub(r"\s*\(.*?\)\s*", "", s)  # quita "(Corners)" etc.
        s = re.sub(r"[^a-z0-9]+", "", s)  # solo alfanum
        return s

    bucket: dict[tuple, dict[str, Any]] = {}
    for r in rows:
        st = r["start_time"]
        key = (
            st.replace(second=0, microsecond=0) if st else None,
            _norm(r.get("home_name") or ""),
            _norm(r.get("away_name") or ""),
        )
        prev = bucket.get(key)
        if prev is None or book_counts.get(int(r["id"]), 0) > book_counts.get(int(prev["id"]), 0):
            bucket[key] = r

    deduped = list(bucket.values())
    if len(deduped) < len(rows):
        logger.info(
            "deep_analysis.match_dedup",
            before=len(rows),
            after=len(deduped),
            removed=len(rows) - len(deduped),
        )
    return deduped


@task
async def collect_odds_for_event(event_id: int, *, freshness_hours: int = 6) -> EventOdds | None:
    """Recolecta odds recientes agrupadas por bookmaker+market.

    Ventana default 6h: medido en producción 2026-04-25, con ventana 90 min solo
    87 de 183 matches (47%) tenían odds. Con 6h cubre 209/209 con odds frescas.
    La frescura no es crítica para line shopping: las odds de soft books son
    "sticky" (no cambian cada min); lo importante es tener QUE comparar.

    `freshness_hours` configurable: el agente on-demand `/analizar` puede pedir
    48h para casos donde el partido se cargó hace 1-2 días pero no hubo refresh
    intermedio (caso Atlético-Arsenal 2026-04-29).

    DISTINCT ON garantiza que tomamos la odds MÁS RECIENTE por (match, book,
    market, outcome, line) en la ventana — no se mezclan precios viejos.
    """
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                WITH recent AS (
                  SELECT DISTINCT ON (match_id, bookmaker, market, outcome, line)
                    match_id, bookmaker, market, outcome, line, odds, ts
                  FROM odds_history
                  WHERE match_id = :mid
                    AND ts >= NOW() - make_interval(hours => :hours)
                  ORDER BY match_id, bookmaker, market, outcome, line, ts DESC
                )
                SELECT r.*, m.external_id, m.start_time, m.sport_code, m.league_id,
                       m.stage, l.name AS league_name
                FROM recent r JOIN matches m ON m.id = r.match_id
                LEFT JOIN leagues l ON l.id = m.league_id
                ORDER BY market, outcome
                """
            ),
            {"mid": event_id, "hours": freshness_hours},
        )
        rows = [dict(r._mapping) for r in result.all()]

    if not rows:
        return None

    # CAMBIO: procesar TODOS los markets disponibles, no solo h2h.
    # Retornamos un EventOdds "múltiple" donde `quotes_by_bookmaker` contiene
    # solo el market h2h (primary). Los demás markets (spreads, totals,
    # team_totals, alternate_totals) se retornan en `additional_markets`
    # para que el flow los procese también.
    sample = rows[0]

    # Siempre empezar con h2h si está disponible
    target_market = "h2h" if any(r["market"] == "h2h" for r in rows) else rows[0]["market"]
    market_rows = [r for r in rows if r["market"] == target_market]
    if not market_rows:
        return None

    from collections import defaultdict as _dd

    def _build_event_for_market(m_rows: list[dict[str, Any]], market_name: str) -> EventOdds | None:
        """Construye un EventOdds para un market específico.

        LINE CANÓNICA: para markets con line (totals/spreads), elige la line más
        frecuente por outcome (moda) y FILTRA quotes a esa line exacta. Evita el
        bug de line-mismatch donde quotes de distintas lines se mezclaban al mismo
        pool, causando EVs inflados al comparar p_pinn de una line vs odds de otra.
        """
        if not m_rows:
            return None
        from collections import Counter as _Counter

        grouped_m: dict[tuple[str, str, float | None], list[dict[str, Any]]] = _dd(list)
        for r in m_rows:
            key = (r["bookmaker"], r["outcome"], r["line"])
            grouped_m[key].append(r)
        dedup_m: list[dict[str, Any]] = []
        for _k, entries in grouped_m.items():
            if len(entries) == 1:
                dedup_m.append(entries[0])
            else:
                normal = [e for e in entries if float(e["odds"]) < 10.0]
                best = max(normal if normal else entries, key=lambda x: x["ts"])
                dedup_m.append(best)

        # Orden explícito para que pinnacle_fair[i] coincida con outcomes[i].
        _all_oc = {r["outcome"] for r in dedup_m}
        _CANONICAL = ["home", "away", "draw", "over", "under"]
        outcomes_m = [o for o in _CANONICAL if o in _all_oc] + sorted(_all_oc - set(_CANONICAL))

        # Line canónica por outcome = moda (más frecuente). Si market no tiene
        # line (h2h) → None. Para totals con multiple lines (6.0, 6.5, 7.0, 7.5),
        # tomamos la que más books ofrezcan → máxima liquidez y mejor de-vig.
        line_counts: dict[str, _Counter[float]] = {}
        for r in dedup_m:
            if r["line"] is None:
                continue
            oc = r["outcome"]
            line_counts.setdefault(oc, _Counter())
            line_counts[oc][float(r["line"])] += 1

        # Para totals/spreads: la line canónica debe ser la misma para todos los
        # outcomes complementarios (over↔under, home spread↔away spread) para
        # que el de-vig sum a 1.0. Elegimos la line más frecuente globalmente.
        if line_counts:
            global_ctr: _Counter[float] = _Counter()
            for ctr in line_counts.values():
                global_ctr.update(ctr)
            canonical_line = global_ctr.most_common(1)[0][0]
        else:
            canonical_line = None

        # Filtrar dedup_m: mantener solo quotes con la line canónica (o sin line si h2h)
        if canonical_line is not None:
            filtered = [
                r
                for r in dedup_m
                if r["line"] is None or abs(float(r["line"]) - canonical_line) < 1e-6
            ]
        else:
            filtered = dedup_m

        if not filtered:
            return None

        # Rebuild outcomes from filtered rows (algunos pueden no tener esa line)
        _all_oc_f = {r["outcome"] for r in filtered}
        outcomes_m = [o for o in _CANONICAL if o in _all_oc_f] + sorted(_all_oc_f - set(_CANONICAL))

        quotes_m: dict[str, list[float]] = {}
        lines_m: list[float | None] = [canonical_line] * len(outcomes_m)
        for r in filtered:
            bm = r["bookmaker"]
            idx = outcomes_m.index(r["outcome"])
            quotes_m.setdefault(bm, [0.0] * len(outcomes_m))
            quotes_m[bm][idx] = float(r["odds"])
        return EventOdds(
            event_id=event_id,
            event_external_id=str(sample["external_id"]),
            market=market_name,
            start_time=sample["start_time"],
            outcomes=outcomes_m,
            quotes_by_bookmaker=quotes_m,
            lines=lines_m if any(v is not None for v in lines_m) else None,
            league_id=sample.get("league_id"),
            league=sample.get("league_name"),
            sport_code=sample.get("sport_code"),
            stage=sample.get("stage"),
        )

    # Construir EventOdds para el market principal
    primary = _build_event_for_market(market_rows, target_market)
    if primary is None:
        return None

    # Procesar también los otros markets disponibles (spreads, totals, team_totals)
    # Guardados en atributo _additional_markets para que el flow los procese.
    other_markets = {r["market"] for r in rows} - {target_market}
    additional: list[EventOdds] = []
    for other_m in other_markets:
        if other_m in ("h2h", "spreads", "totals", "team_totals", "alternate_totals"):
            m_rows = [r for r in rows if r["market"] == other_m]
            ev = _build_event_for_market(m_rows, other_m)
            if ev is not None:
                additional.append(ev)
    # Adjuntar markets secundarios (dataclass tiene `additional_markets` field).
    primary.additional_markets = additional
    return primary


@task
async def run_mirror_validation(event: dict[str, Any]) -> dict[str, Any]:
    """Ejecuta mirror_check y retorna resumen.

    Threshold configurable vía `APUESTAS_MIRROR_MIN_COMPLETENESS` (default 0.7).
    Útil para modo degradado cuando APIs externas están agotadas: bajar a 0.2
    permite emitir picks con badge 'DATOS LIMITADOS' en lugar de bloquearlos.
    """
    import os as _os

    try:
        min_comp = float(_os.environ.get("APUESTAS_MIRROR_MIN_COMPLETENESS", "0.7"))
    except ValueError:
        min_comp = 0.7

    check = await run_mirror_check(
        match_id=int(event["id"]),
        home_team_id=int(event["home_team_id"]),
        away_team_id=int(event["away_team_id"]),
        venue_id=event.get("venue_id"),
        sport_code=str(event["sport_code"]),
        minimum_completeness=min_comp,
    )
    return {
        "analysis_complete": check.analysis_complete,
        "overall_score": check.overall_completeness_score,
        "missing": check.missing,
        "warnings": check.warnings,
        "threshold_used": min_comp,
    }


@task
async def fetch_rag_context(
    event: dict[str, Any],
    *,
    top_k: int = 10,
) -> str:
    """Recuperar snippets RAG relevantes para el evento."""
    try:
        async with EmbedClient() as embed:
            retriever = RAGRetriever(embed_client=embed)
            sport = str(event["sport_code"])
            team_ids = [int(event["home_team_id"]), int(event["away_team_id"])]
            query = (
                f"match preview {sport} home={event['home_team_id']} away={event['away_team_id']}"
            )
            hits = await retriever.hybrid_search(
                query,
                top_k=top_k,
                sports=[sport],
                team_ids=team_ids,
            )
            return retriever.format_snippets(hits)
    except Exception as exc:
        logger.warning("deep_analysis.rag_failed", event_id=event["id"], error=str(exc))
        return "(sin contexto RAG disponible)"


async def _push_props_to_summary(*, prop_picks: list[Any], event: dict[str, Any]) -> None:
    """Envía prop picks al match_summary builder para el mensaje consolidado
    Telegram. No-op si builder flag off.
    """
    import os as _os

    if _os.environ.get("APUESTAS_UNIFIED_MESSAGES", "true").lower() != "true":
        return
    try:
        from apuestas.bot.match_summary import get_builder

        builder = get_builder()
        home_name = event.get("home_team_name", "")
        away_name = event.get("away_team_name", "")
        sport = event.get("sport_code", "")
        start_time_raw = event.get("start_time")

        for p in prop_picks:
            prob = float(getattr(p, "p_model", 0.0))
            line = float(getattr(p, "line", 0.0))
            side = str(getattr(p, "side", "over"))
            prop_code = str(getattr(p, "prop_code", "")).replace("player_", "")
            # PropEntry.model_prob es P(over line). Si side=under, convertir.
            model_prob = prob if side == "over" else 1.0 - prob
            await builder.add_prop(
                match_id=int(event["id"]),
                home=home_name,
                away=away_name,
                sport=sport,
                start_time=start_time_raw,
                prop_type=prop_code,
                team_or_player=str(getattr(p, "player_name", "")),
                line=line,
                model_prob=model_prob,
                bookmaker=str(getattr(p, "bookmaker", "")),
                odds=float(getattr(p, "odds", 0.0)),
                ev=float(getattr(p, "ev", 0.0)),
            )
    except Exception as exc:
        logger.debug("deep.props_summary_push_fail", error=str(exc)[:100])


@task
async def llm_analyze_event(
    event: dict[str, Any],
    rag_snippets: str,
    *,
    correlation_id: str,
) -> dict[str, Any] | None:
    """Llama al LLM (Qwen local o DeepSeek) para enriquecer un pick emitido.

    Optimizaciones aplicadas:
      - System prompt invariante → DeepSeek context caching automático activo
        (input tokens cacheados $0.07/M vs $0.27/M = -74% costo input).
      - Removidos: fetch_memory_context, tier_a_features, steam_moves query.
        Eran ~150-300ms de I/O + setean keys (`memory_context`, `tier_a_features`)
        que el template `pre_match/v1.yaml` NO referencia → el LLM nunca las
        veía. Computar+pasar variables que se descartan = waste puro.
      - Si en el futuro queremos inyectar memoria, hacerlo como user message
        APPEND (no concat al system) para preservar el cache hit del system.
    """
    from apuestas.config import get_settings
    from apuestas.llm.router import run_task
    from apuestas.schemas.llm import PreMatchAnalysis

    backend = get_settings().llm.llm_backend
    llm_cls: Any
    if backend == "deepseek":
        from apuestas.llm.deepseek_client import DeepSeekClient

        llm_cls = DeepSeekClient
    else:
        llm_cls = LlamaClient

    prompt_vars = _build_prompt_vars(event, rag_snippets)

    try:
        async with llm_cls() as llm:
            result = await run_task(
                task_kind="pre_match",
                version="v1",
                client=llm,
                render_vars=prompt_vars,
            )
        if not isinstance(result, PreMatchAnalysis):
            logger.warning(
                "llm_analyze_event.unexpected_type",
                type=type(result).__name__,
                event_id=event.get("id"),
            )
            return None
        return {
            "summary_es": result.summary_es,
            "confidence": result.confidence_in_analysis,
            "edge_direction": result.overall_edge_direction,
            "line_movement": result.line_movement_assessment,
            "home": {
                "team_name": result.home_team_analysis.team_name,
                "key_injuries": [
                    {"player": i.player, "severity": i.severity}
                    for i in result.home_team_analysis.key_injuries
                ],
                "rest_days": result.home_team_analysis.rest_days,
                "b2b": result.home_team_analysis.back_to_back,
                "momentum": result.home_team_analysis.narrative_momentum,
            },
            "away": {
                "team_name": result.away_team_analysis.team_name,
                "key_injuries": [
                    {"player": i.player, "severity": i.severity}
                    for i in result.away_team_analysis.key_injuries
                ],
                "rest_days": result.away_team_analysis.rest_days,
                "b2b": result.away_team_analysis.back_to_back,
                "momentum": result.away_team_analysis.narrative_momentum,
                "travel_km": result.away_team_analysis.travel_km,
                "altitude_delta_m": result.away_team_analysis.altitude_delta_m,
            },
            "contradictions_found": result.contradictions_found,
        }
    except Exception as exc:
        logger.warning(
            "deep_analysis.llm_failed",
            event_id=event["id"],
            cid=correlation_id,
            error=str(exc),
        )
        return None


def _build_prompt_vars(event: dict[str, Any], rag_snippets: str) -> dict[str, Any]:
    """Variables para pre_match/v1.

    Antes enviábamos 13 placeholders `(ver queries en versión productiva)` ≈
    150 tokens/call de basura semántica que el LLM trataba como "data" y
    contribuía a alucinaciones (intenta inferir desde la nada). Ahora pasamos
    cadenas vacías; el template del prompt las renderiza sin contenido y el
    LLM responde sólo sobre los campos que SÍ tienen información (RAG).
    Ahorro: 150 tok × 1655 calls/día ≈ 250k tok/día.
    """
    empty = ""  # placeholder semánticamente neutro (sin "ver queries...")
    return {
        "home_name": f"Team {event['home_team_id']}",
        "away_name": f"Team {event['away_team_id']}",
        "sport": event["sport_code"],
        "league": event.get("league_id", "unknown"),
        "start_time": event["start_time"].isoformat() if event.get("start_time") else "TBD",
        "venue_name": event.get("venue_id", "unknown"),
        "altitude_m": 0,
        "surface": "unknown",
        "roof": "unknown",
        "stats_markdown": empty,
        "home_away_splits_markdown": empty,
        "injuries_markdown": empty,
        "lineups_markdown": empty,
        "transfers_markdown": empty,
        "coaching_markdown": empty,
        "streaks_markdown": empty,
        "h2h_markdown": empty,
        "home_rest_days": 2,
        "home_b2b": False,
        "away_rest_days": 2,
        "away_b2b": False,
        "away_travel_km": 0,
        "tz_delta_h": 0,
        "alt_delta_m": 0,
        "rag_snippets": rag_snippets,
        "line_movement_markdown": empty,
        "weather_summary": empty,
        "official_notes": empty,
    }


@task(retries=1, retry_delay_seconds=5)
async def run_detector(
    event: dict[str, Any],
    event_odds: EventOdds | None,
    *,
    correlation_id: str,
) -> list[Any]:
    """Ejecuta detect_value_bets_for_event + integra mejoras de Fase 2-4.

    Integraciones:
      - 2.3 Shadow Pinnacle: divergence tier se loggea para auditoría.
      - 4.1 RLM signal: tag strong si detectado (solo loggeo, sin resize de stake).
    """
    if event_odds is None:
        return []
    cfg = DetectorConfig()

    # Cargar modelo production y generar model_probs vía feature_store.
    # Sin features reales → model_probs=None; detector skipea (fail-safe).
    model_probs: dict[str, float] | None = None
    conformal_obj: Any = None
    conformal_intervals_local: dict[str, tuple[float, float]] | None = None
    sport_code = event.get("sport_code") or ""
    # Sprint 13 — canonical sport_code para hierarchy resolver
    import os as _os

    from apuestas.sports import canonical_sport_code

    sport_canonical = canonical_sport_code(sport_code)
    # Métricas del modelo production (Brier, n_train) para F2/F7 hardening.
    model_metrics_for_detector: dict[str, Any] | None = None
    try:
        from apuestas.features.feature_store import build_match_features
        from apuestas.ml.registry import load_production_model

        # Sprint 13 Capa 3: hierarchy resolver (opt-in via env, fallback a legacy)
        use_hierarchy = _os.environ.get("APUESTAS_USE_MODEL_HIERARCHY", "false").lower() == "true"
        if use_hierarchy:
            from apuestas.db import session_scope as _ss
            from apuestas.ml.model_hierarchy_resolver import resolve_and_load_model

            async with _ss() as _s:
                resolved = await resolve_and_load_model(
                    _s,
                    sport_code=sport_canonical,
                    market=event_odds.market,
                    league_id=event.get("league_id"),
                )
            if resolved is not None:
                hinfo, raw_obj = resolved
                # raw_obj puede ser:
                #   - dict {estimator, conformal, feature_names, target, sport}  ← MLflow LGBM/CatBoost
                #   - dict {model, ...}  ← Bayesian xG, otros
                #   - objeto sklearn directo  ← legacy
                # Extraer el modelo real + feature_names en cualquier caso.
                if isinstance(raw_obj, dict):
                    real_estimator = raw_obj.get("estimator") or raw_obj.get("model") or raw_obj
                    real_features = raw_obj.get("feature_names") or getattr(
                        real_estimator, "feature_names_in_", []
                    )
                    conformal_obj = raw_obj.get("conformal")
                else:
                    real_estimator = raw_obj
                    real_features = getattr(raw_obj, "feature_names_in_", [])
                loaded = (
                    hinfo,
                    {
                        "estimator": real_estimator,
                        "feature_names": list(real_features) if real_features is not None else [],
                    },
                )
            else:
                loaded = None
        else:
            loaded = await load_production_model(sport_code, event_odds.market)
        if loaded is not None:
            info, model_obj = loaded
            estimator = model_obj.get("estimator") if isinstance(model_obj, dict) else model_obj
            feature_names = (
                model_obj.get("feature_names", []) if isinstance(model_obj, dict) else []
            )
            # Extrae conformal (MAPIE wrapper) si está en el artefacto MLflow.
            if conformal_obj is None and isinstance(model_obj, dict):
                conformal_obj = model_obj.get("conformal")
            # Propaga métricas del modelo (Brier, n_train, log_loss) al detector
            # para activar F2 (adaptive blend) y F7 (sample size guard).
            try:
                model_metrics_for_detector = dict(getattr(info, "performance", {}) or {})
            except Exception:
                model_metrics_for_detector = None

            # ── Dispatch para modelos que NO usan features tabulares rolling ──
            # Estos modelos toman team_ids o pinnacle_fair_prob directamente.
            # Si caen al bloque sklearn-estándar build_match_features los rechaza
            # (coverage 0%) y model_probs queda None → DC fallback.
            home_id = event.get("home_team_id")
            away_id = event.get("away_team_id")

            try:
                from apuestas.ml.bayesian_xg_runtime import BayesianXGModel
            except Exception:  # pragma: no cover
                BayesianXGModel = None  # type: ignore[assignment]
            try:
                from apuestas.ml.catchall_baseline import CatchallBaselineModel
            except Exception:  # pragma: no cover
                CatchallBaselineModel = None  # type: ignore[assignment]
            try:
                from apuestas.ml.dixon_coles_runtime import DixonColesCrossLeagueModel
            except Exception:  # pragma: no cover
                DixonColesCrossLeagueModel = None  # type: ignore[assignment]
            # _IndependentPoissonModel está dentro de train_soccer; lo detectamos
            # por duck-typing (tiene predict(home_id, away_id) → _Pred).
            is_indep_poisson = (
                hasattr(estimator, "predict")
                and not hasattr(estimator, "predict_proba")
                and not isinstance(estimator, type)
                and getattr(estimator, "__class__", type).__name__
                in {
                    "_IndependentPoissonModel",
                    "_DCModelWithMap",
                }
            )

            if (
                BayesianXGModel is not None
                and isinstance(estimator, BayesianXGModel)
                and home_id is not None
                and away_id is not None
            ):
                try:
                    import numpy as _np

                    proba = estimator.predict_proba(_np.array([[int(home_id), int(away_id)]]))
                    model_probs = {
                        "away": float(proba[0, 0]),
                        "draw": float(proba[0, 1]),
                        "home": float(proba[0, 2]),
                    }
                    logger.info(
                        "deep_analysis.bayesian_xg_applied",
                        sport=sport_code,
                        league_id=getattr(estimator, "league_id", None),
                        probs={k: round(v, 4) for k, v in model_probs.items()},
                    )
                except Exception as exc:
                    logger.warning("deep_analysis.bayesian_xg_fail", error=str(exc)[:120])
            elif is_indep_poisson and home_id is not None and away_id is not None:
                try:
                    pred = await asyncio.to_thread(estimator.predict, int(home_id), int(away_id))
                    p_h, p_d, p_a = pred.home_draw_away
                    model_probs = {
                        "home": float(p_h),
                        "draw": float(p_d),
                        "away": float(p_a),
                    }
                    logger.info(
                        "deep_analysis.indep_poisson_applied",
                        sport=sport_code,
                        probs={k: round(v, 4) for k, v in model_probs.items()},
                    )
                except Exception as exc:
                    logger.warning("deep_analysis.indep_poisson_fail", error=str(exc)[:120])
            elif (
                DixonColesCrossLeagueModel is not None
                and isinstance(estimator, DixonColesCrossLeagueModel)
                and home_id is not None
                and away_id is not None
            ):
                try:
                    import numpy as _np

                    market_kind = getattr(event_odds, "market", "h2h")
                    line = getattr(event_odds, "line", None)
                    X = _np.array([[int(home_id), int(away_id)]])
                    if market_kind == "totals" and line is not None:
                        proba = estimator.predict_proba_total(X, line=float(line))
                        model_probs = {"under": float(proba[0, 0]), "over": float(proba[0, 1])}
                    elif market_kind == "btts":
                        proba = estimator.predict_proba_btts(X)
                        model_probs = {"no": float(proba[0, 0]), "yes": float(proba[0, 1])}
                    else:
                        proba = estimator.predict_proba(X)
                        model_probs = {
                            "away": float(proba[0, 0]),
                            "draw": float(proba[0, 1]),
                            "home": float(proba[0, 2]),
                        }
                    logger.info(
                        "deep_analysis.dc_crossleague_applied",
                        sport=sport_code,
                        market=market_kind,
                        league_id=event.get("league_id"),
                        probs={k: round(v, 4) for k, v in model_probs.items()},
                    )
                except Exception as exc:
                    logger.warning("deep_analysis.dc_crossleague_fail", error=str(exc)[:120])
            elif CatchallBaselineModel is not None and isinstance(estimator, CatchallBaselineModel):
                # Catchall requiere pinnacle_fair_prob; lo extraemos del event_odds
                # (consensus o fair_pinn). Si no disponible → skip (no garbage).
                pinn_p = getattr(event_odds, "pinnacle_fair_prob_home", None) or getattr(
                    event_odds, "fair_prob_home", None
                )
                if pinn_p is not None:
                    try:
                        import numpy as _np

                        proba = estimator.predict_proba(_np.array([[float(pinn_p)]]))
                        p_home = float(proba[0, 1])
                        model_probs = {"home": p_home, "away": 1.0 - p_home}
                        logger.info(
                            "deep_analysis.catchall_applied",
                            sport=sport_code,
                            p_home=round(p_home, 4),
                        )
                    except Exception as exc:
                        logger.warning("deep_analysis.catchall_fail", error=str(exc)[:120])
                else:
                    logger.debug("deep_analysis.catchall_no_pinn_fair", sport=sport_code)

            if (
                model_probs is None
                and estimator is not None
                and hasattr(estimator, "predict_proba")
                and feature_names
                and home_id is not None
                and away_id is not None
            ):
                # Path B (preferido): reproduce el pipeline de training en runtime
                # → cero skew train/inference, coverage muy superior al feature_store
                # legacy basado en JSON `team_stats_rolling_*` (5 keys vs 40-60+).
                # Solo aplica a sports con pipeline en runtime_features
                # (nba/mlb/nfl). Para el resto cae directo al fallback legacy.
                X = None
                feature_path: str | None = None
                if sport_code in {"nba", "mlb", "nfl"}:
                    try:
                        from apuestas.features.runtime_features import (
                            build_match_features_from_raw,
                        )

                        X = await build_match_features_from_raw(
                            sport_code=sport_code,
                            home_team_id=int(event["home_team_id"]),
                            away_team_id=int(event["away_team_id"]),
                            match_start=event["start_time"],
                            feature_names=feature_names,
                        )
                        if X is not None:
                            feature_path = "raw"
                    except Exception as exc:
                        logger.warning(
                            "deep_analysis.runtime_features_fail",
                            sport=sport_code,
                            error=str(exc)[:160],
                        )

                # Fallback al feature_store legacy (rolling JSON 5 keys) si raw falló
                if X is None:
                    X = await build_match_features(
                        sport_code=sport_code,
                        home_team_id=int(event["home_team_id"]),
                        away_team_id=int(event["away_team_id"]),
                        match_start=event["start_time"],
                        feature_names=feature_names,
                    )
                    if X is not None:
                        feature_path = "legacy"

                if X is not None:
                    try:
                        proba = estimator.predict_proba(X.reshape(1, -1))
                        # Binary (home_win proba) o 3-way (away/draw/home order).
                        if proba.shape[1] == 2:
                            # class 1 = home_win (convención de train_base.py)
                            p_home = float(proba[0, 1])
                            model_probs = {"home": p_home, "away": 1.0 - p_home}
                        elif proba.shape[1] == 3:
                            # Soccer multiclass: y=0 (away), y=1 (draw), y=2 (home)
                            model_probs = {
                                "away": float(proba[0, 0]),
                                "draw": float(proba[0, 1]),
                                "home": float(proba[0, 2]),
                            }
                        logger.info(
                            "deep_analysis.model_probs_populated",
                            sport=sport_code,
                            model=info.model_name,
                            feature_path=feature_path,
                            probs={k: round(v, 4) for k, v in (model_probs or {}).items()},
                        )
                        # Conformal intervals — solo para binary (2 outcomes).
                        # Soccer 3-way tiene conformal por clase; skip por ahora.
                        if (
                            conformal_obj is not None
                            and model_probs is not None
                            and len(model_probs) == 2
                        ):
                            try:
                                _, p_low_arr, p_up_arr = conformal_obj.predict_intervals(
                                    estimator, X.reshape(1, -1)
                                )
                                _outs = list(model_probs.keys())
                                conformal_intervals_local = {
                                    _outs[0]: (float(p_low_arr[0]), float(p_up_arr[0])),
                                    _outs[1]: (
                                        float(1.0 - float(p_up_arr[0])),
                                        float(1.0 - float(p_low_arr[0])),
                                    ),
                                }
                            except Exception as _cexc:
                                logger.debug(
                                    "deep_analysis.conformal_intervals_fail",
                                    sport=sport_code,
                                    error=str(_cexc)[:100],
                                )
                    except Exception as exc:
                        logger.warning(
                            "deep_analysis.model_predict_fail",
                            sport=sport_code,
                            error=str(exc)[:120],
                        )
                else:
                    # Fail-safe: ambos paths sin features → persist decision_log
                    # con skip_reason='insufficient_history' (visibilidad operacional)
                    # y NO emitir pick. Cero garbage input.
                    logger.info(
                        "deep_analysis.features_insufficient",
                        sport=sport_code,
                        model=info.model_name,
                    )
                    await _persist_insufficient_history(
                        event_id=int(event["id"]),
                        market=event_odds.market,
                        correlation_id=correlation_id,
                    )
    except Exception as exc:
        logger.debug("deep_analysis.model_load_fail", error=str(exc)[:120])

    # Soccer: Dixon-Coles como FALLBACK cuando no hay ML específico de la liga.
    # Antes: solo aplicaba a sport_code=="soccer" literal → 31 ligas SIN modelo
    # quedaban sin p_model → detector hacía `continue` silencioso.
    # Ahora: aplica a TODOS los sport_code que canonicaliza a 'soccer'
    # (laliga, epl, liga_mx, seriea, bundesliga, ligue1, mls, ucl, etc.).
    #
    # Base técnica: Dixon-Coles 1997 (Poisson bivariado + corrección low-score)
    # es modelo Bayesiano calibrado que funciona en cualquier liga soccer con
    # ≥20 partidos históricos. ECE típico 0.04-0.08 (aceptable).
    # Validación calidad:
    #   - DC retorna None si insuficiente histórico → no genera pick
    #   - Conformal_width filter rechaza picks con incertidumbre alta
    #   - EV threshold 3% sigue activo → solo emit picks con edge real
    if sport_canonical == "soccer" and model_probs is None:
        try:
            from apuestas.features.soccer import dixon_coles_predict

            home_id = event.get("home_team_id")
            away_id = event.get("away_team_id")
            if home_id is not None and away_id is not None:
                dc = await asyncio.to_thread(dixon_coles_predict, int(home_id), int(away_id))
                if dc is not None:
                    model_probs = {
                        "home": float(dc["p_home"]),
                        "draw": float(dc["p_draw"]),
                        "away": float(dc["p_away"]),
                    }
                    logger.info(
                        "deep_analysis.dixon_coles_applied",
                        sport=sport_code,
                        sport_canonical=sport_canonical,
                        probs={k: round(v, 4) for k, v in model_probs.items()},
                    )
        except Exception as exc:
            logger.debug("deep_analysis.dixon_coles_fail", error=str(exc)[:120])

    picks = await detect_value_bets_for_event(
        event_odds,
        model_probs=model_probs,
        conformal_intervals=conformal_intervals_local,
        cfg=cfg,
        correlation_id=correlation_id,
        model_metrics=model_metrics_for_detector,
    )

    # RLM tag informativo (sin resize de stake: ya no hay stake).
    if picks:
        try:
            from apuestas.betting.rlm_detector import detect_rlm

            rlm = await detect_rlm(int(event["id"]), market="h2h")
            if rlm and rlm.strength == "strong":
                logger.info("deep_analysis.rlm_strong_detected", event_id=event["id"])
        except Exception as exc:
            logger.debug("deep_analysis.rlm_fail", error=str(exc)[:80])

    return picks


@flow(name="apuestas-deep-analysis", log_prints=True)
async def deep_analysis_flow(
    *,
    hours_ahead: int = 48,
    max_events: int = 300,
    skip_catchup: bool = False,
) -> dict[str, Any]:
    """Entry point del `make analyze`.

    Pre-filtros aplicados antes del bucle de análisis:
      1. `sport_focus.is_emit_enabled(sport)` — descarta deportes off (nhl, tennis,
         nfl off-season, boxing, mma) ANTES de ejecutar mirror/RAG/detector.
      2. Orden por proximidad de kickoff ascendente — los más cercanos primero.
      3. Cap `max_events` (default 300) sobre la lista ya filtrada.
    """
    mcp = MCPClient.get()
    await mcp.start()
    await mcp_memory.jornada_start()

    if not skip_catchup:
        # .fn() evita anidar un flow Prefect dentro de otro (doble run + errores
        # de 'crashed' al cancelar). catchup_flow es un @flow, pero invocado
        # desde otro flow debe correr como función normal.
        await catchup_flow.fn()

    events_raw = await get_upcoming_events(hours_ahead)
    n_raw = len(events_raw)

    from apuestas.betting.sport_focus import is_emit_enabled

    events = [e for e in events_raw if is_emit_enabled(e.get("sport_code"))]
    n_after_focus = len(events)

    events.sort(key=lambda e: e.get("start_time") or datetime.max.replace(tzinfo=UTC))
    events = events[:max_events]

    logger.info(
        "deep_analysis.events",
        n=len(events),
        raw=n_raw,
        after_sport_focus=n_after_focus,
        capped_at=max_events,
    )

    all_picks: list[Any] = []

    # Paralelizar análisis por evento con Semaphore(10).
    # Justificación: DeepSeek API soporta ~300 RPM en plan standard.
    # 10 paralelos × avg 3s = ~3.3 req/s = 200 RPM → margen 33% bajo el límite.
    # Con 200 eventos × 2 llamadas LLM (NER + pre_match) = 400 calls total,
    # a 200 RPM tarda ~2 min vs 8 min con semaphore=5 original.
    # RAG + mirror + detector + LLM analysis cada evento independiente.
    event_sem = asyncio.Semaphore(10)

    async def _analyze_event(event: dict[str, Any]) -> list[Any]:
        async with event_sem:
            correlation_id = uuid.uuid4().hex[:12]
            logger.info(
                "deep_analysis.event.start",
                event_id=event["id"],
                cid=correlation_id,
            )

            mirror, odds, rag = await asyncio.gather(
                run_mirror_validation(event),
                collect_odds_for_event(int(event["id"])),
                fetch_rag_context(event),
                return_exceptions=True,
            )

            # Mirror_check pasa de HARD FILTER a SOFT TAG (decisión técnica
            # 2026-04-25). Razón: el filtro era redundante con los guards del
            # detector (low_hold ≤3%, conformal_width, EV thresholds, draw_guard).
            # Sin features pobladas, p_model ≈ baseline → blend(0.4*0.5 + 0.6*p_pinn)
            # ≈ p_pinn → EV ≈ vig_soft_book. Como vig típico (5-10%) supera
            # rara vez EV threshold (3-5%), el detector skipea naturalmente
            # cuando data está incompleta. Mantener mirror_check como info
            # adicional para tag `low_data_quality` en picks (futuro).
            mirror_score: float | None = None
            mirror_complete: bool = True
            if isinstance(mirror, dict):
                mirror_score = mirror.get("overall_score")
                mirror_complete = mirror.get("analysis_complete", True)
                if not mirror_complete:
                    logger.info(
                        "deep_analysis.partial_data",
                        event_id=event["id"],
                        score=mirror_score,
                        missing_count=len(mirror.get("missing", [])),
                    )

            if isinstance(odds, Exception) or odds is None:
                return []

            # Detector primero (barato, local). LLM enriquecimiento solo se
            # ejecuta si el detector emitió ≥1 pick — antes se llamaba SIEMPRE
            # y se descartaba el resultado (bug de gasto: ~175 pre_match
            # calls/día sin uso). Hoy: pre_match queda gated por picks reales.
            picks = await run_detector(event, odds, correlation_id=correlation_id)
            all_p = [p for p in picks if p.is_bet] if picks else []
            # Procesar markets adicionales del mismo evento (spreads, totals,
            # team_totals, alternate_totals) — multiplica picks por 3-5x.
            for extra_market in getattr(odds, "additional_markets", []) or []:
                try:
                    extra_picks = await run_detector(
                        event, extra_market, correlation_id=correlation_id
                    )
                    if extra_picks:
                        all_p.extend(p for p in extra_picks if p.is_bet)
                except Exception as exc_mkt:
                    logger.debug(
                        "deep.extra_market_fail",
                        market=extra_market.market,
                        error=str(exc_mkt)[:100],
                    )

            # Sprint 10 Fase 1 — correlation filter: picks h2h + spread del
            # mismo side son ~85% correlacionados (Koopman & Lit 2015).
            # Emitir ambos duplica riesgo. Conservar el de mayor edge.
            if len(all_p) > 1:
                from apuestas.betting.correlation_filter import filter_correlated_picks

                all_p, _dropped = filter_correlated_picks(all_p)
                if _dropped:
                    logger.info(
                        "deep.correlation_filter",
                        event_id=event["id"],
                        dropped=len(_dropped),
                    )

            # Props scanning — solo si existen player_prop_lines + historial
            # suficiente. Fail-silent: no bloquea picks principales.
            try:
                prop_picks = await detect_all_player_props_for_match(
                    match_id=int(event["id"]),
                    min_ev=0.04,
                    kelly_fraction_cap=0.15,
                    min_historical_samples=15,
                )
                if prop_picks:
                    logger.info(
                        "deep.props_detected",
                        event_id=event["id"],
                        n_props=len(prop_picks),
                    )
                    await _push_props_to_summary(
                        prop_picks=prop_picks,
                        event=event,
                    )
            except Exception as exc_props:
                logger.debug(
                    "deep.props_scan_fail",
                    event_id=event["id"],
                    error=str(exc_props)[:100],
                )

            # LLM enrichment SOLO si el detector emitió picks. Antes corría
            # incondicional para todos los eventos (waste). Ahora paga el
            # pre_match call (~$0.0006/event) sólo cuando hay valor real.
            if all_p:
                try:
                    _llm_result = await llm_analyze_event(
                        event,
                        rag if isinstance(rag, str) else "",
                        correlation_id=correlation_id,
                    )
                    if _llm_result is not None:
                        llm_results_by_event[int(event["id"])] = _llm_result
                except Exception as exc_llm:
                    logger.debug(
                        "deep.llm_enrich_fail",
                        event_id=event["id"],
                        error=str(exc_llm)[:100],
                    )

            return all_p

    llm_results_by_event: dict[int, dict[str, Any]] = {}
    per_event_picks = await asyncio.gather(
        *[_analyze_event(e) for e in events], return_exceptions=True
    )
    for pe in per_event_picks:
        if isinstance(pe, list):
            all_picks.extend(pe)

    # Emisión directa a pick_alerts (sin staking/Kelly/portfolio).
    # Sprint 2 extenderá esto con alert_store.should_emit_or_upgrade; el
    # stub actual hace dedup contra los unique indexes 0021/0022 y loguea
    # new/upgrade/skip para telemetría.
    new_ids: list[int] = []
    upgraded: int = 0
    skipped: int = 0
    if all_picks:
        new_ids, upgraded, skipped = await emit_alerts(all_picks)
        logger.info(
            "deep_analysis.emit",
            new=len(new_ids),
            upgrade=upgraded,
            skip=skipped,
        )

    # Persistir llm_analysis en predictions para los eventos con picks nuevos.
    # Se hace POST-emit para que el INSERT de predictions ya exista.
    if llm_results_by_event:
        try:
            import json as _json

            from apuestas.db import session_scope as _llm_ss

            async with _llm_ss() as _s:
                for _mid, _llm in llm_results_by_event.items():
                    await _s.execute(
                        text(
                            """UPDATE predictions
                               SET llm_analysis = CAST(:la AS json)
                               WHERE match_id = :mid
                                 AND created_at > now() - interval '2 minutes'"""
                        ),
                        {"la": _json.dumps(_llm), "mid": int(_mid)},
                    )
                await _s.commit()
            logger.info("deep_analysis.llm_persisted", n=len(llm_results_by_event))
        except Exception as _exc:
            logger.warning("deep_analysis.llm_persist_fail", error=str(_exc)[:120])

    summary = {
        "events_checked": len(events),
        "picks_detected": len(all_picks),
        "alerts_new": len(new_ids),
        "alerts_upgrade": upgraded,
        "alerts_skip": skipped,
    }
    logger.info("deep_analysis.done", **summary)
    # NOTE: MCP cleanup se hace fuera del flow (runner wrapper). Si lo haces
    # dentro del @flow, Prefect marca Crashed por el CancelledError del shield.
    return summary


async def _apply_sprint14_market_movement(vb: Any, match_id: int, start_time: Any) -> None:
    """Sprint 14 #158 — tag pick con sharp_confirmation / sharp_disagreement
    si Pinnacle line movió >2% pre-kickoff. Fail-silent.
    """
    import os as _os

    if _os.environ.get("APUESTAS_MARKET_MOVEMENT_TAGS", "true").lower() != "true":
        return
    try:
        from apuestas.betting.market_movement import (
            classify_move_vs_pick,
            compute_line_movement,
        )
        from apuestas.db import session_scope as _ss

        async with _ss() as _s:
            mv = await compute_line_movement(
                _s,
                match_id=match_id,
                market=vb.market,
                outcome=vb.outcome,
                line=getattr(vb, "line", None),
                match_start=start_time,
            )
        tag = classify_move_vs_pick(pick_outcome=vb.outcome, line_move_pct=mv["line_move_pct"])
        flags = list(getattr(vb, "flags", None) or [])
        if tag == "sharp_confirmation" and "sharp_confirmation" not in flags:
            flags.append("sharp_confirmation")
        elif tag == "sharp_disagreement" and "sharp_disagreement" not in flags:
            flags.append("sharp_disagreement")
        if mv.get("has_sharp_move", 0.0) >= 1.0 and "steam_move" not in flags:
            flags.append("steam_move")
        if tag != "stable" or mv.get("has_sharp_move", 0.0) >= 1.0:
            try:
                object.__setattr__(vb, "flags", flags)
            except AttributeError, TypeError:
                pass
    except Exception:
        pass


def _apply_sprint11_soft_tags(vb: Any, sport_code: str) -> None:
    """Sprint 11 — tag soft_tags desde execution_timing + information_edge.

    Fail-silent: cualquier excepción no bloquea el emit.
    """
    import os as _os

    if _os.environ.get("APUESTAS_SPRINT11_SOFT_TAGS", "true").lower() != "true":
        return
    try:
        from apuestas.betting.execution_timing import score_timing

        kickoff = getattr(vb, "start_time", None)
        if kickoff is not None:
            ts = score_timing(sport_code=sport_code, kickoff_utc=kickoff)
            flags = list(getattr(vb, "flags", None) or [])
            if ts.in_optimal_window and "optimal_timing" not in flags:
                flags.append("optimal_timing")
                try:
                    object.__setattr__(vb, "flags", flags)
                except AttributeError, TypeError:
                    pass
            elif ts.edge_multiplier < 0.8 and "late_window" not in flags:
                flags.append("late_window")
                try:
                    object.__setattr__(vb, "flags", flags)
                except AttributeError, TypeError:
                    pass
    except Exception:
        pass

    # Sprint 12 — CLV anticipado via closing_line_predictor.
    # Si el predictor anticipa que el cierre va a ser MEJOR que odds actual
    # (i.e. CLV anticipado negativo), significa que la línea se moverá en
    # contra → baja tier con soft_tag `anticipated_clv_negative`.
    try:
        if _os.environ.get("APUESTAS_ANTICIPATED_CLV", "true").lower() == "true":
            from apuestas.betting.closing_line_predictor import (
                ClosingLineFeatures,
                load_fitted_predictor,
            )

            current_odds = float(getattr(vb, "odds", 0) or 0)
            sharp_consensus = float(
                (getattr(vb, "p_pinnacle_fair", 0) and (1.0 / getattr(vb, "p_pinnacle_fair", 0.5)))
                or current_odds
            )
            kickoff = getattr(vb, "start_time", None)
            if kickoff is not None and current_odds > 1.0:
                from datetime import UTC as _UTC
                from datetime import datetime as _dt

                hrs = (kickoff - _dt.now(tz=_UTC)).total_seconds() / 3600.0
                feats = ClosingLineFeatures(
                    current_odds=current_odds,
                    line_movement_4h=0.0,
                    line_movement_1h=0.0,
                    n_updates_4h=1,
                    n_books_tracking=1,
                    sharp_book_consensus=sharp_consensus,
                    public_pct=0.5,
                    hours_until_start=max(hrs, 0.1),
                    sport_code=sport_code,
                    league_id=None,
                )
                predictor = load_fitted_predictor(sport_code)
                anticipated_clv = predictor.anticipated_clv(feats)
                flags = list(getattr(vb, "flags", None) or [])
                if anticipated_clv < -0.02 and "anticipated_clv_negative" not in flags:
                    flags.append("anticipated_clv_negative")
                    try:
                        object.__setattr__(vb, "flags", flags)
                    except AttributeError, TypeError:
                        pass
                elif anticipated_clv > 0.03 and "anticipated_clv_positive" not in flags:
                    flags.append("anticipated_clv_positive")
                    try:
                        object.__setattr__(vb, "flags", flags)
                    except AttributeError, TypeError:
                        pass
    except Exception:
        pass


def consolidate_picks_per_market(picks: list[Any]) -> list[Any]:
    """Un solo outcome por (match_id, market, line): el de mayor edge.

    Evita emitir simultáneamente home+away del mismo mercado desde distintos
    books. El unique index uq_pick_alerts_market bloquearía la segunda
    inserción, pero esta función se anticipa filtrando en memoria y
    loggeando el descarte para auditoría.
    """
    best: dict[tuple[Any, str, Any], Any] = {}
    dropped = 0
    for p in picks:
        key = (p.event_id, p.market, p.line)
        cur = best.get(key)
        if cur is None or float(p.edge or 0) > float(cur.edge or 0):
            if cur is not None:
                dropped += 1
            best[key] = p
        else:
            dropped += 1
    if dropped:
        logger.info("emit_alerts.consolidated", dropped_by_lower_edge=dropped)
    return list(best.values())


async def emit_alerts(picks: list[Any]) -> tuple[list[int], int, int]:
    """Inserta value picks en `pick_alerts` usando `alert_store` para dedup.

    Pipeline:
      1. `consolidate_picks_per_market` deja un solo outcome por mercado.
      2. `should_emit_or_upgrade` decide new/upgrade/skip con umbrales
         adaptativos por deporte (`config/upgrade_thresholds.yaml`).
      3. 'new' → INSERT prediction + pick_alerts + notify Telegram.
         'upgrade' → UPDATE best_odds_* + upgrade_count, notify compacto.
         'skip' → decision_log (no enviado a Telegram).

    Retorna: (new_ids, upgrade_count, skip_count).
    """
    from datetime import UTC, datetime

    from sqlalchemy import text as _text

    from apuestas.betting.alert_store import mark_upgrade, should_emit_or_upgrade
    from apuestas.db import session_scope as _session_scope
    from apuestas.ml.registry import load_production_model

    picks = consolidate_picks_per_market(picks)

    # Sprint 4e — tag picks con soft_tags de odds_spike (steam/pricing_error/soft_line).
    # Un SpikeAlert reciente en el mismo (match, market, outcome, book) anexa su tag
    # a ValueBet.flags. Fail-safe: un error en el detector no aborta el emit.
    spike_index: dict[tuple[int, str, str, str], str] = {}
    try:
        import os as _os

        if _os.environ.get("ENABLE_ODDS_SPIKE", "true").lower() == "true":
            from apuestas.betting.odds_spike import run_all_detectors

            alerts = await run_all_detectors()
            for a in alerts:
                key = (
                    int(a.match_id),
                    str(a.market).lower(),
                    str(a.outcome).lower(),
                    str(a.bookmaker).lower(),
                )
                spike_index[key] = str(a.tag)
            logger.info("emit_alerts.spike_index_loaded", n=len(spike_index))
    except Exception as exc:
        logger.debug("emit_alerts.spike_load_fail", error=str(exc)[:80])

    def _anotar_spike_tag(vb: Any) -> None:
        key = (
            int(vb.event_id),
            str(vb.market).lower(),
            str(vb.outcome).lower(),
            str(vb.bookmaker).lower(),
        )
        tag = spike_index.get(key)
        if tag is None:
            return
        flags = list(getattr(vb, "flags", None) or [])
        if tag not in flags:
            flags.append(tag)
            try:
                object.__setattr__(vb, "flags", flags)
            except AttributeError, TypeError:
                pass

    new_ids: list[int] = []
    upgrade_count = 0
    skip_count = 0
    _notify_pending: list[tuple[int, dict[str, Any], str]] = []
    _model_cache: dict[tuple[str, str], tuple[str, str]] = {}

    async def _resolve_model_meta(sport_code: str, market: str) -> tuple[str, str]:
        key = (sport_code, market)
        if key in _model_cache:
            return _model_cache[key]
        loaded = await load_production_model(sport_code, market)
        if loaded is None:
            # Soccer sin modelo registrado aún corre Dixon-Coles como fallback.
            from apuestas.sports import canonical_sport_code as _canon

            is_soccer = _canon(sport_code) == "soccer"
            meta = ("dixon_coles_v1", "v1") if is_soccer else ("no_model", "v1")
        else:
            info, raw_obj = loaded
            # Modelos DC guardados con "model_type": "dixon_coles" (train_soccer.py).
            # El sklearn inference se saltea (feature_names=[]) y DC corre como fallback;
            # registrar "dixon_coles_v1" en vez del nombre de registro sklearn.
            if isinstance(raw_obj, dict) and raw_obj.get("model_type") == "dixon_coles":
                meta = ("dixon_coles_v1", info.model_version)
            else:
                meta = (info.model_name, info.model_version)
        _model_cache[key] = meta
        return meta

    async with _session_scope() as s:
        for vb in picks:
            sport_code = getattr(vb, "sport_code", None) or ""
            if not sport_code:
                sport_row = (
                    await s.execute(
                        _text("SELECT sport_code FROM matches WHERE id = :id"),
                        {"id": vb.event_id},
                    )
                ).first()
                sport_code = sport_row.sport_code if sport_row else "unknown"
                try:
                    object.__setattr__(vb, "sport_code", sport_code)
                except AttributeError, TypeError:
                    pass

            _anotar_spike_tag(vb)

            # Sprint 11 — tag soft_tags desde execution_timing + information_edge
            _apply_sprint11_soft_tags(vb, sport_code)

            # Sprint 14 #158 — market movement (Pinnacle line 6h→30min)
            try:
                await _apply_sprint14_market_movement(vb, int(vb.event_id), vb.start_time)
            except Exception:
                pass

            # Sprint 14 #150 — NBA injury penalty (baja tier si >8%)
            if sport_code == "nba":
                try:
                    from apuestas.ingest.nba_injury_report import (
                        compute_team_injury_penalty,
                    )

                    home_penalty = await compute_team_injury_penalty(
                        s, int(getattr(vb, "home_team_id", 0) or 0), vb.start_time
                    )
                    away_penalty = await compute_team_injury_penalty(
                        s, int(getattr(vb, "away_team_id", 0) or 0), vb.start_time
                    )
                    max_penalty = max(home_penalty, away_penalty)
                    if max_penalty >= 0.08:
                        flags = list(getattr(vb, "flags", None) or [])
                        if "injury_heavy" not in flags:
                            flags.append("injury_heavy")
                            try:
                                object.__setattr__(vb, "flags", flags)
                            except AttributeError, TypeError:
                                pass
                except Exception:
                    pass

            decision = await should_emit_or_upgrade(s, vb)
            new_odds = float(vb.odds)

            if decision == "skip":
                skip_count += 1
                try:
                    from apuestas.obs.metrics import inc_alerts_skip as _m_skip

                    _m_skip(sport_code)
                except Exception:
                    pass
                # decision_log ya recibe registro en detector.persist_decision;
                # aquí sólo contamos para telemetría.
                continue

            if decision == "upgrade":
                alert_row = (
                    await s.execute(
                        _text(
                            """
                            SELECT id FROM pick_alerts
                            WHERE match_id = :mid AND market = :mk
                              AND COALESCE(line, -999) = COALESCE(:ln, -999)
                              AND outcome = :oc
                              AND (outcome_result IS NULL OR outcome_result = 'pending')
                            LIMIT 1
                            """
                        ),
                        {
                            "mid": vb.event_id,
                            "mk": vb.market,
                            "ln": vb.line,
                            "oc": vb.outcome,
                        },
                    )
                ).first()
                if alert_row is not None:
                    await mark_upgrade(
                        s,
                        int(alert_row.id),
                        new_odds=new_odds,
                        bookmaker=vb.bookmaker,
                    )
                    upgrade_count += 1
                    try:
                        from apuestas.obs.metrics import inc_alerts_upgrade as _m_up

                        _m_up(sport_code)
                    except Exception:
                        pass
                    logger.info(
                        "emit_alerts.upgrade",
                        alert_id=int(alert_row.id),
                        match_id=vb.event_id,
                        market=vb.market,
                        outcome=vb.outcome,
                        new_best=new_odds,
                        book=vb.bookmaker,
                    )
                    upgrade_detail = _build_notify_detail(vb, sport_code, new_odds)
                    upgrade_detail["_pick_alert_id"] = int(alert_row.id)
                    upgrade_detail = await _enrich_with_consensus(s, upgrade_detail)
                    upgrade_detail = await _enrich_with_regional(s, upgrade_detail)
                    upgrade_detail = await _enrich_with_weather(s, upgrade_detail)
                    upgrade_detail = await _enrich_with_historical_features(s, upgrade_detail)
                    _notify_pending.append((int(alert_row.id), upgrade_detail, "upgrade"))
                continue

            # decision == "new"
            mn, mv = await _resolve_model_meta(sport_code, vb.market)
            pred_row = (
                await s.execute(
                    _text(
                        """
                        INSERT INTO predictions
                          (match_id, model_name, model_version, market, outcome,
                           line, probability, p_lower, p_upper, ev, decision,
                           created_at)
                        VALUES
                          (:mid, :mn, :mv, :mk, :oc, :ln, :pr, :pl, :pu, :ev,
                           'bet', :ts)
                        RETURNING id
                        """
                    ),
                    {
                        "mid": vb.event_id,
                        "mn": mn,
                        "mv": mv,
                        "mk": vb.market,
                        "oc": vb.outcome,
                        "ln": vb.line,
                        "pr": float(vb.p_blended or 0),
                        "pl": float(vb.p_lower) if vb.p_lower is not None else None,
                        "pu": float(vb.p_upper) if vb.p_upper is not None else None,
                        "ev": float(vb.ev),
                        "ts": datetime.now(tz=UTC),
                    },
                )
            ).first()
            if pred_row is None:
                continue
            pred_id = int(pred_row[0])

            try:
                alert_row = (
                    await s.execute(
                        _text(
                            """
                            INSERT INTO pick_alerts
                              (match_id, prediction_id, bookmaker, market,
                               outcome, line, odds_placed, placed_at, status,
                               best_odds_seen, best_odds_book,
                               best_odds_updated_at, last_alert_at)
                            VALUES
                              (:mid, :pid, :bk, :mk, :oc, :ln, :od, :ts,
                               'pending', :od, :bk, :ts, :ts)
                            RETURNING id
                            """
                        ),
                        {
                            "mid": vb.event_id,
                            "pid": pred_id,
                            "bk": vb.bookmaker,
                            "mk": vb.market,
                            "oc": vb.outcome,
                            "ln": vb.line,
                            "od": new_odds,
                            "ts": datetime.now(tz=UTC),
                        },
                    )
                ).first()
            except Exception as exc:
                msg = str(exc).lower()
                if "uq_pick_alerts" in msg or "unique" in msg or "duplicate" in msg:
                    logger.info(
                        "emit_alerts.race_condition_skip",
                        match_id=vb.event_id,
                        outcome=vb.outcome,
                    )
                    await s.rollback()
                    skip_count += 1
                    continue
                raise

            if alert_row is None:
                continue
            alert_id = int(alert_row[0])
            new_ids.append(alert_id)
            try:
                from apuestas.obs.metrics import inc_alerts_new as _m_new

                _m_new(sport_code)
            except Exception:
                pass

            # Sprint 8 wire — SHAP top-5 persistente en pick_alerts.shap_top5.
            # Fail-safe total: cualquier error no impide el emit.
            try:
                await _persist_shap_top5(s, alert_id, vb, sport_code)
            except Exception as exc:
                logger.warning(
                    "emit_alerts.shap_persist_fail",
                    alert_id=alert_id,
                    error=str(exc)[:80],
                )

            # Notificación Telegram (reutiliza send_pick_to_telegram del legacy).
            try:
                meta_row = (
                    await s.execute(
                        _text(
                            """
                            SELECT m.sport_code,
                                   ht.name AS home,
                                   at.name AS away
                            FROM matches m
                            JOIN teams ht ON ht.id = m.home_team_id
                            JOIN teams at ON at.id = m.away_team_id
                            WHERE m.id = :mid
                            """
                        ),
                        {"mid": vb.event_id},
                    )
                ).first()
                detail = _build_notify_detail(
                    vb,
                    meta_row.sport_code if meta_row else sport_code,
                    new_odds,
                    home=meta_row.home if meta_row else "",
                    away=meta_row.away if meta_row else "",
                )
                detail["_pick_alert_id"] = alert_id
                detail = await _enrich_with_consensus(s, detail)

                # Sprint C abr-2026 — filtro p_consensus_sharp opt-in.
                # Backtest 7d: picks con p_sharp<0.40 → ROI -56%; con ≥0.40 → +12.5%.
                # Si flag activo y consensus tiene ≥1 fuente y p_sharp<umbral →
                # cancela alert + persiste decision_log con skip_reason.
                if _should_filter_low_sharp(detail):
                    # `emit_alerts` no recibe correlation_id en signature;
                    # decision_log permite NULL en esa columna.
                    await _cancel_alert_low_sharp(
                        s,
                        alert_id=alert_id,
                        vb=vb,
                        p_sharp=float(detail.get("p_consensus_sharp") or 0.0),
                        correlation_id=None,
                    )
                    continue  # sin notificación Telegram

                # F5 — CLV anti-stale: si Pinnacle movió >2% en contra del pick
                # en últimos 30min, el edge es stale (info pública incorporada).
                clv_ok, drift = await _check_clv_anti_stale(
                    s,
                    match_id=int(vb.event_id),
                    market=str(vb.market),
                    outcome=str(vb.outcome),
                    line=vb.line,
                )
                if not clv_ok:
                    await _cancel_alert_clv(
                        s,
                        alert_id=alert_id,
                        vb=vb,
                        drift=float(drift or 0.0),
                    )
                    continue

                # F4 — slippage guard: re-check odds del book TARGET en odds_history
                # antes de notificar. Si el book ya movió >5% en contra, cancela.
                ok, _curr = await _check_slippage(
                    s,
                    match_id=int(vb.event_id),
                    bookmaker=str(vb.bookmaker),
                    market=str(vb.market),
                    outcome=str(vb.outcome),
                    line=vb.line,
                    odds_emitted=float(vb.odds),
                )
                if not ok:
                    await _cancel_alert_slippage(
                        s,
                        alert_id=alert_id,
                        vb=vb,
                        odds_emitted=float(vb.odds),
                        odds_current=float(_curr or 0.0),
                    )
                    continue

                detail = await _enrich_with_regional(s, detail)
                detail = await _enrich_with_weather(s, detail)
                detail = await _enrich_with_historical_features(s, detail)
                _notify_pending.append((alert_id, detail, "new"))
            except Exception as exc:
                logger.debug("emit_alerts.notify_prep_fail", alert_id=alert_id, error=str(exc)[:80])
        await s.commit()

    # Sync counters con _notify_pending: si un pick fue cancelado post-insert
    # (low_sharp / clv_stale / slippage), NO debe contar en "alerts_new" ni
    # "upgrade_count" ya que nunca se notificó. Antes: el summary mostraba
    # "Picks nuevos: 3" cuando solo 2 mensajes llegaron a Telegram.
    _notified_new = {nid for nid, _, kind in _notify_pending if kind == "new"}
    _notified_upgrades = {nid for nid, _, kind in _notify_pending if kind == "upgrade"}
    new_ids = [nid for nid in new_ids if nid in _notified_new]
    upgrade_count = len(_notified_upgrades)

    if _notify_pending:
        try:
            from apuestas.bot.telegram import send_pick_to_telegram

            # Gap 10 / A11 — batching cuando hay >5 picks: 250 ms entre envíos
            # para no pegarle al rate-limit Telegram (30 msg/s global, 1/s chat).
            batch_delay = 0.25 if len(_notify_pending) > 5 else 0.0
            for alert_id, detail, _kind in _notify_pending:
                try:
                    await send_pick_to_telegram(alert_id, detail)
                except Exception as exc:
                    logger.debug(
                        "emit_alerts.notify_fail",
                        alert_id=alert_id,
                        error=str(exc)[:80],
                    )
                if batch_delay > 0:
                    await asyncio.sleep(batch_delay)
        except ImportError:
            logger.debug("emit_alerts.telegram_unavailable")

    return new_ids, upgrade_count, skip_count


def _build_notify_detail(
    vb: Any,
    sport_code: str,
    new_odds: float,
    *,
    home: str = "",
    away: str = "",
) -> dict[str, Any]:
    """Detalle compartido entre notify new y upgrade."""
    return {
        "match_id": vb.event_id,
        "home": home,
        "away": away,
        "sport": sport_code,
        "market": vb.market,
        "outcome": vb.outcome,
        "line": vb.line,
        "bookmaker": vb.bookmaker,
        "odds": float(new_odds),
        "ev_pct": float(vb.ev),
        "reason": getattr(vb, "llm_reason", "") or getattr(vb, "reason", ""),
        "start_time": vb.start_time,
        "p_blended": getattr(vb, "p_blended", None),
        "p_lower": getattr(vb, "p_lower", None),
        "p_upper": getattr(vb, "p_upper", None),
        "p_model": getattr(vb, "p_model", None),
        "p_pinnacle_fair": getattr(vb, "p_pinnacle_fair", None),
        "implied_prob": getattr(vb, "implied_prob", None),
        "soft_tags": getattr(vb, "flags", []) or [],
    }


async def _persist_shap_top5(session: Any, alert_id: int, vb: Any, sport_code: str) -> None:
    """Calcula SHAP top-5 con el modelo production y persiste en pick_alerts.shap_top5."""
    import os as _os

    if _os.environ.get("ENABLE_SHAP", "true").lower() != "true":
        return
    from sqlalchemy import text as _t

    from apuestas.features.feature_store import build_match_features
    from apuestas.ml.registry import load_production_model
    from apuestas.ml.shap_explain import SHAPExplainer

    loaded = await load_production_model(sport_code, vb.market)
    if loaded is None:
        return
    info, model_obj = loaded
    estimator = model_obj.get("estimator") if isinstance(model_obj, dict) else model_obj
    feature_names = model_obj.get("feature_names", []) if isinstance(model_obj, dict) else []
    if estimator is None or not feature_names:
        return
    home_id = getattr(vb, "home_team_id", None)
    away_id = getattr(vb, "away_team_id", None)
    start_time = getattr(vb, "start_time", None)
    if home_id is None or away_id is None or start_time is None:
        mrow = (
            await session.execute(
                _t("SELECT home_team_id, away_team_id, start_time FROM matches WHERE id = :id"),
                {"id": vb.event_id},
            )
        ).first()
        if mrow is None:
            return
        home_id = home_id or mrow.home_team_id
        away_id = away_id or mrow.away_team_id
        start_time = start_time or mrow.start_time
    # Para NBA/MLB/NFL intentar el pipeline raw primero (más features, menos gaps).
    x: Any = None
    if sport_code in {"nba", "mlb", "nfl"}:
        try:
            from apuestas.features.runtime_features import build_match_features_from_raw

            x = await build_match_features_from_raw(
                sport_code=sport_code,
                home_team_id=int(home_id),
                away_team_id=int(away_id),
                match_start=start_time,
                feature_names=feature_names,
            )
        except Exception:
            pass
    if x is None:
        x = await build_match_features(
            sport_code=sport_code,
            home_team_id=int(home_id),
            away_team_id=int(away_id),
            match_start=start_time,
            feature_names=feature_names,
        )
    if x is None:
        logger.warning("emit_alerts.shap_no_features", alert_id=alert_id, sport=sport_code)
        return
    explainer = SHAPExplainer(estimator, feature_names)
    top5 = explainer.explain_row(x, top_k=5)
    if not top5:
        return
    payload = [t.to_dict() for t in top5]
    import json as _json

    await session.execute(
        _t("UPDATE pick_alerts SET shap_top5 = CAST(:v AS jsonb) WHERE id = :id"),
        {"v": _json.dumps(payload), "id": int(alert_id)},
    )
    logger.info(
        "emit_alerts.shap_persisted",
        alert_id=alert_id,
        model=info.model_name,
        n_features=len(top5),
    )


async def _enrich_with_historical_features(session: Any, detail: dict[str, Any]) -> dict[str, Any]:
    """Sprint 12 — anexa features de tablas históricas backfilled.

    Consulta `odds_history_archive`, `team_elo_daily`, `nfl_epa_plays`,
    `fangraphs_team_stats_daily`, `pitcher_game_stats` y agrega features
    al detail. Fail-silent: cualquier query que falle se ignora.
    """
    try:
        import os as _os

        if _os.environ.get("ENABLE_HIST_FEATURES", "true").lower() != "true":
            return detail

        from apuestas.features.historical_data_features import (
            fetch_closing_odds_implied_prob,
            fetch_clubelo_for_match,
            fetch_fangraphs_team,
            fetch_nfl_epa_rolling,
        )

        sport = (detail.get("sport_code") or "").lower()
        home_team = detail.get("home_team") or detail.get("home_name") or ""
        away_team = detail.get("away_team") or detail.get("away_name") or ""
        match_date = detail.get("start_date")

        if sport == "soccer" and match_date and home_team and away_team:
            elo_feats = await fetch_clubelo_for_match(session, home_team, away_team, match_date)
            if elo_feats:
                detail["hist_elo_clubelo"] = elo_feats
            closing = await fetch_closing_odds_implied_prob(
                session, home_team, away_team, match_date
            )
            if closing:
                detail["hist_closing_implied"] = closing

        if sport == "nfl" and match_date:
            home_abbr = detail.get("home_abbr") or ""
            away_abbr = detail.get("away_abbr") or ""
            if home_abbr:
                epa_h = await fetch_nfl_epa_rolling(session, home_abbr, match_date)
                if epa_h:
                    detail["hist_nfl_epa_home"] = epa_h
            if away_abbr:
                epa_a = await fetch_nfl_epa_rolling(session, away_abbr, match_date)
                if epa_a:
                    detail["hist_nfl_epa_away"] = epa_a

        if sport == "mlb" and match_date:
            home_id = detail.get("home_team_id")
            away_id = detail.get("away_team_id")
            if home_id:
                fg_h = await fetch_fangraphs_team(session, int(home_id), match_date)
                if fg_h:
                    detail["hist_fangraphs_home"] = fg_h
            if away_id:
                fg_a = await fetch_fangraphs_team(session, int(away_id), match_date)
                if fg_a:
                    detail["hist_fangraphs_away"] = fg_a
    except Exception as _exc:
        logger.debug("hist_features.enrich_fail", error=str(_exc)[:100])
    return detail


async def _enrich_with_consensus(session: Any, detail: dict[str, Any]) -> dict[str, Any]:
    """Anexa p_consensus_sharp + market_consensus_delta al detail.

    Fuentes: Pinnacle de-vigged (desde el propio pick), Polymarket midpoint,
    Kalshi yes_midpoint. Si falta alguna, el `compute_consensus_sharp`
    renormaliza pesos sobre las presentes.

    Persiste también en `pick_alerts.p_consensus_sharp` y
    `pick_alerts.market_consensus_delta` para evidencia retrospectiva.
    """
    try:
        import os as _os

        if _os.environ.get("ENABLE_CONSENSUS", "true").lower() != "true":
            return detail
        match_id = detail.get("match_id")
        p_blended = detail.get("p_blended") or detail.get("p_pinnacle_fair")
        if match_id is None or p_blended is None:
            return detail

        from sqlalchemy import text as _t

        from apuestas.betting.consensus_fetch import (
            fetch_kalshi_midpoint,
            fetch_polymarket_midpoint,
        )
        from apuestas.betting.market_consensus import (
            compute_consensus_sharp,
            consensus_delta,
        )

        pm = await fetch_polymarket_midpoint(session, match_id=int(match_id))
        poly_mid, poly_vol = pm if pm else (None, None)
        kalshi_mid = await fetch_kalshi_midpoint(session, match_id=int(match_id))
        pinn_mid = detail.get("p_pinnacle_fair")

        consensus = compute_consensus_sharp(
            pinnacle_devigged=float(pinn_mid) if pinn_mid is not None else None,
            polymarket_mid=poly_mid,
            kalshi_mid=kalshi_mid,
            polymarket_volume_usd=poly_vol,
        )
        if consensus.sources == 0:
            return detail

        delta = consensus_delta(float(p_blended), consensus)
        detail["p_consensus_sharp"] = consensus.p_consensus
        detail["market_consensus_delta"] = delta
        detail["consensus_sources"] = consensus.sources

        # Si el modelo diverge >8pp y hay ≥2 fuentes, tag disagreement.
        if consensus.sources >= 2 and delta >= 0.08:
            flags = list(detail.get("soft_tags") or [])
            if "market_disagreement" not in flags:
                flags.append("market_disagreement")
            detail["soft_tags"] = flags

        # Persiste en DB para trazabilidad (si hay id de la alerta en detail).
        alert_id = detail.get("_pick_alert_id")
        if alert_id is not None:
            try:
                await session.execute(
                    _t(
                        """
                        UPDATE pick_alerts
                        SET p_consensus_sharp = :p,
                            market_consensus_delta = :d
                        WHERE id = :id
                        """
                    ),
                    {
                        "p": float(consensus.p_consensus),
                        "d": float(delta),
                        "id": int(alert_id),
                    },
                )
            except Exception as exc:
                logger.debug("emit_alerts.consensus_persist_fail", error=str(exc)[:80])
    except Exception as exc:
        logger.debug("emit_alerts.consensus_enrich_fail", error=str(exc)[:80])
    return detail


async def _enrich_with_regional(session: Any, detail: dict[str, Any]) -> dict[str, Any]:
    """Anexa bloque `regional: {mx, us, global}` al detail (Sprint 4c).

    Lee `odds_history` recientes (<3h) para el (match, market, outcome),
    construye BookmakerQuote por casa y llama a regional.compare_regions.
    Fail-safe: cualquier excepción retorna detail sin modificar.
    """
    try:
        import os

        if os.environ.get("ENABLE_REGIONAL", "true").lower() != "true":
            return detail
        match_id = detail.get("match_id")
        market = detail.get("market")
        outcome = detail.get("outcome")
        if match_id is None or market is None or outcome is None:
            return detail

        from sqlalchemy import text as _t

        from apuestas.betting.ev import BookmakerQuote
        from apuestas.betting.regional import compare_regions

        rows = (
            await session.execute(
                _t(
                    """
                    SELECT bookmaker, odds, line
                    FROM odds_history
                    WHERE match_id = :mid
                      AND market = :mk
                      AND outcome = :oc
                      AND ts > now() - interval '3 hours'
                      AND odds > 1.0
                    ORDER BY ts DESC
                    """
                ),
                {"mid": int(match_id), "mk": market, "oc": outcome},
            )
        ).all()
        if not rows:
            return detail

        seen: set[str] = set()
        quotes: list[BookmakerQuote] = []
        for r in rows:
            bm = (r.bookmaker or "").lower()
            if not bm or bm in seen:
                continue
            seen.add(bm)
            try:
                quotes.append(
                    BookmakerQuote(
                        bookmaker=bm,
                        odds=float(r.odds),
                        line=float(r.line) if r.line is not None else None,
                    )
                )
            except TypeError, ValueError:
                continue
        if len(quotes) < 2:
            return detail

        p_fair_raw = detail.get("p_blended") or detail.get("p_pinnacle_fair")
        if p_fair_raw is None:
            return detail
        try:
            p_fair = float(p_fair_raw)
        except TypeError, ValueError:
            return detail

        rec = compare_regions(
            event_id=int(match_id),
            market=str(market),
            outcome=str(outcome),
            p_fair=p_fair,
            quotes=quotes,
        )

        block: dict[str, dict[str, Any]] = {}
        if rec.mx.best_offer is not None:
            bo = rec.mx.best_offer
            block["mx"] = {"book": bo.bookmaker, "odds": float(bo.odds)}
        if rec.us.best_offer is not None:
            bo = rec.us.best_offer
            block["us"] = {"book": bo.bookmaker, "odds": float(bo.odds)}
        best_overall = max(quotes, key=lambda q: q.odds)
        block["global"] = {
            "book": best_overall.bookmaker,
            "odds": float(best_overall.odds),
        }
        if block:
            detail["regional"] = block
    except Exception as exc:
        logger.debug("emit_alerts.regional_enrich_fail", error=str(exc)[:80])
    return detail


async def _enrich_with_weather(session: Any, detail: dict[str, Any]) -> dict[str, Any]:
    """Anexa `weather_summary` + `weather_hint` al detail si hay forecast.

    No-op para deportes indoor o sin forecast. Fail-safe: cualquier error
    retorna el detail sin cambios.
    """
    try:
        import os

        if os.environ.get("ENABLE_WEATHER", "true").lower() != "true":
            return detail
        match_id = detail.get("match_id")
        sport = str(detail.get("sport") or "").lower()
        if match_id is None or sport not in ("mlb", "nfl", "soccer"):
            return detail
        from apuestas.features.weather_match import (
            fetch_match_weather_bucket,
            multiplier_hint,
            summarize_for_pick,
        )

        bucket = await fetch_match_weather_bucket(session, int(match_id))
        if bucket is None:
            return detail
        detail["weather_summary"] = summarize_for_pick(bucket)
        mult = multiplier_hint(bucket, sport)
        if mult is not None:
            detail["weather_hint"] = mult

        # Sprint 14 #155 — soccer weather goals O/U adjustment
        if sport == "soccer":
            try:
                from apuestas.features.soccer_weather import (
                    compute_soccer_weather_features,
                )

                match_start = detail.get("start_time")
                venue_id = detail.get("venue_id")
                if match_start is not None:
                    adj = await compute_soccer_weather_features(
                        session, venue_id=venue_id, match_time=match_start
                    )
                    detail["weather_total_adj"] = adj.get("weather_total_adj", 1.0)
                    detail["weather_variance_factor"] = adj.get("weather_variance_factor", 1.0)
            except Exception:
                pass
    except Exception as exc:
        logger.debug("emit_alerts.weather_enrich_fail", error=str(exc)[:80])
    return detail


def _silence_asyncio_cleanup_noise() -> None:
    """Suprime logs ruidosos de asyncio al cerrar async generators del MCP stdio.

    `mcp` library tiene cancel-scope issues al teardown con Python 3.14; los
    warnings son cosméticos (cleanup tras flow exitoso). Silenciamos SOLO
    `asyncio` ERROR level para evitar spam en los logs.
    """
    import logging
    import sys
    import warnings

    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    logging.getLogger("prefect").setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore", category=ResourceWarning)
    # unraisablehook: descarta RuntimeError del stdio_client cancel scope
    _orig_hook = sys.unraisablehook

    def _hook(unraisable: Any) -> None:
        exc = unraisable.exc_value
        if isinstance(exc, RuntimeError) and "cancel scope" in str(exc):
            return  # silence
        if isinstance(exc, asyncio.CancelledError):
            return  # silence
        _orig_hook(unraisable)

    sys.unraisablehook = _hook


async def run_deep_analysis_safe(**kwargs: Any) -> dict[str, Any]:
    """Wrapper que corre deep_analysis_flow + MCP cleanup post-flow.

    Se ejecuta FUERA del @flow para no propagar CancelledError a Prefect
    runtime (evita marcar flow Crashed por el cleanup).
    """
    result = await deep_analysis_flow(**kwargs)
    try:
        from apuestas.mcp.client import MCPClient

        mcp = MCPClient.get()
        await mcp.stop()
    except Exception:
        pass  # cleanup errors son cosméticos
    return result


if __name__ == "__main__":
    _silence_asyncio_cleanup_noise()
    asyncio.run(deep_analysis_flow())
