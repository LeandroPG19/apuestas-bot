"""Pipeline end-to-end de detección de value bets.

Entrada: lista de quotes por evento (dict bookmaker→lista outcomes).
Salida: lista de ValueBet con p_fair, edge, EV, Kelly, flags de
conformal y repetition.

Flujo (blueprint §6):
1. Agrupar quotes por (event, market).
2. De-vigging de consenso sharp (Pinnacle/Circa/Betfair) con Shin.
3. Blend con p_modelo propio (0.4 modelo / 0.6 Pinnacle si disponibles ambos).
4. Filtro conformal: p_lower > implied + margen.
5. Line shopping en soft books (Caliente/Strendus/Codere) excluyendo Pinnacle.
6. Calcular EV, Kelly con correlation-aware si hay múltiples picks por evento.
7. Dedupe por (event, market, outcome, bookmaker) últimos 15 min.
8. Persist cada pick descartado en decision_log con skip_reason.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sqlalchemy import text


def _load_draw_thresholds() -> dict[int, float]:
    """Carga config/soccer_draw_thresholds.yaml (autotune script). Fallback hardcoded."""
    try:
        path = Path(__file__).resolve().parents[3] / "config" / "soccer_draw_thresholds.yaml"
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            raw = data.get("thresholds", {}) or {}
            return {int(k): float(v) for k, v in raw.items()}
    except Exception:
        pass
    # Fallback hardcoded (MLS + default behavior)
    return {22: 0.30, 253: 0.30}


from apuestas.betting.devig import consensus_fair_probs
from apuestas.betting.ev import (
    BookmakerQuote,
    blend_probabilities,
    implied_probability,
    line_shopping,
)
from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class EventOdds:
    """Odds agrupadas por evento y mercado listas para de-vigging."""

    event_id: int
    event_external_id: str
    market: str
    start_time: datetime
    outcomes: list[str]
    # quotes[bookmaker] -> [odds por outcome en el MISMO orden que outcomes]
    quotes_by_bookmaker: dict[str, list[float]] = field(default_factory=dict)
    lines: list[float | None] | None = None
    league_id: int | None = None
    # Nombre canónico de la liga (e.g. "EPL", "MLS"). Necesario para que
    # `book_power_ratings.get_cached_edge()` indexe por (bookmaker, league)
    # en `line_shopping`. Sin este campo el arg `league=` siempre era None y
    # el book_power_ratings nunca priorizaba soft books con edge histórico.
    league: str | None = None
    sport_code: str | None = None
    # Mejora 4: stage del match (regular/playoff/postseason) para threshold
    # adaptativo y playoff guard. Poblado por deep_analysis desde matches.stage.
    stage: str | None = None
    # Markets secundarios del MISMO evento (spreads, totals, team_totals, etc.)
    # para que el detector los analice en paralelo con el market primario.
    additional_markets: list[EventOdds] = field(default_factory=list)


@dataclass(slots=True)
class ValueBet:
    event_id: int
    event_external_id: str
    market: str
    outcome: str
    line: float | None
    bookmaker: str
    odds: float
    p_model: float | None
    p_pinnacle_fair: float | None
    p_blended: float
    p_lower: float | None
    p_upper: float | None
    implied_prob: float
    edge: float
    ev: float
    sport_code: str | None
    league_id: int | None
    start_time: datetime
    skip_reason: str | None = None
    flags: list[str] = field(default_factory=list)

    @property
    def is_bet(self) -> bool:
        return self.skip_reason is None


@dataclass(slots=True)
class DetectorConfig:
    use_shin_devig: bool = True
    blend_weight_model: float = 0.40
    conformal_margin: float = 0.01
    dedupe_window_minutes: int = 15
    min_sharp_books: int = 1
    # Mejora 3 — conformal width filter. Si el intervalo MAPIE es muy ancho
    # (>15pp) la incertidumbre es demasiada para apostar. Defensa extra contra
    # modelos mal calibrados tipo NBA playoffs o MLB inicio de temporada.
    # Sprint 10 Fase 1: bajado 0.15→0.12 default; override por deporte en
    # conformal_max_width_by_sport (mlb 0.10, soccer 0.08, resto 0.12).
    conformal_max_width: float = field(
        default_factory=lambda: float(
            __import__("os").environ.get("APUESTAS_CONFORMAL_MAX_WIDTH", "0.12")
        )
    )
    conformal_max_width_by_sport: dict[str, float] = field(
        default_factory=lambda: {
            "mlb": 0.10,
            "soccer": 0.08,
            "nba": 0.12,
            "nfl": 0.12,
            "nhl": 0.12,
            "tennis": 0.10,
        }
    )
    # Mejora 2 — draw guard. Si sport=soccer y el mercado es 3-way, saltar
    # h2h/home o h2h/away si el empate tiene prob_model > threshold.
    # Evita picks tipo #105 Go Ahead-AZ 0-0 (empate no pagado en 2-way).
    soccer_max_draw_prob: float = field(
        default_factory=lambda: float(
            __import__("os").environ.get("APUESTAS_SOCCER_MAX_DRAW_PROB", "0.25")
        )
    )
    # Per-league override (MLS tiene ~26% empates históricos vs 22% Europa).
    # Evita picks tipo #42/43 NY Red Bulls-DC / NYCFC-Cincinnati empatados 4-4.
    soccer_max_draw_prob_by_league: dict[int, float] = field(
        default_factory=lambda: _load_draw_thresholds()
    )
    # Mejora 4 — late line soft tag. Picks emitidos <= N min antes del kick
    # marcar con soft_tag='late_line' para que classify_confidence baje tier.
    late_line_minutes: int = field(
        default_factory=lambda: int(__import__("os").environ.get("APUESTAS_LATE_LINE_MIN", "90"))
    )
    # Hard cutoff: picks emitidos <= N min antes del kickoff se bloquean
    # completamente (skip_reason="too_close_to_kickoff"). Las odds a <20 min
    # son stale, el sharp money ya movió la línea y no hay tiempo de ejecutar.
    # Configurable vía APUESTAS_HARD_CUTOFF_MIN (0 = desactivado).
    hard_cutoff_minutes: int = field(
        default_factory=lambda: int(__import__("os").environ.get("APUESTAS_HARD_CUTOFF_MIN", "20"))
    )
    # Mejora 4 — NBA playoff guard. Hasta tener train_nba_playoffs.py dedicado,
    # bloquear picks NBA en stage='playoff' porque distribución es distinta.
    block_playoff_sports: frozenset[str] = field(
        default_factory=lambda: frozenset(
            s.strip().lower()
            for s in __import__("os").environ.get("APUESTAS_BLOCK_PLAYOFF_SPORTS", "nba").split(",")
            if s.strip()
        )
    )
    # Solo emitir picks donde el equipo apostado es FAVORITO (p_fair > umbral).
    # Cambiar comportamiento por preferencia del usuario: "quiero siempre ganar,
    # no apostar al underdog aunque haya valor matemático". Env configurable.
    only_favorites: bool = field(
        default_factory=lambda: (
            __import__("os").environ.get("APUESTAS_ONLY_FAVORITES", "false").lower() == "true"
        )
    )
    favorite_min_prob: float = field(
        default_factory=lambda: float(
            __import__("os").environ.get("APUESTAS_FAVORITE_MIN_PROB", "0.55")
        )
    )
    # Fase 1.2 — low-hold filter: rechaza markets con vig agregado >3% (los pros
    # solo juegan cuando el hold lo permite; en Caliente MX el hold típico es
    # 8-10% → edge modelo raramente supera eso). Relajar a 5% si solo hay un
    # book disponible (fallback Pinnacle-only ingest limitado).
    # Env-configurable: `APUESTAS_MAX_HOLD` + `APUESTAS_MAX_HOLD_SINGLE_BOOK`
    max_hold: float = field(
        default_factory=lambda: float(__import__("os").environ.get("APUESTAS_MAX_HOLD", "0.03"))
    )
    max_hold_single_book: float = field(
        default_factory=lambda: float(
            __import__("os").environ.get("APUESTAS_MAX_HOLD_SINGLE_BOOK", "0.05")
        )
    )
    # F3 (hardening abr-2026) — hold del SOFT BOOK específico donde se va a
    # apostar (no del consenso). Si Caliente/Codere ofrecen un mercado con vig
    # >7% en sus 2 outcomes, el edge real es muy probablemente artefacto del
    # margen del book, no value real. Default 0.07 (7%); 0.0 desactiva el filtro.
    max_hold_target_book: float = field(
        default_factory=lambda: float(
            __import__("os").environ.get("APUESTAS_MAX_HOLD_TARGET_BOOK", "0.07")
        )
    )
    # F7 (hardening) — sample size guard del modelo production. Si el modelo
    # tiene <50 matches de training, su calibración es ruidosa → no emit.
    # Por sport (override env: APUESTAS_MIN_TRAIN_SAMPLES=200).
    min_train_samples: int = field(
        default_factory=lambda: int(
            __import__("os").environ.get("APUESTAS_MIN_TRAIN_SAMPLES", "50")
        )
    )
    # F2 (hardening) — blend adaptive según Brier holdout del modelo:
    # Brier 0.20 (excelente) → weight = base × 1.5
    # Brier 0.22 (bueno)     → weight = base × 1.0
    # Brier 0.25 (≈random)   → weight = base × 0.5 (modelo malo, casi solo Pinnacle)
    # Activable: APUESTAS_ADAPTIVE_BLEND=true (default true).
    adaptive_blend_enabled: bool = field(
        default_factory=lambda: (
            __import__("os").environ.get("APUESTAS_ADAPTIVE_BLEND", "true").lower() == "true"
        )
    )
    # Books donde el apostador puede tomar la apuesta. Incluye soft books MX
    # (Caliente/Codere), US regulados (DK/FanDuel/BetMGM/Caesars), offshore
    # (BetUS/BetOnline/Bovada) y books europeos (Bet365/Betway/BWin) + todos
    # los books que aparecen vía OddsJam aggregator (80+). Pinnacle/Circa son
    # SHARP → usados solo como benchmark fair, no para apostar (exclude_sharp=True).
    soft_books_allowed: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                # MX regulados
                "caliente",
                "strendus",
                "codere",
                "winpot",
                "campobet",
                "jugabet",
                # US regulados
                "draftkings",
                "fanduel",
                "betmgm",
                "caesars",
                "betrivers",
                "pointsbet",
                "bet105",
                "betparx",
                "fanatics",
                "hard rock",
                "borgata",
                "twinspires",
                "sugarhouse",
                "wynnbet",
                "betjack",
                "thescore",
                "bally bet",
                "tipico",
                # Offshore
                "betus",
                "betonline",
                "betonlineag",  # The Odds API slug exacto
                "bovada",
                "mybookieag",
                "betwhale",
                "everygame",
                "sportsbetting_ag",
                "bookmaker",
                "lowvig",  # LowVig.ag — The Odds API
                "gtbets",  # GTBets — The Odds API
                "williamhill_us",  # William Hill US — The Odds API
                "williamhill",  # William Hill (EU) — The Odds API slug sin espacio
                "unibet_nl",  # Unibet Holanda
                "unibet_it",  # Unibet Italia
                # redundante (ya en US regulados)
                # EU / internacional
                "bet365",
                "betway",
                "bwin",
                "betano",
                "betsson",
                "william hill",
                "ladbrokes",
                "unibet",
                "888sport",
                "comeon",  # Sprint B abr-2026: Kambi multi-operador validado
                "nordicbet",  # Kambi multi-operador
                "expekt",  # Kambi multi-operador
                "mariacasino",  # Kambi multi-operador
                "pinnacle_alt",
                "leovegas",
                "coolbet",
                "bet99",
                "rivalry",
                "crypto.com",
                "sports interaction",
                "playnow",
                # Exchange / P2P / prediction
                "novig",
                "prophet x",
                "sporttrade",
                "kalshi",
                "polymarket",
                "polymarket (usa)",
                "rebet",
                "robinhood",
                "fliff",
                "underdog predictions",
                # Secondarios OddsJam mapped
                "midnite",
                "four winds",
                "desert diamond",
                "dogg house",
                "jackpot.bet",
                "prime sports",
                "proline",
                "thrillzz",
                "sportzino",
                # Books internacionales via The Odds API — completar cobertura
                "onexbet",
                "1xbet",
                "matchbook",
                "leovegas_se",
                "leovegas_mx",
                "winamax_fr",
                "winamax_de",
                "unibet_fr",
                "unibet_se",
                "tipico_de",
                "sport888",
                "pmu_fr",
                "betanysports",
                "betfair_ex_eu",
                "betfair_ex_uk",
                "betfair_ex_au",
                "lowvig_ag",
                "codere_it",
                "snai",
                "eurobet",
                "betclic_fr",
                "parionssport_fr",
                "zebet_fr",
                "france_pari",
                "betcris_mx",
                "marathonbet",
                "betfred",
                "paddypower",
                "skybet",
                "boylesports",
                "coral",
                "ladbrokes_uk",
                "betway_uk",
                "sportingbet",
                "betsafe",
                "nbet",
                "virginbet",
                "betvictor",
                "sisal",
            }
        )
    )


async def _was_alerted_recently(
    event_id: int,
    market: str,
    outcome: str,
    bookmaker: str,
    window_minutes: int,
) -> bool:
    """Dedupe: evita re-emitir el mismo pick.

    Bug histórico: el dedupe consultaba `decision_log` que puede fallar al
    persistir silenciosamente (try/except con log warning). Eso dejaba al
    detector emitiendo el mismo pick 2-4× en runs consecutivas (ej: match 8
    NBA away emitido 4× en 13 min). Fix: consultar `pick_alerts` directamente,
    que es la fuente persistente real, y bloquear si EXISTE pick para
    (match, market, outcome) sin liquidar — independiente del bookmaker (no
    queremos triple exposición al mismo lado del mismo evento).
    """
    since = datetime.now(tz=UTC) - timedelta(minutes=window_minutes)
    async with session_scope() as session:
        # Bloqueo fuerte: si hay pick PENDING al mismo (match, market, outcome),
        # no re-emitir aunque venga de otro book — evita triple exposición.
        result = await session.execute(
            text(
                """
                SELECT 1 FROM pick_alerts
                WHERE match_id = :event_id
                  AND market = :market
                  AND outcome = :outcome
                  AND (outcome_result IS NULL OR outcome_result = 'pending')
                LIMIT 1
                """
            ),
            {"event_id": event_id, "market": market, "outcome": outcome},
        )
        if result.first() is not None:
            return True
        # Bloqueo blando: pick recientemente emitido (resuelto o no) en el
        # mismo (match, market, outcome, bookmaker) dentro de window_minutes.
        result = await session.execute(
            text(
                """
                SELECT 1 FROM pick_alerts
                WHERE match_id = :event_id
                  AND market = :market
                  AND outcome = :outcome
                  AND bookmaker = :bookmaker
                  AND placed_at >= :since
                LIMIT 1
                """
            ),
            {
                "event_id": event_id,
                "market": market,
                "outcome": outcome,
                "bookmaker": bookmaker,
                "since": since,
            },
        )
        return result.first() is not None


async def persist_decision(bet: ValueBet, *, correlation_id: str | None = None) -> None:
    """Grava decisión (bet o skip) en decision_log. Nunca re-raises."""
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
                      (:event_id, :market, :outcome, :line,
                       :p_model, :p_lower, :p_upper,
                       :fair_odds, :best_offer, :bookmaker,
                       :edge, :decision, :skip_reason, :cid)
                    """
                ),
                {
                    "event_id": bet.event_id,
                    "market": bet.market,
                    "outcome": bet.outcome,
                    "line": bet.line,
                    "p_model": bet.p_model,
                    "p_lower": bet.p_lower,
                    "p_upper": bet.p_upper,
                    "fair_odds": 1.0 / bet.p_blended if bet.p_blended > 0 else None,
                    "best_offer": bet.odds,
                    "bookmaker": bet.bookmaker,
                    "edge": bet.edge,
                    "decision": "bet" if bet.is_bet else "skip",
                    "skip_reason": bet.skip_reason,
                    "cid": correlation_id,
                },
            )
    except Exception as exc:
        logger.warning("detector.persist_decision.failed", error=str(exc))


def build_quotes_list(event: EventOdds, outcome_idx: int) -> list[BookmakerQuote]:
    """Transforma dict quotes_by_bookmaker en lista BookmakerQuote para el outcome_idx."""
    quotes: list[BookmakerQuote] = []
    line = event.lines[outcome_idx] if event.lines else None
    for bm, odds_list in event.quotes_by_bookmaker.items():
        if outcome_idx >= len(odds_list):
            continue
        odds = odds_list[outcome_idx]
        if odds is None or odds <= 1.0:
            continue
        quotes.append(BookmakerQuote(bookmaker=bm, odds=float(odds), line=line))
    return quotes


async def detect_value_bets_for_event(
    event: EventOdds,
    *,
    model_probs: dict[str, float] | None = None,
    conformal_intervals: dict[str, tuple[float, float]] | None = None,
    cfg: DetectorConfig | None = None,
    correlation_id: str | None = None,
    model_metrics: dict[str, Any] | None = None,
) -> list[ValueBet]:
    """Ejecuta detección sobre un evento/mercado específico.

    Args:
        event: EventOdds con quotes por bookmaker.
        model_probs: {outcome: p_model} del modelo ML calibrado.
        conformal_intervals: {outcome: (p_low, p_upper)} del conformal.
        cfg: DetectorConfig con thresholds.
        model_metrics: dict con métricas del modelo production
            (`holdout_brier`, `n_train`, `holdout_log_loss`, etc.). Usado por
            F2 (blend adaptive) y F7 (min sample guard). Si None, defaults
            seguros (brier=0.25 ≈ random; n_train=999999 ≈ no filtra).
    """
    cfg = cfg or DetectorConfig()

    # Sprint 14 — focus mode: sólo emit en sports habilitados (MLB/NBA/soccer core).
    # Permite desactivar NHL/NFL/tennis/boxing/mma sin borrar código.
    if event.sport_code:
        from apuestas.betting.sport_focus import is_emit_enabled

        if not is_emit_enabled(event.sport_code):
            logger.info(
                "detector.skip_sport_disabled",
                sport=event.sport_code,
                event_id=event.event_external_id,
            )
            return []

    # Gap 9 — off-season awareness: si el deporte está fuera de ventana
    # activa, no gastamos API calls ni LLM. Fail-open si la config falla.
    if event.sport_code:
        from apuestas.betting.season import is_sport_active

        if not is_sport_active(event.sport_code):
            logger.info(
                "detector.skip_off_season",
                sport=event.sport_code,
                event_id=event.event_external_id,
            )
            return []

    # F7 — sample size guard. Si el modelo production se entrenó con <N matches,
    # su calibración isotonic es ruidosa y los EVs reportados no son confiables.
    if model_metrics is not None and model_probs is not None:
        n_train = model_metrics.get("n_train") or model_metrics.get("n_train_samples")
        if n_train is not None:
            try:
                if int(n_train) < cfg.min_train_samples:
                    logger.info(
                        "detector.skip_low_train_samples",
                        sport=event.sport_code,
                        market=event.market,
                        n_train=int(n_train),
                        min_required=cfg.min_train_samples,
                    )
                    return []
            except (TypeError, ValueError):
                pass

    # F2 — blend_weight adaptive según Brier holdout del modelo. Brier <0.20
    # = excelente; >0.24 = ruidoso. Multiplica el weight base [0.5x, 1.5x].
    effective_blend_base = cfg.blend_weight_model
    if cfg.adaptive_blend_enabled and model_metrics is not None and model_probs is not None:
        brier = model_metrics.get("holdout_brier") or model_metrics.get("brier")
        try:
            if brier is not None:
                b = float(brier)
                # Linear scale: brier 0.20 → 1.5x, 0.22 → 1.0x, 0.25 → 0.5x
                # quality = clip((0.25 - brier) / 0.05, 0.0, 1.0)
                quality = max(0.0, min(1.0, (0.25 - b) / 0.05))
                multiplier = 0.5 + quality * 1.0  # 0.5x .. 1.5x
                effective_blend_base = cfg.blend_weight_model * multiplier
                logger.debug(
                    "detector.adaptive_blend",
                    sport=event.sport_code,
                    holdout_brier=round(b, 4),
                    base_weight=cfg.blend_weight_model,
                    effective_weight=round(effective_blend_base, 4),
                )
        except (TypeError, ValueError):
            pass

    method = "shin" if cfg.use_shin_devig else "power"

    # Fase 1.2 — Low-hold filter: calcula vig total del mercado consolidado
    # (best odds disponible por outcome) y rechaza si excede max_hold.
    # El best-odds-per-outcome es el hold efectivo que el apostador paga.
    from apuestas.betting.devig import overround as _overround

    best_odds_per_outcome: list[float] = []
    for i in range(len(event.outcomes)):
        odds_i = [
            q[i] for q in event.quotes_by_bookmaker.values() if q[i] is not None and q[i] > 1.0
        ]
        if odds_i:
            best_odds_per_outcome.append(max(odds_i))
    n_books = len(event.quotes_by_bookmaker)
    threshold = cfg.max_hold if n_books > 1 else cfg.max_hold_single_book
    if len(best_odds_per_outcome) >= 2:
        try:
            total_hold = _overround(best_odds_per_outcome)
        except ValueError:
            total_hold = None
        if total_hold is not None and total_hold > threshold:
            logger.info(
                "detector.rejected_high_hold",
                event_id=event.event_external_id,
                market=event.market,
                total_hold=float(total_hold),
                threshold=float(threshold),
                n_books=n_books,
            )
            return []

    # Pinnacle consensus (si disponible)
    pinnacle_fair = consensus_fair_probs(
        event.quotes_by_bookmaker,
        method=method,
    )

    value_bets: list[ValueBet] = []

    for i, outcome in enumerate(event.outcomes):
        p_pinn = float(pinnacle_fair[i]) if pinnacle_fair is not None else None
        p_model = (model_probs or {}).get(outcome)

        # Blend — sin referencia sharp no hay señal calibrada
        if p_model is not None and p_pinn is not None:
            # Shrinkage adaptativo anti-EV-inflado:
            # si |p_model - p_pinn| > 0.05 (5pp de divergencia del consenso sharp),
            # aumentamos el peso de Pinnacle en el blend. Razonamiento: Pinnacle
            # es el consensus market (sharpiest book del mundo) — una divergencia
            # >5pp del modelo es casi siempre ruido de features incompletas,
            # no edge real. Escala lineal: 5pp → weight_model base, 15pp → 0.5×
            # weight_model, 25pp → 0.1× weight_model.
            # F8 — adicional: cuando |Δ|>8pp aplicamos shrinkage cuadrático
            # extra (factor² más agresivo) — el modelo está casi seguro errado.
            # F2 — base weight ya viene ajustado por holdout_brier del modelo
            # (effective_blend_base). Modelos malos (Brier 0.25) → 0.5× weight,
            # modelos buenos (Brier 0.20) → 1.5× weight.
            delta = abs(p_model - p_pinn)
            if delta > 0.08:
                # F8 shrinkage cuadrático: |Δ|=0.08 → 1.0; 0.15 → 0.4; 0.25 → 0.04
                shrink_factor = max(0.04, (1.0 - (delta - 0.05) * 4.5) ** 2)
                effective_weight = effective_blend_base * shrink_factor
            elif delta > 0.05:
                shrink_factor = max(0.1, 1.0 - (delta - 0.05) * 4.5)
                effective_weight = effective_blend_base * shrink_factor
            else:
                effective_weight = effective_blend_base
            p_blended = blend_probabilities(p_model, p_pinn, weight_model=effective_weight)
        elif p_model is None:
            # Sin modelo propio → no emitimos picks basados solo en Pinnacle
            # de-vigged. Post-pivote 2026-04-23: el fallback `p_blended = p_pinn`
            # producía picks con edge ~ ruido (apostar a fair-value Pinnacle
            # desde un soft-book donde la vig del soft es lo único que pagas).
            # Este era el bug de "MLB picks sin modelo" observado el 22/23 abr.
            logger.info(
                "detector.skip_no_model",
                sport=event.sport_code,
                market=event.market,
                event_id=event.event_external_id,
                has_pinnacle=p_pinn is not None,
            )
            await persist_decision(
                ValueBet(
                    event_id=event.event_id,
                    event_external_id=event.event_external_id,
                    market=event.market,
                    outcome=outcome,
                    line=event.lines[i] if event.lines else None,
                    bookmaker="",
                    odds=0.0,
                    p_model=None,
                    p_pinnacle_fair=p_pinn,
                    p_blended=p_pinn or 0.0,
                    p_lower=None,
                    p_upper=None,
                    implied_prob=0.0,
                    edge=0.0,
                    ev=0.0,
                    sport_code=event.sport_code,
                    league_id=getattr(event, "league_id", None),
                    start_time=event.start_time,
                    skip_reason="no_model_for_market",
                    flags=["no_model"],
                ),
                correlation_id=correlation_id,
            )
            continue
        elif p_pinn is None:
            # Sin odds Pinnacle el modelo no tiene calibración sharp → skip.
            # Evita EVs inflados de 8-15% que son artefactos de modelo desconectado.
            logger.info(
                "detector.skip_no_pinnacle",
                sport=event.sport_code,
                market=event.market,
                event_id=event.event_external_id,
            )
            await persist_decision(
                ValueBet(
                    event_id=event.event_id,
                    event_external_id=event.event_external_id,
                    market=event.market,
                    outcome=outcome,
                    line=event.lines[i] if event.lines else None,
                    bookmaker="",
                    odds=0.0,
                    p_model=p_model,
                    p_pinnacle_fair=None,
                    p_blended=0.0,
                    p_lower=None,
                    p_upper=None,
                    implied_prob=0.0,
                    edge=0.0,
                    ev=0.0,
                    sport_code=event.sport_code,
                    league_id=getattr(event, "league_id", None),
                    start_time=event.start_time,
                    skip_reason="no_pinnacle_reference",
                    flags=["no_sharp_anchor"],
                ),
                correlation_id=correlation_id,
            )
            continue
        else:  # pragma: no cover — defensivo
            continue

        p_low = p_up = None
        if conformal_intervals and outcome in conformal_intervals:
            p_low, p_up = conformal_intervals[outcome]

        quotes = build_quotes_list(event, i)
        if not quotes:
            continue

        offer = line_shopping(
            quotes,
            p_fair=p_blended,
            exclude_sharp=True,
            allowed_books=cfg.soft_books_allowed,
            sport=event.sport_code,
            stage=getattr(event, "stage", None),
            market=event.market,
            league_id=getattr(event, "league_id", None),
            league=getattr(event, "league", None),
        )

        vb = ValueBet(
            event_id=event.event_id,
            event_external_id=event.event_external_id,
            market=event.market,
            outcome=outcome,
            line=event.lines[i] if event.lines else None,
            bookmaker=offer.bookmaker if offer else "",
            odds=offer.odds if offer else 0.0,
            p_model=p_model,
            p_pinnacle_fair=p_pinn,
            p_blended=p_blended,
            p_lower=p_low,
            p_upper=p_up,
            implied_prob=implied_probability(offer.odds) if offer and offer.odds else 0.0,
            edge=offer.edge if offer else 0.0,
            ev=offer.ev if offer else 0.0,
            sport_code=event.sport_code,
            league_id=event.league_id,
            start_time=event.start_time,
        )

        # Razones de skip en orden de precedencia
        _stage = getattr(event, "stage", None)
        _is_playoff = _stage is not None and str(_stage).lower() in (
            "playoff",
            "postseason",
            "finals",
        )
        _is_soccer_threeway = (
            event.sport_code == "soccer"
            and len(event.outcomes) >= 3
            and "draw" in [str(o).lower() for o in event.outcomes]
        )
        # Draw guard: usar el MAX entre p_model[draw] y la prob fair del mercado
        # consensus sharp. Si el modelo es 2-way (no estima draw), p_model[draw]=0
        # y el guard nunca disparaba — bug detectado en Milan-Juve 04-26 donde el
        # market draw_implied=31% pero el guard quedó en 0% e omitió el bloqueo.
        _p_draw = 0.0
        if _is_soccer_threeway and model_probs:
            _p_draw = float(model_probs.get("draw", 0.0))
        if _is_soccer_threeway and pinnacle_fair is not None:
            try:
                draw_idx = next(i for i, o in enumerate(event.outcomes) if str(o).lower() == "draw")
                _p_draw = max(_p_draw, float(pinnacle_fair[draw_idx]))
            except (StopIteration, IndexError, TypeError, ValueError):
                pass

        # Late line: minutos desde ahora hasta kickoff. Si por algún motivo el
        # cálculo falla (start_time naive o None), tratamos como "timing desconocido"
        # y bloqueamos por seguridad cuando hard_cutoff > 0 — bug detectado picks
        # 127 (7.6 min) y 207 (15.9 min) que pasaron por debajo del cutoff de 20 min.
        _late_line_flag = False
        _mins_to_kick: float | None = None
        _timing_unknown = False
        if event.start_time is not None:
            try:
                from datetime import UTC as _UTC
                from datetime import datetime as _dt

                st = event.start_time
                # Si start_time es naive, asumir UTC en lugar de fallar silenciosamente
                if st.tzinfo is None:
                    st = st.replace(tzinfo=_UTC)
                _mins_to_kick = (st - _dt.now(tz=_UTC)).total_seconds() / 60.0
                _late_line_flag = 0 < _mins_to_kick <= cfg.late_line_minutes
            except (TypeError, ValueError, AttributeError) as exc:
                logger.warning(
                    "detector.start_time_parse_fail",
                    event_id=event.event_external_id,
                    error=str(exc)[:100],
                )
                _timing_unknown = True
        else:
            _timing_unknown = True

        _conf_width = (p_up - p_low) if (p_up is not None and p_low is not None) else None

        # F3 — hold del soft book TARGET (donde se ejecuta la apuesta).
        # Si el book tiene 2+ outcomes para este market y su overround es
        # > max_hold_target_book, el edge del pick es probablemente artefacto
        # del margen del book, no value real.
        _target_book_hold: float | None = None
        if (
            offer is not None
            and cfg.max_hold_target_book > 0
            and offer.bookmaker in event.quotes_by_bookmaker
        ):
            book_quotes = [
                q for q in event.quotes_by_bookmaker[offer.bookmaker] if q is not None and q > 1.0
            ]
            if len(book_quotes) >= 2:
                try:
                    _target_book_hold = _overround(book_quotes)
                except ValueError:
                    _target_book_hold = None

        if offer is None:
            vb.skip_reason = "no_qualifying_offer"
        elif (
            event.sport_code
            in ("soccer", "epl", "laliga", "bundesliga", "seriea", "ligue1", "liga_mx")
            and model_probs is not None
            and len(model_probs) >= 3
            and all(abs(p - 0.333) < 0.05 for p in model_probs.values())
        ):
            # Guard prior-degenerado: si el modelo soccer 3-way devuelve probs
            # casi uniformes (33%/33%/33% ± 5pp), es señal de que está prediciendo
            # la prior promedio del training set en lugar de hacer una predicción
            # real para los teams. Esto pasa cuando teams no están en posteriors
            # (ej. Palmeiras/Santos cargados al modelo `soccer_liga_mx` por
            # identity rota). Sin este guard, el detector emite picks 100% basados
            # en arbitraje Pinnacle-vs-soft sin aporte del modelo propio.
            vb.skip_reason = "model_prior_degenerated"
            vb.flags.append("no_real_model_signal")
        elif _target_book_hold is not None and _target_book_hold > cfg.max_hold_target_book:
            vb.skip_reason = f"target_book_hold_{int(_target_book_hold * 100)}pct"
            vb.flags.append("high_target_hold")
        elif vb.ev > 0.20:
            # Sanity check EV-extreme: >20% en sharp markets es 99% data corrupta
            # o modelo descalibrado (features incompletas, odds stale, mismatch
            # 2-way vs 3-way). Threshold subido 0.10→0.20 el 2026-04-27 tras
            # detectar que rechazaba picks legítimos con edge real (Espanyol-Levante
            # home @2.14 con EV 13-17% es razonable para un favorito grande).
            # Los pros reales ven 3-15% EV típicamente; >20% es red flag.
            vb.skip_reason = f"ev_unrealistic_{int(vb.ev * 100)}pct"
            vb.flags.append("data_quality_guard")
        elif event.sport_code in cfg.block_playoff_sports and _is_playoff:
            # Mejora 4 — playoff sin modelo dedicado. #104 MIN-DEN perdió
            # con p_model=60% (era Game 3 playoff; regular-season model no
            # transfiere). Bloquear hasta tener train_{sport}_playoffs.
            vb.skip_reason = "playoff_model_missing"
            vb.flags.append("playoff_guard")
        elif (
            outcome.lower() in ("home", "away")
            and _is_soccer_threeway
            and _p_draw
            >= cfg.soccer_max_draw_prob_by_league.get(
                getattr(event, "league_id", None) or -1,
                cfg.soccer_max_draw_prob,
            )
        ):
            # Mejora 2 — draw guard. #105 Go Ahead 0-0 AZ: el bot apostó
            # home pero empate 0-0 matea el pick 2-way. Si el mercado tiene
            # 3 outcomes y el modelo asigna draw alto → skip home/away.
            vb.skip_reason = f"high_draw_risk_{int(_p_draw * 100)}pct"
            vb.flags.append("draw_guard")
        elif _conf_width is not None and _conf_width > cfg.conformal_max_width_by_sport.get(
            event.sport_code, cfg.conformal_max_width
        ):
            # Mejora 3 — conformal width filter. Intervalo muy ancho =
            # modelo poco confiable para este pick; evita picks tipo
            # #104 donde el modelo pretendía certeza que no tenía.
            # Sprint 10 Fase 1: umbral por deporte (mlb 0.10, soccer 0.08).
            vb.skip_reason = f"conformal_width_{int(_conf_width * 100)}pp"
            vb.flags.append("conformal_width_filter")
        elif vb.ev > 0.08:
            # EV 8-10%: alto pero no absurdo. Permitir pero flag para revisión.
            vb.flags.append("high_ev_review")
        elif cfg.only_favorites and p_blended < cfg.favorite_min_prob:
            # Usuario prefiere solo picks a favorito (p_fair > 55% default).
            vb.skip_reason = "not_favorite"
            vb.flags.append("only_favorites_mode")
        elif p_low is not None and p_low <= vb.implied_prob + cfg.conformal_margin:
            vb.skip_reason = "conformal_width"
            vb.flags.append("conformal_filter")
        elif await _was_alerted_recently(
            event.event_id,
            event.market,
            outcome,
            offer.bookmaker,
            cfg.dedupe_window_minutes,
        ):
            vb.skip_reason = "dedupe_recent_alert"
            vb.flags.append("dedupe")
        elif (
            cfg.hard_cutoff_minutes > 0
            and _mins_to_kick is not None
            and 0 < _mins_to_kick <= cfg.hard_cutoff_minutes
        ):
            vb.skip_reason = "too_close_to_kickoff"
            vb.flags.append("hard_cutoff")
        elif cfg.hard_cutoff_minutes > 0 and _timing_unknown:
            # Si timing no se pudo calcular y hay cutoff configurado, ser
            # conservador y bloquear — evita el bypass detectado picks 127/207.
            vb.skip_reason = "timing_unknown"
            vb.flags.append("hard_cutoff")

        # Mejora 4 — late_line siempre anexa soft_tag aunque no skipee.
        # Informa a classify_confidence (en el mensaje Telegram) que el
        # pick es de línea tardía.
        if _late_line_flag and "late_line" not in vb.flags:
            vb.flags.append("late_line")

        await persist_decision(vb, correlation_id=correlation_id)
        value_bets.append(vb)

    return value_bets


async def detect_for_events(
    events: Iterable[EventOdds],
    *,
    model_probs_per_event: dict[int, dict[str, float]] | None = None,
    conformal_per_event: dict[int, dict[str, tuple[float, float]]] | None = None,
    cfg: DetectorConfig | None = None,
) -> list[ValueBet]:
    """Wrapper batch para múltiples eventos."""
    model_probs_per_event = model_probs_per_event or {}
    conformal_per_event = conformal_per_event or {}
    results: list[ValueBet] = []
    for ev in events:
        results.extend(
            await detect_value_bets_for_event(
                ev,
                model_probs=model_probs_per_event.get(ev.event_id),
                conformal_intervals=conformal_per_event.get(ev.event_id),
                cfg=cfg,
            )
        )
    return results


def compute_offered_vs_fair_spread(event: EventOdds) -> dict[str, float]:
    """Diagnóstico: compara odds ofrecidas por book soft vs fair Pinnacle.

    Útil para detectar soft lines (§17.4). Retorna por bookmaker la media del
    (odds_soft − odds_fair) / odds_fair.
    """
    pinnacle_fair = consensus_fair_probs(event.quotes_by_bookmaker)
    if pinnacle_fair is None:
        return {}
    fair_odds = 1.0 / pinnacle_fair
    spread: dict[str, float] = {}
    for bm, odds_list in event.quotes_by_bookmaker.items():
        if bm in {"pinnacle", "circa", "betfair", "bookmaker"}:
            continue
        arr = np.asarray(odds_list, dtype=np.float64)
        if len(arr) != len(fair_odds):
            continue
        mask = arr > 1.0
        if not mask.any():
            continue
        rel = (arr[mask] - fair_odds[mask]) / fair_odds[mask]
        spread[bm] = float(np.mean(rel))
    return spread


def hash_event_signature(event: EventOdds) -> str:
    """Hash deterministico de un evento+odds para tracking y dedupe."""
    payload = f"{event.event_external_id}:{event.market}:{sorted(event.quotes_by_bookmaker)}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ═══════════════════════ Player props (Gap #5) ══════════════════════════


@dataclass(slots=True)
class PropValueBet:
    """Un pick de prop (over/under) con EV > threshold."""

    player_id: int
    player_name: str
    match_id: int
    prop_code: str
    line: float
    side: str  # "over" | "under"
    p_model: float
    odds: float
    bookmaker: str
    ev: float
    kelly_fraction: float


async def detect_value_props(
    *,
    match_id: int,
    player_id: int,
    player_name: str,
    prop_code: str,
    historical_samples: list[float],
    min_ev: float = 0.03,
    kelly_fraction_cap: float = 0.25,
) -> list[PropValueBet]:
    """Dado historial del jugador + prop_code, entrena distribución y evalúa
    líneas existentes en `player_prop_lines` buscando EV > `min_ev`.

    MVP: soporta props con distribución Poisson/NegBin/Gamma (count/continuous).
    Retorna lista de `PropValueBet` (over/under) por bookmaker.
    """
    import numpy as np
    from sqlalchemy import text as _text

    from apuestas.db import session_scope as _session_scope
    from apuestas.ml.train_props import train_prop_parametric

    if len(historical_samples) < 10:
        return []

    arr = np.asarray(historical_samples, dtype=np.float64)
    # Entrena distribución sobre samples completos (MVP)
    result = train_prop_parametric(
        prop_code=prop_code,
        historical_samples=arr,
    )
    # Reconstruir la distribución para p_over/p_under
    from apuestas.ml.props_distributions import fit_gamma, fit_neg_binomial, fit_poisson
    from apuestas.schemas.props import PropDistribution, get_prop

    prop_def = get_prop(prop_code)
    if prop_def.distribution == PropDistribution.POISSON:
        dist = fit_poisson(arr)
    elif prop_def.distribution == PropDistribution.NEG_BINOMIAL:
        dist = fit_neg_binomial(arr)
    elif prop_def.distribution == PropDistribution.GAMMA:
        dist = fit_gamma(arr)
    else:
        return []

    async with _session_scope() as s:
        rows = (
            await s.execute(
                _text(
                    """
                    SELECT line, over_odds, under_odds, bookmaker
                    FROM player_prop_lines
                    WHERE match_id = :mid AND player_id = :pid AND prop_type = :pc
                    """
                ),
                {"mid": match_id, "pid": player_id, "pc": prop_code},
            )
        ).all()

    picks: list[PropValueBet] = []
    for row in rows:
        line = float(row.line)
        for side, odds_raw in (("over", row.over_odds), ("under", row.under_odds)):
            if odds_raw is None:
                continue
            odds = float(odds_raw)
            if odds <= 1.0:
                continue
            p = dist.p_over(line) if side == "over" else 1.0 - dist.p_over(line)
            ev = p * (odds - 1) - (1 - p)
            if ev < min_ev:
                continue
            b = odds - 1
            kelly_full = max(0.0, (p * (b + 1) - 1) / b) if b > 1e-8 else 0.0
            kelly = min(kelly_full * 0.25, kelly_fraction_cap)
            picks.append(
                PropValueBet(
                    player_id=player_id,
                    player_name=player_name,
                    match_id=match_id,
                    prop_code=prop_code,
                    line=line,
                    side=side,
                    p_model=float(p),
                    odds=odds,
                    bookmaker=str(row.bookmaker),
                    ev=float(ev),
                    kelly_fraction=float(kelly),
                )
            )
    logger.info(
        "detector.props_done",
        match_id=match_id,
        player=player_name,
        prop=prop_code,
        picks=len(picks),
        brier=result.brier_holdout,
    )
    return picks


async def detect_all_player_props_for_match(
    *,
    match_id: int,
    min_ev: float = 0.03,
    kelly_fraction_cap: float = 0.25,
    min_historical_samples: int = 15,
) -> list[PropValueBet]:
    """Scan todos los jugadores con player_prop_lines en este match y devuelve
    value picks para cada (player, prop_code).

    Feature principal: recupera historial del jugador desde `player_game_logs`
    para entrenar distribución antes de evaluar lines.
    """
    from sqlalchemy import text as _text

    from apuestas.db import session_scope as _session_scope

    async with _session_scope() as s:
        rows = (
            await s.execute(
                _text(
                    """
                    SELECT DISTINCT ppl.player_id, ppl.prop_type,
                                    p.full_name, p.sport_code
                    FROM player_prop_lines ppl
                    JOIN players p ON p.id = ppl.player_id
                    WHERE ppl.match_id = :mid
                    """
                ),
                {"mid": match_id},
            )
        ).all()
        if not rows:
            return []

        all_picks: list[PropValueBet] = []
        for row in rows:
            player_id = int(row.player_id)
            prop_code = str(row.prop_type)
            player_name = str(row.full_name)
            sport_code = str(row.sport_code)

            # stats JSONB → extract key igual al prop_type (ej. "points")
            # Normalizar prop_type: oddsapi manda "player_points" → key real es "points"
            stat_key = (
                prop_code.replace("player_", "").replace("pitcher_", "").replace("batter_", "")
            )
            samples_r = await s.execute(
                _text(
                    """
                    SELECT (pgl.stats ->> :sk)::numeric AS v
                    FROM player_game_logs pgl
                    JOIN matches m ON m.id = pgl.match_id
                    WHERE pgl.player_id = :pid
                      AND pgl.sport_code = :sc
                      AND pgl.stats ? :sk
                    ORDER BY m.start_time DESC
                    LIMIT 100
                    """
                ),
                {"pid": player_id, "sc": sport_code, "sk": stat_key},
            )
            samples = [float(r.v) for r in samples_r.all() if r.v is not None]
            if len(samples) < min_historical_samples:
                logger.debug(
                    "detector.props_skip_low_samples",
                    player=player_name,
                    prop=prop_code,
                    n=len(samples),
                )
                continue
            try:
                picks = await detect_value_props(
                    match_id=match_id,
                    player_id=player_id,
                    player_name=player_name,
                    prop_code=prop_code,
                    historical_samples=samples,
                    min_ev=min_ev,
                    kelly_fraction_cap=kelly_fraction_cap,
                )
                all_picks.extend(picks)
            except Exception as exc:
                logger.debug(
                    "detector.props_player_fail",
                    player=player_name,
                    prop=prop_code,
                    error=str(exc)[:100],
                )
        return all_picks
