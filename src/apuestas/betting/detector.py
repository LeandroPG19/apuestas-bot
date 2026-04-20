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

import numpy as np
from sqlalchemy import text

from apuestas.betting.devig import consensus_fair_probs
from apuestas.betting.ev import (
    BookmakerQuote,
    blend_probabilities,
    implied_probability,
    line_shopping,
)
from apuestas.config import get_settings
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
    sport_code: str | None = None


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
    kelly_fraction_pct: float
    stake_units: float
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
    soft_books_allowed: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "caliente",
                "strendus",
                "codere",
                "betano",
                "betway",
                "betsson",
                "bwin",
                "draftkings",
                "fanduel",
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
    """Dedupe: evita re-alertar un mismo pick si odds no cambió >1 tick."""
    since = datetime.now(tz=UTC) - timedelta(minutes=window_minutes)
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT 1 FROM decision_log
                WHERE event_id = :event_id
                  AND market = :market
                  AND outcome = :outcome
                  AND best_bookmaker = :bookmaker
                  AND decision = 'bet'
                  AND created_at >= :since
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
    bankroll: float | None = None,
    cfg: DetectorConfig | None = None,
    correlation_id: str | None = None,
) -> list[ValueBet]:
    """Ejecuta detección sobre un evento/mercado específico.

    Args:
        event: EventOdds con quotes por bookmaker.
        model_probs: {outcome: p_model} del modelo ML calibrado.
        conformal_intervals: {outcome: (p_low, p_upper)} del conformal.
        bankroll: usa settings default si None.
        cfg: DetectorConfig con thresholds.
    """
    cfg = cfg or DetectorConfig()
    settings = get_settings()
    if bankroll is None:
        bankroll = settings.betting.default_bankroll_units

    method = "shin" if cfg.use_shin_devig else "power"

    # Pinnacle consensus (si disponible)
    pinnacle_fair = consensus_fair_probs(
        event.quotes_by_bookmaker,
        method=method,
    )

    value_bets: list[ValueBet] = []

    for i, outcome in enumerate(event.outcomes):
        p_pinn = float(pinnacle_fair[i]) if pinnacle_fair is not None else None
        p_model = (model_probs or {}).get(outcome)

        # Blend
        if p_model is not None and p_pinn is not None:
            p_blended = blend_probabilities(p_model, p_pinn, weight_model=cfg.blend_weight_model)
        elif p_pinn is not None:
            p_blended = p_pinn
        elif p_model is not None:
            p_blended = p_model
        else:
            continue  # Sin señal

        p_low = p_up = None
        if conformal_intervals and outcome in conformal_intervals:
            p_low, p_up = conformal_intervals[outcome]

        quotes = build_quotes_list(event, i)
        if not quotes:
            continue

        offer = line_shopping(
            quotes,
            p_fair=p_blended,
            bankroll=bankroll,
            exclude_sharp=True,
            allowed_books=cfg.soft_books_allowed,
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
            implied_prob=(offer.odds and implied_probability(offer.odds)) or 0.0,
            edge=offer.edge if offer else 0.0,
            ev=offer.ev if offer else 0.0,
            kelly_fraction_pct=offer.kelly_fraction_pct if offer else 0.0,
            stake_units=offer.stake_units if offer else 0.0,
            sport_code=event.sport_code,
            league_id=event.league_id,
            start_time=event.start_time,
        )

        # Razones de skip en orden de precedencia
        if offer is None:
            vb.skip_reason = "no_qualifying_offer"
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

        await persist_decision(vb, correlation_id=correlation_id)
        value_bets.append(vb)

    return value_bets


async def detect_for_events(
    events: Iterable[EventOdds],
    *,
    model_probs_per_event: dict[int, dict[str, float]] | None = None,
    conformal_per_event: dict[int, dict[str, tuple[float, float]]] | None = None,
    bankroll: float | None = None,
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
                bankroll=bankroll,
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
