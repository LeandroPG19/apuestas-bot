"""Expected Value + line shopping.

Blueprint §7: EV = P_modelo × cuota − 1. Line shopping: mejor precio
disponible entre soft books, excluyendo Pinnacle (el fair value, no donde
se apuesta).

Nota post-pivote 2026-04-23: el módulo ya no calcula Kelly ni stakes; el
bot es puro detector de alertas de valor. Las funciones `kelly_stake` y
`evaluate_offer` con bankroll/stake fueron eliminadas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from apuestas.config import get_settings

SHARP_BOOKS = frozenset({"pinnacle", "circa", "bookmaker", "betfair"})
MEXICAN_BOOKS = frozenset({"caliente", "strendus", "codere", "betway_mx"})


@dataclass(slots=True, frozen=True)
class BookmakerQuote:
    bookmaker: str
    odds: float
    line: float | None = None
    timestamp: str | None = None


@dataclass(slots=True, frozen=True)
class BestOffer:
    bookmaker: str
    odds: float
    line: float | None
    edge: float
    ev: float


def compute_ev(p: float, odds: float) -> float:
    """EV = P × odds - 1. Positivo = value bet."""
    if odds <= 1.0:
        return -1.0
    return float(p * odds - 1.0)


def implied_probability(odds: float) -> float:
    return 1.0 / odds if odds > 1.0 else 0.0


def edge(p: float, odds: float) -> float:
    """Edge = P_modelo - P_implícita (no lo mismo que EV)."""
    return float(p - implied_probability(odds))


def find_best_price(
    quotes: list[BookmakerQuote],
    *,
    exclude_sharp: bool = True,
    allowed_books: frozenset[str] | None = None,
) -> BookmakerQuote | None:
    """Mejor precio disponible — mayor decimal = mejor para el apostador.

    Por defecto excluye Pinnacle/Circa (donde NO apuestas; sirven para fair).
    Si `allowed_books` dado, solo considera esos.
    """
    candidates = quotes
    if exclude_sharp:
        candidates = [q for q in candidates if q.bookmaker not in SHARP_BOOKS]
    if allowed_books is not None:
        candidates = [q for q in candidates if q.bookmaker in allowed_books]
    if not candidates:
        return None
    return max(candidates, key=lambda q: q.odds)


def evaluate_offer(
    *,
    p_fair: float,
    quote: BookmakerQuote,
    sport: str | None = None,
    stage: str | None = None,
    market: str | None = None,
    league_id: int | None = None,
) -> BestOffer | None:
    """Convierte cuota + p en `BestOffer` si pasa threshold EV y rango odds.

    Mejora 1: `sport`/`stage`/`market`/`league_id` activan threshold adaptativo
    (`config/ev_thresholds.yaml`). Sin argumentos usa el threshold global.
    """
    s = get_settings().betting
    odds = quote.odds
    if odds < s.min_odds or odds > s.max_odds:
        return None
    ev = compute_ev(p_fair, odds)

    # Threshold adaptativo con jerarquía: league > market > stage > sport.
    if sport is not None:
        from apuestas.betting.ev_thresholds import ev_threshold_for

        thr = ev_threshold_for(
            sport=sport,
            stage=stage,
            market=market,
            league_id=league_id,
            fallback=float(s.ev_threshold),
        )
    else:
        thr = float(s.ev_threshold)

    if ev < thr:
        return None
    return BestOffer(
        bookmaker=quote.bookmaker,
        odds=odds,
        line=quote.line,
        edge=edge(p_fair, odds),
        ev=ev,
    )


def line_shopping(
    quotes: list[BookmakerQuote],
    *,
    p_fair: float,
    exclude_sharp: bool = True,
    allowed_books: frozenset[str] | None = None,
    sport: str | None = None,
    stage: str | None = None,
    league: str | None = None,
    market: str | None = None,
    league_id: int | None = None,
) -> BestOffer | None:
    """Encuentra mejor precio Y evalúa EV en una pasada.

    Sprint 11 Fase D — si `league` se provee, pondera quotes por
    `book_power_ratings` cache: libros soft con edge bps histórico > 0
    reciben boost marginal para preferirlos sobre libros calibrados.
    """
    import os as _os

    if league is not None and _os.environ.get("APUESTAS_USE_BOOK_POWER", "true").lower() == "true":
        try:
            from apuestas.betting.book_power_ratings import get_cached_edge

            # Re-ordena quotes por edge histórico descendente
            quotes = sorted(
                quotes,
                key=lambda q: -get_cached_edge(q.bookmaker, league),
            )
        except Exception:
            pass
    best = find_best_price(quotes, exclude_sharp=exclude_sharp, allowed_books=allowed_books)
    if best is None:
        return None
    return evaluate_offer(
        p_fair=p_fair,
        quote=best,
        sport=sport,
        stage=stage,
        market=market,
        league_id=league_id,
    )


def blend_probabilities(
    p_model: float,
    p_pinnacle_fair: float,
    *,
    weight_model: float = 0.40,
) -> float:
    """Ensemble blend entre modelo propio y consenso Pinnacle de-vigged.

    Blueprint §6: cuando modelo tiene baja confianza, Pinnacle domina.
    weight_model=0.40 es el default recomendado.
    """
    w = max(0.0, min(1.0, weight_model))
    return float(w * p_model + (1.0 - w) * p_pinnacle_fair)


Direction = Literal["home", "away", "over", "under", "draw"]
