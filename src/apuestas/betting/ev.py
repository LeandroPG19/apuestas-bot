"""Expected Value + Kelly wrappers + line shopping.

Blueprint §7: EV = P_modelo × cuota − 1. Kelly fraccional = ¼ Kelly con cap 5%.
Line shopping: mejor precio disponible entre soft books, excluyendo Pinnacle
(el fair value, NO donde apuestas).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from apuestas.config import get_settings
from apuestas.risk.kelly import kelly_fraction as _kelly_fraction

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
    kelly_fraction_pct: float
    stake_units: float


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


def kelly_stake(
    p: float,
    odds: float,
    *,
    bankroll: float,
    fraction: float | None = None,
    cap_pct: float | None = None,
) -> tuple[float, float]:
    """Retorna (stake_absoluto, kelly_fraction_pct). Lee defaults de settings."""
    s = get_settings().betting
    f = fraction if fraction is not None else s.kelly_fraction
    cap = cap_pct if cap_pct is not None else s.kelly_max_stake_pct
    k_pct = _kelly_fraction(p, odds, fraction=f, cap=cap)
    return k_pct * bankroll, k_pct


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
    bankroll: float,
) -> BestOffer | None:
    """Convierte cuota + p en `BestOffer` si pasa threshold EV y rango odds."""
    s = get_settings().betting
    odds = quote.odds
    if odds < s.min_odds or odds > s.max_odds:
        return None
    ev = compute_ev(p_fair, odds)
    if ev < s.ev_threshold:
        return None
    edge_val = edge(p_fair, odds)
    stake_abs, kelly_pct = kelly_stake(p_fair, odds, bankroll=bankroll)
    if kelly_pct <= 0:
        return None
    return BestOffer(
        bookmaker=quote.bookmaker,
        odds=odds,
        line=quote.line,
        edge=edge_val,
        ev=ev,
        kelly_fraction_pct=kelly_pct,
        stake_units=stake_abs,
    )


def line_shopping(
    quotes: list[BookmakerQuote],
    *,
    p_fair: float,
    bankroll: float,
    exclude_sharp: bool = True,
    allowed_books: frozenset[str] | None = None,
) -> BestOffer | None:
    """Encuentra mejor precio Y evalúa EV + Kelly en una pasada."""
    best = find_best_price(quotes, exclude_sharp=exclude_sharp, allowed_books=allowed_books)
    if best is None:
        return None
    return evaluate_offer(p_fair=p_fair, quote=best, bankroll=bankroll)


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


def compute_clv(
    *,
    odds_placed: float,
    closing_odds: float,
) -> float:
    """Closing Line Value. >0 = mejor precio que el cierre (sharp territory)."""
    if closing_odds <= 1.0:
        return 0.0
    return float(odds_placed / closing_odds - 1.0)
