"""Fase 4.3 — Alternative lines / buy points analyzer.

NFL/NBA offer spreads en ±0.5 a ±10. Los pros calculan cuándo el half-point
vale más que los cents extras (key numbers: 3, 7, 10, 14 en NFL).

Ejemplo:
  Lakers -3.5 @ 1.91 (base line)
  Lakers -3.0 @ 1.76 (buy 0.5 point)
  Lakers -4.0 @ 2.05 (sell 0.5 point, riskier)

Dado nuestra p_model, calculamos true_value de cada line alternativa → elegimos
la óptima (may alter base line).

Uso:
    from apuestas.betting.alternative_lines import evaluate_alternative_lines
    best_line = evaluate_alternative_lines(p_model, alt_offers)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

Sport = Literal["nba", "nfl", "nhl", "mlb", "soccer"]

# Key numbers por sport (frequency de márgenes exactos)
# NFL: 3 puntos 15%, 7 puntos 9%, 10 puntos 5%, 14 puntos 3%
# NBA: menos concentrado, 5 puntos ~4% base rate
KEY_NUMBERS: dict[Sport, list[float]] = {
    "nfl": [3.0, 7.0, 10.0, 14.0, 4.0, 6.0],
    "nba": [5.0, 7.0, 3.0],
    "nhl": [1.0, 2.0],
    "mlb": [1.0, 2.0],
    "soccer": [1.0, 2.0, 3.0],
}


@dataclass(slots=True, frozen=True)
class AlternativeLineOffer:
    """Una línea alternativa ofrecida por un book."""

    line: float
    odds: float


@dataclass(slots=True, frozen=True)
class LineEvaluation:
    """Evaluación de una línea específica con EV estimado."""

    line: float
    odds: float
    p_cover: float
    ev: float
    crosses_key_number: bool
    key_numbers_crossed: list[float]


def probability_cover_spread(
    p_base: float,
    base_line: float,
    target_line: float,
    *,
    sport: Sport,
) -> float:
    """Estima P(cover target_line) dada P(cover base_line).

    Simplificación: asume margen ~ Normal con std dependiente del sport.
    Std real se calibraría con histórico de márgenes.
    """
    # Std típico de diferenciales por sport (calibración empírica)
    std_by_sport: dict[Sport, float] = {
        "nfl": 13.5,
        "nba": 11.0,
        "nhl": 2.0,
        "mlb": 3.2,
        "soccer": 1.4,
    }
    std = std_by_sport.get(sport, 10.0)

    # Transform: P(cover base) → implied margin needed → shift en target
    # delta = target_line - base_line (positive = harder to cover = lower p)
    delta = target_line - base_line

    # Z-shift: cada delta unidades = delta/std en Z-space
    from scipy.stats import norm  # type: ignore[import-untyped]

    z_base = norm.ppf(max(min(p_base, 0.999), 0.001))
    z_target = z_base - (delta / std)
    return float(max(0.01, min(0.99, norm.cdf(z_target))))


def _count_key_numbers_crossed(base: float, target: float, sport: Sport) -> list[float]:
    """Lista los key numbers cruzados entre base y target."""
    lo = min(abs(base), abs(target))
    hi = max(abs(base), abs(target))
    return [kn for kn in KEY_NUMBERS.get(sport, []) if lo <= kn <= hi]


def evaluate_alternative_lines(
    p_base: float,
    base_line: float,
    alt_offers: list[AlternativeLineOffer],
    *,
    sport: Sport,
) -> list[LineEvaluation]:
    """Evalúa cada offer alternativa → retorna lista ordenada por EV descendente.

    El caller decide qué line tomar (normalmente la de mayor EV).
    """
    evals: list[LineEvaluation] = []
    for offer in alt_offers:
        p_cover = probability_cover_spread(p_base, base_line, offer.line, sport=sport)
        ev = p_cover * offer.odds - 1.0
        key_numbers_crossed = _count_key_numbers_crossed(base_line, offer.line, sport)
        evals.append(
            LineEvaluation(
                line=offer.line,
                odds=offer.odds,
                p_cover=p_cover,
                ev=ev,
                crosses_key_number=bool(key_numbers_crossed),
                key_numbers_crossed=key_numbers_crossed,
            )
        )

    evals.sort(key=lambda e: e.ev, reverse=True)
    return evals


def best_alt_line(
    p_base: float,
    base_line: float,
    alt_offers: list[AlternativeLineOffer],
    *,
    sport: Sport,
    min_ev: float = 0.03,
) -> LineEvaluation | None:
    """Retorna la alt line con mayor EV si supera el umbral. None si ninguna vale."""
    evals = evaluate_alternative_lines(p_base, base_line, alt_offers, sport=sport)
    if not evals or evals[0].ev < min_ev:
        return None
    return evals[0]
