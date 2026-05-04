"""Fase 4.4 — Derivative markets (first-half, first-inning, team totals).

Pinnacle + DK ofrecen first-half totals, first-inning over 0.5, team total runs.
Estos mercados tienen hold 4-6% (vs 2% full-game) pero el modelo tiene edge
aún mayor porque los books le prestan menos atención.

Typical edge: 5-8% vs 2-3% en full game.

Market types soportados:
  - `first_half_total_nba`: NBA primer medio total over/under
  - `first_half_total_nfl`: NFL primer medio
  - `first_inning_over_05_mlb`: MLB over 0.5 runs 1st inning
  - `team_total_home`: home team total > line
  - `team_total_away`: away team total > line
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class DerivativeMarket:
    code: str  # "first_half_total_nba" | ...
    full_game_market: str  # "totals"
    ratio_to_full: float  # ej 0.5 para first-half (half = 50% del full-game total)
    std_adjustment: float  # variance multiplier (first-half tiene menos muestras)


CATALOG: dict[str, DerivativeMarket] = {
    "first_half_total_nba": DerivativeMarket(
        code="first_half_total_nba",
        full_game_market="totals",
        ratio_to_full=0.5,
        std_adjustment=1.4,
    ),
    "first_half_total_nfl": DerivativeMarket(
        code="first_half_total_nfl",
        full_game_market="totals",
        ratio_to_full=0.5,
        std_adjustment=1.4,
    ),
    "first_inning_over_05_mlb": DerivativeMarket(
        code="first_inning_over_05_mlb",
        full_game_market="totals",
        ratio_to_full=0.11,  # 1/9 innings
        std_adjustment=3.0,
    ),
    "team_total_home": DerivativeMarket(
        code="team_total_home",
        full_game_market="totals",
        ratio_to_full=0.5,
        std_adjustment=1.2,
    ),
    "team_total_away": DerivativeMarket(
        code="team_total_away",
        full_game_market="totals",
        ratio_to_full=0.5,
        std_adjustment=1.2,
    ),
}


def estimate_derivative_prob(
    p_full_game_over: float,
    derivative_code: str,
    line_full: float,
    line_derivative: float,
) -> float:
    """Dado P(over full-game) y líneas ambas, estima P(over derivative).

    Método: proyecta ratio hacia el derivative con ajuste de varianza.
    """
    meta = CATALOG.get(derivative_code)
    if meta is None:
        return p_full_game_over

    # Si el derivative_line es proporcional al ratio del full-game line,
    # la probabilidad debería ser similar
    expected_derivative_line = line_full * meta.ratio_to_full
    if expected_derivative_line == 0:
        return p_full_game_over
    # Si line_derivative > expected → prob menor
    ratio = expected_derivative_line / line_derivative if line_derivative > 0 else 1.0
    # Ajuste simple: p_derivative = p_full × ratio × variance_factor
    p = p_full_game_over * ratio * (1.0 / meta.std_adjustment)
    return max(0.05, min(0.95, p))


def is_derivative_market(market: str) -> bool:
    return market in CATALOG or market.startswith(
        ("first_half_", "first_inning_", "team_total_", "first_quarter_")
    )
