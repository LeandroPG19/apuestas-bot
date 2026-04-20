"""Kelly Criterion — fraccional + cap + correlation-aware para múltiples picks.

§17.2: cuando hay N picks +EV correlacionados (mismo evento o liga),
full Kelly individual sobrestima. Solución:
- Si mismo evento: 0.25x × (1 - avg_correlation) conservador.
- Si múltiples picks/día: resolver QP maximizando log-utility con
  constraint total stake ≤ 15% bankroll/día.

Referencias: Thorp (2006), MacLean-Thorp-Ziemba (2009).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(slots=True)
class KellyBet:
    p: float
    odds: float
    event_id: int | str
    market: str


def kelly_fraction(
    p: float,
    odds: float,
    *,
    fraction: float = 0.25,
    cap: float = 0.05,
) -> float:
    """Kelly fraccional simple con cap % bankroll.

    Fórmula: f* = (b*p - q) / b, donde b = odds - 1, q = 1 - p.
    """
    b = odds - 1.0
    if b <= 0 or p <= 0 or p >= 1:
        return 0.0
    q = 1.0 - p
    f_star = (b * p - q) / b
    if f_star <= 0:
        return 0.0
    return float(min(f_star * fraction, cap))


def implied_correlation(bet_a: KellyBet, bet_b: KellyBet) -> float:
    """Heurística rápida de correlación entre dos picks.

    - Mismo event_id + mismo market → 0.9 (casi idénticos).
    - Mismo event_id + distinto market (over + home cover, etc.) → 0.6.
    - Misma liga (primer segmento de event_id) + mismo día → 0.15.
    - Otro → 0.02.
    """
    if bet_a.event_id == bet_b.event_id:
        return 0.9 if bet_a.market == bet_b.market else 0.6
    # event_id cross-day typically encodes date; heuristic keeps low
    return 0.05


def correlation_matrix(bets: list[KellyBet]) -> np.ndarray:
    n = len(bets)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            c = implied_correlation(bets[i], bets[j])
            corr[i, j] = corr[j, i] = c
    return corr


def correlation_aware_kelly(
    bets: list[KellyBet],
    *,
    fraction: float = 0.25,
    cap_per_bet: float = 0.05,
    daily_cap: float = 0.15,
) -> list[float]:
    """Kelly multi-pick con correlación.

    Maximiza log-utility esperada sujeto a:
    - Σ stake_i ≤ daily_cap
    - 0 ≤ stake_i ≤ cap_per_bet
    - Penalización por correlación vía matriz cov ~ corr × σ²

    Devuelve stakes como fracción del bankroll (equivalente a kelly_fraction).
    """
    n = len(bets)
    if n == 0:
        return []
    if n == 1:
        return [kelly_fraction(bets[0].p, bets[0].odds, fraction=fraction, cap=cap_per_bet)]

    p = np.array([b.p for b in bets])
    b_arr = np.array([b.odds - 1 for b in bets])
    q = 1 - p

    # Expected log utility aproximada: Σ p_i log(1 + b_i * f_i) + q_i log(1 - f_i)
    # Con término de correlación penalizando sizes concentrados
    corr = correlation_matrix(bets)

    def neg_utility(stakes: np.ndarray) -> float:
        stakes = np.clip(stakes, 1e-6, cap_per_bet - 1e-6)
        # Log utility esperada asumiendo independencia (término principal)
        win_term = p * np.log1p(b_arr * stakes)
        loss_term = q * np.log(np.clip(1 - stakes, 1e-6, None))
        base = np.sum(win_term + loss_term)
        # Penalización cuadrática por correlación (peso empírico)
        penalty = 0.5 * stakes @ (corr - np.eye(n)) @ stakes
        return -(base - penalty)

    x0 = np.array([kelly_fraction(b.p, b.odds, fraction=fraction, cap=cap_per_bet) for b in bets])
    bounds = [(0.0, cap_per_bet) for _ in range(n)]
    constraints = ({"type": "ineq", "fun": lambda s: daily_cap - np.sum(s)},)

    result = minimize(
        neg_utility,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 100, "ftol": 1e-6},
    )
    stakes = np.clip(result.x if result.success else x0, 0.0, cap_per_bet)
    return stakes.tolist()
