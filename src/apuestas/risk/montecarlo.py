"""Monte Carlo simulator de bankroll — risk of ruin (§17.3).

Simula 10,000 paths × 1,000 bets con:
- Distribución empírica de edges (sample de historial backtest).
- Distribución empírica de odds.
- Correlación entre picks del mismo día (modelada con bloques).

Reporta:
- Prob(drawdown > 25%)
- Prob(drawdown > 40%)
- Prob(bankroll × 2 antes de ruina)
- Bankroll esperado 6M (percentiles 10/50/90)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apuestas.obs.logging import get_logger
from apuestas.risk.kelly import kelly_fraction

logger = get_logger(__name__)


@dataclass(slots=True)
class MCConfig:
    n_simulations: int = 10_000
    n_bets_per_path: int = 1000
    initial_bankroll: float = 1000.0
    edge_distribution: list[float] = None  # type: ignore[assignment]
    odds_distribution: list[float] = None  # type: ignore[assignment]
    kelly_fraction: float = 0.25
    kelly_cap: float = 0.05
    ruin_threshold_pct: float = 0.40
    bets_per_day: int = 3
    same_day_correlation: float = 0.10
    seed: int = 42


@dataclass(slots=True)
class MCResult:
    prob_dd_25pct: float
    prob_dd_40pct: float
    prob_ruin: float
    prob_double: float
    expected_final_bankroll: float
    p10_final_bankroll: float
    p50_final_bankroll: float
    p90_final_bankroll: float
    median_max_drawdown_pct: float
    params: dict[str, float | int]


def _sample_edge_odds(
    cfg: MCConfig, rng: np.random.Generator, n: int
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.asarray(cfg.edge_distribution or [0.02, 0.03, 0.04, 0.05, 0.07])
    odds = np.asarray(cfg.odds_distribution or [1.70, 1.85, 2.00, 2.15, 2.40])
    idx = rng.integers(0, len(edges), size=n)
    return edges[idx], odds[idx % len(odds)]


def simulate(cfg: MCConfig | None = None) -> MCResult:
    cfg = cfg or MCConfig()
    rng = np.random.default_rng(cfg.seed)

    finals = np.empty(cfg.n_simulations)
    max_dd_pct = np.empty(cfg.n_simulations)
    reached_ruin = np.zeros(cfg.n_simulations, dtype=bool)
    reached_dd_25 = np.zeros(cfg.n_simulations, dtype=bool)
    reached_dd_40 = np.zeros(cfg.n_simulations, dtype=bool)
    reached_double = np.zeros(cfg.n_simulations, dtype=bool)

    for s in range(cfg.n_simulations):
        bankroll = cfg.initial_bankroll
        peak = bankroll
        path_min_dd = 0.0

        edges, odds = _sample_edge_odds(cfg, rng, cfg.n_bets_per_path)
        probs = 1.0 / odds + edges
        probs = np.clip(probs, 0.01, 0.99)
        stakes_frac = np.array(
            [
                kelly_fraction(p, o, fraction=cfg.kelly_fraction, cap=cfg.kelly_cap)
                for p, o in zip(probs, odds, strict=True)
            ]
        )

        # Correlación same-day: bloques de `bets_per_day` con outcomes correlacionados
        for day_start in range(0, cfg.n_bets_per_path, cfg.bets_per_day):
            day_end = min(day_start + cfg.bets_per_day, cfg.n_bets_per_path)
            n_day = day_end - day_start

            # Sample un "common factor" del día
            z = rng.normal(0, 1)
            outcomes = np.empty(n_day, dtype=np.int8)
            for i in range(n_day):
                p = probs[day_start + i]
                # p_adj con correlación
                p_noise = p + cfg.same_day_correlation * z * np.sqrt(p * (1 - p))
                p_noise = np.clip(p_noise, 0.01, 0.99)
                outcomes[i] = int(rng.random() < p_noise)

            for i in range(n_day):
                idx = day_start + i
                stake = stakes_frac[idx] * bankroll
                if stake <= 0:
                    continue
                if outcomes[i] == 1:
                    bankroll += stake * (odds[idx] - 1)
                else:
                    bankroll -= stake
                peak = max(peak, bankroll)
                dd = (bankroll - peak) / peak
                path_min_dd = min(path_min_dd, dd)

            # Checks por día (tracking más simple)
            dd_now = (bankroll - peak) / peak if peak > 0 else 0.0
            if dd_now <= -0.25:
                reached_dd_25[s] = True
            if dd_now <= -0.40:
                reached_dd_40[s] = True
            if bankroll <= cfg.initial_bankroll * (1 - cfg.ruin_threshold_pct):
                reached_ruin[s] = True
            if bankroll >= cfg.initial_bankroll * 2:
                reached_double[s] = True

        finals[s] = bankroll
        max_dd_pct[s] = abs(path_min_dd)

    return MCResult(
        prob_dd_25pct=float(reached_dd_25.mean()),
        prob_dd_40pct=float(reached_dd_40.mean()),
        prob_ruin=float(reached_ruin.mean()),
        prob_double=float(reached_double.mean()),
        expected_final_bankroll=float(finals.mean()),
        p10_final_bankroll=float(np.quantile(finals, 0.10)),
        p50_final_bankroll=float(np.quantile(finals, 0.50)),
        p90_final_bankroll=float(np.quantile(finals, 0.90)),
        median_max_drawdown_pct=float(np.quantile(max_dd_pct, 0.50)),
        params={
            "n_simulations": cfg.n_simulations,
            "n_bets_per_path": cfg.n_bets_per_path,
            "kelly_fraction": cfg.kelly_fraction,
            "kelly_cap": cfg.kelly_cap,
            "ruin_threshold_pct": cfg.ruin_threshold_pct,
            "initial_bankroll": cfg.initial_bankroll,
        },
    )


def recommend_kelly_adjustment(result: MCResult) -> str | None:
    """§17.3: si Prob(DD>40%) > 5%, alertar reducir Kelly."""
    if result.prob_dd_40pct > 0.05:
        return (
            f"Prob(DD>40%) = {result.prob_dd_40pct:.1%} excede 5% — "
            "reducir kelly_fraction de ¼ a ⅛ hasta que CLV estabilice."
        )
    if result.prob_dd_25pct > 0.30:
        return (
            f"Prob(DD>25%) = {result.prob_dd_25pct:.1%} — monitorear, podría "
            "ajustarse el EV threshold de 3% a 4%."
        )
    return None
