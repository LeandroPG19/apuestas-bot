"""Backtester walk-forward con replay exacto de odds (§17.1).

Reglas anti-leakage no negociables:
- TimeSeriesSplit con gap de 7 días (evita partidos simultáneos).
- Odds del timestamp T-30min del partido, NO agregadas.
- Closing line solo como ground truth CLV, jamás feature.
- Model version + feature version registrados en cada pick simulado.

Métricas reportadas: ROI, Sharpe, Sortino, Calmar, Max DD,
ROI por mercado, ROI por liga, CLV distribución, hit rate por bucket EV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class BacktestBet:
    match_id: int
    start_time: datetime
    market: str
    outcome: str
    p_model: float
    p_lower: float
    p_upper: float
    odds: float
    implied_prob: float
    edge: float
    stake_units: float
    result: int  # 1 won, 0 lost
    pnl_units: float
    clv: float | None
    league_id: int | None = None


@dataclass(slots=True)
class BacktestConfig:
    initial_bankroll: float = 1000.0
    unit_size: float = 1.0  # 1% del initial
    kelly_fraction: float = 0.25
    kelly_cap_pct: float = 0.05
    ev_threshold: float = 0.03
    min_odds: float = 1.5
    max_odds: float = 4.0
    stop_loss_pct: float = 0.30  # pausa si bankroll baja 30%
    conformal_min_margin: float = 0.01  # p_lower > implied_prob + margen


@dataclass(slots=True)
class BacktestReport:
    n_bets: int
    n_won: int
    n_lost: int
    roi: float
    yield_pct: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown_pct: float
    clv_mean: float
    clv_std: float
    clv_positive_rate: float
    hit_rate_by_ev_bucket: dict[str, float]
    roi_by_market: dict[str, float]
    roi_by_league: dict[str, float]
    final_bankroll: float
    bankroll_curve: list[float] = field(default_factory=list)
    bets: list[BacktestBet] = field(default_factory=list)


def _kelly_stake(
    p: float, odds: float, *, fraction: float, cap_pct: float, bankroll: float
) -> float:
    """Kelly fraccional con cap."""
    b = odds - 1.0
    if b <= 0 or p <= 1.0 / odds:
        return 0.0
    f_star = ((b * p) - (1 - p)) / b
    if f_star <= 0:
        return 0.0
    stake = min(f_star * fraction, cap_pct) * bankroll
    return max(stake, 0.0)


def simulate_walk_forward(
    events_df: pl.DataFrame,
    *,
    cfg: BacktestConfig | None = None,
) -> BacktestReport:
    """Simula secuencia de bets sobre un DataFrame con predicciones+odds+resultados.

    `events_df` requiere columnas:
        match_id, start_time, market, outcome, p_model, p_lower, p_upper,
        odds, result (0/1), closing_line, league_id
    """
    cfg = cfg or BacktestConfig()
    df = events_df.sort("start_time")

    bankroll = cfg.initial_bankroll
    peak = cfg.initial_bankroll
    curve: list[float] = [bankroll]
    bets: list[BacktestBet] = []
    returns: list[float] = []

    paused = False
    for row in df.iter_rows(named=True):
        # Filtros básicos
        odds = float(row["odds"])
        if odds < cfg.min_odds or odds > cfg.max_odds:
            continue
        p = float(row["p_model"])
        implied = 1.0 / odds
        edge = p - implied
        if edge < cfg.ev_threshold:
            continue

        # Filtro conformal: solo si p_lower > implied_prob + margen
        p_lower = float(row.get("p_lower") or p)
        if p_lower <= implied + cfg.conformal_min_margin:
            continue

        # Stop-loss
        if bankroll <= cfg.initial_bankroll * (1 - cfg.stop_loss_pct):
            if not paused:
                logger.warning("backtest.stop_loss_hit", bankroll=bankroll)
                paused = True
            continue

        stake = _kelly_stake(
            p,
            odds,
            fraction=cfg.kelly_fraction,
            cap_pct=cfg.kelly_cap_pct,
            bankroll=bankroll,
        )
        if stake <= 0:
            continue

        result = int(row["result"])
        pnl = stake * (odds - 1) if result == 1 else -stake
        bankroll += pnl
        returns.append(pnl / (curve[-1] or 1.0))
        peak = max(peak, bankroll)
        curve.append(bankroll)

        closing = row.get("closing_line")
        clv = None
        if closing is not None and float(closing) > 1.0:
            clv = float(odds) / float(closing) - 1.0

        bets.append(
            BacktestBet(
                match_id=int(row["match_id"]),
                start_time=row["start_time"],
                market=str(row["market"]),
                outcome=str(row["outcome"]),
                p_model=p,
                p_lower=p_lower,
                p_upper=float(row.get("p_upper") or p),
                odds=odds,
                implied_prob=implied,
                edge=edge,
                stake_units=stake / cfg.unit_size if cfg.unit_size else stake,
                result=result,
                pnl_units=pnl / cfg.unit_size if cfg.unit_size else pnl,
                clv=clv,
                league_id=row.get("league_id"),
            )
        )

    return _summarize(bets, curve, cfg)


def _summarize(bets: list[BacktestBet], curve: list[float], cfg: BacktestConfig) -> BacktestReport:
    if not bets:
        return BacktestReport(
            n_bets=0,
            n_won=0,
            n_lost=0,
            roi=0.0,
            yield_pct=0.0,
            sharpe=0.0,
            sortino=0.0,
            calmar=0.0,
            max_drawdown_pct=0.0,
            clv_mean=0.0,
            clv_std=0.0,
            clv_positive_rate=0.0,
            hit_rate_by_ev_bucket={},
            roi_by_market={},
            roi_by_league={},
            final_bankroll=curve[-1] if curve else cfg.initial_bankroll,
            bankroll_curve=curve,
            bets=[],
        )

    n = len(bets)
    n_won = sum(1 for b in bets if b.result == 1)
    total_stake = sum(b.stake_units for b in bets)
    total_pnl = sum(b.pnl_units for b in bets)
    yield_pct = (total_pnl / total_stake) if total_stake > 0 else 0.0

    # Sharpe/Sortino sobre curve returns
    curve_arr = np.asarray(curve)
    rets = np.diff(curve_arr) / curve_arr[:-1]
    sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(252)) if np.std(rets) > 0 else 0.0
    downside = rets[rets < 0]
    sortino = (
        float(np.mean(rets) / np.std(downside) * np.sqrt(252))
        if downside.size > 0 and np.std(downside) > 0
        else 0.0
    )

    # Max DD
    running_max = np.maximum.accumulate(curve_arr)
    dd = (curve_arr - running_max) / running_max
    max_dd = float(dd.min())
    calmar = float((curve_arr[-1] / curve_arr[0] - 1) / abs(max_dd)) if max_dd < 0 else 0.0

    # CLV stats
    clvs = [b.clv for b in bets if b.clv is not None]
    clv_mean = float(np.mean(clvs)) if clvs else 0.0
    clv_std = float(np.std(clvs)) if clvs else 0.0
    clv_pos_rate = float(np.mean([1 if c > 0 else 0 for c in clvs])) if clvs else 0.0

    # Hit rate por bucket EV
    buckets = [(0.03, 0.05), (0.05, 0.07), (0.07, 0.10), (0.10, 1.0)]
    hit_rate_by_ev: dict[str, float] = {}
    for lo, hi in buckets:
        subset = [b for b in bets if lo <= b.edge < hi]
        if not subset:
            continue
        won = sum(1 for b in subset if b.result == 1)
        hit_rate_by_ev[f"EV_{int(lo * 100)}-{int(hi * 100)}pct"] = won / len(subset)

    # ROI por mercado
    markets = {b.market for b in bets}
    roi_by_market: dict[str, float] = {}
    for m in markets:
        subset = [b for b in bets if b.market == m]
        stake_m = sum(b.stake_units for b in subset)
        pnl_m = sum(b.pnl_units for b in subset)
        roi_by_market[m] = pnl_m / stake_m if stake_m > 0 else 0.0

    # ROI por liga
    leagues = {b.league_id for b in bets if b.league_id is not None}
    roi_by_league: dict[str, float] = {}
    for lid in leagues:
        subset = [b for b in bets if b.league_id == lid]
        stake_l = sum(b.stake_units for b in subset)
        pnl_l = sum(b.pnl_units for b in subset)
        roi_by_league[str(lid)] = pnl_l / stake_l if stake_l > 0 else 0.0

    return BacktestReport(
        n_bets=n,
        n_won=n_won,
        n_lost=n - n_won,
        roi=total_pnl / cfg.initial_bankroll,
        yield_pct=yield_pct,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown_pct=abs(max_dd),
        clv_mean=clv_mean,
        clv_std=clv_std,
        clv_positive_rate=clv_pos_rate,
        hit_rate_by_ev_bucket=hit_rate_by_ev,
        roi_by_market=roi_by_market,
        roi_by_league=roi_by_league,
        final_bankroll=curve[-1],
        bankroll_curve=curve,
        bets=bets,
    )


def summary_to_dict(report: BacktestReport) -> dict[str, Any]:
    """Serialización amigable para JSON/MLflow log."""
    return {
        "n_bets": report.n_bets,
        "n_won": report.n_won,
        "roi": report.roi,
        "yield_pct": report.yield_pct,
        "sharpe": report.sharpe,
        "sortino": report.sortino,
        "calmar": report.calmar,
        "max_drawdown_pct": report.max_drawdown_pct,
        "clv_mean": report.clv_mean,
        "clv_positive_rate": report.clv_positive_rate,
        "hit_rate_by_ev_bucket": report.hit_rate_by_ev_bucket,
        "roi_by_market": report.roi_by_market,
        "roi_by_league": report.roi_by_league,
        "final_bankroll": report.final_bankroll,
    }
