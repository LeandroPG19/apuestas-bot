"""Backtesting walk-forward para validación de modelos post-pivote.

Plan §7.1: sin bankroll/PnL, la única forma de probar que los picks
emitidos son +EV reales es contra histórico con TimeSeriesSplit purgado.
"""

from apuestas.backtest.walk_forward import (
    BacktestResult,
    format_report,
    walk_forward_backtest,
)

__all__ = ["BacktestResult", "format_report", "walk_forward_backtest"]
