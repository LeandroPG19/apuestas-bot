"""Fase 1.5 — CLI ejecutable para backtest walk-forward.

Uso:
    python -m apuestas.scripts.run_backtest --sport nba --seasons 2022,2023
    python -m apuestas.scripts.run_backtest --sport soccer --seasons 2021,2022,2023 --output reports/bt_soccer.json

Depende de que `ml/backtest.py::simulate_walk_forward()` ya exista (framework
library ya completo desde migración inicial).

Criterio de aceptación del modelo según el plan: Sharpe ≥ 1.2. Si no, el
script sale con exit code 1 para bloquear deploy automático.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def _build_events_df(sport: str, seasons: list[int]) -> Any:
    """Construye DataFrame polars con columnas requeridas por simulate_walk_forward.

    Columnas: match_id, start_time, market, outcome, p_model, p_lower, p_upper,
    odds, result (0/1), closing_line, league_id.

    Uses predictions + odds_history + matches para armar el dataset de replay.
    """
    import polars as pl
    from sqlalchemy import text as _t

    from apuestas.db import session_scope

    async with session_scope() as session:
        result = await session.execute(
            _t(
                """
                SELECT
                    p.match_id,
                    m.start_time,
                    p.market,
                    p.outcome,
                    p.probability AS p_model,
                    p.p_lower,
                    p.p_upper,
                    b.odds_placed AS odds,
                    CASE
                        WHEN b.status = 'won' THEN 1
                        WHEN b.status = 'lost' THEN 0
                        ELSE NULL
                    END AS result,
                    b.closing_line,
                    m.league_id
                FROM predictions p
                JOIN matches m ON m.id = p.match_id
                JOIN bets b ON b.prediction_id = p.id
                WHERE m.sport_code = :sport
                  AND EXTRACT(YEAR FROM m.start_time) = ANY(:seasons)
                  AND b.status IN ('won', 'lost')
                  AND b.test_data = false
                ORDER BY m.start_time ASC
                """
            ),
            {"sport": sport, "seasons": seasons},
        )
        rows = [dict(r._mapping) for r in result.all()]

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest walk-forward sobre histórico")
    p.add_argument(
        "--sport", required=True, choices=["nba", "mlb", "nfl", "nhl", "soccer", "tennis"]
    )
    p.add_argument("--seasons", required=True, help="CSV ej: 2021,2022,2023")
    p.add_argument("--output", default=None, help="Path a JSON de reporte")
    p.add_argument(
        "--min-sharpe",
        type=float,
        default=1.2,
        help="Sharpe mínimo para considerar modelo deployable (default 1.2)",
    )
    p.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="TimeSeriesSplit n_splits",
    )
    p.add_argument(
        "--gap-days",
        type=int,
        default=7,
        help="Gap days entre train/test para evitar leakage",
    )
    return p.parse_args(argv)


async def run_backtest_cli(args: argparse.Namespace) -> int:
    """Ejecuta backtest walk-forward.

    Retorna exit code: 0 si Sharpe ≥ min_sharpe, 1 si no se alcanza.
    """
    seasons = [int(s) for s in args.seasons.split(",")]
    try:
        from apuestas.ml.backtest import BacktestConfig, simulate_walk_forward
    except ImportError as exc:
        print(f"❌ Importar backtest dependencies falló: {exc}")
        return 2

    logger.info(
        "backtest.start",
        sport=args.sport,
        seasons=seasons,
    )

    # Construye events_df desde DB: requiere predictions + odds + results ya persistidos
    events_df = await _build_events_df(args.sport, seasons)
    if events_df.is_empty():
        print(
            f"❌ Sin predictions históricas para {args.sport} seasons {seasons}.\n"
            "   Ejecuta primero: `apuestas retrain --sport {args.sport} --full` para\n"
            "   generar predictions sobre el histórico sembrado."
        )
        return 2

    try:
        cfg = BacktestConfig()
        report = simulate_walk_forward(events_df, cfg=cfg)
    except Exception as exc:
        logger.exception("backtest.failed", error=str(exc))
        print(f"❌ Backtest falló: {exc}")
        return 2

    sharpe = float(report.sharpe)
    sortino = float(report.sortino)
    calmar = float(report.calmar)
    roi_annualized = float(report.roi_annualized)
    n_bets = int(report.n_bets)
    result: dict[str, Any] = {
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "roi_annualized": roi_annualized,
        "n_bets": n_bets,
        "final_bankroll": float(report.final_bankroll),
        "max_drawdown": float(report.max_drawdown),
        "win_rate": float(report.win_rate),
        "total_pnl": float(report.total_pnl),
    }

    print("\n═══════════════════════════════════════════════════════════")
    print(f"  Backtest {args.sport} · seasons {args.seasons}")
    print("═══════════════════════════════════════════════════════════")
    print(f"  Bets simuladas:  {n_bets}")
    print(f"  Sharpe:          {sharpe:+.3f}  (target: ≥ {args.min_sharpe})")
    print(f"  Sortino:         {sortino:+.3f}")
    print(f"  Calmar:          {calmar:+.3f}")
    print(f"  ROI anualizado:  {roi_annualized:+.2%}")
    print("═══════════════════════════════════════════════════════════")

    # Output JSON
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, default=str, indent=2), encoding="utf-8")
        print(f"  Report JSON:     {out_path}")

    if n_bets < 10:
        print("⚠  Muy pocos bets (<10). Resultado no estadísticamente significativo.")
        return 2

    if sharpe < args.min_sharpe:
        print(f"\n❌ Sharpe {sharpe:.3f} < target {args.min_sharpe}. Modelo NO deployable.")
        return 1

    print(f"\n✅ Sharpe {sharpe:.3f} ≥ {args.min_sharpe}. Modelo apto para deploy.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(run_backtest_cli(args))


if __name__ == "__main__":
    sys.exit(main())
