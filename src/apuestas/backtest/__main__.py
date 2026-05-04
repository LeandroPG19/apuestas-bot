"""CLI wrapper: `uv run python -m apuestas.backtest --sport nba --since 2025-10-01`.

Genera reporte markdown en `artifacts/backtest_reports/{sport}_{date}.md`.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, date, datetime
from pathlib import Path

from apuestas.backtest.walk_forward import format_report, walk_forward_backtest
from apuestas.obs.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _parse_date(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s).replace(tzinfo=UTC)
    except ValueError:
        return datetime.combine(date.fromisoformat(s), datetime.min.time(), tzinfo=UTC)


async def _run(args: argparse.Namespace) -> int:
    since = _parse_date(args.since)
    until = _parse_date(args.until) if args.until else datetime.now(tz=UTC)

    logger.info(
        "backtest.cli.start",
        sport=args.sport,
        since=since.isoformat(),
        until=until.isoformat(),
        n_splits=args.n_splits,
        gap_days=args.gap_days,
    )

    results = await walk_forward_backtest(
        sport=args.sport,
        start_date=since,
        end_date=until,
        n_splits=args.n_splits,
        gap_days=args.gap_days,
    )

    report = format_report(results, sport=args.sport)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{args.sport}_{stamp}.md"
    out_path.write_text(report, encoding="utf-8")
    logger.info("backtest.cli.done", output=str(out_path), n_folds=len(results))

    print(report)
    print(f"\n→ guardado en: {out_path}")
    return 0 if results else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m apuestas.backtest",
        description="Walk-forward backtest sobre pick_alerts resueltas.",
    )
    parser.add_argument(
        "--sport", required=True, help="sport_code: nba, mlb, nfl, nhl, soccer, ..."
    )
    parser.add_argument("--since", required=True, help="Fecha ISO (YYYY-MM-DD) de inicio.")
    parser.add_argument(
        "--until", default=None, help="Fecha ISO (YYYY-MM-DD) de fin; default: hoy."
    )
    parser.add_argument("--n-splits", type=int, default=10, help="Número de folds temporales.")
    parser.add_argument("--gap-days", type=int, default=7, help="Purga entre train y test (días).")
    parser.add_argument(
        "--out-dir",
        default="artifacts/backtest_reports",
        help="Carpeta de salida del reporte markdown.",
    )
    args = parser.parse_args()

    configure_logging()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
