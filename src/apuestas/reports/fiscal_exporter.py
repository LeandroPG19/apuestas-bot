"""Exportador fiscal MX — CSV mensual compatible con SAT (§17.11).

Genera CSV para declaración ISR por ganancias en apuestas deportivas:

- Casas MX con licencia SEGOB retienen 1% ISR federal + ~6% ISR estatal GTO
  sobre el monto ganado neto (no sobre el total apostado).
- UMA 2026 ≈ 113.07 MXN/día. Retiros >= 325 UMA/mes (≈ $36,750 MXN) activan
  umbral LFPIORPI (Ley Federal para la Prevención e Identificación de
  Operaciones con Recursos de Procedencia Ilícita).
- Si ingresos totales declarables > $600,000 MXN/año → obligación SAT.

Uso:
    make fiscal-export MONTH=2026-04
    → exports/fiscal/fiscal_2026-04.csv
    → exports/fiscal/fiscal_2026-04_summary.json

El CSV NO sustituye asesoría contable; es un apoyo para llevar trazabilidad.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

UMA_2026_DIARIA_MXN = Decimal("113.07")
UMF_LFPIORPI_UMAS = 325
ISR_FEDERAL_RATE = Decimal("0.01")
ISR_GTO_RATE = Decimal("0.06")
SAT_ANUAL_THRESHOLD_MXN = Decimal("600000")


@dataclass(slots=True)
class FiscalSummary:
    period_start: date
    period_end: date
    total_bets: int = 0
    total_stake_mxn: Decimal = Decimal("0")
    total_winnings_gross_mxn: Decimal = Decimal("0")
    total_losses_mxn: Decimal = Decimal("0")
    net_profit_mxn: Decimal = Decimal("0")
    isr_federal_withheld: Decimal = Decimal("0")
    isr_gto_estimated: Decimal = Decimal("0")
    lfpiorpi_threshold_mxn: Decimal = field(init=False)
    lfpiorpi_triggered: bool = False
    sat_anual_alert: bool = False
    by_bookmaker: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.lfpiorpi_threshold_mxn = UMA_2026_DIARIA_MXN * UMF_LFPIORPI_UMAS

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "period": f"{self.period_start.isoformat()}/{self.period_end.isoformat()}",
            "total_bets": self.total_bets,
            "totals_mxn": {
                "stake": float(self.total_stake_mxn),
                "winnings_gross": float(self.total_winnings_gross_mxn),
                "losses": float(self.total_losses_mxn),
                "net_profit": float(self.net_profit_mxn),
            },
            "isr_withheld_mxn": {
                "federal_1pct": float(self.isr_federal_withheld),
                "gto_6pct_estimated": float(self.isr_gto_estimated),
            },
            "compliance": {
                "uma_2026_daily_mxn": float(UMA_2026_DIARIA_MXN),
                "lfpiorpi_threshold_monthly_mxn": float(self.lfpiorpi_threshold_mxn),
                "lfpiorpi_triggered": self.lfpiorpi_triggered,
                "sat_annual_threshold_mxn": float(SAT_ANUAL_THRESHOLD_MXN),
                "sat_annual_alert_monthly_proxy": self.sat_anual_alert,
            },
            "by_bookmaker": self.by_bookmaker,
        }


def _period_bounds(month_iso: str) -> tuple[datetime, datetime]:
    """'2026-04' → (2026-04-01 00:00 UTC, 2026-05-01 00:00 UTC)."""
    year_s, month_s = month_iso.split("-", 1)
    year = int(year_s)
    month = int(month_s)
    start = datetime(year, month, 1, tzinfo=UTC)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=UTC)
    else:
        end = datetime(year, month + 1, 1, tzinfo=UTC)
    return start, end


async def _fetch_settled_bets(*, start: datetime, end: datetime, unit_mxn: Decimal) -> pl.DataFrame:
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT b.id AS bet_id,
                       b.bookmaker,
                       b.market,
                       b.outcome,
                       b.line,
                       b.stake_units,
                       b.odds_placed,
                       b.status,
                       b.pnl_units,
                       b.clv,
                       b.placed_at,
                       b.settled_at,
                       b.is_paper,
                       m.sport_code,
                       m.start_time
                FROM bets b
                JOIN matches m ON m.id = b.match_id
                WHERE b.settled_at >= :s AND b.settled_at < :e
                  AND b.is_paper = false
                  AND b.status IN ('won','lost','halfwon','halflost','void','cashed')
                ORDER BY b.settled_at ASC
                """
            ),
            {"s": start, "e": end},
        )
        rows = [dict(r._mapping) for r in result.all()]

    if not rows:
        return pl.DataFrame()

    df = pl.DataFrame(rows)

    unit_f = float(unit_mxn)
    df = df.with_columns(
        [
            pl.col("stake_units").cast(pl.Float64),
            pl.col("odds_placed").cast(pl.Float64),
            pl.col("pnl_units").cast(pl.Float64).fill_null(0.0),
            (pl.col("stake_units").cast(pl.Float64) * unit_f).alias("stake_mxn"),
            (pl.col("pnl_units").cast(pl.Float64) * unit_f).alias("pnl_mxn"),
        ]
    )
    df = df.with_columns(
        [
            pl.when(pl.col("pnl_mxn") > 0)
            .then(pl.col("pnl_mxn"))
            .otherwise(0.0)
            .alias("winnings_gross_mxn"),
            pl.when(pl.col("pnl_mxn") < 0)
            .then(-pl.col("pnl_mxn"))
            .otherwise(0.0)
            .alias("losses_mxn"),
        ]
    )
    df = df.with_columns(
        [
            (pl.col("winnings_gross_mxn") * float(ISR_FEDERAL_RATE)).alias("isr_federal_mxn"),
            (pl.col("winnings_gross_mxn") * float(ISR_GTO_RATE)).alias("isr_gto_mxn"),
        ]
    )
    return df


def _build_summary(
    df: pl.DataFrame, start: datetime, end: datetime, monthly_income_mxn: Decimal
) -> FiscalSummary:
    summary = FiscalSummary(period_start=start.date(), period_end=end.date())
    if df.is_empty():
        return summary

    summary.total_bets = df.height
    summary.total_stake_mxn = Decimal(str(round(df["stake_mxn"].sum() or 0.0, 2)))
    summary.total_winnings_gross_mxn = Decimal(str(round(df["winnings_gross_mxn"].sum() or 0.0, 2)))
    summary.total_losses_mxn = Decimal(str(round(df["losses_mxn"].sum() or 0.0, 2)))
    summary.net_profit_mxn = Decimal(str(round(df["pnl_mxn"].sum() or 0.0, 2)))
    summary.isr_federal_withheld = Decimal(str(round(df["isr_federal_mxn"].sum() or 0.0, 2)))
    summary.isr_gto_estimated = Decimal(str(round(df["isr_gto_mxn"].sum() or 0.0, 2)))

    by_book = (
        df.group_by("bookmaker")
        .agg(
            [
                pl.len().alias("bets"),
                pl.col("stake_mxn").sum().alias("stake"),
                pl.col("winnings_gross_mxn").sum().alias("winnings"),
                pl.col("pnl_mxn").sum().alias("net"),
            ]
        )
        .sort("net", descending=True)
    )
    summary.by_bookmaker = {
        row["bookmaker"]: {
            "bets": int(row["bets"]),
            "stake": float(row["stake"]),
            "winnings": float(row["winnings"]),
            "net": float(row["net"]),
        }
        for row in by_book.to_dicts()
    }

    summary.lfpiorpi_triggered = summary.total_winnings_gross_mxn >= summary.lfpiorpi_threshold_mxn
    projected_annual = monthly_income_mxn * 12
    summary.sat_anual_alert = projected_annual >= SAT_ANUAL_THRESHOLD_MXN
    return summary


def _write_csv(df: pl.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df.is_empty():
        pl.DataFrame(
            {
                "bet_id": [],
                "fecha": [],
                "casa": [],
                "deporte": [],
                "mercado": [],
                "outcome": [],
                "stake_mxn": [],
                "pnl_mxn": [],
                "winnings_gross_mxn": [],
                "isr_federal_mxn": [],
                "isr_gto_mxn": [],
            }
        ).write_csv(out_path)
        return

    export = df.select(
        [
            pl.col("bet_id"),
            pl.col("settled_at").alias("fecha"),
            pl.col("bookmaker").alias("casa"),
            pl.col("sport_code").alias("deporte"),
            pl.col("market").alias("mercado"),
            pl.col("outcome"),
            pl.col("stake_mxn").round(2),
            pl.col("pnl_mxn").round(2),
            pl.col("winnings_gross_mxn").round(2),
            pl.col("isr_federal_mxn").round(2),
            pl.col("isr_gto_mxn").round(2),
        ]
    )
    export.write_csv(out_path)


async def export_fiscal_month(
    *, month_iso: str, unit_mxn: Decimal = Decimal("100"), out_dir: Path | None = None
) -> dict[str, Any]:
    """Exporta CSV + summary JSON para un mes completo.

    Args:
        month_iso: "YYYY-MM" (ej. "2026-04")
        unit_mxn: valor MXN de 1 unidad de bankroll (default 100 MXN/u)
        out_dir: carpeta destino (default exports/fiscal/)
    """
    start, end = _period_bounds(month_iso)
    out_dir = out_dir or Path("exports/fiscal")

    df = await _fetch_settled_bets(start=start, end=end, unit_mxn=unit_mxn)

    monthly_income_mxn = (
        Decimal(str(round(df["pnl_mxn"].sum() or 0.0, 2))) if not df.is_empty() else Decimal("0")
    )
    summary = _build_summary(df, start, end, monthly_income_mxn)

    csv_path = out_dir / f"fiscal_{month_iso}.csv"
    json_path = out_dir / f"fiscal_{month_iso}_summary.json"
    _write_csv(df, csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(summary.as_jsonable(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info(
        "fiscal_export.done",
        month=month_iso,
        bets=summary.total_bets,
        net_mxn=float(summary.net_profit_mxn),
        csv=str(csv_path),
        lfpiorpi=summary.lfpiorpi_triggered,
    )
    return {
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "summary": summary.as_jsonable(),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Exportador fiscal mensual MX")
    parser.add_argument("--month", required=True, help="YYYY-MM")
    parser.add_argument("--unit-mxn", default="100", help="MXN por unidad bankroll")
    parser.add_argument("--out", default="exports/fiscal", help="Carpeta destino")
    args = parser.parse_args()

    result = asyncio.run(
        export_fiscal_month(
            month_iso=args.month,
            unit_mxn=Decimal(args.unit_mxn),
            out_dir=Path(args.out),
        )
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
