"""Auto-tune soccer_max_draw_prob_by_league — Sprint 14 #156.

Calcula empates históricos por league_id rolling 90d y genera
`config/soccer_draw_thresholds.yaml`. DetectorConfig lee ese YAML en vez
de hardcoded defaults.

Regla: threshold = draw_rate_90d * 1.15 (15% margen). Mínimo 0.22, máximo 0.35.

Uso:
    python scripts/autotune_draw_thresholds.py --days 90
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.pop("PYTHON_GIL", None)

import yaml
from sqlalchemy import text

from apuestas.db import session_scope


async def compute_draw_rates(days: int) -> dict[int, dict]:
    """Rate de empates por league_id en ventana days."""
    since = datetime.now(tz=UTC) - timedelta(days=days)
    async with session_scope() as s:
        rows = (
            await s.execute(
                text(
                    """
                    SELECT m.league_id, l.name,
                           COUNT(*) n,
                           COUNT(*) FILTER (WHERE m.home_score = m.away_score) draws
                    FROM matches m
                    LEFT JOIN leagues l ON l.id = m.league_id
                    WHERE m.sport_code='soccer'
                      AND m.home_score IS NOT NULL AND m.away_score IS NOT NULL
                      AND m.start_time >= :since
                      AND m.league_id IS NOT NULL
                    GROUP BY m.league_id, l.name
                    HAVING COUNT(*) >= 20
                    ORDER BY COUNT(*) FILTER (WHERE m.home_score = m.away_score)::float
                            / NULLIF(COUNT(*), 0) DESC
                    """
                ),
                {"since": since},
            )
        ).fetchall()
    result = {}
    for r in rows:
        rate = float(r.draws) / max(1, r.n)
        # Threshold = rate * 1.15, clipped [0.22, 0.35]
        thr = max(0.22, min(0.35, rate * 1.15))
        result[int(r.league_id)] = {
            "league_name": r.name or f"league_{r.league_id}",
            "n": int(r.n),
            "draws": int(r.draws),
            "draw_rate": round(rate, 4),
            "threshold": round(thr, 3),
        }
    return result


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--out", default="config/soccer_draw_thresholds.yaml")
    args = ap.parse_args()

    rates = await compute_draw_rates(args.days)
    if not rates:
        print("No data.")
        return

    print(f"{'league_id':>10} {'name':30s} {'n':>5} {'draws':>7} {'rate':>8} {'thr':>8}")
    for lg_id, m in sorted(rates.items(), key=lambda x: -x[1]["draw_rate"]):
        print(
            f"{lg_id:>10} {m['league_name'][:30]:30s} "
            f"{m['n']:>5} {m['draws']:>7} {m['draw_rate']:>8.4f} {m['threshold']:>8.3f}"
        )

    yaml_data = {
        "meta": {
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "window_days": args.days,
        },
        "thresholds": {lg: m["threshold"] for lg, m in rates.items()},
        "detail": rates,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(yaml_data, sort_keys=True), encoding="utf-8")
    print(f"\n→ {out_path} ({len(rates)} leagues)")


if __name__ == "__main__":
    asyncio.run(main())
