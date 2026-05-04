"""Error categorization analysis sobre picks emitidos + matches históricos.

Output: artifacts/error_categories/{sport}_{date}.md con breakdown por:
  - sport × market × line_range × favorite_side × odds_bucket
  - identifica categorías perdedoras sistemáticas
  - recomendaciones accionables (thresholds, guards)

Uso:
  python scripts/error_analysis.py --since 2026-04-22
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.pop("PYTHON_GIL", None)

from sqlalchemy import text

from apuestas.db import session_scope


def odds_bucket(odds: float) -> str:
    if odds < 1.50:
        return "heavy_fav(<1.50)"
    if odds < 1.80:
        return "fav(1.50-1.80)"
    if odds < 2.20:
        return "mid(1.80-2.20)"
    if odds < 3.00:
        return "under_dog(2.20-3.00)"
    if odds < 5.00:
        return "big_dog(3.00-5.00)"
    return "longshot(>5.00)"


async def fetch_settled(since: str):
    async with session_scope() as s:
        rows = (
            await s.execute(
                text(
                    """
                    SELECT pa.id, pa.market, pa.outcome, pa.line, pa.odds_placed,
                           pa.outcome_result, m.sport_code, m.league_id, m.stage,
                           ht.name home, at.name away
                    FROM pick_alerts pa
                    JOIN matches m ON m.id=pa.match_id
                    JOIN teams ht ON ht.id=m.home_team_id
                    JOIN teams at ON at.id=m.away_team_id
                    WHERE pa.placed_at >= :since
                      AND pa.outcome_result IN ('won','lost')
                    """
                ),
                {"since": datetime.fromisoformat(since).replace(tzinfo=UTC)},
            )
        ).fetchall()
    return [dict(r._mapping) for r in rows]


def category_report(picks: list[dict]) -> str:
    buckets: dict[tuple, dict] = defaultdict(lambda: {"won": 0, "lost": 0, "profit": 0.0})
    for p in picks:
        cat = (
            p["sport_code"],
            p["market"],
            odds_bucket(float(p["odds_placed"])),
            p["outcome"],
        )
        if p["outcome_result"] == "won":
            buckets[cat]["won"] += 1
            buckets[cat]["profit"] += float(p["odds_placed"]) - 1.0
        else:
            buckets[cat]["lost"] += 1
            buckets[cat]["profit"] -= 1.0

    rows = []
    for cat, m in buckets.items():
        n = m["won"] + m["lost"]
        if n == 0:
            continue
        rows.append(
            {
                "sport": cat[0],
                "market": cat[1],
                "odds": cat[2],
                "outcome": cat[3],
                "n": n,
                "won": m["won"],
                "lost": m["lost"],
                "hr": m["won"] / n,
                "roi": m["profit"] / n,
                "profit": m["profit"],
            }
        )

    # Sort by worst ROI first
    rows.sort(key=lambda r: r["roi"])

    out = ["# Error Analysis — Categorías de picks"]
    out.append(f"\nGenerado: {datetime.now(tz=UTC).isoformat()}")
    out.append(f"\nTotal picks settled: {sum(r['n'] for r in rows)}")
    out.append(
        "\n## Ranking por ROI (peor → mejor)\n\n"
        "| Sport | Market | Odds bucket | Outcome | n | W | L | HR | ROI | Profit |"
        "\n|---|---|---|---|---|---|---|---|---|---|"
    )
    for r in rows:
        out.append(
            f"| {r['sport']} | {r['market']} | {r['odds']} | {r['outcome']} | "
            f"{r['n']} | {r['won']} | {r['lost']} | {r['hr']:.2f} | "
            f"{r['roi']:+.3f} | ${r['profit']:+.2f} |"
        )

    # Accionables
    out.append("\n## Categorías con pérdida severa (n≥3, ROI<-0.20)")
    bad = [r for r in rows if r["n"] >= 3 and r["roi"] < -0.20]
    if bad:
        for r in bad:
            out.append(
                f"- **{r['sport']}/{r['market']}/{r['odds']}/{r['outcome']}**: "
                f"{r['won']}W {r['lost']}L ROI {r['roi']:+.3f} → **bloquear o subir threshold**"
            )
    else:
        out.append("- (ninguna — sample insuficiente o bot está OK)")

    out.append("\n## Categorías ganadoras (n≥2, ROI>+0.10)")
    good = [r for r in rows if r["n"] >= 2 and r["roi"] > 0.10]
    if good:
        for r in good:
            out.append(
                f"- **{r['sport']}/{r['market']}/{r['odds']}/{r['outcome']}**: "
                f"{r['won']}W {r['lost']}L ROI {r['roi']:+.3f} → **mantener o bajar threshold**"
            )
    else:
        out.append("- (sample insuficiente)")

    return "\n".join(out)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-04-22")
    ap.add_argument("--out", default="artifacts/error_categories")
    args = ap.parse_args()

    picks = await fetch_settled(args.since)
    if not picks:
        print("Sin picks resueltos en la ventana.")
        return
    report = category_report(picks)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M")
    out_path = out_dir / f"errors_{stamp}.md"
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
