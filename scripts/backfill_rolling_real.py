"""Backfill real de team_stats_rolling_{home,away} — Fase B #137.

Schema existente:
  (team_id, sport_code, window_size) PK
  metrics json — payload con todas las métricas rolling del team

Este script:
  1. Carga matches finalizados con scores por sport
  2. Para cada team, calcula últimas N={5,10,20} games stats
  3. UPSERT en team_stats_rolling_{home,away} con metrics jsonb

Métricas incluidas:
  win_margin_mean, goals_for_mean, goals_against_mean, total_points_mean,
  win_rate, last_match_date.

Uso:
  python scripts/backfill_rolling_real.py --sport nba
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.pop("PYTHON_GIL", None)

from sqlalchemy import text

from apuestas.db import session_scope


async def load_team_games(sport: str) -> list[dict]:
    """Para cada team, sus matches como home y como away por separado."""
    async with session_scope() as s:
        rows = (
            await s.execute(
                text(
                    """
                    SELECT m.id, m.start_time, m.home_team_id, m.away_team_id,
                           m.home_score::float hs, m.away_score::float as_
                    FROM matches m
                    WHERE m.sport_code=:sp AND m.home_score IS NOT NULL
                      AND m.away_score IS NOT NULL
                    ORDER BY m.start_time
                    """
                ),
                {"sp": sport},
            )
        ).fetchall()
    return [dict(r._mapping) for r in rows]


def compute_team_window_stats(
    rows: list[dict], team_id: int, side: str, windows: tuple[int, ...] = (5, 10, 20)
) -> dict[int, dict]:
    """Para un team y side (home/away), calcula stats sobre últimas N games.

    Retorna {window_size: metrics_dict}.
    """
    filtered = [
        r
        for r in rows
        if (side == "home" and r["home_team_id"] == team_id)
        or (side == "away" and r["away_team_id"] == team_id)
    ]
    if not filtered:
        return {}

    # Cronológico descendente — tomamos las últimas N
    filtered.sort(key=lambda r: r["start_time"], reverse=True)

    out: dict[int, dict] = {}
    for w in windows:
        sample = filtered[:w]
        if not sample:
            continue
        goals_for = [float(r["hs"] if side == "home" else r["as_"]) for r in sample]
        goals_against = [float(r["as_"] if side == "home" else r["hs"]) for r in sample]
        margins = [gf - ga for gf, ga in zip(goals_for, goals_against, strict=False)]
        totals = [gf + ga for gf, ga in zip(goals_for, goals_against, strict=False)]
        wins = sum(1 for m in margins if m > 0)
        out[w] = {
            "goals_for_mean": round(sum(goals_for) / len(goals_for), 3),
            "goals_against_mean": round(sum(goals_against) / len(goals_against), 3),
            "win_margin_mean": round(sum(margins) / len(margins), 3),
            "total_points_mean": round(sum(totals) / len(totals), 3),
            "win_rate": round(wins / len(sample), 3),
            "sample_size": len(sample),
            "last_match": sample[0]["start_time"].isoformat()
            if hasattr(sample[0]["start_time"], "isoformat")
            else str(sample[0]["start_time"]),
        }
    return out


async def upsert_team_rolling(
    sport: str, team_id: int, side: str, windows_data: dict[int, dict]
) -> int:
    if not windows_data:
        return 0
    table = f"team_stats_rolling_{side}"
    n = 0
    async with session_scope() as s:
        for w, metrics in windows_data.items():
            try:
                await s.execute(
                    text(
                        f"""
                        INSERT INTO {table}
                          (team_id, sport_code, window_size, metrics, sample_size, last_computed)
                        VALUES (:tid, :sp, :w, CAST(:m AS json), :sz, NOW())
                        ON CONFLICT (team_id, sport_code, window_size) DO UPDATE SET
                          metrics = EXCLUDED.metrics,
                          sample_size = EXCLUDED.sample_size,
                          last_computed = NOW()
                        """
                    ),
                    {
                        "tid": team_id,
                        "sp": sport,
                        "w": w,
                        "m": json.dumps(metrics),
                        "sz": metrics.get("sample_size", 0),
                    },
                )
                n += 1
            except Exception as exc:
                print(f"Upsert fail team={team_id} w={w}: {str(exc)[:80]}")
    return n


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", required=True)
    args = ap.parse_args()

    rows = await load_team_games(args.sport)
    print(f"[{args.sport}] matches loaded: {len(rows)}")
    if not rows:
        return

    teams = sorted({r["home_team_id"] for r in rows} | {r["away_team_id"] for r in rows})
    print(f"[{args.sport}] unique teams: {len(teams)}")

    n_home = n_away = 0
    for i, team_id in enumerate(teams):
        home_stats = compute_team_window_stats(rows, team_id, "home")
        away_stats = compute_team_window_stats(rows, team_id, "away")
        n_home += await upsert_team_rolling(args.sport, team_id, "home", home_stats)
        n_away += await upsert_team_rolling(args.sport, team_id, "away", away_stats)
        if (i + 1) % 10 == 0:
            print(f"  progress: {i + 1}/{len(teams)} teams done")

    print(f"[{args.sport}] rolling rows: home={n_home} away={n_away}")


if __name__ == "__main__":
    asyncio.run(main())
