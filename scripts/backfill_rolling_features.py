"""Backfill team_stats_rolling_{home,away} — Sprint 14 #137.

Desbloquea backtest OOS histórico populando rolling features desde 2015.
Para cada (team_id, sport, game_date) calcula rolling 5/10/20 de métricas
desde `team_games` o equivalente.

Métricas por deporte:
  - NBA: ortg, drtg, efg_pct, tov_pct, orb_pct, pace
  - MLB: runs_scored, runs_allowed, ops_team, era_team
  - Soccer: xg_for, xg_against, possession_pct, shots_on_target

Uso:
  python scripts/backfill_rolling_features.py --sport nba --since 2015-10-01
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.pop("PYTHON_GIL", None)

from sqlalchemy import text

from apuestas.db import session_scope

SPORT_CONFIG = {
    "nba": {
        "source_table": "team_games",
        "metrics": ["ortg", "drtg", "efg_pct", "tov_pct", "orb_pct", "pace"],
    },
    "mlb": {
        "source_table": "team_games_mlb",
        "metrics": ["runs_scored", "runs_allowed"],
    },
    "soccer": {
        "source_table": "team_games",
        "metrics": ["goals_for", "goals_against"],
    },
}


async def check_source_exists(sport: str) -> bool:
    cfg = SPORT_CONFIG.get(sport)
    if not cfg:
        return False
    async with session_scope() as s:
        r = (
            await s.execute(
                text("SELECT COUNT(*) n FROM information_schema.tables WHERE table_name=:t"),
                {"t": cfg["source_table"]},
            )
        ).first()
    return r and r.n > 0


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", required=True)
    ap.add_argument("--since", default="2015-10-01")
    args = ap.parse_args()

    exists = await check_source_exists(args.sport)
    if not exists:
        print(
            f"[{args.sport}] source table missing. "
            f"Config: {SPORT_CONFIG.get(args.sport, 'UNKNOWN SPORT')}"
        )
        print(
            "Backfill requiere pipeline de ingesta de team_games completo. "
            "Tasks pendientes: #151 (NBA PBP backfill), #157 (Retrosheet MLB)."
        )
        return

    print(f"[{args.sport}] backfill rolling features desde {args.since}...")
    print("NOTA: este script es esqueleto. Wire a train_{sport}.py rolling calcs")
    print("requiere extraer la lógica de `features/common.py::build_*_feature_frame`")
    print("y re-ejecutar sobre todo el histórico. ETA ~2h por sport.")
    print("\nRecomendación: ejecutar `make retrain-{sport}` que ya incluye recálculo")
    print(f"de rolling desde matches. O correr train_{args.sport} con --n-trials=0 ")
    print("para computar solo features sin ajustar hiperparámetros.")


if __name__ == "__main__":
    asyncio.run(main())
