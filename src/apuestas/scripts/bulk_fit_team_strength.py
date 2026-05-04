"""Bulk-fit team_strength_bayesian desde matches finalizados.

Re-procesa todos los matches finished de una o varias ligas soccer y aplica
`process_match_settlement` cronológicamente para que `team_strength_bayesian`
refleje el histórico real (no shrinkage uniforme attack=defense=1.0).

Uso:
    apuestas bulk-fit-team-strength --leagues 17,15,18,19
    apuestas bulk-fit-team-strength --all
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.ml.bayesian_dc import process_match_settlement
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def fit_league(league_id: int, *, reset: bool = False) -> dict[str, int]:
    """Fit strength para todos los matches finished de `league_id`.

    Si `reset=True`, primero borra los entries existentes para teams de esa
    liga (evita acumular sobre prior previo).
    """
    async with session_scope() as session:
        if reset:
            await session.execute(
                text(
                    """
                    DELETE FROM team_strength_bayesian
                    WHERE team_id IN (SELECT id FROM teams WHERE league_id = :lid)
                    """
                ),
                {"lid": league_id},
            )

        rows = (
            await session.execute(
                text(
                    """
                    SELECT id FROM matches
                    WHERE sport_code = 'soccer' AND league_id = :lid
                      AND status = 'finished'
                      AND home_score IS NOT NULL AND away_score IS NOT NULL
                    ORDER BY start_time ASC
                    """
                ),
                {"lid": league_id},
            )
        ).fetchall()

    match_ids = [int(r.id) for r in rows]
    processed = 0
    skipped = 0
    for mid in match_ids:
        result = await process_match_settlement(mid)
        if result.get("skipped"):
            skipped += 1
        else:
            processed += 1

    logger.info(
        "bulk_fit_team_strength.league_done",
        league_id=league_id,
        n_matches=len(match_ids),
        processed=processed,
        skipped=skipped,
    )
    return {"league_id": league_id, "processed": processed, "skipped": skipped}


async def main(args: argparse.Namespace) -> None:
    if args.all:
        async with session_scope() as session:
            rows = (
                await session.execute(
                    text(
                        """
                        SELECT DISTINCT league_id FROM matches
                        WHERE sport_code = 'soccer' AND league_id IS NOT NULL
                          AND status = 'finished'
                        """
                    )
                )
            ).fetchall()
            league_ids = [int(r.league_id) for r in rows]
    else:
        league_ids = [int(x.strip()) for x in args.leagues.split(",") if x.strip()]

    print(f"Bulk fitting {len(league_ids)} leagues...")
    for lid in league_ids:
        r = await fit_league(lid, reset=args.reset)
        print(f"  league_id={lid}: processed={r['processed']} skipped={r['skipped']}")
    print("✅ Done")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--leagues", default="", help="CSV league_ids (e.g. '17,15,18,19')")
    p.add_argument("--all", action="store_true")
    p.add_argument(
        "--reset",
        action="store_true",
        help="Borrar strengths previas antes de fittear (recomendado primera vez)",
    )
    asyncio.run(main(p.parse_args()))
