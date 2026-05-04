"""B3 complemento: bootstrap priors desde ClubElo para teams no cubiertos por FBref.

Flujo:
1. Fetch snapshot ClubElo del día (CSV gratuito sin auth).
2. Para cada club, resolver team_id interno via team_resolver (sport='soccer'
   o específico de liga).
3. Convertir Elo → (attack_rating, defense_rating) vía `elo_to_dc_prior()`.
4. Upsert `team_strength_bayesian` SOLO si el team_id aún no tiene prior
   (prioridad FBref > ClubElo; Elo es proxy más débil).

Variance=0.15 (vs 0.10 FBref) porque Elo no es fit directo de goles.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from apuestas.db import session_scope
from apuestas.ingest.clubelo import ClubEloClient, elo_to_dc_prior
from apuestas.ingest.team_resolver import resolve_team_id
from apuestas.obs.logging import configure_logging, get_logger

logger = get_logger(__name__)


async def team_has_prior(team_id: int) -> bool:
    async with session_scope() as session:
        r = await session.execute(
            text("SELECT 1 FROM team_strength_bayesian WHERE team_id = :tid LIMIT 1"),
            {"tid": team_id},
        )
        return r.first() is not None


async def upsert_prior(
    *, team_id: int, attack: float, defense: float, variance: float, n_matches: int
) -> None:
    async with session_scope() as session:
        await session.execute(
            text(
                """
                INSERT INTO team_strength_bayesian
                    (team_id, attack_rating, defense_rating, variance, n_matches, updated_at)
                VALUES (:tid, :a, :d, :v, :n, NOW())
                ON CONFLICT (team_id) DO NOTHING
                """
            ),
            {"tid": team_id, "a": attack, "d": defense, "v": variance, "n": n_matches},
        )


async def main() -> None:
    configure_logging()
    client = ClubEloClient()
    async with client.session():
        snapshot = await client.fetch_snapshot()
    logger.info("clubelo.snapshot_fetched", n_clubs=len(snapshot))

    resolved = 0
    inserted = 0
    skipped_has_prior = 0

    for club in snapshot:
        if not club["club"]:
            continue
        # Intentar resolver con sport=soccer (la mayoría de clubes ClubElo son europeos).
        try:
            team_id = await resolve_team_id(
                source="clubelo",
                external_id=club["club"],
                external_name=club["club"],
                sport_code="soccer",
                auto_link_threshold=90.0,  # conservador: nombres ClubElo compactos
            )
        except Exception as exc:
            logger.debug("clubelo.resolve_fail", club=club["club"], error=str(exc)[:100])
            continue

        if team_id is None:
            # Retry con sport específico si matchea por country
            country_to_sport = {
                "ENG": "epl",
                "ESP": "laliga",
                "GER": "bundesliga",
                "ITA": "seriea",
                "FRA": "ligue1",
                "MEX": "liga_mx",
            }
            alt_sport = country_to_sport.get(club["country"])
            if alt_sport:
                try:
                    team_id = await resolve_team_id(
                        source="clubelo",
                        external_id=club["club"],
                        external_name=club["club"],
                        sport_code=alt_sport,
                        auto_link_threshold=90.0,
                    )
                except Exception:
                    team_id = None
            if team_id is None:
                continue
        resolved += 1

        # Skip si ya hay prior (FBref tiene prioridad — más preciso)
        if await team_has_prior(team_id):
            skipped_has_prior += 1
            continue

        attack, defense = elo_to_dc_prior(club["elo"])
        try:
            await upsert_prior(
                team_id=team_id,
                attack=attack,
                defense=defense,
                variance=0.15,
                n_matches=10,  # Elo proxy: menos "evidencia sintética" que FBref
            )
            inserted += 1
        except Exception as exc:
            logger.warning("clubelo.upsert_fail", team_id=team_id, error=str(exc)[:100])

    logger.info(
        "clubelo.done",
        n_total=len(snapshot),
        resolved=resolved,
        inserted=inserted,
        skipped_has_prior=skipped_has_prior,
    )


if __name__ == "__main__":
    asyncio.run(main())
