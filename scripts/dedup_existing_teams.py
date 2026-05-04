"""Bootstrap dedup: resuelve team_ids inconsistentes entre Sofascore e interno.

Estrategia:
1. Identifica "teams canónicos" (los que tienen rolling stats y nombres oficiales).
2. Para cada team sin rolling stats, busca matching fuzzy canónico con RapidFuzz.
3. Auto-link si score ≥ 92 (más conservador para bootstrap).
4. Las parejas 75-92 van a team_match_review (manual).
5. Para links auto: poblar team_external_id(source='sofascore', external_id=old_id, team_id=canonical).

NO reasigna rolling stats ni borra teams duplicados — solo crea el mapping.
Eso se hará en un paso posterior tras review manual.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from rapidfuzz import fuzz, process
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from apuestas.db import session_scope
from apuestas.ingest.team_resolver import _normalize
from apuestas.obs.logging import configure_logging, get_logger

logger = get_logger(__name__)


async def identify_canonical_teams(sport_code: str) -> dict[int, str]:
    """Teams con rolling stats → son canónicos (tienen historia)."""
    async with session_scope() as session:
        r = await session.execute(
            text(
                """
                SELECT DISTINCT t.id, t.name
                FROM teams t
                WHERE t.sport_code = :sp
                  AND (
                    EXISTS (SELECT 1 FROM team_stats_rolling_home h
                            WHERE h.team_id = t.id AND h.sport_code = :sp)
                    OR EXISTS (SELECT 1 FROM team_stats_rolling_away a
                               WHERE a.team_id = t.id AND a.sport_code = :sp)
                  )
                """
            ),
            {"sp": sport_code},
        )
        return {int(row.id): str(row.name) for row in r.all()}


async def identify_orphan_teams(sport_code: str) -> dict[int, str]:
    """Teams SIN rolling stats pero con matches recientes → candidatos a link."""
    async with session_scope() as session:
        r = await session.execute(
            text(
                """
                SELECT DISTINCT t.id, t.name
                FROM teams t
                WHERE t.sport_code = :sp
                  AND EXISTS (
                    SELECT 1 FROM matches m
                    WHERE m.sport_code = :sp
                      AND (m.home_team_id = t.id OR m.away_team_id = t.id)
                      AND m.start_time > NOW() - INTERVAL '30 days'
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM team_stats_rolling_home h WHERE h.team_id = t.id
                  )
                """
            ),
            {"sp": sport_code},
        )
        return {int(row.id): str(row.name) for row in r.all()}


async def dedup_sport(sport_code: str) -> dict[str, int]:
    """Procesa un sport: retorna {auto_linked, review_queued, no_match}."""
    canonical = await identify_canonical_teams(sport_code)
    orphans = await identify_orphan_teams(sport_code)

    if not canonical:
        logger.warning("dedup.no_canonical", sport=sport_code)
        return {"auto_linked": 0, "review_queued": 0, "no_match": 0}

    logger.info(
        "dedup.start",
        sport=sport_code,
        canonical=len(canonical),
        orphans=len(orphans),
    )

    # Normalize canonical names for fuzzy matching
    canonical_norm = {tid: _normalize(name) for tid, name in canonical.items()}

    counts = {"auto_linked": 0, "review_queued": 0, "no_match": 0}

    for orphan_id, orphan_name in orphans.items():
        query_norm = _normalize(orphan_name)
        best = process.extractOne(
            query_norm,
            canonical_norm,
            scorer=fuzz.WRatio,
            score_cutoff=75,
        )

        if best is None:
            counts["no_match"] += 1
            continue

        _matched_norm, score, canonical_team_id = best
        canonical_name = canonical[canonical_team_id]

        if score >= 92.0:
            # Auto-link: el orphan es un alias del canónico
            async with session_scope() as session:
                await session.execute(
                    text(
                        """
                        INSERT INTO team_external_id
                            (team_id, source, external_id, confidence, verified)
                        VALUES (:tid, 'sofascore', :eid, :c, true)
                        ON CONFLICT (source, external_id) DO UPDATE
                          SET team_id = EXCLUDED.team_id,
                              confidence = EXCLUDED.confidence,
                              verified = true
                        """
                    ),
                    {
                        "tid": canonical_team_id,
                        "eid": str(orphan_id),
                        "c": score / 100.0,
                    },
                )
            logger.info(
                "dedup.auto_linked",
                sport=sport_code,
                orphan=orphan_name,
                canonical=canonical_name,
                score=round(score, 1),
            )
            counts["auto_linked"] += 1
        else:
            # 75-92: review queue
            async with session_scope() as session:
                await session.execute(
                    text(
                        """
                        INSERT INTO team_match_review
                            (source, external_id, external_name, sport_code,
                             candidate_team_id, candidate_name, score, status)
                        VALUES ('sofascore', :eid, :en, :sp, :ctid, :cn, :sc, 'pending')
                        ON CONFLICT (source, external_id) DO NOTHING
                        """
                    ),
                    {
                        "eid": str(orphan_id),
                        "en": orphan_name,
                        "sp": sport_code,
                        "ctid": canonical_team_id,
                        "cn": canonical_name,
                        "sc": score / 100.0,
                    },
                )
            counts["review_queued"] += 1

    return counts


async def main() -> None:
    configure_logging()
    total = {"auto_linked": 0, "review_queued": 0, "no_match": 0}
    for sport in ("nba", "nfl", "mlb", "nhl", "soccer"):
        c = await dedup_sport(sport)
        logger.info("dedup.sport_done", sport=sport, **c)
        for k, v in c.items():
            total[k] += v
    logger.info("dedup.all_done", **total)


if __name__ == "__main__":
    asyncio.run(main())
