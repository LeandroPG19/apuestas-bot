"""Mapea team_id local → statsbomb team_id en team_external_id.

Sin este mapeo, `compute_team_rolling_from_sb` no puede ser llamado en
runtime (no sabe qué SB id corresponde a un team_id local).

Estrategia:
1. Extrae distinct (sb_team_id, sb_team_name) de statsbomb_events.event_jsonb.
2. Fuzzy match contra teams.name (case-insensitive, normalizado).
3. INSERT en team_external_id source='statsbomb', verified=true cuando
   match exacto, verified=false (ratio 0.85+) cuando aproximado.

Idempotente — usa ON CONFLICT.

Uso:
    apuestas map-statsbomb-teams
"""

from __future__ import annotations

import asyncio
import unicodedata
from difflib import SequenceMatcher

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


def _norm(name: str) -> str:
    n = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode().lower().strip()
    n = n.replace(" fc", "").replace(" cf", "").replace(" sc", "")
    n = " ".join(n.split())
    return n


async def main() -> None:
    async with session_scope() as session:
        sb_rows = (
            await session.execute(
                text(
                    """
                    SELECT DISTINCT team_id AS sb_id, event_jsonb->'team'->>'name' AS sb_name
                    FROM statsbomb_events
                    WHERE event_jsonb->'team'->>'name' IS NOT NULL
                      AND team_id IS NOT NULL
                    """
                )
            )
        ).fetchall()
        local_rows = (
            await session.execute(
                text(
                    """
                    SELECT id, name FROM teams WHERE sport_code = 'soccer'
                    """
                )
            )
        ).fetchall()

    local_by_norm: dict[str, list[tuple[int, str]]] = {}
    for r in local_rows:
        if not r.name:
            continue
        local_by_norm.setdefault(_norm(r.name), []).append((int(r.id), r.name))

    inserted_exact = 0
    inserted_fuzzy = 0
    skipped = 0

    async with session_scope() as session:
        for sb in sb_rows:
            sb_norm = _norm(sb.sb_name)
            local_match = local_by_norm.get(sb_norm)
            if local_match:
                for tid, _ in local_match:
                    await session.execute(
                        text(
                            """
                            INSERT INTO team_external_id (team_id, source, external_id, verified, confidence)
                            VALUES (:tid, 'statsbomb', :sbid, true, 1.0)
                            ON CONFLICT DO NOTHING
                            """
                        ),
                        {"tid": tid, "sbid": str(sb.sb_id)},
                    )
                    inserted_exact += 1
                continue

            best_ratio = 0.0
            best_local: tuple[int, str] | None = None
            for norm_local, candidates in local_by_norm.items():
                ratio = SequenceMatcher(None, sb_norm, norm_local).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_local = candidates[0]

            if best_local and best_ratio >= 0.85:
                tid, _ = best_local
                await session.execute(
                    text(
                        """
                        INSERT INTO team_external_id (team_id, source, external_id, verified, confidence)
                        VALUES (:tid, 'statsbomb', :sbid, false, :conf)
                        ON CONFLICT DO NOTHING
                        """
                    ),
                    {"tid": tid, "sbid": str(sb.sb_id), "conf": float(best_ratio)},
                )
                inserted_fuzzy += 1
            else:
                skipped += 1
        await session.commit()

    print(
        f"✅ statsbomb mapping: exact={inserted_exact} fuzzy={inserted_fuzzy} skipped={skipped} "
        f"(total sb_teams={len(sb_rows)})"
    )


if __name__ == "__main__":
    asyncio.run(main())
