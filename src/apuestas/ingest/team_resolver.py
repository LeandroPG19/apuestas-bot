"""Resolver de team_id cross-source vía RapidFuzz + tabla team_external_id.

Flujo de resolución (hot-path en ingesta):
1. Hit directo `team_external_id (source, external_id)` → canonical team_id.
2. Si no hit: fuzzy match sobre `teams.name` filtrado por sport_code.
   - score ≥ 95 → auto-link (guarda en team_external_id, verified=true).
   - 75 ≤ score < 95 → encola en team_match_review, retorna None.
   - score < 75 → retorna None (el caller crea team nuevo si procede).

Implementado con `rapidfuzz` (C++ backend, 10-100x más rápido que thefuzz).
"""

from __future__ import annotations

import unicodedata

from rapidfuzz import fuzz, process
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

# Thresholds calibrados: ≥95 auto, ≥75 review, <75 new candidate.
AUTO_LINK_THRESHOLD = 95.0
REVIEW_THRESHOLD = 75.0


def _normalize(name: str) -> str:
    """Lowercase + strip accents + collapse whitespace + remove punctuation."""
    import re

    s = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("ASCII")
    s = re.sub(r"[^a-zA-Z0-9 ]", " ", s).lower()
    return re.sub(r"\s+", " ", s).strip()


async def _fetch_direct(source: str, external_id: str) -> int | None:
    async with session_scope() as session:
        r = await session.execute(
            text("SELECT team_id FROM team_external_id WHERE source = :s AND external_id = :e"),
            {"s": source, "e": str(external_id)},
        )
        row = r.first()
    return int(row.team_id) if row else None


async def _fetch_candidates(sport_code: str) -> dict[int, str]:
    async with session_scope() as session:
        r = await session.execute(
            text("SELECT id, name FROM teams WHERE sport_code = :sp"),
            {"sp": sport_code},
        )
        return {int(row.id): str(row.name) for row in r.all()}


async def _upsert_external_id(
    source: str,
    external_id: str,
    team_id: int,
    *,
    confidence: float,
    verified: bool,
) -> None:
    async with session_scope() as session:
        await session.execute(
            text(
                """
                INSERT INTO team_external_id
                    (team_id, source, external_id, confidence, verified)
                VALUES (:tid, :s, :e, :c, :v)
                ON CONFLICT (source, external_id) DO UPDATE
                  SET team_id = EXCLUDED.team_id,
                      confidence = EXCLUDED.confidence,
                      verified = EXCLUDED.verified
                """
            ),
            {"tid": team_id, "s": source, "e": str(external_id), "c": confidence, "v": verified},
        )


async def _enqueue_review(
    source: str,
    external_id: str,
    external_name: str,
    sport_code: str,
    *,
    candidate_team_id: int | None,
    candidate_name: str | None,
    score: float,
) -> None:
    async with session_scope() as session:
        await session.execute(
            text(
                """
                INSERT INTO team_match_review
                    (source, external_id, external_name, sport_code,
                     candidate_team_id, candidate_name, score, status)
                VALUES (:s, :e, :n, :sp, :ctid, :cn, :sc, 'pending')
                ON CONFLICT (source, external_id) DO NOTHING
                """
            ),
            {
                "s": source,
                "e": str(external_id),
                "n": external_name,
                "sp": sport_code,
                "ctid": candidate_team_id,
                "cn": candidate_name,
                "sc": score / 100.0,
            },
        )


async def resolve_team_id(
    *,
    source: str,
    external_id: str,
    external_name: str,
    sport_code: str,
    auto_link_threshold: float = AUTO_LINK_THRESHOLD,
    review_threshold: float = REVIEW_THRESHOLD,
) -> int | None:
    """Resuelve un team_id interno desde (source, external_id, name).

    Returns `int` si hay hit directo o fuzzy ≥ auto_link_threshold.
    Returns `None` si fuzzy 75-95 (encolado para review) o < 75 (candidato new).
    """
    direct = await _fetch_direct(source, external_id)
    if direct is not None:
        return direct

    candidates = await _fetch_candidates(sport_code)
    if not candidates:
        return None

    normalized_query = _normalize(external_name)
    # Build normalized candidate map
    norm_map = {tid: _normalize(name) for tid, name in candidates.items()}

    # RapidFuzz process.extractOne sobre normalized strings
    best = process.extractOne(
        normalized_query,
        norm_map,
        scorer=fuzz.WRatio,
        score_cutoff=review_threshold,
    )

    if best is None:
        logger.debug(
            "team_resolver.no_match",
            source=source,
            external_name=external_name,
            sport=sport_code,
        )
        return None

    matched_norm_name, score, team_id = best
    candidate_name = candidates[team_id]

    if score >= auto_link_threshold:
        await _upsert_external_id(
            source, external_id, team_id, confidence=score / 100.0, verified=True
        )
        logger.info(
            "team_resolver.auto_link",
            source=source,
            external_name=external_name,
            matched=candidate_name,
            score=round(score, 1),
        )
        return team_id

    # Review queue
    await _enqueue_review(
        source,
        external_id,
        external_name,
        sport_code,
        candidate_team_id=team_id,
        candidate_name=candidate_name,
        score=score,
    )
    logger.info(
        "team_resolver.review_enqueued",
        source=source,
        external_name=external_name,
        candidate=candidate_name,
        score=round(score, 1),
    )
    return None


async def link_external_id(
    *, source: str, external_id: str, team_id: int, confidence: float = 1.0
) -> None:
    """Link manual: usado por admin CLI tras aprobar review."""
    await _upsert_external_id(source, external_id, team_id, confidence=confidence, verified=True)
