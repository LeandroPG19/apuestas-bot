"""Match fingerprint — Sprint 13 Capa 1.

Calcula hash canónico para deduplicar matches del mismo partido que
distintas fuentes insertaron como rows separadas.

Fingerprint = SHA256(sport_canonical | normalized_home | normalized_away | date_2h_bucket)

Reglas normalización:
- Lowercase + strip espacios
- Remover diacríticos (Sevilla/Sëvilla → sevilla)
- Remover FC/CF/SC/etc suffixes comunes
- Remover paréntesis "(U19)", "(Women)", "(Cyber)"
- Ordenar home/away alfabéticamente para matches sin side (cup neutral)

Date bucket de 2h = match con 1:59h de diferencia = mismo partido.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime
from typing import Any

from apuestas.obs.logging import get_logger
from apuestas.sports import canonical_sport_code

logger = get_logger(__name__)

_SUFFIX_RE = re.compile(
    r"\s+(fc|cf|sc|ac|as|bc|cd|sd|ud|rc|club|de|del|la|los|sport)$", re.IGNORECASE
)
_PREFIX_RE = re.compile(r"^(fc|cf|sc|ac|as|bc|cd|sd|ud|rc|real)\s+", re.IGNORECASE)
_PAREN_RE = re.compile(r"\s*\([^)]*\)\s*")
_MULTISPACE_RE = re.compile(r"\s+")


def normalize_team_name(raw: str) -> str:
    """Normaliza team name para comparación fuzzy."""
    if not raw:
        return ""
    # Strip diacríticos
    s = unicodedata.normalize("NFKD", str(raw)).encode("ASCII", "ignore").decode()
    s = s.lower().strip()
    # Remover paréntesis "(U19)", "(Cyber)", "(W)"
    s = _PAREN_RE.sub(" ", s)
    # Remover suffixes + prefixes FC/CF/Real
    for _ in range(3):
        s = _SUFFIX_RE.sub("", s).strip()
        s = _PREFIX_RE.sub("", s).strip()
    # Remover puntuación residual
    s = re.sub(r"[^\w\s]", "", s)
    s = _MULTISPACE_RE.sub(" ", s).strip()
    return s


def _bucket_time(ts: datetime, window_hours: int = 2) -> datetime:
    """Redondea timestamp al bucket window_hours para match fuzzy de fechas."""
    if ts.tzinfo is None:
        from datetime import UTC

        ts = ts.replace(tzinfo=UTC)
    hours_since_epoch = int(ts.timestamp() / 3600)
    bucket_hours = (hours_since_epoch // window_hours) * window_hours
    from datetime import UTC

    return datetime.fromtimestamp(bucket_hours * 3600, tz=UTC)


def match_fingerprint(
    *,
    sport_code: str,
    home_team: str,
    away_team: str,
    start_time: datetime,
    window_hours: int = 2,
) -> str:
    """Genera fingerprint determinístico para un match.

    Dos matches con el mismo (sport canonical, teams, hora±window) producen
    el mismo fingerprint → dedup.
    """
    canonical = canonical_sport_code(sport_code)
    nh = normalize_team_name(home_team)
    na = normalize_team_name(away_team)
    bucket = _bucket_time(start_time, window_hours).isoformat()
    material = f"{canonical}|{nh}|{na}|{bucket}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


async def upsert_canonical(
    session: Any,
    *,
    match_id: int,
    sport_code: str,
    home_team_id: int,
    away_team_id: int,
    home_team_name: str,
    away_team_name: str,
    start_time: datetime,
) -> tuple[int, bool]:
    """Upsert en `match_canonical`. Devuelve (canonical_id, is_new).

    Si ya existe un canonical para este fingerprint, `match_id` se añade
    a `alternate_match_ids` (lista de duplicados).
    """
    from sqlalchemy import text

    canonical = canonical_sport_code(sport_code)
    fp = match_fingerprint(
        sport_code=sport_code,
        home_team=home_team_name,
        away_team=away_team_name,
        start_time=start_time,
    )
    bucket = _bucket_time(start_time)

    try:
        existing = (
            await session.execute(
                text(
                    """
                    SELECT id, primary_match_id, alternate_match_ids
                    FROM match_canonical WHERE fingerprint = :fp
                    """
                ),
                {"fp": fp},
            )
        ).first()

        if existing is not None:
            # Añadir match_id como alternate si no es el primary
            if int(existing.primary_match_id) != match_id:
                alts = list(existing.alternate_match_ids or [])
                if match_id not in alts:
                    alts.append(match_id)
                    import json as _json

                    await session.execute(
                        text(
                            "UPDATE match_canonical SET alternate_match_ids = CAST(:alts AS jsonb), "
                            "updated_at = NOW() WHERE id = :cid"
                        ),
                        {"alts": _json.dumps(alts), "cid": existing.id},
                    )
            return int(existing.id), False

        # Nuevo canonical
        r = await session.execute(
            text(
                """
                INSERT INTO match_canonical (
                    fingerprint, primary_match_id, sport_code_canonical,
                    home_team_id, away_team_id, start_time_bucket
                ) VALUES (:fp, :mid, :sport, :ht, :at, :bucket)
                RETURNING id
                """
            ),
            {
                "fp": fp,
                "mid": match_id,
                "sport": canonical,
                "ht": home_team_id,
                "at": away_team_id,
                "bucket": bucket,
            },
        )
        new_id = int(r.scalar() or 0)
        return new_id, True
    except Exception as exc:
        logger.debug("fingerprint.upsert_fail", error=str(exc)[:100])
        return 0, False


__all__ = [
    "match_fingerprint",
    "normalize_team_name",
    "upsert_canonical",
]
