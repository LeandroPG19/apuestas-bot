"""Off-season awareness por deporte (Gap 9 / A12).

Lee `config/sport_seasons.yaml` y expone `is_sport_active(sport, now)`.
El detector evita gastar API calls y LLM en deportes fuera de temporada.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import yaml

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

_CFG_PATH = Path(__file__).resolve().parents[3] / "config" / "sport_seasons.yaml"
_CACHE: dict[str, list[dict[str, int]]] | None = None


def _load() -> dict[str, list[dict[str, int]]]:
    global _CACHE
    if _CACHE is None:
        try:
            with _CFG_PATH.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            _CACHE = {k.lower(): v or [] for k, v in data.items()}
        except Exception as exc:
            logger.warning("season.load_fail", error=str(exc)[:80])
            _CACHE = {}
    return _CACHE


def reset_season_cache() -> None:
    global _CACHE
    _CACHE = None


def is_sport_active(sport: str, now: datetime | None = None) -> bool:
    """True si `sport` está dentro de al menos una ventana activa hoy.

    Si el deporte no está configurado o la config falla al cargarse, asume
    activo (fail-open) para no bloquear picks por data faltante.
    """
    seasons = _load().get(sport.lower())
    if not seasons:
        return True  # fail-open
    month = (now or datetime.now(tz=UTC)).month
    for window in seasons:
        start = int(window.get("start_month", 1))
        end = int(window.get("end_month", 12))
        if start <= end:
            if start <= month <= end:
                return True
        # Wrap around año (ej. NBA Oct-Jun)
        elif month >= start or month <= end:
            return True
    return False


__all__ = ["is_sport_active", "reset_season_cache"]
