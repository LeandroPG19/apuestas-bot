"""Sport focus mode — Sprint 14.

Gate emit + retrain por deporte desde `config/enabled_sports.yaml`.
Permite desactivar deportes sin eliminar código.
"""

from __future__ import annotations

import os
from datetime import UTC, date, datetime
from pathlib import Path

import yaml

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

_CFG_PATH = Path(__file__).resolve().parents[3] / "config" / "enabled_sports.yaml"
_CACHE: dict[str, dict] | None = None


def _load() -> dict[str, dict]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    try:
        with _CFG_PATH.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        _CACHE = {
            "emit_enabled": raw.get("emit_enabled", {}),
            "retrain_enabled": raw.get("retrain_enabled", {}),
            "scheduled_reactivation": raw.get("scheduled_reactivation", {}),
            "odds_api_disabled_keys": set(raw.get("odds_api_disabled_keys", []) or []),
        }
    except Exception as exc:
        logger.warning("sport_focus.load_fail", error=str(exc)[:80])
        # Fail-safe: permitir todos si YAML falla
        _CACHE = {
            "emit_enabled": dict.fromkeys(
                ("mlb", "nba", "soccer", "nhl", "tennis", "nfl", "boxing", "mma"), True
            ),
            "retrain_enabled": dict.fromkeys(
                ("mlb", "nba", "soccer", "nhl", "tennis", "nfl", "boxing", "mma"), True
            ),
            "scheduled_reactivation": {},
            "odds_api_disabled_keys": set(),
        }
    return _CACHE


def reset_cache() -> None:
    global _CACHE
    _CACHE = None


def _check_scheduled(sport: str, cfg: dict) -> bool:
    """Si hay fecha scheduled y ya pasó → override a enabled."""
    sched = cfg.get("scheduled_reactivation", {})
    if sport not in sched:
        return False
    try:
        reactivate = date.fromisoformat(str(sched[sport]))
        today = datetime.now(tz=UTC).date()
        return today >= reactivate
    except Exception:
        return False


def _env_disable_list() -> set[str]:
    raw = os.environ.get("APUESTAS_DISABLE_SPORTS", "")
    if not raw:
        return set()
    return {s.strip().lower() for s in raw.split(",") if s.strip()}


def is_emit_enabled(sport_code: str | None) -> bool:
    """True si el bot debe emitir picks para este sport."""
    if not sport_code:
        return False
    from apuestas.sports import canonical_sport_code

    canonical = canonical_sport_code(sport_code)
    # Env override tiene prioridad máxima
    if canonical in _env_disable_list():
        return False
    cfg = _load()
    if _check_scheduled(canonical, cfg):
        return True  # scheduled reactivation override
    return bool(cfg["emit_enabled"].get(canonical, False))


def is_retrain_enabled(sport_code: str | None) -> bool:
    if not sport_code:
        return False
    from apuestas.sports import canonical_sport_code

    canonical = canonical_sport_code(sport_code)
    if canonical in _env_disable_list():
        return False
    cfg = _load()
    if _check_scheduled(canonical, cfg):
        return True
    return bool(cfg["retrain_enabled"].get(canonical, False))


def is_odds_api_key_disabled(sport_key: str) -> bool:
    """True si este sport_key específico (ej. `wnba`, `soccer_china_superleague`)
    está en la block-list de Odds API, aunque su sport canónico esté habilitado.

    Override granular: permite tener `soccer: true` global pero excluir ligas
    secundarias que queman créditos sin generar picks.
    """
    if not sport_key:
        return False
    cfg = _load()
    return sport_key in cfg.get("odds_api_disabled_keys", set())


__all__ = [
    "is_emit_enabled",
    "is_odds_api_key_disabled",
    "is_retrain_enabled",
    "reset_cache",
]
