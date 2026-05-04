"""EV threshold adaptativo por (sport, stage) — Mejora 1.

Reemplaza el threshold fijo `settings.betting.ev_threshold` (0.03 global)
por una tabla YAML que considera el deporte y la fase del match (regular,
playoff, postseason). Motivación: los 7 picks del 23 abr revelaron que
picks marginales en MLB (EV 3-4%, modelo sin features) pierden sistemática.

Uso:
    from apuestas.betting.ev_thresholds import ev_threshold_for
    thr = ev_threshold_for(sport="nba", stage="playoff")  # → 0.08
"""

from __future__ import annotations

from pathlib import Path

import yaml

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

_CFG_PATH = Path(__file__).resolve().parents[3] / "config" / "ev_thresholds.yaml"
_CACHE: dict[str, dict[str, float]] | None = None


def _load() -> dict[str, dict[str, float]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    try:
        with _CFG_PATH.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        out: dict[str, dict[str, float]] = {}
        for key, block in raw.items():
            if not isinstance(block, dict):
                continue
            thr = block.get("ev_threshold")
            if thr is None:
                continue
            out[str(key).lower()] = {"ev_threshold": float(thr)}
        _CACHE = out
    except Exception as exc:
        logger.warning("ev_thresholds.load_fail", error=str(exc)[:80])
        _CACHE = {"defaults": {"ev_threshold": 0.03}}
    return _CACHE


def reset_cache() -> None:
    """Para tests que modifican el YAML en runtime."""
    global _CACHE
    _CACHE = None


def ev_threshold_for(
    *,
    sport: str | None,
    stage: str | None = None,
    market: str | None = None,
    league_id: int | None = None,
    fallback: float = 0.03,
) -> float:
    """Umbral EV recomendado.

    Resolución por especificidad descendente:
      {sport}_league_{league_id} → {sport}_{market} → {sport}_{stage} →
      {sport}_playoff → {sport} → defaults → fallback.
    """
    cfg = _load()
    sport_l = (sport or "").lower() or None
    stage_l = (stage or "").lower() or None
    market_l = (market or "").lower() or None

    if sport_l:
        if league_id is not None:
            key_lg = f"{sport_l}_league_{league_id}"
            if key_lg in cfg:
                return float(cfg[key_lg]["ev_threshold"])
        if market_l:
            key_mkt = f"{sport_l}_{market_l}"
            if key_mkt in cfg:
                return float(cfg[key_mkt]["ev_threshold"])
        if stage_l:
            if stage_l in ("playoff", "postseason", "finals"):
                key = f"{sport_l}_playoff"
                if key in cfg:
                    return float(cfg[key]["ev_threshold"])
            key2 = f"{sport_l}_{stage_l}"
            if key2 in cfg:
                return float(cfg[key2]["ev_threshold"])
        if sport_l in cfg:
            return float(cfg[sport_l]["ev_threshold"])

    default_cfg = cfg.get("defaults")
    if default_cfg is not None:
        return float(default_cfg["ev_threshold"])
    return fallback


__all__ = ["ev_threshold_for", "reset_cache"]
