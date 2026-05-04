"""Alert store — one-alert-per-identity con upgrade adaptativo por deporte.

Post-pivote 2026-04-23: reemplaza el dedup ad-hoc que antes vivía en
`persist_allocations`. Decide, para cada `ValueBet` recién detectado, si:
  - 'new'     → no existe alerta viva con su identidad → crear
  - 'upgrade' → existe + odds/EV mejoran según umbral adaptativo → actualizar
  - 'skip'    → existe y la diferencia es ruido de mercado → decision_log

Los umbrales por deporte viven en `config/upgrade_thresholds.yaml`.
La identidad de un pick se define como (match_id, market, line, outcome).
La tabla `pick_alerts` tiene unique index parcial sobre esa tupla
(migración 0021), y otro sobre (match, market, line) sin outcome
(migración 0022) — que impide home+away simultáneos.

Referencias:
  - plan §4.1-§4.4 (Sprint 2: Alert store + confidence)
  - migration 0020 añadió columnas best_odds_seen, best_odds_book,
    best_odds_updated_at, upgrade_count, last_alert_at, outcome_result.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import yaml
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from apuestas.betting.detector import ValueBet
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

Decision = Literal["new", "upgrade", "skip"]

_CFG_PATH = Path(__file__).resolve().parents[3] / "config" / "upgrade_thresholds.yaml"


@dataclass(frozen=True, slots=True)
class UpgradeConfig:
    """Umbrales por deporte. `from_yaml` valida presencia de `defaults`."""

    ev_delta_min_pp: float  # pp (ej. 1.5 = +0.015 en EV)
    odds_delta_min: float  # decimal (ej. 0.07 = precio sube de 1.92 a 1.99+)
    cooldown_min: int  # minutos desde last_alert_at para permitir upgrade


_CFG_CACHE: dict[str, UpgradeConfig] | None = None


def _load_thresholds_file(path: Path = _CFG_PATH) -> dict[str, UpgradeConfig]:
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if "defaults" not in raw:
        msg = f"upgrade_thresholds.yaml debe incluir bloque 'defaults': {path}"
        raise ValueError(msg)
    out: dict[str, UpgradeConfig] = {}
    for sport, cfg in raw.items():
        out[sport] = UpgradeConfig(
            ev_delta_min_pp=float(cfg["ev_delta_min_pp"]),
            odds_delta_min=float(cfg["odds_delta_min"]),
            cooldown_min=int(cfg["cooldown_min"]),
        )
    return out


def get_upgrade_config(sport: str | None) -> UpgradeConfig:
    """Devuelve la config del deporte o `defaults` si no existe.

    Cacheado por proceso; para releer (p. ej. en tests) invocar
    `reset_upgrade_config_cache()`.
    """
    global _CFG_CACHE
    if _CFG_CACHE is None:
        _CFG_CACHE = _load_thresholds_file()
    key = (sport or "").lower()
    return _CFG_CACHE.get(key) or _CFG_CACHE["defaults"]


def reset_upgrade_config_cache() -> None:
    """Para tests que modifican el YAML en tiempo de ejecución."""
    global _CFG_CACHE
    _CFG_CACHE = None


async def should_emit_or_upgrade(
    session: AsyncSession,
    pick: ValueBet,
    *,
    now: datetime | None = None,
) -> Decision:
    """Consulta `pick_alerts` y decide la acción sobre el pick entrante.

    Esta función NO hace el UPDATE/INSERT — sólo retorna la decisión. El
    llamador (emit_alerts) es responsable de aplicarla, manteniendo la
    transacción bajo su control.

    Args:
        session: AsyncSession activa.
        pick: ValueBet recién detectado.
        now: para tests; default `datetime.now(UTC)`.

    Returns:
        'new'     → no hay alerta viva con esa identidad
        'upgrade' → hay alerta viva y odds nuevas superan umbrales
        'skip'    → hay alerta viva y el cambio no justifica re-alertar
    """
    now_ts = now or datetime.now(tz=UTC)
    cfg = get_upgrade_config(pick.sport_code)

    row = (
        await session.execute(
            text(
                """
                SELECT id, odds_placed, best_odds_seen, bookmaker,
                       upgrade_count, last_alert_at, prediction_id
                FROM pick_alerts
                WHERE match_id = :mid
                  AND market = :mk
                  AND COALESCE(line, -999) = COALESCE(:ln, -999)
                  AND outcome = :oc
                  AND (outcome_result IS NULL OR outcome_result = 'pending')
                LIMIT 1
                """
            ),
            {
                "mid": pick.event_id,
                "mk": pick.market,
                "ln": pick.line,
                "oc": pick.outcome,
            },
        )
    ).first()

    if row is None:
        return "new"

    current_best = float(row.best_odds_seen or row.odds_placed)
    new_odds = float(pick.odds)

    # 1) Odds no mejoran lo suficiente → skip.
    odds_delta = new_odds - current_best
    if odds_delta < cfg.odds_delta_min:
        _log_skip(pick, row.id, "odds_delta_below_threshold", odds_delta, cfg)
        return "skip"

    # 2) EV nuevo vs EV del mejor previo. Aproximamos EV_prev con
    #    best_odds_seen * p_blended_nuevo - 1 (usamos la misma p para
    #    mantener la comparación entre odds, no entre modelos).
    p = float(pick.p_blended)
    ev_new = p * new_odds - 1.0
    ev_prev = p * current_best - 1.0
    ev_delta_pp = (ev_new - ev_prev) * 100.0
    if ev_delta_pp < cfg.ev_delta_min_pp:
        _log_skip(pick, row.id, "ev_delta_below_threshold", ev_delta_pp, cfg)
        return "skip"

    # 3) Cooldown: protege contra flapping entre books dentro de una ventana
    #    corta (el mercado suele cruzar de un book a otro en segundos).
    last = row.last_alert_at
    if last is not None:
        elapsed = (now_ts - last).total_seconds() / 60.0
        if elapsed < cfg.cooldown_min:
            _log_skip(pick, row.id, "cooldown_active", elapsed, cfg)
            return "skip"

    return "upgrade"


def _log_skip(
    pick: ValueBet,
    alert_id: int,
    reason: str,
    measured: float,
    cfg: UpgradeConfig,
) -> None:
    logger.info(
        "alert_store.skip",
        alert_id=alert_id,
        match_id=pick.event_id,
        market=pick.market,
        outcome=pick.outcome,
        sport=pick.sport_code,
        reason=reason,
        measured=round(float(measured), 4),
        ev_delta_min_pp=cfg.ev_delta_min_pp,
        odds_delta_min=cfg.odds_delta_min,
        cooldown_min=cfg.cooldown_min,
    )


async def mark_upgrade(
    session: AsyncSession,
    alert_id: int,
    *,
    new_odds: float,
    bookmaker: str,
    now: datetime | None = None,
) -> None:
    """Aplica el upgrade (UPDATE de best_odds_* + upgrade_count++)."""
    now_ts = now or datetime.now(tz=UTC)
    await session.execute(
        text(
            """
            UPDATE pick_alerts
            SET best_odds_seen = :best,
                best_odds_book = :bk,
                best_odds_updated_at = :ts,
                upgrade_count = upgrade_count + 1,
                last_alert_at = :ts
            WHERE id = :id
            """
        ),
        {
            "id": alert_id,
            "best": float(new_odds),
            "bk": bookmaker,
            "ts": now_ts,
        },
    )


def time_since_last(last_alert_at: datetime | None, now: datetime | None = None) -> timedelta:
    """Utilidad pública para tests/telemetría."""
    if last_alert_at is None:
        return timedelta.max
    return (now or datetime.now(tz=UTC)) - last_alert_at


__all__ = [
    "Decision",
    "UpgradeConfig",
    "get_upgrade_config",
    "mark_upgrade",
    "reset_upgrade_config_cache",
    "should_emit_or_upgrade",
    "time_since_last",
]
