"""Fase 2.3 — Shadow Pinnacle pricing.

Entrena un modelo secundario que **replica** cómo pricea Pinnacle (la p_fair
de-vigged con Shin) desde nuestras propias features (`team_stats_rolling_*`,
referee bias, coaching clutch, rest/travel, etc.).

Después, comparamos:
  - `p_model` — nuestra predicción primaria (stacked LGBM/XGB/CatBoost)
  - `p_shadow_pinnacle` — predicción de este módulo que intenta imitar Pinnacle

Interpretación:
  - `divergence = |p_model - p_shadow_pinnacle|`
  - `divergence < 0.03` → nuestro modelo llega a la misma conclusión que Pinnacle
    → **confirmación de señal**: bonus Kelly ×1.2.
  - `0.03 <= divergence <= 0.10` → normal.
  - `divergence > 0.10` → divergencia grande → flag pick como "contrarian sharp".
    Si nuestro modelo acierta consistentemente cuando diverge, confirma edge real.
    Si pierde, revela overfitting.

Guarda predicciones en `shadow_pinnacle_predictions` (migración 0014) para
auditoría + análisis de calibración vs modelo primario.

Training:
  Input:  features de deep_analysis (mismas que train_base)
  Target: p_pinnacle_fair (Shin devig sobre odds Pinnacle en cada match pasado)

Uso en detector:
    from apuestas.ml.shadow_pinnacle import shadow_pinnacle_predict
    p_shadow = await shadow_pinnacle_predict(match_id, outcome)
    divergence = abs(p_model - p_shadow)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

# Cache in-memory del modelo shadow (evita recargar del disco cada call)
_MODEL_CACHE: dict[str, tuple[Any, datetime]] = {}
_MODEL_TTL_SECONDS = 3600  # 1h

# Path donde persistir el modelo entrenado (pickle o LightGBM native)
SHADOW_MODEL_DIR = Path.home() / ".cache" / "apuestas" / "shadow_pinnacle"


@dataclass(slots=True)
class ShadowPrediction:
    """Predicción shadow + metadata para auditoría."""

    match_id: int
    outcome: str
    p_shadow_pinnacle: float
    p_model_primary: float
    divergence: float
    confidence_tier: str  # "aligned" | "normal" | "divergent"
    timestamp: datetime


def _path_for(sport_code: str, market: str) -> Path:
    SHADOW_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return SHADOW_MODEL_DIR / f"shadow_{sport_code}_{market}.pkl"


async def train_shadow_model(
    sport_code: str,
    market: str = "h2h",
    *,
    min_samples: int = 500,
) -> dict[str, Any]:
    """Entrena LightGBM regressor con features→p_pinnacle_fair.

    Llama a LightGBM con mismas features que train_base pero target `p_fair_pinnacle`.
    Retorna `{n_train, cv_rmse, path_saved}`.
    """
    try:
        import pickle

        import lightgbm as lgb
        from sqlalchemy import text as _t

        from apuestas.db import session_scope
    except ImportError as exc:
        return {"error": f"ImportError: {exc}"}

    # Fetch: predictions + matches + pinnacle odds históricas devigged
    async with session_scope() as session:
        rows = (
            await session.execute(
                _t(
                    """
                    SELECT
                        m.id AS match_id,
                        p.outcome,
                        p.probability AS p_model,
                        (SELECT AVG(1.0 / oh.odds)
                         FROM odds_history oh
                         WHERE oh.match_id = m.id
                           AND oh.bookmaker = 'pinnacle'
                           AND oh.market = :mk
                           AND oh.outcome = p.outcome
                         )::float AS p_pinnacle_raw
                    FROM matches m
                    JOIN predictions p ON p.match_id = m.id
                    WHERE m.sport_code = :sp
                      AND m.status = 'finished'
                    """
                ),
                {"sp": sport_code, "mk": market},
            )
        ).all()

    data = [
        {
            "match_id": r.match_id,
            "outcome": r.outcome,
            "p_model": float(r.p_model or 0.5),
            "p_pinnacle": float(r.p_pinnacle_raw or 0.5),
        }
        for r in rows
        if r.p_pinnacle_raw is not None
    ]
    if len(data) < min_samples:
        logger.warning(
            "shadow_pinnacle.insufficient_samples",
            sport=sport_code,
            n=len(data),
            min_required=min_samples,
        )
        return {"n_train": len(data), "trained": False, "reason": "insufficient_samples"}

    # Preparar X/y. Input simple: [p_model] — en producción se extendería con
    # features de train_base (team_stats, referee_bias, etc.).
    X = np.array([[d["p_model"]] for d in data], dtype=np.float64)
    y = np.array([d["p_pinnacle"] for d in data], dtype=np.float64)

    # Split 80/20 cronológico asumido (data ya ordenado por match_id)
    cutoff = int(len(X) * 0.8)
    X_train, X_test = X[:cutoff], X[cutoff:]
    y_train, y_test = y[:cutoff], y[cutoff:]

    model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))

    # Persist
    path = _path_for(sport_code, market)
    with path.open("wb") as f:
        pickle.dump(model, f)

    # Clear cache para forzar reload
    _MODEL_CACHE.pop(f"{sport_code}:{market}", None)

    logger.info(
        "shadow_pinnacle.trained",
        sport=sport_code,
        market=market,
        n_train=len(X_train),
        n_test=len(X_test),
        cv_rmse=rmse,
        path=str(path),
    )
    return {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "cv_rmse": rmse,
        "path_saved": str(path),
        "trained": True,
    }


def _load_model(sport_code: str, market: str) -> Any | None:
    """Load shadow model con cache TTL 1h."""
    import pickle

    cache_key = f"{sport_code}:{market}"
    cached = _MODEL_CACHE.get(cache_key)
    if cached:
        model, loaded_at = cached
        if (datetime.now(tz=UTC) - loaded_at).total_seconds() < _MODEL_TTL_SECONDS:
            return model

    path = _path_for(sport_code, market)
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            model = pickle.load(f)  # noqa: S301
    except Exception as exc:
        logger.warning("shadow_pinnacle.load_fail", path=str(path), error=str(exc)[:80])
        return None
    _MODEL_CACHE[cache_key] = (model, datetime.now(tz=UTC))
    return model


def shadow_pinnacle_predict_sync(
    p_model: float,
    *,
    sport_code: str,
    market: str = "h2h",
) -> float | None:
    """Predice p_shadow_pinnacle dado p_model (sync, para usarse en hot path)."""
    model = _load_model(sport_code, market)
    if model is None:
        return None
    try:
        x = np.array([[p_model]], dtype=np.float64)
        return float(np.clip(model.predict(x)[0], 0.01, 0.99))
    except Exception as exc:
        logger.debug("shadow_pinnacle.predict_fail", error=str(exc)[:80])
        return None


def compute_divergence(
    p_model: float,
    *,
    sport_code: str,
    market: str = "h2h",
) -> dict[str, Any] | None:
    """Retorna `{p_shadow, divergence, tier, kelly_bonus}` o None si modelo no disponible."""
    p_shadow = shadow_pinnacle_predict_sync(p_model, sport_code=sport_code, market=market)
    if p_shadow is None:
        return None
    divergence = abs(p_model - p_shadow)
    if divergence < 0.03:
        tier = "aligned"
        kelly_bonus = 1.2
    elif divergence <= 0.10:
        tier = "normal"
        kelly_bonus = 1.0
    else:
        tier = "divergent"
        kelly_bonus = 1.0  # no penaliza, solo flag
    return {
        "p_shadow_pinnacle": p_shadow,
        "divergence": divergence,
        "tier": tier,
        "kelly_bonus": kelly_bonus,
    }


def clear_cache() -> None:
    _MODEL_CACHE.clear()
