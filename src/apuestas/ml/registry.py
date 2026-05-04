"""MLflow model registry: champion/shadow + promote + rollback + model cards.

Flujo según §17.6 del plan:
- Cada modelo nuevo entra como stage='shadow'.
- Corre en paralelo al champion registrando picks en `decision_log`.
- Job semanal `calibration_audit` compara CLV(shadow, 60d) vs CLV(champion, 60d).
- Si shadow > champion + 0.5% con p<0.05 (Wilcoxon) → promote automático.
- Rollback manual siempre disponible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient
from scipy import stats as scipy_stats
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class ModelInfo:
    run_id: str
    model_name: str
    model_version: str
    sport_code: str
    stage: str
    promoted_at: datetime | None
    performance: dict[str, Any]


@dataclass(slots=True)
class PromotionDecision:
    should_promote: bool
    reason: str
    champion_clv: float
    shadow_clv: float
    delta: float
    p_value: float | None
    n_picks: int


def _ensure_mlflow_tracking_uri() -> None:
    """Asegura que mlflow use el server HTTP, no filesystem local.

    Si MLFLOW_TRACKING_URI no está en env, defaults a localhost:5000.
    Defensivo: previene el bug donde un script sin env vars cae a ./mlruns.
    """
    import os as _os

    import mlflow as _mlflow

    uri = _os.environ.get("MLFLOW_TRACKING_URI") or "http://localhost:5000"
    current = _mlflow.get_tracking_uri()
    # Si no es http (es filesystem o none), force set
    if not str(current).startswith("http"):
        _mlflow.set_tracking_uri(uri)

    # Asegurar S3 endpoint para MinIO.
    # Si el valor tiene hostname Docker interno (minio:9000) y estamos fuera de Docker,
    # reemplazar con localhost para que el proceso local pueda conectar al puerto mapeado.
    s3_url = _os.environ.get("MLFLOW_S3_ENDPOINT_URL", "")
    if not s3_url or "minio:9000" in s3_url or "minio:9001" in s3_url:
        _os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
    if not _os.environ.get("AWS_ACCESS_KEY_ID"):
        _os.environ["AWS_ACCESS_KEY_ID"] = "minio-admin"
    if not _os.environ.get("AWS_SECRET_ACCESS_KEY"):
        _os.environ["AWS_SECRET_ACCESS_KEY"] = "change-me-minio-password"  # noqa: S105


def _mlflow_client() -> MlflowClient:
    _ensure_mlflow_tracking_uri()
    return MlflowClient()


# Cache in-memory de modelos production. Key: (sport_code, market).
# Value: (ModelInfo, loaded_object, loaded_at). TTL 1h.
_PRODUCTION_CACHE: dict[tuple[str, str], tuple[ModelInfo, Any, datetime]] = {}
_PRODUCTION_TTL_SECONDS = 3600


_MARKET_TO_MODEL_SUFFIX: dict[str, tuple[str, ...]] = {
    # h2h/moneyline → modelos *_moneyline o *_h2h o *_league_X (soccer)
    "h2h": ("moneyline", "h2h", "1x2", "league_"),
    "moneyline": ("moneyline", "h2h", "1x2", "league_"),
    "1x2": ("moneyline", "h2h", "1x2", "league_"),
    # spreads/handicap → modelos *_spread, *_ats (NFL), *_runline (MLB), *_puckline (NHL)
    "spreads": ("spread", "ats", "runline", "puckline", "ah", "handicap"),
    "handicap": ("spread", "ats", "runline", "puckline", "ah", "handicap"),
    "ah": ("spread", "ats", "runline", "puckline", "ah", "handicap"),
    "runline": ("runline", "spread"),
    "puckline": ("puckline", "spread"),
    # totals → modelos *_total
    "totals": ("total",),
    "total": ("total",),
    "over_under": ("total",),
}


def _market_matches_model(market: str, model_name: str) -> bool:
    """¿El nombre del modelo es compatible con el market solicitado?"""
    name_lower = model_name.lower()
    suffixes = _MARKET_TO_MODEL_SUFFIX.get(market.lower())
    if not suffixes:
        # Market desconocido — permisivo (legacy behavior)
        return True
    return any(suf in name_lower for suf in suffixes)


async def load_production_model(
    sport_code: str,
    market: str = "h2h",
    *,
    model_name_pattern: str | None = None,
) -> tuple[ModelInfo, Any] | None:
    """Carga el modelo `stage='production'` más reciente para (sport, market).

    Gap #3: elimina el hardcode `ensemble_v1/v1` sin modelo real. Si no hay
    modelo production, retorna None y el caller cae a `pinnacle_proxy`.

    Bug fix 2026-04-27: el filtro previo NO consideraba `market` y devolvía
    cualquier modelo de ese sport. Para MLB spreads cargaba `mlb_moneyline` —
    probabilidad de ganar el partido, NO de cubrir el spread → 12 picks MLB
    spreads con overconfidence +33pp y ROI -68%. Fix: filtrar `model_name`
    por sufijo compatible con el market (h2h vs spread vs total).

    Cache in-memory 1h para no golpear MLflow en cada pick.
    """
    _ensure_mlflow_tracking_uri()
    name_pattern = model_name_pattern or f"%{sport_code}%"
    key = (sport_code, market, name_pattern)
    now = datetime.now(tz=UTC)
    cached = _PRODUCTION_CACHE.get(key)
    if cached is not None:
        info, obj, loaded_at = cached
        if (now - loaded_at).total_seconds() < _PRODUCTION_TTL_SECONDS:
            return info, obj
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT mlflow_run_id, model_name, model_version, sport_code,
                       stage, promoted_at, performance_30d
                FROM model_registry_meta
                WHERE stage = 'production'
                  AND sport_code = :sport
                  AND model_name LIKE :np
                ORDER BY promoted_at DESC NULLS LAST
                """
            ),
            {"sport": sport_code, "np": name_pattern},
        )
        rows = result.all()

    # Filtrar por market_type compatible con model_name
    matched_rows = [r for r in rows if _market_matches_model(market, r.model_name)]
    row = matched_rows[0] if matched_rows else None
    if row is None:
        logger.info(
            "registry.no_production_model",
            sport=sport_code,
            market=market,
            n_models_for_sport=len(rows),
        )
        return None

    info = ModelInfo(
        run_id=row.mlflow_run_id,
        model_name=row.model_name,
        model_version=row.model_version,
        sport_code=row.sport_code,
        stage=row.stage,
        promoted_at=row.promoted_at,
        performance=dict(row.performance_30d or {}),
    )

    loaded: Any | None = None
    try:
        import mlflow.sklearn  # type: ignore[import-untyped]

        model_uri = f"runs:/{info.run_id}/model"
        # `mlflow.sklearn.load_model` es sync; envolvemos en to_thread.
        import asyncio as _asyncio

        loaded = await _asyncio.to_thread(mlflow.sklearn.load_model, model_uri)
    except Exception as exc:
        logger.info(
            "registry.mlflow_load_fallback",
            run_id=info.run_id,
            error=str(exc)[:120],
        )
        # Fallback: el artifact se guardó via log_artifact (no log_model), así
        # que falta MLmodel metadata. Descargar pkl directamente y cloudpickle.load.
        try:
            from pathlib import Path as _Path

            import cloudpickle as _cp  # type: ignore[import-untyped]

            path = await _asyncio.to_thread(
                download_artifacts, run_id=info.run_id, artifact_path="model"
            )
            # Buscar el primer .pkl dentro del dir model/
            pkl_files = list(_Path(path).glob("*.pkl"))
            if not pkl_files:
                logger.warning("registry.no_pkl_in_artifacts", path=path[:80])
                return None
            with pkl_files[0].open("rb") as f:
                loaded_raw = _cp.load(f)
            # Distintos trainers guardan schemas distintos:
            #   - {"estimator": sklearn_model, "conformal": ..., ...} (NBA/NFL/MLB trainers)
            #   - {"model": ..., "config": ...} (soccer DC)
            #   - modelo directo (casos legacy)
            if isinstance(loaded_raw, dict):
                if "estimator" in loaded_raw:
                    loaded = loaded_raw  # keep full dict — caller destructures
                elif "model" in loaded_raw:
                    loaded = loaded_raw["model"]
                else:
                    loaded = loaded_raw
            else:
                loaded = loaded_raw
            logger.info(
                "registry.cloudpickle_fallback_ok",
                run_id=info.run_id,
                pkl=str(pkl_files[0])[-60:],
            )
        except Exception as exc2:
            logger.warning(
                "registry.load_model_failed",
                run_id=info.run_id,
                error=str(exc2)[:120],
            )
            return None

    _PRODUCTION_CACHE[key] = (info, loaded, now)
    logger.info(
        "registry.production_model_loaded",
        sport=sport_code,
        market=market,
        model_name=info.model_name,
        version=info.model_version,
    )
    return info, loaded


def clear_production_cache() -> None:
    """Limpia el cache (útil para tests o tras promote manual)."""
    _PRODUCTION_CACHE.clear()


async def register_model(
    *,
    run_id: str,
    model_name: str,
    sport_code: str,
    stage: str = "shadow",
    performance: dict[str, Any] | None = None,
) -> None:
    """Alta en `model_registry_meta`. Idempotente por (mlflow_run_id)."""
    import json as _json

    async with session_scope() as session:
        await session.execute(
            text(
                """
                INSERT INTO model_registry_meta
                  (mlflow_run_id, model_name, model_version, sport_code,
                   stage, promoted_at, performance_30d)
                VALUES
                  (:run_id, :name, :version, :sport, :stage, NOW(), CAST(:perf AS jsonb))
                ON CONFLICT (mlflow_run_id) DO UPDATE
                  SET stage = EXCLUDED.stage,
                      model_version = EXCLUDED.model_version,
                      performance_30d = EXCLUDED.performance_30d
                """
            ),
            {
                "run_id": run_id,
                "name": model_name,
                "version": datetime.now(tz=UTC).strftime("%Y%m%d_%H%M"),
                "sport": sport_code,
                "stage": stage,
                "perf": _json.dumps(performance or {}),
            },
        )
    logger.info("registry.model_registered", run_id=run_id, name=model_name, stage=stage)


async def get_champion(model_name: str) -> ModelInfo | None:
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT mlflow_run_id, model_name, model_version, sport_code,
                       stage, promoted_at, performance_30d
                FROM model_registry_meta
                WHERE model_name = :name AND stage = 'production'
                ORDER BY promoted_at DESC NULLS LAST
                LIMIT 1
                """
            ),
            {"name": model_name},
        )
        row = result.first()
    if row is None:
        return None
    return ModelInfo(
        run_id=row.mlflow_run_id,
        model_name=row.model_name,
        model_version=row.model_version,
        sport_code=row.sport_code,
        stage=row.stage,
        promoted_at=row.promoted_at,
        performance=dict(row.performance_30d or {}),
    )


async def get_shadow(model_name: str) -> ModelInfo | None:
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT mlflow_run_id, model_name, model_version, sport_code,
                       stage, promoted_at, performance_30d
                FROM model_registry_meta
                WHERE model_name = :name AND stage = 'shadow'
                ORDER BY promoted_at DESC NULLS LAST
                LIMIT 1
                """
            ),
            {"name": model_name},
        )
        row = result.first()
    if row is None:
        return None
    return ModelInfo(
        run_id=row.mlflow_run_id,
        model_name=row.model_name,
        model_version=row.model_version,
        sport_code=row.sport_code,
        stage=row.stage,
        promoted_at=row.promoted_at,
        performance=dict(row.performance_30d or {}),
    )


async def list_versions(model_name: str) -> list[ModelInfo]:
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT mlflow_run_id, model_name, model_version, sport_code,
                       stage, promoted_at, performance_30d
                FROM model_registry_meta
                WHERE model_name = :name
                ORDER BY promoted_at DESC NULLS LAST
                """
            ),
            {"name": model_name},
        )
        rows = result.all()
    return [
        ModelInfo(
            run_id=r.mlflow_run_id,
            model_name=r.model_name,
            model_version=r.model_version,
            sport_code=r.sport_code,
            stage=r.stage,
            promoted_at=r.promoted_at,
            performance=dict(r.performance_30d or {}),
        )
        for r in rows
    ]


async def compute_clv_over_window(run_id: str, *, days: int = 60) -> tuple[float, int, list[float]]:
    """CLV promedio de bets tomadas a partir de predictions de este run_id."""
    since = datetime.now(tz=UTC) - timedelta(days=days)
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT b.clv
                FROM bets b
                JOIN predictions p ON p.id = b.prediction_id
                WHERE p.model_version LIKE :run_pattern
                  AND b.placed_at >= :since
                  AND b.clv IS NOT NULL
                """
            ),
            {"run_pattern": f"%{run_id[:12]}%", "since": since},
        )
        clvs = [float(r[0]) for r in result.all()]
    if not clvs:
        return 0.0, 0, []
    return sum(clvs) / len(clvs), len(clvs), clvs


async def evaluate_promotion(
    model_name: str,
    *,
    min_picks: int = 100,
    min_delta: float = 0.005,
    p_threshold: float = 0.05,
    window_days: int = 60,
) -> PromotionDecision:
    """§17.6: decide si shadow debe reemplazar champion.

    Criterios:
    - Shadow tiene ≥ min_picks.
    - Shadow CLV > Champion CLV + min_delta.
    - Wilcoxon signed-rank p < p_threshold (significancia estadística).
    """
    champion = await get_champion(model_name)
    shadow = await get_shadow(model_name)
    if champion is None or shadow is None:
        return PromotionDecision(
            should_promote=False,
            reason="no_champion_or_shadow",
            champion_clv=0.0,
            shadow_clv=0.0,
            delta=0.0,
            p_value=None,
            n_picks=0,
        )

    champ_clv, champ_n, champ_samples = await compute_clv_over_window(
        champion.run_id, days=window_days
    )
    shadow_clv, shadow_n, shadow_samples = await compute_clv_over_window(
        shadow.run_id, days=window_days
    )

    if shadow_n < min_picks:
        return PromotionDecision(
            should_promote=False,
            reason=f"insufficient_shadow_picks_{shadow_n}<{min_picks}",
            champion_clv=champ_clv,
            shadow_clv=shadow_clv,
            delta=shadow_clv - champ_clv,
            p_value=None,
            n_picks=shadow_n,
        )

    delta = shadow_clv - champ_clv
    if delta <= min_delta:
        return PromotionDecision(
            should_promote=False,
            reason=f"delta_too_small_{delta:.4f}<={min_delta}",
            champion_clv=champ_clv,
            shadow_clv=shadow_clv,
            delta=delta,
            p_value=None,
            n_picks=shadow_n,
        )

    # Wilcoxon requiere mismas muestras pareadas. Como champion y shadow
    # evalúan eventos diferentes, usamos Mann-Whitney como aproximación.
    try:
        u_stat, p_value = scipy_stats.mannwhitneyu(
            shadow_samples, champ_samples, alternative="greater"
        )
    except ValueError:
        return PromotionDecision(
            should_promote=False,
            reason="mwhit_failed",
            champion_clv=champ_clv,
            shadow_clv=shadow_clv,
            delta=delta,
            p_value=None,
            n_picks=shadow_n,
        )

    decision = PromotionDecision(
        should_promote=bool(p_value < p_threshold),
        reason="significant" if p_value < p_threshold else f"p_{p_value:.3f}>={p_threshold}",
        champion_clv=champ_clv,
        shadow_clv=shadow_clv,
        delta=delta,
        p_value=float(p_value),
        n_picks=shadow_n,
    )
    logger.info(
        "registry.promotion_eval",
        model=model_name,
        decision=decision.should_promote,
        champion_clv=champ_clv,
        shadow_clv=shadow_clv,
        delta=delta,
        p=p_value,
    )
    return decision


async def promote_shadow(model_name: str, *, dry_run: bool = False) -> bool:
    """Promueve shadow → production. Archiva champion previo."""
    decision = await evaluate_promotion(model_name)
    if not decision.should_promote:
        logger.info("registry.promote.skipped", model=model_name, reason=decision.reason)
        return False
    if dry_run:
        logger.info("registry.promote.dry_run", model=model_name, decision=decision)
        return True

    champion = await get_champion(model_name)
    shadow = await get_shadow(model_name)
    if shadow is None:
        return False

    async with session_scope() as session:
        if champion is not None:
            await session.execute(
                text(
                    """
                    UPDATE model_registry_meta
                    SET stage = 'archived', retired_at = NOW()
                    WHERE mlflow_run_id = :run_id
                    """
                ),
                {"run_id": champion.run_id},
            )
        await session.execute(
            text(
                """
                UPDATE model_registry_meta
                SET stage = 'production', promoted_at = NOW(), promoted_by = 'auto'
                WHERE mlflow_run_id = :run_id
                """
            ),
            {"run_id": shadow.run_id},
        )
    logger.info(
        "registry.promote.done",
        model=model_name,
        new_champion=shadow.run_id,
        old_champion=champion.run_id if champion else None,
    )
    return True


async def rollback_to(model_name: str, *, version_run_id: str) -> bool:
    """Reactiva un run_id histórico como champion. Archiva el champion actual."""
    champion = await get_champion(model_name)
    async with session_scope() as session:
        if champion is not None and champion.run_id != version_run_id:
            await session.execute(
                text(
                    """
                    UPDATE model_registry_meta
                    SET stage = 'archived', retired_at = NOW()
                    WHERE mlflow_run_id = :run_id
                    """
                ),
                {"run_id": champion.run_id},
            )
        result = await session.execute(
            text(
                """
                UPDATE model_registry_meta
                SET stage = 'production', promoted_at = NOW(),
                    promoted_by = 'rollback', retired_at = NULL
                WHERE mlflow_run_id = :run_id AND model_name = :name
                RETURNING 1
                """
            ),
            {"run_id": version_run_id, "name": model_name},
        )
        updated = result.first() is not None

    logger.info(
        "registry.rollback",
        model=model_name,
        to=version_run_id,
        success=updated,
    )
    return updated


def load_model(run_id: str) -> dict[str, Any]:
    """Descarga el artifact del modelo desde MLflow y deserializa pickle."""
    try:
        local_path = download_artifacts(run_id=run_id, artifact_path="model/calibrated_model.pkl")
    except Exception as exc:
        logger.exception("registry.load_failed", run_id=run_id, error=str(exc))
        raise

    with open(local_path, "rb") as f:
        import cloudpickle

        return cloudpickle.load(f)


def generate_model_card(run_id: str) -> str:
    """Genera Markdown con performance, feature list, known limitations.

    §19.16 del plan.
    """
    client = _mlflow_client()
    run = client.get_run(run_id)
    params = run.data.params
    metrics = run.data.metrics
    tags = run.data.tags

    lines = [
        f"# Model Card — {params.get('sport', '?')}/{tags.get('market', '?')}",
        f"**Run ID**: `{run_id}`",
        f"**Created**: {datetime.fromtimestamp(run.info.start_time / 1000, tz=UTC).isoformat()}",
        "",
        "## Training",
        f"- Seasons: {params.get('seasons', '?')}",
        f"- N train / cal / holdout: {params.get('n_train', '?')} / {params.get('n_cal', '?')} / {params.get('n_holdout', '?')}",
        f"- Feature set: `{params.get('feature_set', '?')}` ({params.get('n_features', '?')} features)",
        f"- Optuna trials: {params.get('n_trials', '?')}",
        f"- Random state: {params.get('random_state', '?')}",
        "",
        "## Holdout metrics",
        f"- Log-loss: **{metrics.get('holdout_log_loss', '?'):.4f}** (objetivo ≤ 0.67)",
        f"- Brier score: **{metrics.get('holdout_brier', '?'):.4f}**",
        f"- ECE: **{metrics.get('holdout_ece', '?'):.4f}** (objetivo < 0.03)",
        "",
        "## Known limitations",
        "- Features rolling requieren >10 juegos históricos por equipo.",
        "- Calibración validada sólo para este rango de probabilidad.",
        "- Re-evaluar tras cualquier cambio de reglas de la liga.",
        "",
        "## Status",
        f"- Meets log-loss target: {tags.get('meets_logloss_target', '?')}",
        f"- Meets ECE target: {tags.get('meets_ece_target', '?')}",
        f"- Calibration method: {tags.get('calibration', '?')}",
    ]
    return "\n".join(lines)
