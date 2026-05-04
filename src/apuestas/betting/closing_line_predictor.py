"""Closing line predictor — Sprint 11 Fase C.

Buchdahl 2023 "Betting Smart": predecir el cierre (Pinnacle closing de-vigged)
es más skillful que predecir el resultado. El cierre agrega la info de todo
el mercado; anticiparlo es el objetivo real del value betting.

Modelo: Ridge regression sobre features de line movement (últimas 4h) +
sharp book migration + public betting percentage + sport/league dummies.

Target: Pinnacle closing price de-vigged (Shin o Power dependiendo de market).

Uso:
    pred = await predict_closing_odds(
        match_id=123,
        market="h2h",
        outcome="home",
        current_odds=1.95,
        hours_until_start=3,
    )
    # pred = {"expected_closing": 1.88, "clv_anticipated": +0.035}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class ClosingLineFeatures:
    """Features computadas desde odds_history + market signals."""

    current_odds: float
    line_movement_4h: float  # Δ odds últimas 4h (positivo = drift up)
    line_movement_1h: float  # Δ odds última hora
    n_updates_4h: int  # frecuencia de cambios
    n_books_tracking: int
    sharp_book_consensus: float  # Pinnacle+Circa+Polymarket avg
    public_pct: float  # % de tickets al lado (0-1), default 0.5 si no data
    hours_until_start: float
    sport_code: str
    league_id: int | None


@dataclass(slots=True)
class ClosingLinePredictor:
    """Ridge predictor simple; entrenado con líneas históricas.

    Features normalizadas + log-odds transform para estabilidad numérica.
    Suficiente para capturar tendencia sin sobre-fittear por deporte.
    """

    sport: str
    coef_: np.ndarray | None = field(default=None, init=False)
    intercept_: float = field(default=0.0, init=False)
    _feature_mean: np.ndarray | None = field(default=None, init=False)
    _feature_std: np.ndarray | None = field(default=None, init=False)

    FEATURE_ORDER: tuple[str, ...] = (
        "log_current_odds",
        "line_movement_4h",
        "line_movement_1h",
        "n_updates_4h",
        "sharp_book_consensus_delta",
        "public_pct",
        "hours_until_start_log",
    )

    def _vectorize(self, f: ClosingLineFeatures) -> np.ndarray:
        log_odds = float(np.log(max(f.current_odds, 1.001)))
        sharp_delta = float(f.sharp_book_consensus - f.current_odds)
        hrs_log = float(np.log(max(f.hours_until_start, 0.1)))
        return np.array(
            [
                log_odds,
                f.line_movement_4h,
                f.line_movement_1h,
                f.n_updates_4h,
                sharp_delta,
                f.public_pct,
                hrs_log,
            ],
            dtype=float,
        )

    def fit(self, X: list[ClosingLineFeatures], y_closing: list[float]) -> ClosingLinePredictor:
        """Ajusta Ridge regression target=log(closing_odds).

        Normalización z-score por feature para estabilidad.
        """
        if not X:
            raise ValueError("ClosingLinePredictor.fit: X vacío")
        mat = np.array([self._vectorize(f) for f in X])
        y = np.log(np.clip(np.array(y_closing, dtype=float), 1.001, None))
        self._feature_mean = mat.mean(axis=0)
        self._feature_std = mat.std(axis=0) + 1e-8
        mat_z = (mat - self._feature_mean) / self._feature_std
        # Ridge closed-form
        lam = 1.0
        A = mat_z.T @ mat_z + lam * np.eye(mat_z.shape[1])
        b = mat_z.T @ (y - y.mean())
        self.coef_ = np.linalg.solve(A, b)
        self.intercept_ = float(y.mean())
        logger.info(
            "closing_line_predictor.fit",
            sport=self.sport,
            n=len(X),
            coef_nonzero=int(np.sum(np.abs(self.coef_) > 1e-4)),
        )
        return self

    def predict(self, f: ClosingLineFeatures) -> float:
        """Devuelve odds esperadas al cierre."""
        if self.coef_ is None or self._feature_mean is None or self._feature_std is None:
            # Sin entrenar: heurística = current_odds con leve drift hacia sharp
            return 0.7 * f.current_odds + 0.3 * f.sharp_book_consensus
        x = self._vectorize(f)
        x_z = (x - self._feature_mean) / self._feature_std
        log_closing = float(self.intercept_ + x_z @ self.coef_)
        return float(np.exp(log_closing))

    def anticipated_clv(self, f: ClosingLineFeatures) -> float:
        """CLV anticipado = (current − expected_closing) / expected_closing.

        Positivo = tomamos odds mejores que el cierre (esperado) = +CLV.
        """
        exp_closing = self.predict(f)
        if exp_closing <= 0:
            return 0.0
        return (f.current_odds - exp_closing) / exp_closing


async def extract_features_for_match(
    *,
    session: Any,
    match_id: int,
    market: str,
    outcome: str,
    current_odds: float,
    sharp_book_consensus: float,
    hours_until_start: float,
    sport_code: str,
    league_id: int | None = None,
    public_pct: float = 0.5,
) -> ClosingLineFeatures:
    """Extrae features desde odds_history en DB.

    Requiere session SQLAlchemy async + tabla odds_history con:
    (match_id, market, outcome, bookmaker, odds, ts).
    """
    from sqlalchemy import text

    try:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT odds, ts, bookmaker
                    FROM odds_history
                    WHERE match_id = :mid AND market = :mkt AND outcome = :out
                      AND ts >= NOW() - INTERVAL '4 hours'
                    ORDER BY ts ASC
                    """
                ),
                {"mid": match_id, "mkt": market, "out": outcome},
            )
        ).fetchall()
    except Exception as exc:
        logger.warning("closing_line.extract_fail", error=str(exc)[:100])
        rows = []

    odds_4h = [float(r[0]) for r in rows if r[0] is not None]
    line_mv_4h = (odds_4h[-1] - odds_4h[0]) if len(odds_4h) >= 2 else 0.0
    # 1h window
    import datetime as _dt

    now_utc = _dt.datetime.now(_dt.UTC)
    odds_1h = [float(r[0]) for r in rows if r[1] and r[1] >= now_utc - _dt.timedelta(hours=1)]
    line_mv_1h = (odds_1h[-1] - odds_1h[0]) if len(odds_1h) >= 2 else 0.0
    n_books = len({r[2] for r in rows if r[2]})

    return ClosingLineFeatures(
        current_odds=current_odds,
        line_movement_4h=line_mv_4h,
        line_movement_1h=line_mv_1h,
        n_updates_4h=len(odds_4h),
        n_books_tracking=n_books,
        sharp_book_consensus=sharp_book_consensus,
        public_pct=public_pct,
        hours_until_start=hours_until_start,
        sport_code=sport_code,
        league_id=league_id,
    )


_LOADED_PREDICTORS: dict[str, ClosingLinePredictor] = {}


def load_fitted_predictor(sport: str) -> ClosingLinePredictor:
    """Lazy-load ClosingLinePredictor fiteado desde artifacts/.

    Si no existe archivo fit, devuelve predictor sin entrenar (heurístico).
    Cache en memoria por sport para evitar reloads.
    """
    sport_l = sport.lower()
    if sport_l in _LOADED_PREDICTORS:
        return _LOADED_PREDICTORS[sport_l]

    try:
        import pickle
        from pathlib import Path

        path = (
            Path(__file__).resolve().parents[3]
            / "artifacts"
            / "closing_line_predictor"
            / f"{sport_l}.pkl"
        )
        if path.exists():
            with path.open("rb") as f:
                predictor = pickle.load(f)  # noqa: S301 — artifact trusted (local)
            _LOADED_PREDICTORS[sport_l] = predictor
            logger.info("clp.loaded_from_artifact", sport=sport_l, path=str(path))
            return predictor
    except Exception as exc:
        logger.debug("clp.load_fail", sport=sport_l, error=str(exc)[:80])

    # Fallback: predictor sin entrenar (heurística)
    predictor = ClosingLinePredictor(sport=sport_l)
    _LOADED_PREDICTORS[sport_l] = predictor
    return predictor


__all__ = [
    "ClosingLineFeatures",
    "ClosingLinePredictor",
    "extract_features_for_match",
    "load_fitted_predictor",
]
