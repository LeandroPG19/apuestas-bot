"""Concept drift detector por (sport, market) — Sprint 7 + Deuda 1.

Plan §9.1 usa `river.drift.ADWIN` (Bifet & Gavalda 2007) cuando está
disponible; fallback a Page-Hinkley local si `river` no instalado.

Integración:
  - `mark_alert_results` invoca `drift_monitor.update(p_pred, y_true)` por
    cada alerta que se resuelve.
  - Si drift_detected, se dispara un Prefect event `concept_drift_{sport}_{market}`
    que el deployment `retrain_on_drift` escucha.
  - Cooldown 24h entre retrains por (sport, market) (R5 del plan).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover — dep opcional
    from river.drift import ADWIN as _RiverADWIN  # type: ignore[import-untyped]  # noqa: N811

    _RIVER_AVAILABLE = True
except ImportError:
    _RiverADWIN = None  # type: ignore[assignment,misc]
    _RIVER_AVAILABLE = False


@dataclass(slots=True)
class PageHinkleyDetector:
    """Detector Page-Hinkley one-sided (upper).

    Detecta incrementos sostenidos de la métrica observada respecto a la
    media histórica. Parámetros estándar para Brier residuales:
      - delta = 0.005 (sensibilidad; más bajo = más sensible)
      - lambda_threshold = 50.0 (umbral acumulado)
      - alpha = 0.9999 (factor de olvido; 1.0 = sin olvido)
    """

    delta: float = 0.005
    lambda_threshold: float = 50.0
    alpha: float = 0.9999

    n: int = 0
    mean: float = 0.0
    m_t: float = 0.0  # suma acumulada ajustada
    min_m_t: float = 0.0

    def update(self, value: float) -> bool:
        """Devuelve True si se detecta drift ascendente."""
        self.n += 1
        # Running mean con factor de olvido.
        self.mean = self.alpha * self.mean + (1.0 - self.alpha) * value
        if self.n < 30:
            # Precalentamiento: evita falsos positivos con pocas muestras.
            return False
        x = value - self.mean - self.delta
        self.m_t += x
        self.min_m_t = min(self.min_m_t, self.m_t)
        ph = self.m_t - self.min_m_t
        return ph > self.lambda_threshold

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m_t = 0.0
        self.min_m_t = 0.0


def _build_detector() -> Any:
    """Factoría: usa river.drift.ADWIN si está disponible, sino Page-Hinkley.

    ADWIN es más sensible a cambios locales; Page-Hinkley cubre el caso
    offline puro. Ambos comparten la interfaz update(float) → bool.
    """
    if _RIVER_AVAILABLE and _RiverADWIN is not None:
        return _RiverAdapter(_RiverADWIN(delta=0.002))
    return PageHinkleyDetector()


class _RiverAdapter:
    """Adapta `river.drift.ADWIN` a la interfaz `update(val) → bool`."""

    __slots__ = ("_adwin", "n")

    def __init__(self, adwin: Any) -> None:
        self._adwin = adwin
        self.n: int = 0

    def update(self, value: float) -> bool:
        self.n += 1
        self._adwin.update(float(value))
        return bool(getattr(self._adwin, "drift_detected", False))

    def reset(self) -> None:
        if _RiverADWIN is not None:
            self._adwin = _RiverADWIN(delta=0.002)
        self.n = 0


@dataclass(slots=True)
class DriftState:
    detector: Any = field(default_factory=_build_detector)
    last_drift_at: datetime | None = None
    last_retrain_at: datetime | None = None


class BrierDriftMonitor:
    """Monitor global por `(sport, market)`. Singleton por proceso.

    Usage:
        monitor = BrierDriftMonitor.get()
        drifted = monitor.update("nba", "h2h", pred=0.6, actual=1)
        if drifted: ...  # dispara retrain
    """

    _instance: BrierDriftMonitor | None = None
    RETRAIN_COOLDOWN = timedelta(hours=24)

    def __init__(self) -> None:
        self._state: dict[tuple[str, str], DriftState] = {}

    @classmethod
    def get(cls) -> BrierDriftMonitor:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _key(self, sport: str, market: str) -> tuple[str, str]:
        return ((sport or "").lower(), (market or "").lower())

    def update(
        self,
        sport: str,
        market: str,
        *,
        pred: float,
        actual: int,
    ) -> bool:
        """Alimenta el detector con un brier residual (p - y)².

        Retorna True si se detecta drift Y ha pasado el cooldown.
        """
        key = self._key(sport, market)
        state = self._state.setdefault(key, DriftState())
        residual = float((pred - actual) ** 2)
        drifted = state.detector.update(residual)
        if not drifted:
            return False

        now = datetime.now(tz=UTC)
        if state.last_retrain_at is not None:
            since = now - state.last_retrain_at
            if since < self.RETRAIN_COOLDOWN:
                logger.info(
                    "drift.cooldown_active",
                    sport=sport,
                    market=market,
                    since_hours=round(since.total_seconds() / 3600, 2),
                )
                return False

        state.last_drift_at = now
        state.last_retrain_at = now
        # Reset post-drift: evita disparar continuamente.
        state.detector.reset()
        logger.warning(
            "drift.detected",
            sport=sport,
            market=market,
            n=state.detector.n,
        )
        try:
            from apuestas.obs.metrics import inc_drift_detected

            inc_drift_detected(sport, market)
        except Exception:
            pass
        return True

    def snapshot(self) -> dict[tuple[str, str], dict[str, Any]]:
        return {
            key: {
                "n": st.detector.n,
                "mean": st.detector.mean,
                "last_drift_at": st.last_drift_at.isoformat() if st.last_drift_at else None,
                "last_retrain_at": st.last_retrain_at.isoformat() if st.last_retrain_at else None,
            }
            for key, st in self._state.items()
        }


async def trigger_retrain_event(sport: str, market: str) -> None:
    """Emite evento Prefect para `retrain_on_drift` deployment.

    Fail-safe: si Prefect no está disponible o no hay deployment,
    sólo logea. Sprint 7b añade el deployment con schedule event-driven.
    """
    try:
        from prefect.events import emit_event

        emit_event(
            event=f"concept_drift.{sport}.{market}",
            resource={"prefect.resource.id": f"apuestas.drift.{sport}.{market}"},
        )
        logger.info("drift.event_emitted", sport=sport, market=market)
    except Exception as exc:
        logger.debug("drift.event_emit_fail", error=str(exc)[:100])


__all__ = [
    "BrierDriftMonitor",
    "DriftState",
    "PageHinkleyDetector",
    "trigger_retrain_event",
]
