"""Prometheus metrics export (Deuda 2).

Expone los KPIs primarios del bot para que Prometheus los scrape:

    apuestas_pick_brier{sport=...}          Gauge — Brier rolling por pick resuelto
    apuestas_pick_ece{sport=...}            Gauge — ECE rolling
    apuestas_pick_bss{sport=...}            Gauge — Brier Skill Score
    apuestas_hit_rate_minus_implied{sport=} Gauge
    apuestas_alerts_new_total{sport=...}    Counter
    apuestas_alerts_upgrade_total{sport=...}Counter
    apuestas_alerts_skip_total{sport=...}   Counter
    apuestas_drift_detected_total{sport,market} Counter
    apuestas_last_alert_emit_timestamp_seconds Gauge
    apuestas_live_scores_updated_total{sport=} Counter

Las alerting rules (`config/prometheus/alerting_rules.yml`) referencian
estos métricos. Si `prometheus_client` no está disponible, todas las
operaciones son no-ops (fail-safe).
"""

from __future__ import annotations

import time
from typing import Any

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

try:
    from prometheus_client import (  # type: ignore[import-untyped]
        CollectorRegistry,
        Counter,
        Gauge,
        generate_latest,
    )

    _PROM_AVAILABLE = True
except ImportError:  # pragma: no cover
    CollectorRegistry = None  # type: ignore[assignment,misc]
    Counter = None  # type: ignore[assignment,misc]
    Gauge = None  # type: ignore[assignment,misc]

    def generate_latest(*_a: Any, **_kw: Any) -> bytes:  # type: ignore[misc]
        return b""

    _PROM_AVAILABLE = False


_registry: Any = None
_metrics: dict[str, Any] = {}


def _get_registry() -> Any:
    global _registry
    if _registry is None and _PROM_AVAILABLE and CollectorRegistry is not None:
        _registry = CollectorRegistry()
    return _registry


def _metric(kind: str, name: str, documentation: str, labels: list[str]) -> Any:
    if not _PROM_AVAILABLE:
        return None
    if name in _metrics:
        return _metrics[name]
    registry = _get_registry()
    if kind == "counter":
        m = Counter(name, documentation, labelnames=labels, registry=registry)
    elif kind == "gauge":
        m = Gauge(name, documentation, labelnames=labels, registry=registry)
    else:
        return None
    _metrics[name] = m
    return m


# ──────────── Calibration KPIs ────────────


def set_pick_brier(sport: str, value: float) -> None:
    g = _metric("gauge", "apuestas_pick_brier", "Rolling Brier por sport", ["sport"])
    if g is not None:
        g.labels(sport=sport).set(float(value))


def set_pick_ece(sport: str, value: float) -> None:
    g = _metric("gauge", "apuestas_pick_ece", "Rolling ECE por sport", ["sport"])
    if g is not None:
        g.labels(sport=sport).set(float(value))


def set_pick_bss(sport: str, value: float) -> None:
    g = _metric("gauge", "apuestas_pick_bss", "Rolling Brier Skill Score", ["sport"])
    if g is not None:
        g.labels(sport=sport).set(float(value))


def set_hit_rate_minus_implied(sport: str, value: float) -> None:
    g = _metric(
        "gauge",
        "apuestas_hit_rate_minus_implied",
        "Hit rate - implied rate por sport",
        ["sport"],
    )
    if g is not None:
        g.labels(sport=sport).set(float(value))


# ──────────── Emit counters ────────────


def inc_alerts_new(sport: str, n: int = 1) -> None:
    c = _metric("counter", "apuestas_alerts_new_total", "Alertas nuevas emitidas", ["sport"])
    if c is not None:
        c.labels(sport=sport).inc(n)
    set_last_alert_emit()


def inc_alerts_upgrade(sport: str, n: int = 1) -> None:
    c = _metric(
        "counter",
        "apuestas_alerts_upgrade_total",
        "Alertas upgraded (nueva mejor odds)",
        ["sport"],
    )
    if c is not None:
        c.labels(sport=sport).inc(n)


def inc_alerts_skip(sport: str, n: int = 1) -> None:
    c = _metric(
        "counter",
        "apuestas_alerts_skip_total",
        "Alertas skipped (ruido/cooldown)",
        ["sport"],
    )
    if c is not None:
        c.labels(sport=sport).inc(n)


def inc_drift_detected(sport: str, market: str) -> None:
    c = _metric(
        "counter",
        "apuestas_drift_detected_total",
        "Drift events por (sport, market)",
        ["sport", "market"],
    )
    if c is not None:
        c.labels(sport=sport, market=market).inc()


def inc_live_scores_updated(sport: str, n: int = 1) -> None:
    c = _metric(
        "counter",
        "apuestas_live_scores_updated_total",
        "Matches con score actualizado",
        ["sport"],
    )
    if c is not None:
        c.labels(sport=sport).inc(n)


def set_last_alert_emit(ts: float | None = None) -> None:
    g = _metric(
        "gauge",
        "apuestas_last_alert_emit_timestamp_seconds",
        "Timestamp epoch del último pick emitido (watchdog 4h)",
        [],
    )
    if g is not None:
        g.set(ts if ts is not None else time.time())


def set_clv_rolling(sport: str, days: int, value: float) -> None:
    """CLV rolling avg pct (e.g. +1.5 = +1.5% avg)."""
    g = _metric(
        "gauge",
        "apuestas_clv_rolling_pct",
        "CLV rolling avg %, positivo = skill (Buchdahl 2023)",
        ["sport", "window_days"],
    )
    if g is not None:
        g.labels(sport=sport, window_days=str(days)).set(value)


def set_clv_positive_ratio(sport: str, days: int, value: float) -> None:
    """Fracción de picks con CLV+ (0-1)."""
    g = _metric(
        "gauge",
        "apuestas_clv_positive_ratio",
        "Fracción picks con CLV > 0 rolling window",
        ["sport", "window_days"],
    )
    if g is not None:
        g.labels(sport=sport, window_days=str(days)).set(value)


def inc_closing_lines_captured(sport: str, n: int = 1) -> None:
    c = _metric(
        "counter",
        "apuestas_closing_lines_captured_total",
        "Closing lines Pinnacle snapshot capturadas",
        ["sport"],
    )
    if c is not None:
        c.labels(sport=sport).inc(n)


def render_metrics() -> bytes:
    """Para `/metrics` endpoint de FastAPI."""
    registry = _get_registry()
    if registry is None:
        return b""
    return bytes(generate_latest(registry))


__all__ = [
    "inc_alerts_new",
    "inc_alerts_skip",
    "inc_alerts_upgrade",
    "inc_drift_detected",
    "inc_live_scores_updated",
    "render_metrics",
    "set_hit_rate_minus_implied",
    "set_last_alert_emit",
    "set_pick_brier",
    "set_pick_bss",
    "set_pick_ece",
]
