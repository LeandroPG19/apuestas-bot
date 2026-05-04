"""Tests for `_fuse_signals` shrinkage logic in agents.match_analyzer.

Cubre:
- Promedio ponderado básico con pesos uniformes.
- Shrinkage cuadrático cuando la señal diverge >8pp del Pinnacle anchor.
- Fallback pseudo-sharp cuando no hay Pinnacle (umbrales 0.10/0.15).
- Empty / single signal edge cases.
"""

from __future__ import annotations

from apuestas.agents.match_analyzer import SignalProbs, _fuse_signals


def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


def test_empty_signals_returns_empty() -> None:
    assert _fuse_signals([]) == {}


def test_signals_without_common_outcomes_returns_empty() -> None:
    a = SignalProbs(name="a", probs={"home": 0.5, "away": 0.5}, weight=1.0, confidence=1.0)
    b = SignalProbs(name="b", probs={"over": 0.5, "under": 0.5}, weight=1.0, confidence=1.0)
    assert _fuse_signals([a, b]) == {}


def test_single_signal_passthrough() -> None:
    s = SignalProbs(
        name="dixon_coles", probs={"home": 0.6, "away": 0.4}, weight=1.0, confidence=1.0
    )
    out = _fuse_signals([s])
    assert _approx(out["home"], 0.6)
    assert _approx(out["away"], 0.4)


def test_two_signals_no_anchor_no_divergence() -> None:
    """Sin Pinnacle, dos señales con probs casi iguales → promedio ponderado normal."""
    a = SignalProbs(name="dc", probs={"home": 0.55, "away": 0.45}, weight=1.0, confidence=1.0)
    b = SignalProbs(name="llm", probs={"home": 0.50, "away": 0.50}, weight=1.0, confidence=1.0)
    # Δ vs promedio (~0.525) = 0.025 → < 0.10, sin shrinkage
    out = _fuse_signals([a, b])
    assert _approx(out["home"], 0.525)
    assert _approx(out["away"], 0.475)


def test_pinnacle_anchor_blocks_divergent_signal() -> None:
    """Pinnacle anchor + señal divergente >8pp recibe shrinkage cuadrático."""
    pinn = SignalProbs(
        name="pinnacle_devig", probs={"home": 0.50, "away": 0.50}, weight=1.5, confidence=0.85
    )
    diver = SignalProbs(
        name="dixon_coles", probs={"home": 0.80, "away": 0.20}, weight=1.0, confidence=0.7
    )
    out = _fuse_signals([pinn, diver])
    # delta_max = 0.30, shrink = max(0.04, (1 - 0.25*4.5)^2) = (1 - 1.125)^2 → 0.04 floor
    # Pinnacle peso efectivo = 1.5*0.85 = 1.275
    # DC peso efectivo post-shrink = 1.0*0.7 * 0.04 = 0.028
    # Fusion ≈ Pinnacle dominante
    assert out["home"] < 0.55  # cerca del anchor 0.50
    assert out["home"] > 0.49


def test_pseudo_sharp_anchor_when_no_pinnacle() -> None:
    """Sin Pinnacle, dos señales con divergencia >15pp aplica shrinkage."""
    a = SignalProbs(name="dc", probs={"home": 0.30, "away": 0.70}, weight=1.0, confidence=0.7)
    b = SignalProbs(name="llm", probs={"home": 0.70, "away": 0.30}, weight=1.0, confidence=0.7)
    # promedio = 0.50, delta_max = 0.20 → shrink ambas
    out = _fuse_signals([a, b])
    # con shrinkage equivalente, fusion ≈ promedio
    assert _approx(out["home"], 0.5, tol=0.05)


def test_zero_weight_signal_filtered_out() -> None:
    a = SignalProbs(name="dc", probs={"home": 0.6, "away": 0.4}, weight=0.0, confidence=1.0)
    b = SignalProbs(
        name="pinnacle_devig", probs={"home": 0.5, "away": 0.5}, weight=1.0, confidence=1.0
    )
    out = _fuse_signals([a, b])
    assert _approx(out["home"], 0.5)


def test_zero_confidence_signal_filtered_out() -> None:
    a = SignalProbs(name="dc", probs={"home": 0.6, "away": 0.4}, weight=1.0, confidence=0.0)
    b = SignalProbs(
        name="pinnacle_devig", probs={"home": 0.5, "away": 0.5}, weight=1.0, confidence=1.0
    )
    out = _fuse_signals([a, b])
    assert _approx(out["home"], 0.5)


def test_normalization_sums_to_one() -> None:
    a = SignalProbs(
        name="dc", probs={"home": 0.4, "draw": 0.3, "away": 0.3}, weight=1.0, confidence=1.0
    )
    b = SignalProbs(
        name="llm", probs={"home": 0.5, "draw": 0.2, "away": 0.3}, weight=1.0, confidence=1.0
    )
    out = _fuse_signals([a, b])
    assert _approx(sum(out.values()), 1.0)
