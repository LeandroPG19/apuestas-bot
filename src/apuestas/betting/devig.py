"""De-vigging (remove margin) — Multiplicative + Power + Shin.

Blueprint §7: es CRÍTICO usar el método correcto. Shin (1991/1993) asume
insider trading en el bookmaker y suele ser el más preciso para líneas
sharp (Pinnacle/Circa). Power (iterative) balancea favoritos vs. underdogs.
Multiplicative es simple pero sesga hacia favoritos.

Para el pipeline de value bet detection usamos Shin como default y Power
como fallback si la convergencia numérica falla.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.optimize import brentq

Method = Literal["multiplicative", "power", "shin"]


def overround(odds: list[float] | np.ndarray) -> float:
    """Margen bruto del bookmaker: Σ (1/odds_i) − 1."""
    arr = np.asarray(odds, dtype=np.float64)
    if (arr <= 1.0).any():
        msg = f"Odds decimal deben ser >1, got {arr}"
        raise ValueError(msg)
    return float(np.sum(1.0 / arr) - 1.0)


def multiplicative(odds: list[float] | np.ndarray) -> np.ndarray:
    """p_fair = (1/odds) / Σ (1/odds). Método más simple pero sesgado."""
    arr = np.asarray(odds, dtype=np.float64)
    if (arr <= 1.0).any():
        msg = "Odds decimal deben ser >1"
        raise ValueError(msg)
    implied = 1.0 / arr
    return implied / implied.sum()


def power(odds: list[float] | np.ndarray, *, tolerance: float = 1e-8) -> np.ndarray:
    """Power method: encontrar k tal que Σ (1/odds)^k = 1.

    Ajusta la suma preservando relaciones no-lineales, menos sesgado que
    multiplicative. Root-finding con brentq.
    """
    arr = np.asarray(odds, dtype=np.float64)
    if (arr <= 1.0).any():
        msg = "Odds decimal deben ser >1"
        raise ValueError(msg)
    implied = 1.0 / arr

    def f(k: float) -> float:
        return float(np.sum(implied**k) - 1.0)

    # Con margen positivo, sum(implied) > 1 → k > 1; con <1 necesitamos k < 1
    if abs(f(1.0)) < tolerance:
        return implied

    try:
        if f(1.0) > 0:
            k = brentq(f, 1.0, 5.0, xtol=tolerance, maxiter=200)
        else:
            k = brentq(f, 0.2, 1.0, xtol=tolerance, maxiter=200)
    except ValueError:
        # Sin convergencia, fallback a multiplicative
        return implied / implied.sum()

    return implied**k


def shin(
    odds: list[float] | np.ndarray,
    *,
    tolerance: float = 1e-10,
    max_iter: int = 1000,
) -> np.ndarray:
    """Shin method (1991, 1993): asume insider trading z ∈ [0,1].

    Fórmula: p_i = (sqrt(z² + 4*(1-z)*b_i²/sum_b) - z) / (2*(1-z))
    donde b_i = 1/odds_i.

    Root-find z tal que sum(p_i) = 1. Estándar de-vigging en literatura
    sharp (Buchdahl, Miller&Davidow).
    """
    arr = np.asarray(odds, dtype=np.float64)
    if (arr <= 1.0).any():
        msg = "Odds decimal deben ser >1"
        raise ValueError(msg)
    b = 1.0 / arr
    sum_b = b.sum()

    if sum_b <= 1.0:
        # Margen no positivo; caer a multiplicative normalizado
        return b / sum_b

    def _p_given_z(z: float) -> np.ndarray:
        denom = 2.0 * (1.0 - z)
        if abs(denom) < 1e-15:
            return b / sum_b
        numer = np.sqrt(np.maximum(z**2 + 4.0 * (1.0 - z) * b**2 / sum_b, 0.0)) - z
        return numer / denom

    def f(z: float) -> float:
        return float(_p_given_z(z).sum() - 1.0)

    try:
        # z ∈ [0, 1); en práctica <0.15 para Pinnacle
        z_star = brentq(f, 1e-10, 0.9, xtol=tolerance, maxiter=max_iter)
    except ValueError:
        # Sin convergencia, fallback a power
        return power(arr)

    p = _p_given_z(z_star)
    # Normalización defensiva (errores numéricos)
    return p / p.sum()


def devig(odds: list[float] | np.ndarray, *, method: Method = "shin") -> np.ndarray:
    """Dispatcher. Retorna probabilidades justas suma=1."""
    if method == "multiplicative":
        return multiplicative(odds)
    if method == "power":
        return power(odds)
    if method == "shin":
        return shin(odds)
    msg = f"Método devig desconocido: {method}"
    raise ValueError(msg)


def select_devig_method(
    *,
    market: str,
    n_outcomes: int,
    overround_value: float | None = None,
) -> Method:
    """Selección adaptativa del método de devig (plan §7.6 / Sprint 5 G3).

    Decisión basada en literatura:
      - 2-way moneyline/spreads/totals → **Power** (Clarke, Kovalchik & Ingram
        2017): mejor manejo del favorite-longshot bias sin sobre-corregir.
        Antes el default era Shin para 2-way, lo cual sub-corregía ~2-3pp
        en MLB lopsided lines.
      - 3-way (soccer 1X2) / futures / outright → **Shin** (Shin 1993):
        diseñado para mercados con posibilidad de insider trading.
      - overround alto retail (> 7%) → **Power** igualmente pero el caller
        debe saber que la incertidumbre es alta (retail books suelen tener
        hold 8-10%). No downgrade a multiplicative porque ese asume sharp.
      - overround muy bajo (≤ 3%, típico Pinnacle/Circa) → **Multiplicative**
        (los 3 métodos convergen; el más barato computacionalmente).

    Args:
        market: "h2h" | "moneyline" | "spreads" | "totals" | "1x2" | "outright" | ...
        n_outcomes: número de outcomes del mercado (2 o 3 principalmente).
        overround_value: hold del mercado (opcional). Si None, se decide
            solo por tipo de mercado.

    Returns:
        "multiplicative" | "power" | "shin"
    """
    market_lower = (market or "").lower()
    is_three_way = n_outcomes >= 3 or market_lower in ("1x2", "three_way", "3way", "outright")

    if is_three_way:
        return "shin"

    if overround_value is not None:
        if overround_value <= 0.03:
            return "multiplicative"
        # Hold alto o normal 2-way → Power es el default recomendado.
        return "power"

    # Sin overround proporcionado, asume 2-way → Power.
    return "power"


_DEFAULT_SHARP_BOOKS: tuple[str, ...] = (
    "pinnacle",
    "pinnacle_alt",
    "circa",
    "bookmaker",
    "betfair",
    "betfair_ex_eu",
    "betfair_ex_uk",
    "bet105",
    "matchbook",
    # Sprint B abr-2026: prediction markets CLOB peer-to-peer sin margin
    "polymarket",
    "kalshi",
)
_DEFAULT_SHARP_WEIGHTS: dict[str, float] = {
    # Pinnacle 50% — más líquido, respuesta más rápida
    "pinnacle": 0.50,
    "pinnacle_alt": 0.50,
    # Circa/Bookmaker/Bet105 15% — sharps US
    "circa": 0.15,
    "bookmaker": 0.15,
    "bet105": 0.15,
    # Exchanges 10% — sharp pero con comisión 2-5%
    "betfair": 0.10,
    "betfair_ex_eu": 0.10,
    "betfair_ex_uk": 0.10,
    "matchbook": 0.10,
    # Prediction markets 10% — sin bookmaker margin, alta liquidez en eventos top
    "polymarket": 0.10,
    "kalshi": 0.05,  # menor liquidez que Polymarket
}


def consensus_fair_probs(
    odds_by_bookmaker: dict[str, list[float]],
    *,
    method: Method = "shin",
    sharp_books: tuple[str, ...] = _DEFAULT_SHARP_BOOKS,
    weights: dict[str, float] | None = None,
) -> np.ndarray | None:
    """De-vig y promedia entre libros sharp — MULTI-SHARP CONSENSUS.

    Promedia Pinnacle (50% peso) + Circa/Bookmaker/Bet105 (15% cada uno) +
    Betfair Exchange/Matchbook (10%). Reduce varianza ~40% vs solo Pinnacle.
    Si Pinnacle está roto puntualmente, el consenso sigue dando fair válido.

    Si ningún libro sharp disponible, devuelve None (señal para skip).
    """
    # Filtrar odds inválidas antes de devig — algunos books envían 0 o 1.0
    # cuando el mercado está cerrado/suspendido. Ignorar silenciosamente.
    sharp_present = {
        bm: o
        for bm, o in odds_by_bookmaker.items()
        if bm in sharp_books and all(v > 1.0 for v in o)
    }
    if not sharp_present:
        return None

    if weights is None:
        weights = {bm: _DEFAULT_SHARP_WEIGHTS.get(bm, 0.10) for bm in sharp_present}

    fair_probs = []
    total_weight = 0.0
    for bm, odds in sharp_present.items():
        w = weights.get(bm, 0.10)
        try:
            fair_probs.append(devig(odds, method=method) * w)
            total_weight += w
        except (ValueError, ZeroDivisionError):
            # Odds corruptas — skip este book silenciosamente
            continue

    if total_weight == 0:
        return None
    combined = np.sum(fair_probs, axis=0) / total_weight
    return combined / combined.sum()
