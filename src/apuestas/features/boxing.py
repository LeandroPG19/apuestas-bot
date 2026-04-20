"""Features boxeo — BoxRec + narrative LLM + Elo/Glicko.

Blueprint §6: deporte con muestras limitadas; modelo Elo con K alto
post-inactividad y regresión por edad. LLM extrae features cualitativas
(sparring reports, camp quality, pesaje drama) vía cuba-search.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl

from apuestas.features.common import rolling_mean_prev

FEATURE_SET_NAME = "boxing_v1"


def compute_fighter_age(birthdate: date | None, fight_date: date) -> float:
    if birthdate is None:
        return 30.0  # mediana edad boxeador profesional
    return (fight_date - birthdate).days / 365.25


def age_curve_adjustment(age: float) -> float:
    """Curva empírica: peak ~27-30, declive >32 (Canelo outlier +34)."""
    if age < 25:
        return 0.95 - (25 - age) * 0.01
    if age <= 30:
        return 1.0
    if age <= 32:
        return 0.98
    return 1.0 - (age - 32) * 0.03


def inactivity_penalty(days_since_last_fight: int) -> float:
    """Más de 12 meses de inactividad penaliza fuertemente. Monotónica no creciente."""
    if days_since_last_fight < 180:
        return 1.0
    if days_since_last_fight < 365:
        return 0.95
    if days_since_last_fight < 730:
        return 0.88
    if days_since_last_fight < 1095:
        return 0.78
    if days_since_last_fight < 1460:
        return 0.70
    return 0.60


def fighter_rolling_features(fighter_logs: pl.DataFrame) -> pl.DataFrame:
    """Rolling sobre últimas 5 peleas de métricas cuantitativas."""
    metrics = (
        "knockdowns_scored",
        "knockdowns_received",
        "punches_landed_per_round",
        "punches_thrown_per_round",
        "rounds_completed",
        "ko_probability",
    )
    result = fighter_logs.sort(["fighter_id", "fight_date"])
    for metric in metrics:
        if metric in result.columns:
            result = rolling_mean_prev(
                result,
                by="fighter_id",
                order="fight_date",
                value=metric,
                windows=[3, 5, 10],
            )
    return result


def build_boxing_features(
    *,
    fighter_a_id: int,
    fighter_b_id: int,
    fight_date: date,
    fighter_profiles: dict[int, dict[str, Any]],
) -> dict[str, float]:
    """Features para un combate específico.

    fighter_profiles[id] = {
        "birthdate": date, "record_wins": int, "record_losses": int, "ko_pct": float,
        "reach_cm": float, "stance": "orthodox"|"southpaw",
        "last_fight_date": date, "ranking": int, ...
    }
    """

    def _f(pid: int, key: str, default: float = 0.0) -> float:
        val = fighter_profiles.get(pid, {}).get(key, default)
        try:
            return float(val) if val is not None else default
        except (TypeError, ValueError):  # fmt: skip
            return default

    a_profile = fighter_profiles.get(fighter_a_id, {})
    b_profile = fighter_profiles.get(fighter_b_id, {})
    a_last = a_profile.get("last_fight_date") or fight_date
    b_last = b_profile.get("last_fight_date") or fight_date

    a_age = compute_fighter_age(a_profile.get("birthdate"), fight_date)
    b_age = compute_fighter_age(b_profile.get("birthdate"), fight_date)

    a_inactivity = (fight_date - a_last).days if isinstance(a_last, date) else 180
    b_inactivity = (fight_date - b_last).days if isinstance(b_last, date) else 180

    return {
        # Record
        "a_win_pct": _f(fighter_a_id, "record_wins")
        / max(_f(fighter_a_id, "record_wins") + _f(fighter_a_id, "record_losses"), 1),
        "b_win_pct": _f(fighter_b_id, "record_wins")
        / max(_f(fighter_b_id, "record_wins") + _f(fighter_b_id, "record_losses"), 1),
        "a_ko_pct": _f(fighter_a_id, "ko_pct"),
        "b_ko_pct": _f(fighter_b_id, "ko_pct"),
        # Physical
        "reach_diff_cm": _f(fighter_a_id, "reach_cm") - _f(fighter_b_id, "reach_cm"),
        "height_diff_cm": _f(fighter_a_id, "height_cm") - _f(fighter_b_id, "height_cm"),
        # Stance matchup
        "orthodox_vs_southpaw": float(a_profile.get("stance") != b_profile.get("stance")),
        # Edad + ajuste curva
        "a_age": a_age,
        "b_age": b_age,
        "age_diff": a_age - b_age,
        "a_age_curve": age_curve_adjustment(a_age),
        "b_age_curve": age_curve_adjustment(b_age),
        # Inactividad
        "a_inactivity_days": float(a_inactivity),
        "b_inactivity_days": float(b_inactivity),
        "a_inactivity_penalty": inactivity_penalty(a_inactivity),
        "b_inactivity_penalty": inactivity_penalty(b_inactivity),
        # Ranking (BoxRec)
        "a_ranking": _f(fighter_a_id, "ranking", 999),
        "b_ranking": _f(fighter_b_id, "ranking", 999),
        "ranking_diff": _f(fighter_a_id, "ranking", 999) - _f(fighter_b_id, "ranking", 999),
    }


def elo_update(
    *,
    rating_a: float,
    rating_b: float,
    outcome_a: int,  # 1 win, 0 loss, 0.5 draw
    k: float = 32.0,
) -> tuple[float, float]:
    """Elo update simple. Boxeo usa K alto (32-40) por baja frecuencia."""
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    expected_b = 1 - expected_a
    new_a = rating_a + k * (outcome_a - expected_a)
    new_b = rating_b + k * ((1 - outcome_a) - expected_b)
    return new_a, new_b
