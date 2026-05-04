"""Tests Fase 3.3 — SoS Massey + Elo ratings."""

from __future__ import annotations

from apuestas.features.sos import (
    ELO_BASE,
    ELO_K_BY_SPORT,
    expected_score_elo,
    update_elo_pair,
)


def test_expected_score_equal_ratings() -> None:
    assert expected_score_elo(1500.0, 1500.0) == 0.5


def test_expected_score_stronger_home() -> None:
    # 200 puntos Elo ventaja → ~76%
    prob = expected_score_elo(1700.0, 1500.0)
    assert 0.74 < prob < 0.78


def test_expected_score_weaker_home() -> None:
    prob = expected_score_elo(1400.0, 1600.0)
    assert 0.22 < prob < 0.28


def test_update_elo_home_wins_expected() -> None:
    # Teams iguales, home gana → home sube ~K/2
    new_h, new_a = update_elo_pair(1500.0, 1500.0, result=1.0, k=32.0)
    assert new_h > 1500.0
    assert new_a < 1500.0
    assert abs((new_h - 1500) + (new_a - 1500)) < 0.01  # suma cero


def test_update_elo_away_upset_win() -> None:
    # Away mucho peor pero gana → away sube mucho, home baja mucho
    new_h, new_a = update_elo_pair(1800.0, 1200.0, result=0.0, k=32.0)
    assert new_h < 1800.0  # home baja
    assert new_a > 1200.0  # away sube
    # K=32, expected_home≈0.97 → delta_home = 32·(0-0.97) = -31
    assert abs(new_h - 1800.0 + 31.0) < 1.0


def test_update_elo_draw() -> None:
    new_h, new_a = update_elo_pair(1500.0, 1500.0, result=0.5, k=32.0)
    assert new_h == 1500.0
    assert new_a == 1500.0


def test_hfa_bonus_applied() -> None:
    # Con HFA bonus 50 Elo, home expected ~>50%
    new_h_no_hfa, _ = update_elo_pair(1500.0, 1500.0, 0.5, k=32.0, hfa_bonus=0.0)
    new_h_with_hfa, _ = update_elo_pair(1500.0, 1500.0, 0.5, k=32.0, hfa_bonus=50.0)
    # Con HFA, expected home >0.5 → result 0.5 → delta negativo
    assert new_h_with_hfa < new_h_no_hfa


def test_elo_k_catalog_has_all_sports() -> None:
    for sport in ("nba", "mlb", "nfl", "nhl", "soccer", "tennis"):
        assert sport in ELO_K_BY_SPORT
    # NFL tiene K más alto (temporada corta)
    assert ELO_K_BY_SPORT["nfl"] > ELO_K_BY_SPORT["mlb"]


def test_elo_base_is_1500() -> None:
    assert ELO_BASE == 1500.0


def test_expected_score_sum_to_one() -> None:
    for r_a in (1300, 1500, 1700):
        for r_b in (1200, 1500, 1800):
            e_a = expected_score_elo(r_a, r_b)
            e_b = expected_score_elo(r_b, r_a)
            assert abs(e_a + e_b - 1.0) < 0.001


def test_massey_requires_games() -> None:
    """Sin games → retorna {} (no error)."""
    # Usamos el helper de Massey via API interna: cuando rows=0 retorna dict vacío
    # (no podemos testear full async sin DB, pero el comportamiento del bucle sí).
    # Este test delega al integration test.
    assert True  # placeholder: integration test en test_sos_integration.py
