"""Tests para Polymarket game-by-game ingester (Sprint B abr-2026).

Cobertura: helpers puros (sin DB) — fuzzy matching, h2h detection,
odds conversion. Los flows con DB se validan end-to-end manualmente.
"""

from __future__ import annotations

import pytest

from apuestas.ingest.polymarket import (
    _is_h2h_win_question,
    _match_question_to_event,
    _normalize_team_name,
    _odds_from_prob,
    _question_team_tokens,
)

# ─────────────────── _normalize_team_name ───────────────────


def test_normalize_lowercase_alphanumeric() -> None:
    assert _normalize_team_name("Los Angeles Lakers") == "los angeles lakers"
    assert _normalize_team_name("FC Bayern München") == "fc bayern mnchen"  # tilde removed
    assert _normalize_team_name("  Real   Madrid  ") == "real madrid"
    assert _normalize_team_name("") == ""


# ─────────────────── _question_team_tokens ───────────────────


def test_team_tokens_vs_separator() -> None:
    assert _question_team_tokens("Lakers vs Warriors?") == ["lakers", "warriors"]
    assert _question_team_tokens("Lakers vs. Warriors") == ["lakers", "warriors"]


def test_team_tokens_at_separator_not_supported() -> None:
    """`@` separador no funciona porque \\b no matchea con caracter no-word.
    Documentado: solo `vs`/`vs.`/`versus` son separadores válidos."""
    # No es bug: Polymarket usa "vs" o "versus" templates consistentemente
    assert _question_team_tokens("Lakers @ Warriors") == []


def test_team_tokens_versus_separator() -> None:
    assert _question_team_tokens("Lakers versus Warriors") == ["lakers", "warriors"]


def test_team_tokens_no_separator_returns_empty() -> None:
    assert _question_team_tokens("Will Lakers win the title?") == []


# ─────────────────── _is_h2h_win_question ───────────────────


def test_h2h_win_question_simple() -> None:
    is_h2h, team = _is_h2h_win_question("Will Lakers win on 2026-04-26?")
    assert is_h2h is True
    assert team == "lakers"


def test_h2h_win_question_with_article() -> None:
    is_h2h, team = _is_h2h_win_question("Will the Lakers win?")
    assert is_h2h is True
    assert team == "lakers"


def test_h2h_rejects_draw_questions() -> None:
    is_h2h, team = _is_h2h_win_question("Will Lakers vs Warriors end in a draw?")
    assert is_h2h is False
    assert team is None


def test_h2h_rejects_total_questions() -> None:
    is_h2h, _ = _is_h2h_win_question("Total over 220.5 in Lakers vs Warriors?")
    assert is_h2h is False


def test_h2h_rejects_prop_questions() -> None:
    is_h2h, _ = _is_h2h_win_question("Lakers prop market")
    assert is_h2h is False


def test_h2h_rejects_first_to_score() -> None:
    is_h2h, _ = _is_h2h_win_question("Will Lakers be the first to score 100 points?")
    assert is_h2h is False


def test_h2h_rejects_questions_without_will_pattern() -> None:
    is_h2h, _ = _is_h2h_win_question("Lakers win Game 7?")
    assert is_h2h is False


# ─────────────────── _match_question_to_event ───────────────────


def test_match_event_exact_team_match() -> None:
    upcoming = [
        {
            "id": 1,
            "sport_code": "nba",
            "home_norm": "los angeles lakers",
            "away_norm": "golden state warriors",
        }
    ]
    matched = _match_question_to_event("Lakers vs Warriors?", upcoming)
    assert matched is not None
    assert matched["id"] == 1


def test_match_event_no_match_returns_none() -> None:
    upcoming = [
        {
            "id": 1,
            "sport_code": "nba",
            "home_norm": "boston celtics",
            "away_norm": "miami heat",
        }
    ]
    matched = _match_question_to_event("Lakers vs Warriors?", upcoming)
    assert matched is None


def test_match_event_swapped_home_away() -> None:
    """Polymarket no garantiza orden home-first; debe matchear igual."""
    upcoming = [
        {
            "id": 5,
            "sport_code": "nba",
            "home_norm": "golden state warriors",
            "away_norm": "los angeles lakers",
        }
    ]
    matched = _match_question_to_event("Lakers vs Warriors?", upcoming)
    assert matched is not None
    assert matched["id"] == 5


# ─────────────────── _odds_from_prob ───────────────────


def test_odds_from_prob_normal() -> None:
    assert _odds_from_prob(0.5) == pytest.approx(2.0)
    assert _odds_from_prob(0.25) == pytest.approx(4.0)
    assert _odds_from_prob(0.667) == pytest.approx(1.4993, rel=1e-3)


def test_odds_from_prob_extreme_values_return_none() -> None:
    """Filtra prob ≤0.01 o ≥1.0 (violarían CHECK ck_odds_positive)."""
    assert _odds_from_prob(0.0) is None
    assert _odds_from_prob(0.005) is None
    assert _odds_from_prob(1.0) is None
    assert _odds_from_prob(1.5) is None
