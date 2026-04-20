"""Tests de validación msgspec sobre schemas LLM."""

from __future__ import annotations

import msgspec
import pytest

from apuestas.schemas.llm import (
    InjuryEntry,
    NERExtraction,
    PostMortemNarrative,
    PreMatchAnalysis,
    TeamAnalysis,
)


def _minimal_team_analysis(name: str) -> dict:
    return {
        "team_name": name,
        "key_injuries": [],
        "lineup_changes": [],
        "recent_transfers_impact": [],
        "coaching_change_flags": [],
        "streak_home_away": None,
        "streak_overall": None,
        "player_streaks_notable": [],
        "rest_days": 3,
        "back_to_back": False,
        "narrative_momentum": "neutral",
    }


def test_pre_match_analysis_valid_minimal() -> None:
    payload = {
        "home_team_analysis": _minimal_team_analysis("Home FC"),
        "away_team_analysis": _minimal_team_analysis("Away FC"),
        "matchup_context": {
            "h2h_recent": [],
            "h2h_at_venue": [],
            "venue_factors": None,
            "weather": None,
            "referee_or_umpire_notes": "",
        },
        "contradictions_found": [],
        "line_movement_assessment": "neutral",
        "overall_edge_direction": "neutral",
        "confidence_in_analysis": "medium",
        "summary_es": "Sin factores decisivos detectados.",
    }
    decoded = msgspec.json.decode(msgspec.json.encode(payload), type=PreMatchAnalysis)
    assert decoded.home_team_analysis.team_name == "Home FC"
    assert decoded.confidence_in_analysis == "medium"


def test_pre_match_analysis_rejects_invalid_confidence() -> None:
    payload = {
        "home_team_analysis": _minimal_team_analysis("A"),
        "away_team_analysis": _minimal_team_analysis("B"),
        "matchup_context": {
            "h2h_recent": [],
            "h2h_at_venue": [],
            "venue_factors": None,
            "weather": None,
            "referee_or_umpire_notes": "",
        },
        "contradictions_found": [],
        "line_movement_assessment": "neutral",
        "overall_edge_direction": "neutral",
        "confidence_in_analysis": "altisima",  # inválido
        "summary_es": "x",
    }
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(msgspec.json.encode(payload), type=PreMatchAnalysis)


def test_injury_entry_validates_severity_enum() -> None:
    ok = InjuryEntry(player="X", team="Y", severity="out", impact="cambio rotación")
    assert ok.severity == "out"

    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(
            b'{"player":"X","team":"Y","severity":"muerto","impact":"x"}',
            type=InjuryEntry,
        )


def test_team_analysis_away_travel_optional() -> None:
    """Home no tiene travel_km; away sí. Ambos son opcionales en el tipo."""
    home = TeamAnalysis(
        team_name="Home",
        key_injuries=[],
        lineup_changes=[],
        recent_transfers_impact=[],
        coaching_change_flags=[],
        streak_home_away=None,
        streak_overall=None,
        player_streaks_notable=[],
        rest_days=4,
        back_to_back=False,
        narrative_momentum="positive",
    )
    assert home.travel_km is None


def test_ner_extraction_valid() -> None:
    payload = {
        "persons": [{"name": "Juan Pérez", "role": "player", "team": "América"}],
        "teams": ["América", "Chivas"],
        "injuries": [],
        "suspensions": [],
        "transfers": [],
        "sentiment": "positive",
        "sentiment_score": 0.4,
    }
    d = msgspec.json.decode(msgspec.json.encode(payload), type=NERExtraction)
    assert d.persons[0].role == "player"
    assert -1.0 <= d.sentiment_score <= 1.0


def test_post_mortem_narrative_tags_required() -> None:
    payload = {
        "outcome": "lost",
        "prediction_quality": "off",
        "what_went_right": ["línea sharp identificada"],
        "what_went_wrong": ["injury ignorada"],
        "unexpected_factors": ["OT inesperado"],
        "if_we_had_known": "habríamos skipped",
        "transferable_lesson": "validar injury reports T-30min",
        "tag_for_pattern_detection": ["injury_ignored", "ot_game"],
    }
    d = msgspec.json.decode(msgspec.json.encode(payload), type=PostMortemNarrative)
    assert "injury_ignored" in d.tag_for_pattern_detection
