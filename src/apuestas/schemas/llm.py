"""Schemas msgspec para outputs estructurados del LLM.

msgspec.Struct es 2-5x más rápido que Pydantic v2 en hot paths.
Estos structs son el contrato estricto con el LLM: cualquier JSON que el
modelo emita debe parsear limpio aquí o se considera guardrail failure.
"""

from __future__ import annotations

from typing import Literal

import msgspec

Severity = Literal["out", "doubtful", "questionable", "probable", "active"]
Confidence = Literal["low", "medium", "high"]
Momentum = Literal["positive", "neutral", "negative"]
LineAssessment = Literal["sharp", "public", "neutral", "unknown"]
EdgeDirection = Literal["favors_home", "favors_away", "neutral"]
ImpactRating = Literal["low", "medium", "high"]
PredictionQuality = Literal["accurate", "off", "very_off"]


class InjuryEntry(msgspec.Struct, frozen=True, gc=False):
    player: str
    team: str
    severity: Severity
    impact: str


class LineupChange(msgspec.Struct, frozen=True, gc=False):
    team: str
    change: str
    magnitude: ImpactRating = "medium"


class TransferImpact(msgspec.Struct, frozen=True, gc=False):
    player: str
    direction: Literal["in", "out"]
    team: str
    impact: ImpactRating


class CoachingChangeFlag(msgspec.Struct, frozen=True, gc=False):
    team: str
    change_type: str
    tenure_days: int | None = None
    adaptation_status: str | None = None


class StreakSummary(msgspec.Struct, frozen=True, gc=False):
    metric: str
    current_length: int
    direction: str
    note: str = ""


class TeamAnalysis(msgspec.Struct, frozen=True, gc=False):
    team_name: str
    key_injuries: list[InjuryEntry]
    lineup_changes: list[LineupChange]
    recent_transfers_impact: list[TransferImpact]
    coaching_change_flags: list[CoachingChangeFlag]
    streak_home_away: StreakSummary | None
    streak_overall: StreakSummary | None
    player_streaks_notable: list[StreakSummary]
    rest_days: int
    back_to_back: bool
    narrative_momentum: Momentum
    # Solo en visitante:
    travel_km: float | None = None
    timezone_delta_hours: int | None = None
    altitude_delta_m: int | None = None


class VenueFactors(msgspec.Struct, frozen=True, gc=False):
    venue_name: str
    altitude_m: int | None
    surface: str | None
    roof: str | None
    home_advantage_estimate: float


class WeatherSnapshot(msgspec.Struct, frozen=True, gc=False):
    temp_c: float | None
    wind_kph: float | None
    precip_mm: float | None
    conditions: str | None


class MatchupContext(msgspec.Struct, frozen=True, gc=False):
    h2h_recent: list[str]
    h2h_at_venue: list[str]
    venue_factors: VenueFactors | None
    weather: WeatherSnapshot | None
    referee_or_umpire_notes: str = ""


class PreMatchAnalysis(msgspec.Struct, frozen=True, gc=False):
    """Output canónico del flow deep_analysis. Espejo home/away obligatorio."""

    home_team_analysis: TeamAnalysis
    away_team_analysis: TeamAnalysis
    matchup_context: MatchupContext
    contradictions_found: list[str]
    line_movement_assessment: LineAssessment
    overall_edge_direction: EdgeDirection
    confidence_in_analysis: Confidence
    summary_es: str


class NERPerson(msgspec.Struct, frozen=True, gc=False):
    name: str
    role: Literal["player", "coach", "referee", "executive", "other"]
    team: str | None = None


class NERExtraction(msgspec.Struct, frozen=True, gc=False):
    persons: list[NERPerson]
    teams: list[str]
    injuries: list[InjuryEntry]
    suspensions: list[str]
    transfers: list[str]
    sentiment: Literal["positive", "neutral", "negative"]
    sentiment_score: float  # -1..+1


class PostMortemNarrative(msgspec.Struct, frozen=True, gc=False):
    """Narrativa generada tras cada bet settleada."""

    outcome: Literal["won", "lost", "void"]
    prediction_quality: PredictionQuality
    what_went_right: list[str]
    what_went_wrong: list[str]
    unexpected_factors: list[str]
    if_we_had_known: str
    transferable_lesson: str
    tag_for_pattern_detection: list[str]
