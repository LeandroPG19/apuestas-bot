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
    player: str = ""
    team: str = ""
    severity: Severity = "questionable"
    impact: str = ""


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
    """Análisis por equipo. Campos con defaults para tolerar respuestas LLM
    donde no hay data (e.g. sin news/injuries disponibles)."""

    team_name: str
    narrative_momentum: Momentum = "neutral"
    rest_days: int = 0
    back_to_back: bool = False
    key_injuries: list[InjuryEntry] = msgspec.field(default_factory=list)
    lineup_changes: list[LineupChange] = msgspec.field(default_factory=list)
    recent_transfers_impact: list[TransferImpact] = msgspec.field(default_factory=list)
    coaching_change_flags: list[CoachingChangeFlag] = msgspec.field(default_factory=list)
    streak_home_away: StreakSummary | None = None
    streak_overall: StreakSummary | None = None
    player_streaks_notable: list[StreakSummary] = msgspec.field(default_factory=list)
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
    h2h_recent: list[str] = msgspec.field(default_factory=list)
    h2h_at_venue: list[str] = msgspec.field(default_factory=list)
    venue_factors: VenueFactors | None = None
    weather: WeatherSnapshot | None = None
    referee_or_umpire_notes: str = ""


class PreMatchAnalysis(msgspec.Struct, frozen=True, gc=False):
    """Output canónico del flow deep_analysis. Espejo home/away obligatorio."""

    home_team_analysis: TeamAnalysis
    away_team_analysis: TeamAnalysis
    summary_es: str = ""  # opcional: DeepSeek a veces no lo genera en primera respuesta
    matchup_context: MatchupContext = msgspec.field(default_factory=MatchupContext)
    contradictions_found: list[str] = msgspec.field(default_factory=list)
    line_movement_assessment: LineAssessment = "unknown"
    overall_edge_direction: EdgeDirection = "neutral"
    confidence_in_analysis: Confidence = "low"


class NERPerson(msgspec.Struct, frozen=True, gc=False):
    name: str
    role: Literal["player", "coach", "referee", "executive", "other"]
    team: str | None = None


class NERExtraction(msgspec.Struct, frozen=True, gc=False):
    persons: list[NERPerson] = msgspec.field(default_factory=list)
    teams: list[str] = msgspec.field(default_factory=list)
    injuries: list[InjuryEntry] = msgspec.field(default_factory=list)
    suspensions: list[str] = msgspec.field(default_factory=list)
    transfers: list[str] = msgspec.field(default_factory=list)
    sentiment: Literal["positive", "neutral", "negative"] = "neutral"
    sentiment_score: float = 0.0


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
