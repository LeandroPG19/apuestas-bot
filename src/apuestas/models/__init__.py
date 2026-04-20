"""Modelos ORM SQLAlchemy 2.0.

Importados colectivamente aquí para que Alembic detecte todas las tablas
vía `Base.metadata` al correr autogenerate.
"""

from apuestas.models.analysis import (
    BotState,
    CalibrationRolling,
    CoachingChange,
    H2HHistory,
    Injury,
    Lineup,
    MatchOfficial,
    NewsArticle,
    Official,
    PatternBlacklist,
    PlayerNews,
    PlayerStatsRolling,
    PlayerStreak,
    PostMortem,
    TeamStatsRollingAway,
    TeamStatsRollingHome,
    TeamStreak,
    Transfer,
    TravelLog,
    VenueFactor,
    WeatherForecast,
)
from apuestas.models.catalog import League, Player, Sport, Team, Venue
from apuestas.models.matches import (
    Bet,
    IngestCheckpoint,
    Match,
    OddsHistory,
    Prediction,
)

__all__ = [
    "Bet",
    "BotState",
    "CalibrationRolling",
    "CoachingChange",
    "H2HHistory",
    "IngestCheckpoint",
    "Injury",
    "League",
    "Lineup",
    "Match",
    "MatchOfficial",
    "NewsArticle",
    "OddsHistory",
    "Official",
    "PatternBlacklist",
    "Player",
    "PlayerNews",
    "PlayerStatsRolling",
    "PlayerStreak",
    "PostMortem",
    "Prediction",
    "Sport",
    "Team",
    "TeamStatsRollingAway",
    "TeamStatsRollingHome",
    "TeamStreak",
    "Transfer",
    "TravelLog",
    "Venue",
    "VenueFactor",
    "WeatherForecast",
]
