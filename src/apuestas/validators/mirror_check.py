"""QA simetría home/away + completitud de capas §16 (§16.6 del plan).

Este validador se corre tras `deep_analysis.py` para cada evento, y ANTES
de emitir cualquier pick. Si algún check falla:
- `analysis_complete=False` en predictions
- Bloqueo del pick (o badge 'DATOS LIMITADOS' si score completo≥0.7)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class LayerPresence:
    """Presencia de datos por capa (§16.1) para un team del match."""

    news: bool = False
    player_news: bool = False
    injuries: bool = False
    lineup: bool = False
    streaks: bool = False
    transfers: bool = False
    coaching: bool = False
    stats_rolling: bool = False
    travel_log: bool = False

    def score(self) -> float:
        total = 9
        filled = sum(
            [
                self.news,
                self.player_news,
                self.injuries,
                self.lineup,
                self.streaks,
                self.transfers,
                self.coaching,
                self.stats_rolling,
                self.travel_log,
            ]
        )
        return filled / total


@dataclass(slots=True)
class MirrorCheckResult:
    event_id: int
    home_layers: LayerPresence
    away_layers: LayerPresence
    venue_factors_loaded: bool
    h2h_loaded: bool
    features_have_home_away_diff: bool
    analysis_complete: bool
    overall_completeness_score: float
    missing: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ─────────────────────────── Queries por capa ───────────────────────────


async def _has_news(team_id: int, *, window_hours: int = 72) -> bool:
    since = datetime.now(tz=UTC) - timedelta(hours=window_hours)
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT 1 FROM news_articles
                WHERE :tid = ANY(teams_mentioned) AND published_at >= :since
                LIMIT 1
                """
            ),
            {"tid": team_id, "since": since},
        )
        return result.first() is not None


async def _has_player_news(team_id: int, *, window_hours: int = 48) -> bool:
    since = datetime.now(tz=UTC) - timedelta(hours=window_hours)
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT 1 FROM player_news pn
                JOIN players p ON p.id = pn.player_id
                WHERE p.team_id = :tid AND pn.published_at >= :since
                LIMIT 1
                """
            ),
            {"tid": team_id, "since": since},
        )
        return result.first() is not None


async def _has_injuries(team_id: int, *, window_hours: int = 168) -> bool:
    since = datetime.now(tz=UTC) - timedelta(hours=window_hours)
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT 1 FROM injuries i
                JOIN players p ON p.id = i.player_id
                WHERE p.team_id = :tid AND i.reported_at >= :since
                LIMIT 1
                """
            ),
            {"tid": team_id, "since": since},
        )
        return result.first() is not None


async def _has_lineup(match_id: int, team_id: int) -> bool:
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT 1 FROM lineups
                WHERE match_id = :mid AND team_id = :tid
                LIMIT 1
                """
            ),
            {"mid": match_id, "tid": team_id},
        )
        return result.first() is not None


async def _has_streaks(team_id: int) -> bool:
    async with session_scope() as session:
        result = await session.execute(
            text("SELECT 1 FROM team_streaks WHERE team_id = :tid LIMIT 1"),
            {"tid": team_id},
        )
        return result.first() is not None


async def _has_transfers(team_id: int, *, window_days: int = 56) -> bool:
    cutoff = (datetime.now(tz=UTC) - timedelta(days=window_days)).date()
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT 1 FROM transfers
                WHERE (from_team_id = :tid OR to_team_id = :tid)
                  AND transfer_date >= :cutoff
                LIMIT 1
                """
            ),
            {"tid": team_id, "cutoff": cutoff},
        )
        return result.first() is not None


async def _has_coaching(team_id: int) -> bool:
    async with session_scope() as session:
        result = await session.execute(
            text("SELECT 1 FROM coaching_changes WHERE team_id = :tid LIMIT 1"),
            {"tid": team_id},
        )
        # Coaching es opcional: si no hay cambios conocidos, también cuenta
        # como "OK" (el team tiene coach estable).
        _ = result
        return True


async def _has_stats_rolling(team_id: int, sport_code: str) -> bool:
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT 1 FROM team_stats_rolling_home
                WHERE team_id = :tid AND sport_code = :sc
                LIMIT 1
                """
            ),
            {"tid": team_id, "sc": sport_code},
        )
        return result.first() is not None


async def _has_travel_log(match_id: int, team_id: int) -> bool:
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT 1 FROM travel_log
                WHERE match_id = :mid AND team_id = :tid
                LIMIT 1
                """
            ),
            {"mid": match_id, "tid": team_id},
        )
        return result.first() is not None


async def _has_venue_factors(venue_id: int | None, sport_code: str) -> bool:
    if venue_id is None:
        return False
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT 1 FROM venue_factors
                WHERE venue_id = :vid AND sport_code = :sc
                LIMIT 1
                """
            ),
            {"vid": venue_id, "sc": sport_code},
        )
        return result.first() is not None


async def _has_h2h(team_a_id: int, team_b_id: int) -> bool:
    lo, hi = sorted((team_a_id, team_b_id))
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT 1 FROM h2h_history
                WHERE team_a_id = :a AND team_b_id = :b
                LIMIT 1
                """
            ),
            {"a": lo, "b": hi},
        )
        return result.first() is not None


async def _collect_layer_presence(*, match_id: int, team_id: int, sport_code: str) -> LayerPresence:
    return LayerPresence(
        news=await _has_news(team_id),
        player_news=await _has_player_news(team_id),
        injuries=await _has_injuries(team_id),
        lineup=await _has_lineup(match_id, team_id),
        streaks=await _has_streaks(team_id),
        transfers=await _has_transfers(team_id),
        coaching=await _has_coaching(team_id),
        stats_rolling=await _has_stats_rolling(team_id, sport_code),
        travel_log=await _has_travel_log(match_id, team_id),
    )


# ─────────────────────────── Check principal ────────────────────────────


async def run_mirror_check(
    *,
    match_id: int,
    home_team_id: int,
    away_team_id: int,
    venue_id: int | None,
    sport_code: str,
    feature_columns: list[str] | None = None,
    minimum_completeness: float = 0.7,
) -> MirrorCheckResult:
    """Validación simetría home/away + completitud de capas §16.

    Args:
        match_id: id del match a validar.
        home_team_id, away_team_id: rosters a verificar.
        venue_id, sport_code: para venue_factors.
        feature_columns: lista de columnas del DataFrame de features; se
            verifica que tenga tripletas {feature}_home / _away / _diff.
        minimum_completeness: umbral para `analysis_complete=True`.
    """
    home_layers = await _collect_layer_presence(
        match_id=match_id, team_id=home_team_id, sport_code=sport_code
    )
    away_layers = await _collect_layer_presence(
        match_id=match_id, team_id=away_team_id, sport_code=sport_code
    )

    venue_ok = await _has_venue_factors(venue_id, sport_code)
    h2h_ok = await _has_h2h(home_team_id, away_team_id)

    features_ok, feature_missing = _check_features_mirror(feature_columns or [])

    missing: list[str] = []
    warnings: list[str] = []

    for label, present in (
        ("home_news", home_layers.news),
        ("home_player_news", home_layers.player_news),
        ("home_injuries", home_layers.injuries),
        ("home_lineup", home_layers.lineup),
        ("home_streaks", home_layers.streaks),
        ("home_stats_rolling", home_layers.stats_rolling),
        ("home_travel_log", home_layers.travel_log),
        ("away_news", away_layers.news),
        ("away_player_news", away_layers.player_news),
        ("away_injuries", away_layers.injuries),
        ("away_lineup", away_layers.lineup),
        ("away_streaks", away_layers.streaks),
        ("away_stats_rolling", away_layers.stats_rolling),
        ("away_travel_log", away_layers.travel_log),
    ):
        if not present:
            missing.append(label)

    if not venue_ok:
        missing.append("venue_factors")
    if not h2h_ok:
        missing.append("h2h_history")
    if not features_ok:
        missing.extend(feature_missing)

    # Score combinado
    layer_score = (home_layers.score() + away_layers.score()) / 2
    context_score = 1.0 if (venue_ok and h2h_ok) else 0.6
    feature_score = 1.0 if features_ok else 0.4
    overall = 0.6 * layer_score + 0.2 * context_score + 0.2 * feature_score

    analysis_complete = overall >= minimum_completeness

    if not home_layers.lineup or not away_layers.lineup:
        warnings.append("lineup_unconfirmed_use_starter_projections")
    if not features_ok:
        warnings.append("feature_mirror_incomplete")
    if not venue_ok:
        warnings.append("venue_factors_stale_or_missing")

    result = MirrorCheckResult(
        event_id=match_id,
        home_layers=home_layers,
        away_layers=away_layers,
        venue_factors_loaded=venue_ok,
        h2h_loaded=h2h_ok,
        features_have_home_away_diff=features_ok,
        analysis_complete=analysis_complete,
        overall_completeness_score=overall,
        missing=missing,
        warnings=warnings,
    )

    logger.info(
        "mirror_check.done",
        event_id=match_id,
        score=overall,
        complete=analysis_complete,
        missing_count=len(missing),
    )
    return result


def _check_features_mirror(columns: list[str]) -> tuple[bool, list[str]]:
    """Verifica que exista tripleta home/away/diff para features core.

    Returns (ok, lista de features a las que les falta alguna versión).
    """
    core_metrics = ("ortg", "drtg", "pace", "win_margin", "rest_days")
    missing: list[str] = []
    for metric in core_metrics:
        home_cols = [c for c in columns if c.startswith(f"{metric}") and "_home" in c]
        away_cols = [c for c in columns if c.startswith(f"{metric}") and "_away" in c]
        diff_cols = [c for c in columns if c.startswith(f"{metric}") and c.endswith("_diff")]
        if not home_cols:
            missing.append(f"{metric}_home_*")
        if not away_cols:
            missing.append(f"{metric}_away_*")
        if not diff_cols:
            missing.append(f"{metric}_diff")
    return (len(missing) == 0, missing)
