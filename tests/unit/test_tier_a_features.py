"""Tests Tier A features — referee bias, coaching clutch, half-period, steam, proxies, sentiment."""

from __future__ import annotations

import pytest

from apuestas.betting.half_period_markets import (
    HISTORICAL_FIRST_HALF_SHARE,
    compute_period_edge,
    detect_asymmetric_picks,
)
from apuestas.betting.steam_detector import SHARP_BOOKS, SteamCandidate

# ─── half_period_markets ────────────────────────────────────────────


def test_nba_h1_symmetric_no_edge() -> None:
    """Si línea H1 es 50/50 del total game, no hay edge."""
    edge = compute_period_edge(
        sport_code="nba", period_market="H1", game_total=220, period_total=110
    )
    assert edge is None


def test_nba_q1_asymmetric_detects_edge() -> None:
    """Si libro pone Q1 como 25% pero histórico NBA es 24.6%, no gap relevante."""
    edge = compute_period_edge(
        sport_code="nba", period_market="Q1", game_total=220, period_total=50
    )
    # 50/220 = 22.7% vs 24.6% esperado → gap +1.9% < threshold 2%
    assert edge is None


def test_nba_q1_big_asymmetry() -> None:
    edge = compute_period_edge(
        sport_code="nba", period_market="Q1", game_total=220, period_total=66
    )
    # 66/220 = 30% vs 24.6% esperado → gap -5.4% → under
    assert edge is not None
    assert edge.edge_direction == "under"
    assert edge.edge_magnitude > 0.05


def test_mlb_f5_under_share() -> None:
    """F5 debería ser ~52.7% del game total; si libro pone 50/50 → edge over."""
    edge = compute_period_edge(
        sport_code="mlb", period_market="F5", game_total=8.5, period_total=4.25
    )
    assert edge is not None
    assert edge.edge_direction == "over"
    assert edge.edge_magnitude > 0.02


def test_detect_asymmetric_picks_batch() -> None:
    picks = detect_asymmetric_picks(
        sport_code="nba",
        game_total_line=220,
        period_odds=[
            {"period": "Q1", "line": 66},  # asimétrico
            {"period": "H1", "line": 110},  # simétrico
        ],
    )
    assert len(picks) == 1
    assert picks[0].market_code == "q1_total"


def test_historical_shares_sum_coherent() -> None:
    """Para NBA: Q1+Q2 debería acercarse a H1 en 0.5."""
    nba = HISTORICAL_FIRST_HALF_SHARE["nba"]
    assert 0.49 <= nba["H1"] <= 0.51
    assert 0.23 <= nba["Q1"] <= 0.26


# ─── steam_detector ──────────────────────────────────────────────────


def test_sharp_books_set() -> None:
    assert "pinnacle" in SHARP_BOOKS
    assert "draftkings" not in SHARP_BOOKS


def test_steam_candidate_dataclass() -> None:
    c = SteamCandidate(
        match_id=1,
        market="h2h",
        outcome="home",
        direction="up",
        magnitude_pct=0.045,
        n_books=4,
        pinnacle_led=True,
        books_moved=["pinnacle", "draftkings", "fanduel", "caliente"],
    )
    assert c.pinnacle_led
    assert len(c.books_moved) == 4
    assert c.magnitude_pct > 0.03


# ─── nba_pbp helpers ─────────────────────────────────────────────────


def test_pbp_clock_parser() -> None:
    from apuestas.ingest.nba_pbp import _parse_clock

    assert _parse_clock("PT11M23.50S") == 683
    assert _parse_clock("PT0M3.20S") == 3
    assert _parse_clock("") is None
    assert _parse_clock(None) is None  # type: ignore[arg-type]


def test_pbp_event_type_map() -> None:
    from apuestas.ingest.nba_pbp import EVENT_TYPE_MAP

    assert EVENT_TYPE_MAP[1] == "made_shot"
    assert EVENT_TYPE_MAP[6] == "foul"
    assert EVENT_TYPE_MAP[9] == "timeout"


# ─── injury_feed helpers ─────────────────────────────────────────────


def test_severity_map() -> None:
    from apuestas.ingest.injury_feed import SEVERITY_MAP

    assert SEVERITY_MAP["out"] == "major"
    assert SEVERITY_MAP["probable"] == "minor"
    assert SEVERITY_MAP["active"] == "none"


def test_infer_status_keywords() -> None:
    from apuestas.ingest.injury_feed import _infer_status

    assert _infer_status("Player is out for season") == "out"
    assert _infer_status("Listed as doubtful tonight") == "doubtful"
    assert _infer_status("Back in the lineup, cleared to return") == "active"


def test_extract_player_name_capitalized() -> None:
    from apuestas.ingest.injury_feed import _extract_player_name

    assert _extract_player_name("Luka Doncic scored 40") == "Luka Doncic"
    assert _extract_player_name("LeBron James ruled out") == "LeBron James"


# ─── bluesky_sentiment ───────────────────────────────────────────────


def test_simple_sentiment_positive() -> None:
    from apuestas.ingest.bluesky_sentiment import simple_sentiment

    score = simple_sentiment("Healthy and cleared to return, dominant last 5 games")
    assert score > 0


def test_simple_sentiment_negative() -> None:
    from apuestas.ingest.bluesky_sentiment import simple_sentiment

    score = simple_sentiment("Star player injured, out for season, trade request")
    assert score < 0


def test_simple_sentiment_neutral() -> None:
    from apuestas.ingest.bluesky_sentiment import simple_sentiment

    score = simple_sentiment("Team plays at 7pm ET tonight")
    # Permite pequeña desviación por VADER
    assert -0.3 < score < 0.3


def test_beat_writers_by_sport() -> None:
    from apuestas.ingest.injury_feed import BEAT_WRITERS

    assert len(BEAT_WRITERS["nba"]) >= 2
    assert len(BEAT_WRITERS["soccer"]) >= 1
    assert all(".bsky.social" in h for h in BEAT_WRITERS["nba"])


# ─── polymarket helpers ──────────────────────────────────────────────


def test_polymarket_event_type_infer() -> None:
    from apuestas.ingest.polymarket import _infer_event_type

    assert _infer_event_type("Who wins NBA MVP 2025-26?") == "mvp"
    assert _infer_event_type("Ballon d'Or winner") == "ballon_dor"
    assert _infer_event_type("Premier League champion 2026") == "champion"
    assert _infer_event_type("Cy Young winner AL") == "cy_young"
    assert _infer_event_type("Rookie of the Year") == "roy"
    assert _infer_event_type("Random question") == "other"


def test_polymarket_sport_tag_map() -> None:
    from apuestas.ingest.polymarket import SPORT_TAG_MAP

    assert "basketball" in SPORT_TAG_MAP["nba"]
    assert "super-bowl" in SPORT_TAG_MAP["nfl"]
    assert "champions-league" in SPORT_TAG_MAP["soccer"]


# ─── referee_bias signature smoke ────────────────────────────────────


@pytest.mark.asyncio
async def test_referee_bias_module_importable() -> None:
    from apuestas.features.referee_bias import (
        compute_referee_bias_features,
        fetch_match_referees,
        recompute_all_referee_profiles,
    )

    assert callable(compute_referee_bias_features)
    assert callable(fetch_match_referees)
    assert callable(recompute_all_referee_profiles)


@pytest.mark.asyncio
async def test_coaching_clutch_module_importable() -> None:
    from apuestas.features.coaching_clutch import (
        compute_coaching_features,
        fetch_match_coaches,
        recompute_nba_coaching_from_pbp,
    )

    assert callable(compute_coaching_features)
    assert callable(fetch_match_coaches)
    assert callable(recompute_nba_coaching_from_pbp)


@pytest.mark.asyncio
async def test_tracking_proxies_module_importable() -> None:
    from apuestas.features.tracking_proxies import (
        batch_compute_for_match,
        compute_fatigue_index,
        compute_player_proxies,
    )

    assert callable(compute_player_proxies)
    assert callable(batch_compute_for_match)
    assert callable(compute_fatigue_index)
