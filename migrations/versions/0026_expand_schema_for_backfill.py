"""Expand schema for Sprint 11+ backfill — 11 tablas nuevas data histórica.

Revision ID: 0026
Revises: 0025
Create Date: 2026-04-24

Tablas nuevas para soportar bulk ingest masivo:
- team_elo_daily: Elo pre-calculado multi-source (clubelo, 538, massey)
- odds_history_archive: bulk football-data.co.uk closing odds 1993+
- statsbomb_events: event-level soccer (JSONB)
- injury_reports_normalized: reportes parseados NLP
- weather_stadium_archive: NOAA weather por ts/estadio
- power_rankings_external: Massey/Sagarin/Colley/538
- nfl_epa_plays: play-level EPA/CPOE nflfastR
- nba_lineup_5man_efficiency: net rating 5-man units
- nba_hustle_stats: deflections/contested/screens/charges
- fangraphs_team_stats_daily: wRC+/FIP/WAR rolling
- pitcher_game_stats: ya existe (creación idempotente)
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0026"
down_revision: str | None = "0025"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. team_elo_daily (multi-source)
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS apuestas.team_elo_daily (
            id bigserial PRIMARY KEY,
            team_id bigint REFERENCES apuestas.teams(id),
            sport_code text NOT NULL,
            rating_date date NOT NULL,
            source text NOT NULL,  -- 'clubelo', '538_spi', '538_raptor', etc.
            elo_rating numeric(7,2) NOT NULL,
            ingested_at timestamptz DEFAULT now(),
            UNIQUE (team_id, source, rating_date)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_elo_daily_lookup "
        "ON apuestas.team_elo_daily (team_id, rating_date DESC, source)"
    )

    # 2. odds_history_archive (bulk football-data.co.uk)
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS apuestas.odds_history_archive (
            id bigserial PRIMARY KEY,
            sport_code text NOT NULL,
            league text,
            season text,
            match_date date NOT NULL,
            home_team text NOT NULL,
            away_team text NOT NULL,
            home_score integer,
            away_score integer,
            closing_odds jsonb,  -- {home, draw, away, over25, under25, bts_yes, bts_no}
            source text NOT NULL DEFAULT 'football-data.co.uk',
            ingested_at timestamptz DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_odds_archive_brin "
        "ON apuestas.odds_history_archive USING BRIN (match_date)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_odds_archive_league "
        "ON apuestas.odds_history_archive (league, season, match_date)"
    )

    # 3. statsbomb_events (event-level)
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS apuestas.statsbomb_events (
            id bigserial PRIMARY KEY,
            competition_id integer NOT NULL,
            season_id integer NOT NULL,
            match_id integer NOT NULL,
            period smallint NOT NULL,
            minute smallint,
            team_id integer,
            player_id integer,
            event_type text,
            event_jsonb jsonb NOT NULL,
            ingested_at timestamptz DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_sb_events_match "
        "ON apuestas.statsbomb_events (match_id, period, minute)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_sb_events_jsonb "
        "ON apuestas.statsbomb_events USING GIN (event_jsonb)"
    )

    # 4. injury_reports_normalized
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS apuestas.injury_reports_normalized (
            id bigserial PRIMARY KEY,
            match_id bigint REFERENCES apuestas.matches(id),
            player_id bigint REFERENCES apuestas.players(id),
            player_name_raw text,
            team_id bigint,
            status text NOT NULL,  -- out/questionable/probable/available
            injury_type text,
            reported_at timestamptz NOT NULL,
            source text,
            confidence numeric(3,2),
            ingested_at timestamptz DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_injury_match "
        "ON apuestas.injury_reports_normalized (match_id, reported_at DESC)"
    )

    # 5. weather_stadium_archive
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS apuestas.weather_stadium_archive (
            id bigserial PRIMARY KEY,
            venue_id bigint REFERENCES apuestas.venues(id),
            ts timestamptz NOT NULL,
            temp_f numeric(5,2),
            wind_mph numeric(5,2),
            wind_dir_deg smallint,
            humidity_pct smallint,
            precip_prob numeric(4,3),
            precip_mm numeric(5,2),
            source text DEFAULT 'noaa',
            ingested_at timestamptz DEFAULT now(),
            UNIQUE (venue_id, ts)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_weather_venue_ts "
        "ON apuestas.weather_stadium_archive (venue_id, ts DESC)"
    )

    # 6. power_rankings_external
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS apuestas.power_rankings_external (
            id bigserial PRIMARY KEY,
            team_id bigint REFERENCES apuestas.teams(id),
            sport_code text NOT NULL,
            rating_date date NOT NULL,
            source text NOT NULL,  -- 'massey', 'sagarin', 'colley', '538', 'spi'
            rating numeric(8,3) NOT NULL,
            rank integer,
            ingested_at timestamptz DEFAULT now(),
            UNIQUE (team_id, source, rating_date)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_power_lookup "
        "ON apuestas.power_rankings_external (team_id, rating_date DESC)"
    )

    # 7. nfl_epa_plays
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS apuestas.nfl_epa_plays (
            id bigserial PRIMARY KEY,
            match_id bigint,
            game_id_nflverse text NOT NULL,
            play_id integer NOT NULL,
            offense_team text,
            defense_team text,
            down smallint,
            ydstogo smallint,
            yardline_100 smallint,
            play_type text,
            epa numeric(7,4),
            cpoe numeric(7,4),
            success smallint,
            ingested_at timestamptz DEFAULT now(),
            UNIQUE (game_id_nflverse, play_id)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_nfl_epa_game ON apuestas.nfl_epa_plays (game_id_nflverse)"
    )

    # 8. nba_lineup_5man_efficiency
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS apuestas.nba_lineup_5man_efficiency (
            id bigserial PRIMARY KEY,
            match_id bigint REFERENCES apuestas.matches(id),
            team_id bigint REFERENCES apuestas.teams(id),
            lineup_hash text NOT NULL,
            player_ids jsonb NOT NULL,
            minutes_played numeric(6,2),
            points_for integer,
            points_against integer,
            possessions integer,
            net_rating numeric(7,2),
            ingested_at timestamptz DEFAULT now(),
            UNIQUE (match_id, team_id, lineup_hash)
        )
        """
    )

    # 9. nba_hustle_stats
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS apuestas.nba_hustle_stats (
            id bigserial PRIMARY KEY,
            match_id bigint REFERENCES apuestas.matches(id),
            player_id bigint REFERENCES apuestas.players(id),
            team_id bigint REFERENCES apuestas.teams(id),
            deflections integer,
            contested_shots integer,
            contested_shots_2pt integer,
            contested_shots_3pt integer,
            screen_assists integer,
            charges_drawn integer,
            loose_balls_recovered integer,
            box_outs integer,
            ingested_at timestamptz DEFAULT now(),
            UNIQUE (match_id, player_id)
        )
        """
    )

    # 10. fangraphs_team_stats_daily
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS apuestas.fangraphs_team_stats_daily (
            id bigserial PRIMARY KEY,
            team_id bigint REFERENCES apuestas.teams(id),
            stat_date date NOT NULL,
            wrc_plus numeric(5,2),
            fip numeric(4,2),
            xfip numeric(4,2),
            war_rolling_30 numeric(4,2),
            bsr_rolling_30 numeric(5,2),
            ingested_at timestamptz DEFAULT now(),
            UNIQUE (team_id, stat_date)
        )
        """
    )

    # 11. pitcher_game_stats (idempotente, ya puede existir)
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS apuestas.pitcher_game_stats (
            id bigserial PRIMARY KEY,
            pitcher_mlbam_id bigint NOT NULL,
            game_pk bigint NOT NULL,
            game_date date,
            spin_rate_avg numeric(8,2),
            velo_avg numeric(6,2),
            whiff_pct numeric(5,4),
            release_consistency numeric(6,4),
            n_pitches integer,
            ingested_at timestamptz DEFAULT now(),
            UNIQUE (pitcher_mlbam_id, game_pk)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_pgs_pitcher_date "
        "ON apuestas.pitcher_game_stats (pitcher_mlbam_id, game_date DESC)"
    )


def downgrade() -> None:
    for tbl in (
        "pitcher_game_stats",
        "fangraphs_team_stats_daily",
        "nba_hustle_stats",
        "nba_lineup_5man_efficiency",
        "nfl_epa_plays",
        "power_rankings_external",
        "weather_stadium_archive",
        "injury_reports_normalized",
        "statsbomb_events",
        "odds_history_archive",
        "team_elo_daily",
    ):
        op.execute(f"DROP TABLE IF EXISTS apuestas.{tbl} CASCADE")
