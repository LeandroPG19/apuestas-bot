"""Sprint 13 Capa 1-5: Identity resolution + sport taxonomy + model hierarchy.

Revision ID: 0029
Revises: 0028
Create Date: 2026-04-24

Crea 5 tablas arquitectónicas:
- match_canonical: dedup matches por fingerprint
- team_leagues: N:M relación team × league × season
- model_hierarchy: priority-based model fallback chain
- model_features_registry: fill strategies por feature por modelo
- sport_code_canonical column: normaliza sport_code (laliga → soccer)

Habilita que el bot detecte y analice TODOS los partidos, no solo los
que tienen modelo exacto para su liga.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0029"
down_revision: str | None = "0028"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ═══ Capa 1: Identity Resolution ═══
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS apuestas.match_canonical (
            id bigserial PRIMARY KEY,
            fingerprint text UNIQUE NOT NULL,
            primary_match_id bigint REFERENCES apuestas.matches(id),
            sport_code_canonical text NOT NULL,
            home_team_id bigint,
            away_team_id bigint,
            start_time_bucket timestamptz NOT NULL,
            alternate_match_ids jsonb DEFAULT '[]'::jsonb,
            created_at timestamptz DEFAULT now(),
            updated_at timestamptz DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_mc_fingerprint ON apuestas.match_canonical (fingerprint)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_mc_primary ON apuestas.match_canonical (primary_match_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_mc_bucket "
        "ON apuestas.match_canonical (sport_code_canonical, start_time_bucket)"
    )

    # ═══ Capa 2: team_leagues N:M ═══
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS apuestas.team_leagues (
            team_id bigint REFERENCES apuestas.teams(id),
            league_id bigint REFERENCES apuestas.leagues(id),
            season text NOT NULL,
            confidence numeric(3,2) DEFAULT 1.0,
            first_seen date DEFAULT CURRENT_DATE,
            last_seen date DEFAULT CURRENT_DATE,
            PRIMARY KEY (team_id, league_id, season)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_team_leagues_team "
        "ON apuestas.team_leagues (team_id, last_seen DESC)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_team_leagues_league "
        "ON apuestas.team_leagues (league_id, season)"
    )

    # ═══ Capa 3: model_hierarchy (priority-based fallback) ═══
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS apuestas.model_hierarchy (
            id bigserial PRIMARY KEY,
            sport_code text NOT NULL,
            market text NOT NULL,
            league_id bigint REFERENCES apuestas.leagues(id),
            model_name text NOT NULL,
            priority integer NOT NULL DEFAULT 50,
            active boolean NOT NULL DEFAULT true,
            created_at timestamptz DEFAULT now(),
            UNIQUE (sport_code, market, league_id, model_name)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_mh_lookup "
        "ON apuestas.model_hierarchy (sport_code, market, priority) "
        "WHERE active = true"
    )

    # Popular model_hierarchy con modelos actuales
    op.execute(
        """
        INSERT INTO apuestas.model_hierarchy (sport_code, market, league_id, model_name, priority)
        VALUES
            -- Specific league models
            ('soccer', 'h2h', 4,  'soccer_league_4',  0),   -- Premier
            ('soccer', 'h2h', 5,  'soccer_league_5',  0),   -- Championship
            ('soccer', 'h2h', 6,  'soccer_league_6',  0),   -- LaLiga
            ('soccer', 'h2h', 7,  'soccer_league_7',  0),   -- LaLiga 2
            ('soccer', 'h2h', 11, 'soccer_league_11', 0),   -- Serie B
            -- Liga MX lookup dinámico
            ('soccer', 'h2h', NULL, 'soccer_liga_mx',  50),
            -- Sport-wide fallback (league_id NULL pero mayor priority)
            ('soccer', 'h2h', NULL, 'catchall_baseline', 99),
            -- Otros sports
            ('nba', 'h2h', NULL, 'nba_moneyline', 10),
            ('nba', 'h2h', NULL, 'catchall_baseline', 99),
            ('mlb', 'h2h', NULL, 'mlb_moneyline', 10),
            ('mlb', 'h2h', NULL, 'catchall_baseline', 99),
            ('nfl', 'ats', NULL, 'nfl_ats', 10),
            ('nfl', 'ats', NULL, 'catchall_baseline', 99),
            ('tennis', 'h2h', NULL, 'tennis_sackmann', 20),
            ('tennis', 'h2h', NULL, 'catchall_baseline', 99)
        ON CONFLICT (sport_code, market, league_id, model_name) DO NOTHING
        """
    )

    # ═══ Capa 4: model_features_registry (fill strategies) ═══
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS apuestas.model_features_registry (
            model_name text NOT NULL,
            model_version text NOT NULL,
            feature_name text NOT NULL,
            required boolean NOT NULL DEFAULT true,
            fill_strategy text NOT NULL DEFAULT 'skip',
            default_value numeric,
            PRIMARY KEY (model_name, model_version, feature_name),
            CHECK (fill_strategy IN ('skip', 'zero', 'mean', 'default', 'mode'))
        )
        """
    )

    # ═══ Capa 5: sport_code_canonical column ═══
    op.execute("ALTER TABLE apuestas.matches ADD COLUMN IF NOT EXISTS sport_code_canonical text")
    # Populate canonical codes
    op.execute(
        """
        UPDATE apuestas.matches SET sport_code_canonical = CASE
            WHEN sport_code IN ('soccer', 'laliga', 'epl', 'liga_mx', 'mls',
                                'bundesliga', 'seriea', 'ligue1', 'eredivisie',
                                'liga_portugal', 'liga_expansion') THEN 'soccer'
            WHEN sport_code IN ('nba', 'basketball') THEN 'nba'
            WHEN sport_code IN ('mlb', 'baseball') THEN 'mlb'
            WHEN sport_code IN ('nhl', 'ice-hockey', 'hockey') THEN 'nhl'
            WHEN sport_code IN ('nfl', 'american-football') THEN 'nfl'
            WHEN sport_code IN ('tennis', 'atp', 'wta') THEN 'tennis'
            WHEN sport_code IN ('mma', 'ufc') THEN 'mma'
            WHEN sport_code = 'boxing' THEN 'boxing'
            ELSE sport_code
        END
        WHERE sport_code_canonical IS NULL
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_matches_sport_canonical "
        "ON apuestas.matches (sport_code_canonical, start_time) "
        "WHERE sport_code_canonical IS NOT NULL"
    )

    # Filtrar esports: marcar como 'esports' matches con team names "cyber"
    op.execute(
        """
        UPDATE apuestas.matches m SET sport_code_canonical = 'esports'
        WHERE EXISTS (
            SELECT 1 FROM apuestas.teams t
            WHERE (t.id = m.home_team_id OR t.id = m.away_team_id)
            AND (LOWER(t.name) LIKE '%cyber%' OR LOWER(t.name) LIKE '%esports%'
                 OR LOWER(t.name) LIKE '%(cs)%' OR LOWER(t.name) LIKE '%virtual%'
                 OR LOWER(t.name) LIKE '%e-football%' OR LOWER(t.name) LIKE '%efootball%'
                 OR LOWER(t.name) LIKE '%fifa%' OR LOWER(t.name) LIKE '%nba 2k%')
        )
        """
    )

    # ═══ Capa 2 backfill: team_leagues desde matches existentes ═══
    op.execute(
        """
        INSERT INTO apuestas.team_leagues (team_id, league_id, season, confidence, first_seen, last_seen)
        SELECT
            t.team_id, m.league_id,
            COALESCE(m.season, '2024-25') AS season,
            0.90 AS confidence,
            MIN(m.start_time::date) AS first_seen,
            MAX(m.start_time::date) AS last_seen
        FROM apuestas.matches m
        CROSS JOIN LATERAL (
            VALUES (m.home_team_id), (m.away_team_id)
        ) AS t(team_id)
        WHERE m.league_id IS NOT NULL AND t.team_id IS NOT NULL
        GROUP BY t.team_id, m.league_id, COALESCE(m.season, '2024-25')
        ON CONFLICT (team_id, league_id, season) DO UPDATE SET
            last_seen = EXCLUDED.last_seen,
            confidence = GREATEST(team_leagues.confidence, EXCLUDED.confidence)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS apuestas.idx_matches_sport_canonical")
    op.execute("ALTER TABLE apuestas.matches DROP COLUMN IF EXISTS sport_code_canonical")
    op.execute("DROP TABLE IF EXISTS apuestas.model_features_registry CASCADE")
    op.execute("DROP TABLE IF EXISTS apuestas.model_hierarchy CASCADE")
    op.execute("DROP TABLE IF EXISTS apuestas.team_leagues CASCADE")
    op.execute("DROP TABLE IF EXISTS apuestas.match_canonical CASCADE")
