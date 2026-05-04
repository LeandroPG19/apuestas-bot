"""Backfill league_id + add missing leagues — Sprint 13 quirúrgico.

Revision ID: 0028
Revises: 0027
Create Date: 2026-04-24

Añade ligas faltantes (Liga MX, MLS, Liga Expansión MX, etc.) y backfillea
`matches.league_id` para los 2655 matches sin league_id asignado.

Mapping heurístico por `sport_code`:
  'laliga'  → Premier Esp (id=6)
  'epl'     → Premier Eng (id=4)
  'liga_mx' → Liga MX (nueva)
  'soccer'  → fuzzy por team names

Bug fix: 99.7% matches tenían league_id NULL → detector.skip_no_model.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0028"
down_revision: str | None = "0027"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Crear leagues faltantes (con ON CONFLICT para idempotencia)
    op.execute(
        """
        INSERT INTO apuestas.leagues (sport_code, name, country, tier)
        VALUES
            ('soccer', 'Liga MX', 'MEX', 1),
            ('soccer', 'Liga de Expansión MX', 'MEX', 2),
            ('soccer', 'MLS', 'USA', 1),
            ('soccer', 'USL Championship', 'USA', 2),
            ('soccer', 'Champions League', 'EUR', 1),
            ('soccer', 'Europa League', 'EUR', 1),
            ('soccer', 'Copa Libertadores', 'SAM', 1),
            ('soccer', 'Copa Sudamericana', 'SAM', 1),
            ('soccer', 'Brasileirão Serie A', 'BRA', 1),
            ('soccer', 'Liga Argentina', 'ARG', 1),
            ('soccer', 'J-League', 'JPN', 1),
            ('soccer', 'K-League', 'KOR', 1),
            ('soccer', 'A-League', 'AUS', 1),
            ('soccer', 'Chile Primera', 'CHI', 1),
            ('soccer', 'Saudi Pro League', 'SAU', 1),
            ('soccer', 'FA Cup', 'ENG', 1),
            ('soccer', 'UEFA Conference League', 'EUR', 1)
        ON CONFLICT (external_id) DO NOTHING
        """
    )

    # 2. Backfill por sport_code directo (matches con sport_code específico)
    op.execute(
        """
        UPDATE apuestas.matches m
        SET league_id = (SELECT id FROM apuestas.leagues WHERE name = 'Premier League' AND country = 'ENG' LIMIT 1)
        WHERE m.sport_code = 'epl' AND m.league_id IS NULL
        """
    )
    op.execute(
        """
        UPDATE apuestas.matches m
        SET league_id = (SELECT id FROM apuestas.leagues WHERE name = 'La Liga' AND country = 'ESP' LIMIT 1)
        WHERE m.sport_code = 'laliga' AND m.league_id IS NULL
        """
    )
    op.execute(
        """
        UPDATE apuestas.matches m
        SET league_id = (SELECT id FROM apuestas.leagues WHERE name = 'Liga MX' LIMIT 1)
        WHERE m.sport_code = 'liga_mx' AND m.league_id IS NULL
        """
    )

    # 3. Backfill por team name (matches con sport_code='soccer' genérico)
    # Premier League teams
    op.execute(
        """
        UPDATE apuestas.matches m
        SET league_id = (SELECT id FROM apuestas.leagues WHERE name = 'Premier League' AND country = 'ENG' LIMIT 1)
        WHERE m.sport_code = 'soccer' AND m.league_id IS NULL
        AND EXISTS (
            SELECT 1 FROM apuestas.teams t
            WHERE (t.id = m.home_team_id OR t.id = m.away_team_id)
            AND LOWER(t.name) SIMILAR TO
                '%(arsenal|chelsea|liverpool|manchester|tottenham|newcastle|west ham|brighton|aston villa|crystal palace|fulham|brentford|wolves|everton|bournemouth|nottingham|leicester|leeds|sheffield|burnley|ipswich|southampton)%'
        )
        """
    )

    # La Liga teams
    op.execute(
        """
        UPDATE apuestas.matches m
        SET league_id = (SELECT id FROM apuestas.leagues WHERE name = 'La Liga' AND country = 'ESP' LIMIT 1)
        WHERE m.sport_code = 'soccer' AND m.league_id IS NULL
        AND EXISTS (
            SELECT 1 FROM apuestas.teams t
            WHERE (t.id = m.home_team_id OR t.id = m.away_team_id)
            AND LOWER(t.name) SIMILAR TO
                '%(barcelona|real madrid|atletico|atlético|sevilla|valencia|villarreal|betis|athletic bilbao|real sociedad|celta|espanyol|getafe|osasuna|mallorca|alaves|alavés|rayo|leganes|leganés|girona)%'
        )
        """
    )

    # Liga MX teams
    op.execute(
        """
        UPDATE apuestas.matches m
        SET league_id = (SELECT id FROM apuestas.leagues WHERE name = 'Liga MX' LIMIT 1)
        WHERE m.sport_code = 'soccer' AND m.league_id IS NULL
        AND EXISTS (
            SELECT 1 FROM apuestas.teams t
            WHERE (t.id = m.home_team_id OR t.id = m.away_team_id)
            AND LOWER(t.name) SIMILAR TO
                '%(chivas|guadalajara|américa|america|cruz azul|pumas|tigres|monterrey|toluca|león|leon|atlas|santos|pachuca|necaxa|puebla|juárez|juarez|querétaro|queretaro|mazatlán|mazatlan|tijuana|san luis)%'
        )
        """
    )

    # Serie A Italia
    op.execute(
        """
        UPDATE apuestas.matches m
        SET league_id = (SELECT id FROM apuestas.leagues WHERE name = 'Serie A' AND country = 'ITA' LIMIT 1)
        WHERE m.sport_code = 'soccer' AND m.league_id IS NULL
        AND EXISTS (
            SELECT 1 FROM apuestas.teams t
            WHERE (t.id = m.home_team_id OR t.id = m.away_team_id)
            AND LOWER(t.name) SIMILAR TO
                '%(juventus|milan|inter|napoli|roma|lazio|fiorentina|bologna|torino|atalanta|udinese|sassuolo|genoa|cremonese|monza|empoli|cagliari|verona|parma|lecce)%'
        )
        """
    )

    # Bundesliga
    op.execute(
        """
        UPDATE apuestas.matches m
        SET league_id = (SELECT id FROM apuestas.leagues WHERE name = 'Bundesliga' AND country = 'GER' LIMIT 1)
        WHERE m.sport_code = 'soccer' AND m.league_id IS NULL
        AND EXISTS (
            SELECT 1 FROM apuestas.teams t
            WHERE (t.id = m.home_team_id OR t.id = m.away_team_id)
            AND LOWER(t.name) SIMILAR TO
                '%(bayern|dortmund|leverkusen|leipzig|frankfurt|stuttgart|wolfsburg|hoffenheim|gladbach|freiburg|mainz|bremen|union berlin|augsburg|heidenheim|bochum|holstein kiel|st\\. pauli|kiel)%'
        )
        """
    )

    # Ligue 1
    op.execute(
        """
        UPDATE apuestas.matches m
        SET league_id = (SELECT id FROM apuestas.leagues WHERE name = 'Ligue 1' AND country = 'FRA' LIMIT 1)
        WHERE m.sport_code = 'soccer' AND m.league_id IS NULL
        AND EXISTS (
            SELECT 1 FROM apuestas.teams t
            WHERE (t.id = m.home_team_id OR t.id = m.away_team_id)
            AND LOWER(t.name) SIMILAR TO
                '%(paris saint|psg|marseille|lyon|monaco|nice|rennes|lens|lille|strasbourg|nantes|montpellier|toulouse|auxerre|reims|angers|brest|le havre|saint-etienne|saint-étienne)%'
        )
        """
    )

    # MLS
    op.execute(
        """
        UPDATE apuestas.matches m
        SET league_id = (SELECT id FROM apuestas.leagues WHERE name = 'MLS' LIMIT 1)
        WHERE m.sport_code = 'soccer' AND m.league_id IS NULL
        AND EXISTS (
            SELECT 1 FROM apuestas.teams t
            WHERE (t.id = m.home_team_id OR t.id = m.away_team_id)
            AND LOWER(t.name) SIMILAR TO
                '%(lafc|los angeles fc|inter miami|atlanta united|new york red bulls|new york city|seattle sounders|columbus crew|portland timbers|philadelphia union|toronto fc|chicago fire|dc united|orlando city|austin fc|minnesota united|colorado rapids|real salt lake|houston dynamo|fc cincinnati|montreal|vancouver whitecaps|san jose earthquakes|st\\. louis city|charlotte fc|dallas fc|new england)%'
        )
        """
    )

    # 4. Crear índice para fast lookup en detector
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_matches_league_sport "
        "ON apuestas.matches (league_id, sport_code, start_time) "
        "WHERE league_id IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS apuestas.idx_matches_league_sport")
    # No se elimina league_id porque podría haber data de otros flujos
