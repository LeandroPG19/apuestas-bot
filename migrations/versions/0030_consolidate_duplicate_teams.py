"""Consolida teams duplicados (309 detectados) post-Sprint 13 identity resolution.

Revision ID: 0030
Revises: 0029
Create Date: 2026-04-25

Consolidación pragmática:
  1. Para cada grupo de teams con mismo `normalized_name`, escoge un canónico:
     - Preferencia 1: team SIN sufijo `(Corners)/(Bookings)/(Cards)/(Goals)`
     - Preferencia 2: team con sport_code canónico (`soccer` sobre `laliga/epl/liga_mx`)
     - Preferencia 3: menor `id` (más antiguo)
  2. Crea tabla temporal `_team_consolidation_map (old_id, new_id)`.
  3. UPDATE en cascada sobre las 26 FK que referencian `teams.id`.
  4. UPDATE `sport_code` del canónico al sport canonical (`laliga` → `soccer`).
  5. DELETE los teams duplicados (los que quedaron sin referencias).

Trigger del bug: Pinnacle scraper crea `Getafe (Corners)` como team distinto
de `Getafe`; OddsAPI usa sport_code='laliga' mientras Pinnacle usa 'soccer'
→ 4-12 versiones del mismo team → 4-12 matches paralelos del mismo partido
→ detector elige el match con menor cobertura de soft books → no_qualifying_offer.

Tablas FK afectadas (26):
  coaches, coaching_changes, fangraphs_team_stats_daily, h2h_history (×2),
  injury_feed, lineups, match_coaches, matches (×2), nba_hustle_stats,
  nba_lineup_5man_efficiency, play_by_play, players, power_rankings_external,
  team_elo_daily, team_external_id, team_leagues, team_match_review,
  team_stats_rolling_away, team_stats_rolling_home, team_streaks,
  team_strength_bayesian, transfers (×2), travel_log

Idempotente: si una tabla no existe, sigue. Si los duplicados ya fueron
consolidados, queda no-op.

DOWN: NO reversible. Backup pg_dump antes de aplicar (ver scripts/backup.sh).
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0030"
down_revision: str | None = "0029"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# (table, column) pairs que referencian teams.id — confirmados via
# information_schema.table_constraints en producción 2026-04-25.
_TEAMS_FK_REFERENCES: tuple[tuple[str, str], ...] = (
    ("coaches", "current_team_id"),
    ("coaching_changes", "team_id"),
    ("fangraphs_team_stats_daily", "team_id"),
    ("h2h_history", "team_a_id"),
    ("h2h_history", "team_b_id"),
    ("injury_feed", "team_id"),
    ("lineups", "team_id"),
    ("match_coaches", "team_id"),
    ("matches", "away_team_id"),
    ("matches", "home_team_id"),
    ("nba_hustle_stats", "team_id"),
    ("nba_lineup_5man_efficiency", "team_id"),
    ("play_by_play", "team_id"),
    ("players", "team_id"),
    ("power_rankings_external", "team_id"),
    ("team_elo_daily", "team_id"),
    ("team_external_id", "team_id"),
    ("team_leagues", "team_id"),
    ("team_match_review", "candidate_team_id"),
    ("team_stats_rolling_away", "team_id"),
    ("team_stats_rolling_home", "team_id"),
    ("team_streaks", "team_id"),
    ("team_strength_bayesian", "team_id"),
    ("transfers", "from_team_id"),
    ("transfers", "to_team_id"),
    ("travel_log", "team_id"),
)


def upgrade() -> None:
    # 1. Construir tabla temporal con (old_id, new_id) consolidation map.
    op.execute(
        """
        DROP TABLE IF EXISTS _team_consolidation_map;
        CREATE UNLOGGED TABLE _team_consolidation_map AS
        WITH normalized AS (
            SELECT
                id,
                name,
                sport_code,
                LOWER(REGEXP_REPLACE(
                    REGEXP_REPLACE(name, '\\s*\\(.*?\\)\\s*', '', 'g'),
                    '[^a-zA-Z0-9]', '', 'g'
                )) AS norm_name,
                -- Sport canónico (laliga/epl/liga_mx → soccer)
                CASE
                    WHEN sport_code IN (
                        'laliga', 'epl', 'liga_mx', 'soccer', 'seriea',
                        'bundesliga', 'ligue1', 'mls', 'ucl', 'uel', 'efl_champ',
                        'eredivisie', 'primeira_liga', 'a_league', 'j_league',
                        'k_league', 'brasileirao', 'argentine_primera',
                        'copa_libertadores', 'copa_sudamericana', 'fa_cup',
                        'super_lig', 'super_league'
                    ) THEN 'soccer'
                    WHEN sport_code IN ('nba', 'wnba', 'ncaab', 'euroleague') THEN 'nba'
                    WHEN sport_code IN ('mlb', 'kbo', 'npb') THEN 'mlb'
                    WHEN sport_code IN ('nfl', 'ncaaf', 'cfl', 'ufl') THEN 'nfl'
                    WHEN sport_code IN ('nhl', 'ahl') THEN 'nhl'
                    ELSE sport_code
                END AS canonical_sport,
                -- Score de "canonicidad" — el de mayor score se queda
                CASE
                    -- Penaliza variantes de market paralelo
                    WHEN name ~* '\\((Corners|Bookings|Cards|Goals|Shots|Sets|Points|Games|Cyber)\\)$' THEN 0
                    -- Bonus si sport_code ya es canónico
                    WHEN sport_code IN ('soccer', 'nba', 'mlb', 'nfl', 'nhl') THEN 100
                    ELSE 50
                END AS canonicity_score
            FROM teams
            WHERE name IS NOT NULL
        ),
        canonical_per_group AS (
            SELECT
                norm_name,
                -- Para cada grupo, escoger el team canónico:
                -- 1. Mayor canonicity_score (sin paréntesis + sport canónico)
                -- 2. Menor id como tiebreaker (más antiguo)
                FIRST_VALUE(id) OVER (
                    PARTITION BY norm_name
                    ORDER BY canonicity_score DESC, id ASC
                ) AS canonical_id,
                FIRST_VALUE(canonical_sport) OVER (
                    PARTITION BY norm_name
                    ORDER BY canonicity_score DESC, id ASC
                ) AS canonical_sport,
                id AS old_id
            FROM normalized
            WHERE norm_name != ''
        )
        SELECT
            old_id,
            canonical_id AS new_id,
            canonical_sport
        FROM canonical_per_group
        WHERE old_id != canonical_id;

        CREATE INDEX idx_tcm_old ON _team_consolidation_map (old_id);
        CREATE INDEX idx_tcm_new ON _team_consolidation_map (new_id);
        """
    )

    # 2. UPDATE en cascada sobre cada FK. Usamos SAVEPOINT por tabla para que
    # un fallo (ej. unique-violation en h2h_history(team_a, team_b)) no aborte
    # toda la migración — la tabla simplemente queda con la fila old apuntando
    # al old_id huérfano, y el DELETE final lo respeta.
    from sqlalchemy import text as _sa_text

    bind = op.get_bind()
    for table, column in _TEAMS_FK_REFERENCES:
        # Verificar que la tabla existe (algunas son opcionales)
        check = bind.execute(
            _sa_text(
                """
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = current_schema() AND table_name = :t
                """
            ),
            {"t": table},
        ).first()
        if check is None:
            continue
        sp_name = f"sp_consolidate_{table}".replace(".", "_")
        try:
            bind.execute(_sa_text(f"SAVEPOINT {sp_name}"))
            bind.execute(
                _sa_text(
                    f"""
                    UPDATE {table} t SET {column} = m.new_id
                    FROM _team_consolidation_map m
                    WHERE t.{column} = m.old_id
                    """
                )
            )
            bind.execute(_sa_text(f"RELEASE SAVEPOINT {sp_name}"))
        except Exception:
            # Conflict (ej. UNIQUE violation): rollback solo este UPDATE,
            # las demás tablas siguen. Las filas huérfanas en esta tabla
            # bloquearán el DELETE de su old_team_id (deseable: no perdemos
            # datos del team duplicado, solo no se consolida 100%).
            bind.execute(_sa_text(f"ROLLBACK TO SAVEPOINT {sp_name}"))

    # 3. Actualizar sport_code de los teams canónicos al canonical sport
    op.execute(
        """
        UPDATE teams t SET sport_code = m.canonical_sport
        FROM _team_consolidation_map m
        WHERE t.id = m.new_id
          AND t.sport_code != m.canonical_sport
          AND EXISTS (SELECT 1 FROM sports s WHERE s.code = m.canonical_sport);
        """
    )

    # 4. DELETE los teams duplicados (now que ya no son referenciados)
    # Con ON DELETE no-cascade, las FK que aún apuntan al old_id (por conflicto
    # de unique en step 2) bloquean el DELETE — eso es DESEABLE (no perdemos
    # datos). El team duplicado queda en DB pero sin tráfico nuevo.
    op.execute(
        """
        DELETE FROM teams
        WHERE id IN (SELECT old_id FROM _team_consolidation_map)
          AND NOT EXISTS (
              SELECT 1 FROM matches m
              WHERE m.home_team_id = teams.id OR m.away_team_id = teams.id
          )
          AND NOT EXISTS (SELECT 1 FROM players p WHERE p.team_id = teams.id)
          AND NOT EXISTS (SELECT 1 FROM lineups l WHERE l.team_id = teams.id)
          AND NOT EXISTS (
              SELECT 1 FROM h2h_history h
              WHERE h.team_a_id = teams.id OR h.team_b_id = teams.id
          );
        """
    )

    # 5. Opcional: cleanup de matches que quedaron con home_team_id == away_team_id
    # tras la consolidación (caso raro: si dos teams duplicados eran el mismo).
    op.execute(
        """
        UPDATE matches SET status = 'cancelled'
        WHERE home_team_id = away_team_id AND status != 'finished';
        """
    )

    # 6. Drop tabla temporal — ya no la necesitamos
    op.execute("DROP TABLE IF EXISTS _team_consolidation_map;")


def downgrade() -> None:
    # NO reversible — la consolidación destruye duplicados. Usar pg_restore
    # de un backup previo si hace falta rollback.
    raise NotImplementedError("0030 consolidation is destructive; restore from backup to rollback.")
