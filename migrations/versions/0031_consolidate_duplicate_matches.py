"""Consolida matches duplicados (161 detectados) post-Sprint 13 + 0030.

Revision ID: 0031
Revises: 0030
Create Date: 2026-04-25

Tras consolidar teams duplicados (0030), aún quedan ~161 grupos de matches
duplicados: el mismo partido (mismo start_time + home + away) ingestado por
distintos scrapers (Pinnacle, OddsAPI, Sofascore, Caliente) cuando los
teams todavía vivían como entidades separadas.

Trigger del bug: ya consolidados los teams, dos matches con identico
(start_time, home_team_id, away_team_id) coexisten porque NO existía
unique constraint sobre la triada y el detector elegía el match con menor
cobertura de odds → no_qualifying_offer.

Algoritmo:
  1. Para cada grupo (start_time, home_team_id, away_team_id), elegir el
     canónico:
     - Preferencia 1: el que tenga `external_id` Pinnacle (sharp source).
     - Preferencia 2: el de menor `id` (más antiguo).
  2. Migrar FK que referencian `matches.id` al canónico (odds_history,
     pick_alerts, decisions, lineups, weather_forecasts, etc.).
  3. DELETE los matches duplicados sin references restantes.

Idempotente: si una tabla no existe, sigue. Si los duplicados ya fueron
consolidados (después de la primera aplicación SQL ad-hoc en producción
2026-04-25), queda no-op.

DOWN: NO reversible. Backup `backups/pre_0030_consolidate_20260425_074029.sql`
ya cubre el estado pre-consolidación.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0031"
down_revision: str | None = "0030"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# (table, column) que referencian matches.id — confirmados en producción
# 2026-04-25 via information_schema.table_constraints. Si una tabla no
# existe en este deploy, el SAVEPOINT skip la deja intacta.
_MATCHES_FK_REFERENCES: tuple[tuple[str, str], ...] = (
    ("arb_opportunities", "match_id"),
    ("decision_log", "event_id"),
    ("injury_reports_normalized", "match_id"),
    ("lineups", "match_id"),
    ("match_canonical", "primary_match_id"),
    ("match_coaches", "match_id"),
    ("match_officials", "match_id"),
    ("match_referees", "match_id"),
    ("nba_hustle_stats", "match_id"),
    ("nba_lineup_5man_efficiency", "match_id"),
    ("odds_history", "match_id"),
    ("pick_alerts", "match_id"),
    ("pick_analysis", "event_id"),
    ("pick_closing_lines", "match_id"),
    ("play_by_play", "match_id"),
    ("player_game_logs", "match_id"),
    ("player_prop_lines", "match_id"),
    ("player_tracking_proxies", "match_id"),
    ("predictions", "match_id"),
    ("public_betting_snapshots", "match_id"),
    ("settlement_queue", "match_id"),
    ("shadow_pinnacle_predictions", "match_id"),
    ("steam_moves", "match_id"),
    ("tennis_match_details", "match_id"),
    ("travel_log", "match_id"),
    ("travel_log", "previous_match_id"),
    ("weather_forecast", "match_id"),
)


def upgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS _match_consolidation_map;
        CREATE UNLOGGED TABLE _match_consolidation_map AS
        WITH grouped AS (
            SELECT
                id,
                start_time,
                home_team_id,
                away_team_id,
                external_id,
                -- Score canonicidad: external_id tipo pinnacle = más alto
                CASE
                    WHEN external_id LIKE 'pinnacle:%' THEN 200
                    WHEN external_id LIKE 'odds_api:%' THEN 150
                    WHEN external_id LIKE 'sofascore:%' THEN 100
                    WHEN external_id IS NOT NULL AND external_id != '' THEN 50
                    ELSE 0
                END AS canonicity_score
            FROM matches
            WHERE start_time IS NOT NULL
              AND home_team_id IS NOT NULL
              AND away_team_id IS NOT NULL
              AND home_team_id != away_team_id
        ),
        canonical_per_group AS (
            SELECT
                start_time,
                home_team_id,
                away_team_id,
                FIRST_VALUE(id) OVER (
                    PARTITION BY start_time, home_team_id, away_team_id
                    ORDER BY canonicity_score DESC, id ASC
                ) AS canonical_id,
                id AS old_id
            FROM grouped
        )
        SELECT
            old_id,
            canonical_id AS new_id
        FROM canonical_per_group
        WHERE old_id != canonical_id;

        CREATE INDEX idx_mcm_old ON _match_consolidation_map (old_id);
        CREATE INDEX idx_mcm_new ON _match_consolidation_map (new_id);
        """
    )

    from sqlalchemy import text as _sa_text

    bind = op.get_bind()
    for table, column in _MATCHES_FK_REFERENCES:
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
        sp_name = f"sp_match_consolidate_{table}".replace(".", "_")
        try:
            bind.execute(_sa_text(f"SAVEPOINT {sp_name}"))
            bind.execute(
                _sa_text(
                    f"""
                    UPDATE {table} t SET {column} = m.new_id
                    FROM _match_consolidation_map m
                    WHERE t.{column} = m.old_id
                    """
                )
            )
            bind.execute(_sa_text(f"RELEASE SAVEPOINT {sp_name}"))
        except Exception:
            bind.execute(_sa_text(f"ROLLBACK TO SAVEPOINT {sp_name}"))

    # DELETE matches duplicados row-por-row con SAVEPOINT. Si una FK queda
    # apuntando al old_id (caso raro: tabla nueva no listada en _MATCHES_FK_REFERENCES),
    # el DELETE de esa fila individual rollback al SAVEPOINT y la migración
    # sigue. NO abortamos la migración entera por una fila huérfana.
    bind.execute(
        _sa_text(
            """
            DO $do$
            DECLARE
                _old_id BIGINT;
            BEGIN
                FOR _old_id IN SELECT old_id FROM _match_consolidation_map LOOP
                    BEGIN
                        DELETE FROM matches WHERE id = _old_id;
                    EXCEPTION
                        WHEN foreign_key_violation THEN
                            -- Mark as cancelled (no se procesa más) en lugar de borrar
                            UPDATE matches SET status='cancelled' WHERE id = _old_id;
                    END;
                END LOOP;
            END;
            $do$;
            """
        )
    )

    # Unique constraint preventiva: evita reaparición de duplicados.
    # WHERE clause excluye filas con NULL (no aplican a la regla).
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_matches_identity
        ON matches (start_time, home_team_id, away_team_id)
        WHERE start_time IS NOT NULL
          AND home_team_id IS NOT NULL
          AND away_team_id IS NOT NULL
          AND status != 'cancelled';
        """
    )

    op.execute("DROP TABLE IF EXISTS _match_consolidation_map;")


def downgrade() -> None:
    # Drop solo el unique index (la consolidación es destructiva e
    # irreversible). Si se necesita rollback completo, restaurar
    # backups/pre_0030_consolidate_20260425_074029.sql con pg_restore.
    op.execute("DROP INDEX IF EXISTS uq_matches_identity;")
