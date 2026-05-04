"""Auto-repair de identity resolver: leagues mal-asignadas a teams.

Detectado en post-mortem 25-26 abr 2026: matches europeos regionales (Suiza,
Noruega, Turquía, segundas) tenían `league_id=NULL` o apuntaban a una liga
incorrecta (Lillestrom→Ligue1, Servette→Serie A). El flow live_scores filtra
sport_keys por league.external_id para ahorrar créditos OddsAPI; con misassign
nunca llamaba a las ligas correctas y los picks quedaban pending.

Extensión 2026-04-28: detectado bug PSG-Bayern (#116617) donde el team
`Paris Saint-Germain (Corners)` se persistió como home_team del match
principal, bloqueando el filtro `t1.name NOT LIKE '%(Corners)%'` en
`get_upcoming_events`. Ahora canonicaliza FK a teams sin sufijo derivativo
cuando la versión canónica existe.

También extiende mapping a Conmebol/Brasil/Argentina (Boca/Cruzeiro/Sao
Paulo/Santos) que llegaban con `league_id=NULL` y caían al fallback
`soccer_liga_mx` (que predecía la prior promedio Liga MX para todo, causando
draw guard del 26.1% que bloqueaba todos los picks Sudamericanos).

Este módulo se invoca automáticamente desde `catchup_flow` antes de
`live_scores_flow`. Es idempotente: si las ligas ya existen y los teams ya
están asignados a su liga real, no hace cambios.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

# Sufijos de mercado derivativo que aparecen en el feed de Pinnacle como
# "team name" en matches paralelos (corners, cards, goals, etc.). Cuando un
# match principal queda con un team así, el filtro de `get_upcoming_events`
# lo excluye y el detector nunca lo evalúa.
_DERIVATIVE_SUFFIXES: tuple[str, ...] = (
    "(Corners)",
    "(Bookings)",
    "(Cards)",
    "(Goals)",
    "(Shots)",
    "(Sets)",
    "(Games)",
    "(Points)",
)


# Ligas que el seed inicial no incluyó pero tenemos cobertura en TheOddsAPI.
_NEW_LEAGUES: list[dict[str, str]] = [
    {
        "name": "Eliteserien",
        "country": "NOR",
        "external_id": "fd_norway_eliteserien",
        "sport_code": "soccer",
    },
    {
        "name": "Swiss Super League",
        "country": "SUI",
        "external_id": "fd_switzerland_super",
        "sport_code": "soccer",
    },
]

# Mapping team-name → league.external_id. Solo equipos confirmados con
# misassignment en auditoría. Conservador: añadimos más sólo cuando se
# detecten nuevos casos vía watchdog.stuck_picks.
_TEAM_TO_LEAGUE: dict[str, str] = {
    # Eliteserien (Noruega)
    "Lillestrom": "fd_norway_eliteserien",
    "Bodø/Glimt": "fd_norway_eliteserien",
    "Aalesunds FK": "fd_norway_eliteserien",
    "Kristiansund BK": "fd_norway_eliteserien",
    # Swiss Super League
    "Servette": "fd_switzerland_super",
    "FC Winterthur": "fd_switzerland_super",
    # Turkey Super Lig (id ya existe en seed)
    "Galatasaray": "fd_turkey_super",
    "Fenerbahce": "fd_turkey_super",
    # La Liga 2 — equipos en segunda mal-etiquetados a primera
    "Leganes": "fd_la_liga_2",
    "Andorra": "fd_la_liga_2",
}

# Mapping team-name → league_id directo (cuando la liga ya existe en seed
# pero el team venía sin asignación). Para Conmebol los `external_id` no son
# triviales porque un mismo team aparece en varias copas; usamos el liga
# DOMÉSTICA como ancla (el match Copa se reasigna por context cuando
# `_repair_specific_matches` detecta external_id `theoddsapi:soccer:...`
# proveniente del key Conmebol).
_TEAM_TO_LEAGUE_ID: dict[str, int] = {
    # Brasileirão Serie A (id=28)
    "Cruzeiro": 28,
    "Sao Paulo": 28,
    "São Paulo": 28,
    "Santos": 28,
    "Flamengo": 28,
    "Palmeiras": 28,
    "Corinthians": 28,
    "Fluminense": 28,
    "Internacional": 28,
    "Gremio": 28,
    "Grêmio": 28,
    "Atletico Mineiro": 28,
    "Atlético Mineiro": 28,
    "Botafogo": 28,
    # Liga Argentina (id=29)
    "Boca Juniors": 29,
    "River Plate": 29,
    "Racing Club": 29,
    "Independiente": 29,
    "San Lorenzo": 29,
    "CA Lanús": 29,
    "CA Lanus": 29,
    "Estudiantes": 29,
    "Velez Sarsfield": 29,
    "Vélez Sarsfield": 29,
    "Rosario Central": 29,
    "Newell's Old Boys": 29,
    "Talleres": 29,
}

# Mapping match `external_id` prefix (sport_key del feed Odds API) → league_id.
# Cuando el ingester persistió matches con sport_key sin asignación de liga,
# este mapping permite asignar retroactivamente. La regla aplica solo a matches
# con `league_id IS NULL`.
_SPORT_KEY_TO_LEAGUE_ID: dict[str, int] = {
    "soccer_uefa_champs_league": 24,
    "soccer_uefa_europa_league": 36,
    "soccer_brazil_campeonato": 28,
    "soccer_argentina_primera_division": 29,
    "soccer_conmebol_copa_libertadores": 26,
    "soccer_conmebol_copa_sudamericana": 27,
    "soccer_mexico_ligamx": 20,
    "soccer_usa_mls": 22,
}


async def _upsert_leagues() -> int:
    inserted = 0
    async with session_scope() as session:
        for league in _NEW_LEAGUES:
            existing = (
                await session.execute(
                    text("SELECT id FROM leagues WHERE external_id = :ext"),
                    {"ext": league["external_id"]},
                )
            ).first()
            if existing:
                continue
            await session.execute(
                text(
                    """
                    INSERT INTO leagues (name, country, external_id, sport_code)
                    VALUES (:name, :country, :ext, :sport)
                    ON CONFLICT (external_id) DO NOTHING
                    """
                ),
                {
                    "name": league["name"],
                    "country": league["country"],
                    "ext": league["external_id"],
                    "sport": league["sport_code"],
                },
            )
            inserted += 1
            logger.info("identity_repair.league_inserted", **league)
    return inserted


async def _reassign() -> dict[str, int]:
    teams_updated = 0
    matches_updated = 0
    async with session_scope() as session:
        for team_name, target_ext in _TEAM_TO_LEAGUE.items():
            league_row = (
                await session.execute(
                    text("SELECT id FROM leagues WHERE external_id = :ext"),
                    {"ext": target_ext},
                )
            ).first()
            if league_row is None:
                continue
            target_lid = int(league_row.id)

            team_row = (
                await session.execute(
                    text(
                        """
                        SELECT id, league_id FROM teams
                        WHERE name = :n AND sport_code = 'soccer'
                        """
                    ),
                    {"n": team_name},
                )
            ).first()
            if team_row is None:
                continue
            if team_row.league_id == target_lid:
                continue

            await session.execute(
                text("UPDATE teams SET league_id = :lid WHERE id = :tid"),
                {"lid": target_lid, "tid": int(team_row.id)},
            )
            teams_updated += 1
            logger.info(
                "identity_repair.team_reassigned",
                team=team_name,
                from_league=team_row.league_id,
                to_league=target_lid,
            )

            # Cascadear a matches donde el team es HOME (source of truth).
            result = await session.execute(
                text(
                    """
                    UPDATE matches SET league_id = :lid
                    WHERE home_team_id = :tid
                      AND (league_id IS NULL OR league_id <> :lid)
                    """
                ),
                {"lid": target_lid, "tid": int(team_row.id)},
            )
            n = int(result.rowcount or 0)
            matches_updated += n
            if n > 0:
                logger.info("identity_repair.matches_reassigned", team=team_name, n=n)

    return {"teams": teams_updated, "matches": matches_updated}


async def _migrate_team_aux_data(
    session: Any, derivative_id: int, canonical_id: int
) -> dict[str, int]:
    """Migra datos auxiliares del team derivativo al canónico cuando el canónico
    no tenga su propia versión. Sin esto, canonicalizar `home_team_id` rompía
    el lookup de `team_strength_bayesian` (PSG: la entry vivía en team 927
    `(Corners)`, no en 21797 canónico → DC fallback retornaba None y los matches
    se quedaban sin predicción).

    Idempotente: solo migra cuando el canónico NO tiene entry. Si ambos
    tienen, prefiere el canónico (más reciente, más confiable).
    """
    migrated = {}
    # team_strength_bayesian
    res = await session.execute(
        text(
            """
            INSERT INTO team_strength_bayesian
              (team_id, attack_rating, defense_rating, variance, n_matches, updated_at)
            SELECT :cid, attack_rating, defense_rating, variance, n_matches, updated_at
            FROM team_strength_bayesian
            WHERE team_id = :did
            ON CONFLICT (team_id) DO NOTHING
            """
        ),
        {"did": derivative_id, "cid": canonical_id},
    )
    migrated["team_strength_bayesian"] = int(res.rowcount or 0)

    # Propagar teams.league_id del derivativo al canónico si éste no lo tiene.
    res2 = await session.execute(
        text(
            """
            UPDATE teams SET league_id = (
                SELECT league_id FROM teams WHERE id = :did
            )
            WHERE id = :cid AND league_id IS NULL
              AND (SELECT league_id FROM teams WHERE id = :did) IS NOT NULL
            """
        ),
        {"did": derivative_id, "cid": canonical_id},
    )
    migrated["team_league_id"] = int(res2.rowcount or 0)
    return migrated


async def _canonicalize_derivative_team_fks() -> int:
    """Reasigna FK home/away de matches scheduled cuyo team tiene sufijo
    derivativo `(Corners)`/`(Goals)`/etc. al team canónico (mismo nombre sin
    el sufijo) cuando este último existe en `teams`.

    Maneja la collision contra `uq_matches_identity` (start_time, home, away):
    si tras canonicalizar el match resultaría duplicado de otro existente,
    cancelamos el derivativo (status='cancelled') en vez de update — así
    el match canónico (con sus odds reales) queda como source of truth y el
    derivativo deja de aparecer en `get_upcoming_events`.

    Conserva el team derivativo intacto (no lo borra; otros matches históricos
    de markets paralelos pueden depender de él). Solo cambia el FK del match
    principal o lo cancela.

    Idempotente: si el match ya apunta al team canónico, no toca.
    """
    repaired_updated = 0
    repaired_cancelled = 0
    async with session_scope() as session:
        for suffix in _DERIVATIVE_SUFFIXES:
            pattern = f"% {suffix}"
            rows = await session.execute(
                text(
                    """
                    SELECT id, name, sport_code FROM teams
                    WHERE name LIKE :p
                    """
                ),
                {"p": pattern},
            )
            for r in rows:
                derivative_id = int(r.id)
                canonical_name = r.name.replace(f" {suffix}", "").strip()
                if not canonical_name:
                    continue
                canon = (
                    await session.execute(
                        text(
                            """
                            SELECT id FROM teams
                            WHERE name = :n AND sport_code = :sp
                              AND id <> :did
                            LIMIT 1
                            """
                        ),
                        {"n": canonical_name, "sp": r.sport_code, "did": derivative_id},
                    )
                ).first()
                if canon is None:
                    continue
                canonical_id = int(canon.id)

                # Migrar datos auxiliares (strength bayesian, etc.) ANTES de
                # canonicalizar. Sin esto, la entry vivía en el team derivativo
                # y el lookup post-canonicalize fallaba.
                aux = await _migrate_team_aux_data(session, derivative_id, canonical_id)
                if any(v > 0 for v in aux.values()):
                    logger.info(
                        "identity_repair.aux_data_migrated",
                        derivative_id=derivative_id,
                        canonical_id=canonical_id,
                        **aux,
                    )

                # Match-by-match: detectar colisión con uq_matches_identity
                # antes de UPDATE. Si existe match canónico paralelo → cancel
                # el derivativo. Si no → safe to update.
                derivative_matches = await session.execute(
                    text(
                        """
                        SELECT id, home_team_id, away_team_id, start_time, status
                        FROM matches
                        WHERE (home_team_id = :did OR away_team_id = :did)
                          AND status = 'scheduled'
                          AND start_time BETWEEN NOW() - INTERVAL '1 hour'
                                              AND NOW() + INTERVAL '7 days'
                        """
                    ),
                    {"did": derivative_id},
                )
                for dm in derivative_matches.fetchall():
                    new_home = canonical_id if dm.home_team_id == derivative_id else dm.home_team_id
                    new_away = canonical_id if dm.away_team_id == derivative_id else dm.away_team_id
                    collision = (
                        await session.execute(
                            text(
                                """
                                SELECT id FROM matches
                                WHERE start_time = :st
                                  AND home_team_id = :h
                                  AND away_team_id = :a
                                  AND id <> :mid
                                LIMIT 1
                                """
                            ),
                            {
                                "st": dm.start_time,
                                "h": new_home,
                                "a": new_away,
                                "mid": dm.id,
                            },
                        )
                    ).first()
                    if collision is not None:
                        await session.execute(
                            text(
                                """
                                UPDATE matches SET status = 'cancelled'
                                WHERE id = :mid AND status = 'scheduled'
                                """
                            ),
                            {"mid": dm.id},
                        )
                        repaired_cancelled += 1
                        logger.info(
                            "identity_repair.derivative_match_cancelled",
                            derivative_match_id=int(dm.id),
                            canonical_match_id=int(collision.id),
                            derivative_team=r.name,
                        )
                    else:
                        await session.execute(
                            text(
                                """
                                UPDATE matches
                                SET home_team_id = :h, away_team_id = :a
                                WHERE id = :mid
                                """
                            ),
                            {"h": new_home, "a": new_away, "mid": dm.id},
                        )
                        repaired_updated += 1
                        logger.info(
                            "identity_repair.derivative_team_fk_canonicalized",
                            match_id=int(dm.id),
                            derivative_id=derivative_id,
                            derivative_name=r.name,
                            canonical_id=canonical_id,
                            canonical_name=canonical_name,
                        )
    return repaired_updated + repaired_cancelled


async def _assign_league_id_by_team_name() -> int:
    """Asigna league_id a matches con `league_id IS NULL` usando dos fuentes
    en cascada:

    1. **Source of truth dinámica**: `teams.league_id` ya asignado.
       Si el equipo home tiene league_id en la tabla `teams`, propagamos.
       Esto **auto-aprende** del trabajo previo de identity_repair y de los
       UPSERTs del ingester. Sin hardcodear nada nuevo.

    2. **Fallback hardcoded** (`_TEAM_TO_LEAGUE_ID`): solo aplica cuando el
       team aún no tiene `league_id` en `teams` (caso initial-bootstrap o
       teams nuevos del feed sin league_id resuelto). Tras aplicar, también
       persiste a `teams.league_id` para que la próxima ejecución use la
       ruta dinámica.

    Tras esto, matches Brasileirão/Argentina caen al chain hierarchy correcto
    en lugar de al fallback `soccer_liga_mx` (que predecía la prior promedio
    Liga MX para todo y disparaba el draw guard).
    """
    matches_updated = 0
    async with session_scope() as session:
        # Capa 1 — propagar teams.league_id → matches.league_id
        result_dynamic = await session.execute(
            text(
                """
                UPDATE matches m
                SET league_id = t.league_id
                FROM teams t
                WHERE t.id = m.home_team_id
                  AND t.league_id IS NOT NULL
                  AND m.league_id IS NULL
                  AND m.status = 'scheduled'
                  AND m.start_time > NOW() - INTERVAL '1 day'
                """
            )
        )
        n_dynamic = int(result_dynamic.rowcount or 0)
        if n_dynamic > 0:
            matches_updated += n_dynamic
            logger.info(
                "identity_repair.league_propagated_from_team_table",
                matches=n_dynamic,
            )

        # Capa 2 — hardcoded bootstrap. También persiste a teams.league_id
        # para que la próxima vez no necesite el hardcode.
        for team_name, lid in _TEAM_TO_LEAGUE_ID.items():
            await session.execute(
                text(
                    """
                    UPDATE teams SET league_id = :lid
                    WHERE name = :n AND sport_code = 'soccer'
                      AND league_id IS NULL
                    """
                ),
                {"lid": lid, "n": team_name},
            )
            result = await session.execute(
                text(
                    """
                    UPDATE matches m
                    SET league_id = :lid
                    FROM teams t
                    WHERE t.id = m.home_team_id
                      AND t.name = :n
                      AND t.sport_code = 'soccer'
                      AND m.league_id IS NULL
                      AND m.status = 'scheduled'
                      AND m.start_time > NOW() - INTERVAL '1 day'
                    """
                ),
                {"lid": lid, "n": team_name},
            )
            n = int(result.rowcount or 0)
            if n > 0:
                matches_updated += n
                logger.info(
                    "identity_repair.league_assigned_by_team",
                    team=team_name,
                    league_id=lid,
                    matches=n,
                )
    return matches_updated


async def _assign_league_id_by_external_id() -> int:
    """Cuando el match `external_id` empieza con `theoddsapi:{sport_key}:...`,
    inferimos league_id desde el sport_key del feed.

    Hoy, el ingester persiste `external_id` como `theoddsapi:soccer:...` (sin
    sport_key). Esta función prepara el camino para cuando lo hagamos: si
    detectamos `theoddsapi:soccer_brazil_campeonato:...` extraemos la liga.

    Idempotente y fail-safe: si el formato no calza, no hace nada.
    """
    matches_updated = 0
    async with session_scope() as session:
        for sport_key, lid in _SPORT_KEY_TO_LEAGUE_ID.items():
            pattern = f"theoddsapi:{sport_key}:%"
            result = await session.execute(
                text(
                    """
                    UPDATE matches
                    SET league_id = :lid
                    WHERE external_id LIKE :p
                      AND league_id IS NULL
                      AND status = 'scheduled'
                    """
                ),
                {"lid": lid, "p": pattern},
            )
            n = int(result.rowcount or 0)
            if n > 0:
                matches_updated += n
                logger.info(
                    "identity_repair.league_assigned_by_sport_key",
                    sport_key=sport_key,
                    league_id=lid,
                    matches=n,
                )
    return matches_updated


async def _propagate_league_id_from_derivative_teams() -> int:
    """Para cada team con sufijo derivativo `(Corners)/(Goals)/...` que ya tiene
    `teams.league_id`, propaga el valor al team canónico (mismo nombre sin
    sufijo) cuando éste tenga `league_id IS NULL`.

    Sin esto, después de canonicalizar el FK del match (`team_id 927 → 21797`)
    el team canónico queda sin `league_id` y `_correct_misassigned_ucl_uel_matches`
    no puede aplicar la regla cross-league.
    """
    fixed = 0
    async with session_scope() as session:
        for suffix in _DERIVATIVE_SUFFIXES:
            pattern = f"% {suffix}"
            result = await session.execute(
                text(
                    """
                    UPDATE teams canon
                    SET league_id = deriv.league_id
                    FROM teams deriv
                    WHERE deriv.name LIKE :p
                      AND deriv.league_id IS NOT NULL
                      AND canon.sport_code = deriv.sport_code
                      AND canon.name = REPLACE(deriv.name, ' ' || :s, '')
                      AND canon.id <> deriv.id
                      AND canon.league_id IS NULL
                    """
                ),
                {"p": pattern, "s": suffix},
            )
            n = int(result.rowcount or 0)
            if n > 0:
                fixed += n
                logger.info(
                    "identity_repair.league_propagated_derivative_to_canonical",
                    suffix=suffix,
                    count=n,
                )
    return fixed


async def _infer_team_leagues_from_match_history() -> int:
    """Auto-aprende `teams.league_id` desde el histórico de matches finished.

    Reglas:
      - Solo aplica a teams con `league_id IS NULL` (no sobrescribe asignación
        explícita).
      - El `league_id` se infiere desde el modo (most-frequent) de los matches
        en los que ese team participó como home o away **y** la liga es
        doméstica (no UCL/UEL/cups que mezclan teams de varias ligas, esos
        están en `_CUP_LEAGUE_IDS` y se excluyen del cómputo).
      - Solo aplica si el modo cubre ≥60% de los matches del team y hay ≥3
        matches → evita ruido por friendly único o ad-hoc.

    Tras esto, `_correct_misassigned_ucl_uel_matches` puede detectar matches
    cross-liga (PSG-Bayern) sin necesidad de hardcoded mapping.
    """
    _CUP_LEAGUE_IDS = [24, 26, 27, 36]  # UCL, Libertadores, Sudamericana, UEL
    fixed = 0
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                WITH team_match_leagues AS (
                    SELECT t.id AS team_id, m.league_id, COUNT(*) AS n
                    FROM teams t
                    JOIN matches m ON m.home_team_id = t.id OR m.away_team_id = t.id
                    WHERE t.league_id IS NULL
                      AND t.sport_code = 'soccer'
                      AND m.league_id IS NOT NULL
                      AND m.league_id <> ALL(:cups)
                      AND m.start_time > NOW() - INTERVAL '2 years'
                    GROUP BY t.id, m.league_id
                ),
                ranked AS (
                    SELECT team_id, league_id, n,
                           SUM(n) OVER (PARTITION BY team_id) AS total,
                           ROW_NUMBER() OVER (PARTITION BY team_id ORDER BY n DESC) AS rk
                    FROM team_match_leagues
                ),
                modal AS (
                    SELECT team_id, league_id
                    FROM ranked
                    WHERE rk = 1 AND total >= 3 AND n::float / total >= 0.6
                )
                UPDATE teams SET league_id = modal.league_id
                FROM modal
                WHERE teams.id = modal.team_id AND teams.league_id IS NULL
                """
            ),
            {"cups": _CUP_LEAGUE_IDS},
        )
        fixed = int(result.rowcount or 0)
        if fixed > 0:
            logger.info("identity_repair.team_leagues_inferred", count=fixed)
    return fixed


async def _correct_misassigned_domestic_matches() -> int:
    """Cuando home y away pertenecen a la MISMA liga (en `teams.league_id`)
    pero el match tiene `matches.league_id` distinto → reasignar match a la
    liga real de los teams.

    Caso típico (bug 2026-05-02): match #116557 Palmeiras vs Santos llegó
    con `matches.league_id=20` (Liga MX) pero ambos teams están en
    Brasileirão (id=28). El hierarchy resolver cargaba `soccer_liga_mx`
    (modelo entrenado con teams MX) que NO conoce a Palmeiras/Santos →
    predicción degenerada al prior constante 33%/33%/33% de Liga MX.
    Resultado: pick emit con EV +7.8% que en realidad es solo arbitraje
    Pinnacle-vs-soft sin aporte de modelo propio.

    Idempotente. Solo reasigna matches scheduled próximos (con margen 12 h
    antes para captar matches ya in-play que se están analizando).
    """
    fixed = 0
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                UPDATE matches m
                SET league_id = ht.league_id
                FROM teams ht, teams at
                WHERE ht.id = m.home_team_id
                  AND at.id = m.away_team_id
                  AND ht.league_id IS NOT NULL
                  AND ht.league_id = at.league_id
                  AND m.league_id IS DISTINCT FROM ht.league_id
                  AND m.status = 'scheduled'
                  AND m.start_time BETWEEN NOW() - INTERVAL '12 hours'
                                       AND NOW() + INTERVAL '14 days'
                """
            )
        )
        fixed = int(result.rowcount or 0)
        if fixed > 0:
            logger.info(
                "identity_repair.misassigned_domestic_corrected",
                count=fixed,
            )
    return fixed


async def _correct_misassigned_ucl_uel_matches() -> int:
    """Detecta matches con teams de **ligas distintas** y los reasigna a UCL
    (league_id=24) si actualmente apuntan a una de las top 5 europeas.

    Lógica: PSG (Ligue 1) vs Bayern (Bundesliga) NO juegan en Bundesliga ni
    en Ligue 1. Son UCL/UEL por definición. Sin esta corrección el resolver
    aplica el modelo Bundesliga y predice ~16% para PSG (no está en su
    posterior) en lugar de la prob real ~50% que daría Dixon-Coles cross-liga.

    Idempotente y conservador: solo reasigna matches scheduled próximos.
    """
    fixed = 0
    async with session_scope() as session:
        # Top 5 european domestic leagues (PL, Bundesliga, Serie A, La Liga,
        # Ligue 1) y similares. Si home_league != away_league y el match
        # apunta a una de éstas, es UCL.
        result = await session.execute(
            text(
                """
                UPDATE matches m
                SET league_id = 24
                FROM teams ht, teams at
                WHERE ht.id = m.home_team_id
                  AND at.id = m.away_team_id
                  AND ht.league_id IS NOT NULL
                  AND at.league_id IS NOT NULL
                  AND ht.league_id <> at.league_id
                  AND m.league_id IN (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)
                  AND m.status = 'scheduled'
                  AND m.start_time BETWEEN NOW() - INTERVAL '1 hour'
                                       AND NOW() + INTERVAL '7 days'
                """
            )
        )
        fixed = int(result.rowcount or 0)
        if fixed > 0:
            logger.info("identity_repair.misassigned_ucl_corrected", count=fixed)
    return fixed


async def _purge_orphan_cancelled_duplicates() -> int:
    """Elimina matches `cancelled` que son duplicates exactos (mismo
    start_time + mismos teams o teams equivalentes) de un match `scheduled`
    activo y no tienen FK references (odds_history, pick_alerts, predictions).

    Caso típico: match #167526 (PSG canónico cancelled) duplicate de #116617
    (PSG (Corners) que ahora ya está canonicalizado y con odds reales).

    Idempotente y safe: solo borra cuando NO hay FK refs.
    """
    deleted = 0
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                WITH duplicates AS (
                    SELECT m1.id AS cancelled_id,
                           m2.id AS scheduled_id
                    FROM matches m1
                    JOIN matches m2
                      ON m1.start_time = m2.start_time
                     AND m1.home_team_id = m2.home_team_id
                     AND m1.away_team_id = m2.away_team_id
                     AND m1.id <> m2.id
                    WHERE m1.status = 'cancelled'
                      AND m2.status = 'scheduled'
                      AND m1.start_time > NOW() - INTERVAL '7 days'
                ),
                purgeable AS (
                    SELECT cancelled_id
                    FROM duplicates d
                    WHERE NOT EXISTS (
                        SELECT 1 FROM odds_history WHERE match_id = d.cancelled_id
                    )
                    AND NOT EXISTS (
                        SELECT 1 FROM pick_alerts WHERE match_id = d.cancelled_id
                    )
                    AND NOT EXISTS (
                        SELECT 1 FROM predictions WHERE match_id = d.cancelled_id
                    )
                )
                DELETE FROM matches
                WHERE id IN (SELECT cancelled_id FROM purgeable)
                RETURNING id
                """
            ),
        )
        rows = result.fetchall()
        deleted = len(rows)
        if deleted > 0:
            logger.info(
                "identity_repair.orphan_duplicates_purged",
                count=deleted,
                ids=[int(r.id) for r in rows][:10],
            )
    return deleted


async def repair_league_assignments() -> dict[str, int]:
    """Idempotente. Ejecutar al inicio de catchup. Sin args, sin side-effects extra.

    Ordenado por blast radius: ligas → teams → derivativos → matches. Cada
    capa puede activar la siguiente (un match con team canonicalizado puede
    necesitar reasignar league_id).
    """
    n_leagues = await _upsert_leagues()
    counts = await _reassign()
    counts["leagues_inserted"] = n_leagues
    counts["derivative_fk_canonicalized"] = await _canonicalize_derivative_team_fks()
    counts["matches_league_by_team"] = await _assign_league_id_by_team_name()
    counts["matches_league_by_sport_key"] = await _assign_league_id_by_external_id()
    counts["team_leagues_inferred"] = await _infer_team_leagues_from_match_history()
    counts["league_id_propagated_derivative"] = await _propagate_league_id_from_derivative_teams()
    counts["misassigned_domestic_corrected"] = await _correct_misassigned_domestic_matches()
    counts["misassigned_ucl_corrected"] = await _correct_misassigned_ucl_uel_matches()
    counts["orphan_duplicates_purged"] = await _purge_orphan_cancelled_duplicates()
    if any(v > 0 for v in counts.values()):
        logger.info("identity_repair.summary", **counts)
    return counts


__all__ = ["repair_league_assignments"]
