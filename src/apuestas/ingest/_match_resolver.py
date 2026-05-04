"""Fuzzy resolver compartido: normaliza nombres y hace pg_trgm match.

Usado por scrapers que no tienen external_id único (Caliente.mx, DK, FD, MGM,
Codere). Todos siembran sobre el catálogo de teams creado por Pinnacle con
external_id canónico; aquí solo fuzzy-match y, si no existe, crea el team con
external_id propio del scraper (`{source}:{sport}:{slug}`).

Hardening 2026-04-25 contra duplicados (309 teams duplicados detectados):
  1. Sport_code se normaliza con `canonical_sport_code` antes de buscar/crear.
     Antes: `laliga` y `soccer` eran sport_codes distintos → mismo team
     (Barcelona) creado dos veces. Ahora ambos resuelven a `soccer`.
  2. Nombres con sufijo `(Corners)/(Bookings)/(Cards)/(Goals)/(Sets)/(Shots)`
     se rechazan con ValueError — son markets paralelos del mismo team y
     causaban matches duplicados (match_id 116680 'Getafe (Corners)' vs
     match_id 455 'Getafe').
  3. Búsqueda fuzzy CROSS-SPORT con sport_code canónico, no estricto.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text

# Sufijos de markets paralelos que NO son teams reales — son derivative markets
# (corners, tarjetas, etc.) que algunos scrapers (Pinnacle especialmente)
# devuelven con el formato "TeamName (Market)". Crear teams con estos nombres
# generaba matches duplicados con cero cobertura de soft books.
_PROP_MARKET_SUFFIX_RE = re.compile(
    r"\s*\((Corners|Bookings|Cards|Goals|Shots|Sets|Points|Games|Cyber)\)\s*$",
    re.IGNORECASE,
)


def _is_prop_market_alias(name: str) -> bool:
    """True si el nombre es una variante de market paralelo, no un team real."""
    return bool(_PROP_MARKET_SUFFIX_RE.search(name))


def _strip_prop_suffix(name: str) -> str:
    """Quita sufijo `(Corners)` etc. para fuzzy match contra el team canónico."""
    return _PROP_MARKET_SUFFIX_RE.sub("", name).strip()


def normalize_name(name: str) -> str:
    """Lowercase + strip diacritics + collapse whitespace + strip prop suffix."""
    cleaned = _strip_prop_suffix(name)
    nfkd = unicodedata.normalize("NFKD", cleaned)
    ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(ascii_only.lower().split())


def _canonical_sport(sport_code: str) -> str:
    """Wrapper local de canonical_sport_code para evitar import circular.

    Mapea laliga/epl/liga_mx/seriea/bundesliga/etc. → 'soccer' canónico.
    """
    try:
        from apuestas.sports import canonical_sport_code

        return canonical_sport_code(sport_code)
    except Exception:
        return sport_code


async def resolve_or_create_team(
    session: Any,
    *,
    sport_code: str,
    name: str,
    source: str,
    threshold: float = 0.55,
) -> int:
    """Fuzzy match contra `teams` (pg_trgm similarity). Si no hay match, crea.

    Sport_code se canonicaliza antes de buscar/crear (laliga → soccer) para
    evitar duplicados. Nombres con sufijo `(Corners)` etc. se redirigen al
    team canónico (sin sufijo).

    OPTIMIZACIÓN: cache Valkey TTL 1h. La query similarity() es la #1 más
    cara del bot (350k calls/día @ 7.7ms = 45 min DB time). Con cache hit
    rate >85% (mismos teams aparecen en cada poll), reduce a ~7 min.
    """
    if _is_prop_market_alias(name):
        # Resuelve contra el team SIN sufijo. No creamos un team nuevo para
        # market paralelo — apuntamos al team principal.
        canonical_name = _strip_prop_suffix(name)
        if canonical_name and canonical_name != name:
            return await resolve_or_create_team(
                session,
                sport_code=sport_code,
                name=canonical_name,
                source=source,
                threshold=threshold,
            )

    canonical_sp = _canonical_sport(sport_code)
    normalized = normalize_name(name)

    # Cache lookup
    from apuestas.cache import cache_get as _cg
    from apuestas.cache import cache_set as _cs

    cache_key = f"team:resolve:{canonical_sp}:{normalized}"
    cached = await _cg(cache_key)
    if cached is not None and isinstance(cached, int):
        return cached

    result = await session.execute(
        text(
            """
            SELECT id, similarity(unaccent(lower(name)), :q) AS sim
            FROM teams
            WHERE sport_code = :sp
              AND similarity(unaccent(lower(name)), :q) > :th
              -- Excluir teams que son markets paralelos (no team real)
              AND name !~* '\\((Corners|Bookings|Cards|Goals|Shots|Sets|Points|Games|Cyber)\\)$'
            ORDER BY sim DESC
            LIMIT 1
            """
        ),
        {"sp": canonical_sp, "q": normalized, "th": threshold},
    )
    row = result.first()
    if row is not None:
        team_id = int(row.id)
        await _cs(cache_key, team_id, ttl_seconds=3600)
        return team_id

    ext = f"{source}:{canonical_sp}:{normalized.replace(' ', '-')}"
    insert = await session.execute(
        text(
            """
            INSERT INTO teams (external_id, sport_code, name, active)
            VALUES (:ext, :sp, :name, true)
            ON CONFLICT (external_id) DO UPDATE SET name = EXCLUDED.name
            RETURNING id
            """
        ),
        {"ext": ext, "sp": canonical_sp, "name": _strip_prop_suffix(name).strip() or name.strip()},
    )
    created = insert.first()
    if created is None:
        msg = f"INSERT teams RETURNING falló para {canonical_sp}/{name}"
        raise RuntimeError(msg)
    team_id = int(created.id)
    await _cs(cache_key, team_id, ttl_seconds=3600)
    return team_id


async def resolve_or_create_match(
    session: Any,
    *,
    sport_code: str,
    home_name: str,
    away_name: str,
    start_time: datetime | None,
    source: str,
    window_hours: int = 12,
    season: str | None = None,
    external_ids: dict[str, str] | None = None,
) -> int | None:
    """Fuzzy resolve de ambos equipos + ventana ±N h → match_id.

    Si el match no existe en la ventana se crea con status='scheduled'.
    Retorna None si home == away tras fuzzy (caso de error del scraper).

    `external_ids` (plan §5.1) permite al ingester pasar los IDs nativos
    de su fuente para popular las columnas `external_id_odds_api`,
    `external_id_nba`, `external_id_nhl` añadidas en la migración 0015.
    Esto es la **precondición crítica** para que `live_scores` resuelva
    por lookup exacto (capa 1) antes de caer a fuzzy — el bug observado
    el 2026-04-22/23 venía de que estas columnas siempre quedaban NULL.

    Sources aceptados en el dict:
      - "odds_api"  → external_id_odds_api
      - "nba"       → external_id_nba
      - "nhl"       → external_id_nhl

    Keys desconocidos se ignoran silenciosamente.
    """
    # Canonicalizar sport_code para evitar matches duplicados (laliga vs soccer).
    canonical_sp = _canonical_sport(sport_code)

    # Rechazar matches con teams alias de market paralelo. Esos derivative markets
    # son secundarios del partido principal y se ingestan por otra vía. Crear
    # match con `Getafe (Corners)` generaba duplicados con cero cobertura.
    if _is_prop_market_alias(home_name) or _is_prop_market_alias(away_name):
        return None

    home_id = await resolve_or_create_team(
        session, sport_code=canonical_sp, name=home_name, source=source
    )
    away_id = await resolve_or_create_team(
        session, sport_code=canonical_sp, name=away_name, source=source
    )
    if home_id == away_id:
        return None

    # Capa 1 — lookup exacto por external_id_* si el ingester lo proveyó.
    # Más rápido y robusto que fuzzy; maneja rescheduled games (un partido
    # que se mueve +24h conserva el mismo external_id en OddsAPI/nba_api).
    if external_ids:
        for src_key, col_name in (
            ("odds_api", "external_id_odds_api"),
            ("nba", "external_id_nba"),
            ("nhl", "external_id_nhl"),
        ):
            ext_val = external_ids.get(src_key)
            if not ext_val:
                continue
            direct = await session.execute(
                text(f"SELECT id FROM matches WHERE {col_name} = :v LIMIT 1"),
                {"v": str(ext_val)},
            )
            hit = direct.first()
            if hit is not None:
                return int(hit.id)

    if start_time is not None:
        window_start = start_time - timedelta(hours=window_hours)
        window_end = start_time + timedelta(hours=window_hours)
        # Busca coincidencia exacta O home/away flipped (mismo partido,
        # diferente fuente del scraper). Prioriza status='scheduled' y más cercano
        # al start_time recibido.
        existing = await session.execute(
            text(
                """
                SELECT id FROM matches
                WHERE sport_code = :sp
                  AND (
                    (home_team_id = :h AND away_team_id = :a)
                    OR (home_team_id = :a AND away_team_id = :h)
                  )
                  AND start_time BETWEEN :ws AND :we
                  AND status != 'cancelled'
                ORDER BY ABS(EXTRACT(EPOCH FROM (start_time - :target))) ASC
                LIMIT 1
                """
            ),
            {
                "sp": canonical_sp,
                "h": home_id,
                "a": away_id,
                "ws": window_start,
                "we": window_end,
                "target": start_time,
            },
        )
        row = existing.first()
        if row is not None:
            # Capa 1b — si encontramos match por fuzzy+ventana y el ingester
            # traía external_id_*, actualizamos columnas NULL (no sobrescribe
            # valores existentes, que podrían ser de otra fuente canónica).
            if external_ids:
                await _fill_null_external_ids(session, int(row.id), external_ids)
            return int(row.id)

    effective_start = start_time or datetime.now(tz=UTC)
    ext = f"{source}:{canonical_sp}:{home_id}-{away_id}:{effective_start.date().isoformat()}"
    # Auto-derive season si no se pasa: cross-year sports (NBA/NFL/NHL) usan "YYYY-YY",
    # single-year (MLB/tennis) usan "YYYY". Asume month-based cutoff por sport.
    if season is None:
        effective_season: str | None = None
        y = effective_start.year
        m = effective_start.month
        if canonical_sp in ("nba", "nhl"):
            effective_season = f"{y}-{str(y + 1)[-2:]}" if m >= 10 else f"{y - 1}-{str(y)[-2:]}"
        elif canonical_sp == "nfl":
            effective_season = f"{y}-{str(y + 1)[-2:]}" if m >= 8 else f"{y - 1}-{str(y)[-2:]}"
        elif canonical_sp == "soccer":
            # European calendar: July-May → season = Y-Y+1
            effective_season = f"{y}-{y + 1}" if m >= 7 else f"{y - 1}-{y}"
        elif canonical_sp in ("mlb", "tennis"):
            effective_season = str(y)
    else:
        effective_season = season
    ext_odds_api = (external_ids or {}).get("odds_api")
    ext_nba = (external_ids or {}).get("nba")
    ext_nhl = (external_ids or {}).get("nhl")

    insert = await session.execute(
        text(
            """
            INSERT INTO matches
              (external_id, sport_code, home_team_id, away_team_id, start_time,
               status, season, external_id_odds_api, external_id_nba, external_id_nhl)
            VALUES
              (:ext, :sp, :h, :a, :st, 'scheduled', :season,
               :ext_odds_api, :ext_nba, :ext_nhl)
            ON CONFLICT (external_id) DO UPDATE SET
              start_time = EXCLUDED.start_time,
              season = COALESCE(matches.season, EXCLUDED.season),
              -- Nunca sobrescribe un external_id_* ya poblado (distintas fuentes
              -- podrían reclamar el mismo match). Solo rellena NULLs.
              external_id_odds_api = COALESCE(matches.external_id_odds_api,
                                              EXCLUDED.external_id_odds_api),
              external_id_nba = COALESCE(matches.external_id_nba, EXCLUDED.external_id_nba),
              external_id_nhl = COALESCE(matches.external_id_nhl, EXCLUDED.external_id_nhl)
            RETURNING id
            """
        ),
        {
            "ext": ext,
            "sp": canonical_sp,
            "h": home_id,
            "a": away_id,
            "st": effective_start,
            "season": effective_season,
            "ext_odds_api": ext_odds_api,
            "ext_nba": ext_nba,
            "ext_nhl": ext_nhl,
        },
    )
    created = insert.first()
    if created is None:
        msg = f"INSERT matches RETURNING falló para {canonical_sp} {home_name} vs {away_name}"
        raise RuntimeError(msg)
    return int(created.id)


async def _fill_null_external_ids(
    session: Any,
    match_id: int,
    external_ids: dict[str, str],
) -> None:
    """Actualiza columnas `external_id_*` que estén NULL (nunca sobrescribe).

    Invocado cuando el match ya existe (hit fuzzy en capa 2/3) pero el
    ingester nuevo trae IDs que aún no habíamos capturado.
    """
    pairs: list[tuple[str, str]] = []
    for src_key, col_name in (
        ("odds_api", "external_id_odds_api"),
        ("nba", "external_id_nba"),
        ("nhl", "external_id_nhl"),
    ):
        val = external_ids.get(src_key)
        if val:
            pairs.append((col_name, str(val)))
    if not pairs:
        return
    # SQL SET dinámico construido solo con nombres de columna del whitelist
    # de arriba (no vienen de input externo); los valores van parametrizados.
    set_clause = ", ".join(f"{col} = COALESCE({col}, :v_{i})" for i, (col, _) in enumerate(pairs))
    params: dict[str, Any] = {"id": match_id}
    for i, (_, v) in enumerate(pairs):
        params[f"v_{i}"] = v
    await session.execute(
        text(f"UPDATE matches SET {set_clause} WHERE id = :id"),
        params,
    )
