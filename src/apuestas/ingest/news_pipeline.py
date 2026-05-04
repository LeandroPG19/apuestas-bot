"""Pipeline end-to-end para noticias: ingest → embed → NER → persist.

Secuencia:
1. Consolidar RSS + Reddit + Bluesky (dedupe por url/hash).
2. Para cada artículo nuevo:
    - Embedding BGE-M3 (cache sha256).
    - NER via Qwen con grammar ner_extraction → sentiment + entidades.
    - Resolver team_ids y player_ids por nombre (pg_trgm similarity).
3. INSERT en news_articles (con embedding + metadata).
4. Para jugadores mencionados: también insertar en player_news.

Se ejecuta como TaskIQ task en worker-ml cada 30 min cuando el bot
está activo.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.ingest.news_rss import ingest_all_rss
from apuestas.ingest.reddit_social import fetch_all_bluesky, fetch_all_reddit
from apuestas.llm.client import LlamaClient
from apuestas.llm.embed import EmbedClient
from apuestas.llm.router import run_task
from apuestas.obs.logging import get_logger
from apuestas.schemas.llm import NERExtraction

logger = get_logger(__name__)


async def consolidate_sources() -> list[dict[str, Any]]:
    """Ejecuta los 3 ingestores en paralelo y dedupe."""
    results = await asyncio.gather(
        ingest_all_rss(),
        fetch_all_reddit(),
        fetch_all_bluesky(),
        return_exceptions=True,
    )
    all_entries: list[dict[str, Any]] = []
    for r in results:
        if isinstance(r, Exception):
            logger.warning("news_pipeline.source_failed", error=str(r))
            continue
        all_entries.extend(r)

    # Dedupe por URL; si URL nula, por hash de contenido
    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()
    unique: list[dict[str, Any]] = []
    for e in all_entries:
        url = e.get("url")
        if url:
            if url in seen_urls:
                continue
            seen_urls.add(url)
        else:
            content_hash = hashlib.sha256(
                (e.get("title", "") + e.get("content", "")).encode()
            ).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
        unique.append(e)

    logger.info("news_pipeline.consolidated", total=len(all_entries), unique=len(unique))
    return unique


def _filter_by_sport_focus(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Descarta noticias cuyos `sports` están todos en deportes desactivados.

    Si la noticia no declara `sports` (lista vacía o None), se mantiene
    (fallback conservador — preferimos un NER de más a perder un injury crítico).
    Si declara sports pero ninguno está habilitado, se descarta — no gastamos
    LLM en noticias de NHL/NFL/tennis/boxing cuando esos deportes están off.
    """
    from apuestas.betting.sport_focus import is_emit_enabled

    kept: list[dict[str, Any]] = []
    dropped = 0
    for e in entries:
        sports = e.get("sports") or []
        if not sports:
            kept.append(e)
            continue
        if any(is_emit_enabled(s) for s in sports):
            kept.append(e)
        else:
            dropped += 1
    if dropped:
        logger.info("news_pipeline.sport_focus_filter", kept=len(kept), dropped=dropped)
    return kept


# Sources con waste rate >40% (DB query 7 días: ESPN 45%, BBC 44%, Marca 38%).
# Aún se ingestan pero el embedding+search los recupera vía RAG; saltamos NER
# porque el ROI extracting entidades de generic news es bajo.
_NER_SKIP_SOURCES = frozenset(
    {
        "espn.com",
        "feeds.bbci.co.uk",
        "estaticos.marca.com",
    }
)

# Mínimo de caracteres para procesar NER. Median content actual = 133 chars,
# p95 = 381. Noticias <50 chars son típicamente headlines truncados sin
# contexto suficiente para extraer entidades fiables.
_NER_MIN_CONTENT_LEN = 50


async def _load_active_team_keywords() -> set[str]:
    """Carga set de tokens (lower-case) de teams activos en DB.

    Usado para pre-filter de relevancia: si el contenido no menciona ningún
    team activo, el NER probablemente devolverá [] (32% del waste actual).
    Cache process-local: se reconstruye al reiniciar el flow.
    """
    keywords: set[str] = set()
    try:
        async with session_scope() as session:
            result = await session.execute(
                text(
                    """
                    SELECT DISTINCT t.name, t.short_name, t.city
                    FROM teams t
                    JOIN matches m ON (m.home_team_id = t.id OR m.away_team_id = t.id)
                    WHERE m.start_time > NOW() - INTERVAL '14 days'
                       OR m.start_time < NOW() + INTERVAL '14 days'
                    LIMIT 5000
                    """
                )
            )
            for row in result.all():
                for field in (row.name, row.short_name, row.city):
                    if field and len(str(field)) >= 3:
                        keywords.add(str(field).lower())
    except Exception as exc:
        logger.debug("news_pipeline.keywords_load_fail", error=str(exc))
    return keywords


def _is_relevant_to_active_teams(entry: dict[str, Any], keywords: set[str]) -> bool:
    """True si título o contenido contiene al menos 1 keyword de team activo.

    Si keywords está vacío (carga falló) → fail-open (todas relevantes).
    Heurística cheap antes del LLM call (~$0.0001 vs $0.0002 NER).
    """
    if not keywords:
        return True
    text_blob = ((entry.get("title") or "") + " " + (entry.get("content") or "")).lower()
    return any(kw in text_blob for kw in keywords)


async def resolve_team_ids(team_names: list[str]) -> list[int]:
    """Busca team_ids por similaridad trigram. Devuelve [] si ninguno."""
    if not team_names:
        return []
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT DISTINCT t.id
                FROM teams t, unnest(CAST(:names AS text[])) AS n
                WHERE t.name = n
                   OR similarity(t.name, n) > 0.55
                LIMIT 20
                """
            ),
            {"names": team_names},
        )
        return [row[0] for row in result.all()]


async def resolve_player_ids(player_names: list[str]) -> dict[str, int]:
    """Match player_name → player_id por similaridad."""
    if not player_names:
        return {}
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT DISTINCT ON (LOWER(full_name)) id, full_name
                FROM players
                WHERE active = true
                  AND full_name ILIKE ANY(:patterns)
                LIMIT 50
                """
            ),
            {"patterns": [f"%{n}%" for n in player_names]},
        )
        matches = {row.full_name.lower(): row.id for row in result.all()}

    # Resolve cada input al best match
    out: dict[str, int] = {}
    for name in player_names:
        key = name.lower()
        if key in matches:
            out[name] = matches[key]
            continue
        # Fallback: find any substring match
        for full_name, pid in matches.items():
            if key in full_name or full_name in key:
                out[name] = pid
                break
    return out


async def process_article(
    entry: dict[str, Any],
    *,
    llm: LlamaClient,
    embed: EmbedClient,
) -> tuple[int | None, NERExtraction | None]:
    """Procesa un artículo: embedding + NER + INSERT.

    Returns: (news_article_id_insertado, extracción_ner) o (None, None) si falla.
    """
    title = entry.get("title") or ""
    content = entry.get("content") or ""
    if not (title or content):
        return None, None

    full_text = f"{title}\n\n{content}".strip()

    # NER via LLM
    try:
        ner_result = await run_task(
            task_kind="nlp/ner",
            version="v1",
            client=llm,
            render_vars={
                "content": full_text[:800],  # lead-paragraph; NER no necesita body completo
                "lang": entry.get("lang", "en"),
                "source": entry.get("source", "unknown"),
            },
        )
    except Exception as exc:
        logger.warning("news_pipeline.ner_failed", url=entry.get("url"), error=str(exc))
        return None, None

    if not isinstance(ner_result, NERExtraction):
        logger.warning(
            "news_pipeline.ner_unexpected_type",
            type=type(ner_result).__name__,
            url=entry.get("url"),
        )
        return None, None

    # Embedding (con cache)
    try:
        embedding = await embed.embed_one(full_text[:4000])
    except Exception as exc:
        logger.warning("news_pipeline.embed_failed", error=str(exc))
        embedding = None

    # Resolver entidades a ids
    team_ids = await resolve_team_ids(ner_result.teams)
    player_map = await resolve_player_ids(
        [p.name for p in ner_result.persons if p.role == "player"]
    )

    # INSERT news_articles. Separar path cuando embedding es None para evitar
    # AmbiguousParameterError (asyncpg no puede inferir tipo de NULL en CASE).
    vec_str = "[" + ",".join(f"{x:.6f}" for x in embedding) + "]" if embedding else None
    base_params = {
        "source": entry.get("source"),
        "url": entry.get("url"),
        "title": title,
        "content": content,
        "lang": entry.get("lang"),
        "published_at": entry.get("published_at"),
        "sports": entry.get("sports") or [],
        "teams_mentioned": team_ids,
        "players_mentioned": list(player_map.values()),
        "sentiment_score": float(ner_result.sentiment_score),
    }

    if vec_str is None:
        sql = """
            INSERT INTO news_articles
              (source, url, title, content, lang, published_at,
               sports, teams_mentioned, players_mentioned, sentiment_score, embedding)
            VALUES
              (:source, :url, :title, :content, :lang, :published_at,
               :sports, :teams_mentioned, :players_mentioned, :sentiment_score, NULL)
            ON CONFLICT (url) DO UPDATE
              SET title = EXCLUDED.title,
                  content = EXCLUDED.content,
                  sentiment_score = EXCLUDED.sentiment_score,
                  teams_mentioned = EXCLUDED.teams_mentioned,
                  players_mentioned = EXCLUDED.players_mentioned
            RETURNING id
        """
        params = base_params
    else:
        sql = """
            INSERT INTO news_articles
              (source, url, title, content, lang, published_at,
               sports, teams_mentioned, players_mentioned, sentiment_score, embedding)
            VALUES
              (:source, :url, :title, :content, :lang, :published_at,
               :sports, :teams_mentioned, :players_mentioned, :sentiment_score,
               CAST(:embedding AS vector))
            ON CONFLICT (url) DO UPDATE
              SET title = EXCLUDED.title,
                  content = EXCLUDED.content,
                  sentiment_score = EXCLUDED.sentiment_score,
                  teams_mentioned = EXCLUDED.teams_mentioned,
                  players_mentioned = EXCLUDED.players_mentioned,
                  embedding = EXCLUDED.embedding
            RETURNING id
        """
        params = {**base_params, "embedding": vec_str}

    async with session_scope() as session:
        result = await session.execute(text(sql), params)
        row = result.first()
        article_id = int(row[0]) if row else None

    # Por cada jugador mencionado en persons: también insertar en player_news
    for p in ner_result.persons:
        if p.role != "player" or p.name not in player_map:
            continue
        pid = player_map[p.name]
        async with session_scope() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO player_news
                      (player_id, player_name, source, url, title, content,
                       published_at, sentiment_score)
                    VALUES
                      (:player_id, :player_name, :source, :url, :title, :content,
                       :published_at, :sentiment_score)
                    ON CONFLICT DO NOTHING
                    """
                ),
                {
                    "player_id": pid,
                    "player_name": p.name,
                    "source": entry.get("source"),
                    "url": entry.get("url"),
                    "title": title,
                    "content": content,
                    "published_at": entry.get("published_at"),
                    "sentiment_score": float(ner_result.sentiment_score),
                },
            )

    # Persistir injuries extraídas por NER → injury_reports_normalized.
    # Hasta hoy se descartaban después de news_articles. Ahora cierran el gap
    # de soccer/cualquier sport sin feed dedicado: Reddit + RSS + Bluesky →
    # NER → injuries → tabla unificada que el detector consume.
    sport_codes = entry.get("sports") or []
    primary_sport = sport_codes[0] if sport_codes else None
    if primary_sport:
        from apuestas.sports import canonical_sport_code as _csc

        primary_sport = _csc(primary_sport)

    if ner_result.injuries and primary_sport:
        await _persist_ner_injuries(
            ner_result.injuries,
            sport_code=primary_sport,
            team_ids_in_article=team_ids,
            source=entry.get("source") or "news_pipeline",
        )

    return article_id, ner_result


async def _persist_ner_injuries(
    injuries: list,
    *,
    sport_code: str,
    team_ids_in_article: list[int],
    source: str,
) -> int:
    """Persist injuries extraídas por NER al store unificado.

    Si el NER reporta un team específico para la lesión, busca team_id por
    nombre. Si no, usa el primer team_id del artículo (heurística OK porque
    >90% noticias hablan de UN team principal).
    """
    n = 0
    for inj in injuries:
        try:
            player = (getattr(inj, "player", None) or "").strip()
            if not player:
                continue
            inj_team_name = (getattr(inj, "team", None) or "").strip()
            severity = (getattr(inj, "severity", None) or "questionable").lower()
            impact = (getattr(inj, "impact", None) or "")[:200]

            async with session_scope() as session:
                team_id: int | None = None
                if inj_team_name:
                    row = (
                        await session.execute(
                            text(
                                """
                                SELECT id FROM teams
                                WHERE sport_code = :sp
                                  AND (name = :tn OR similarity(name, :tn) > 0.55)
                                ORDER BY similarity(name, :tn) DESC NULLS LAST
                                LIMIT 1
                                """
                            ),
                            {"sp": sport_code, "tn": inj_team_name},
                        )
                    ).first()
                    team_id = int(row.id) if row else None
                # Fallback: primer team del artículo
                if team_id is None and team_ids_in_article:
                    team_id = team_ids_in_article[0]

                await session.execute(
                    text(
                        """
                        INSERT INTO injury_reports_normalized
                          (sport_code, team_id, player_name, status, reason,
                           reported_at, source)
                        VALUES (:sp, :tid, :p, :st, :rsn, NOW(), :src)
                        ON CONFLICT (team_id, player_name) WHERE team_id IS NOT NULL
                        DO UPDATE SET status = EXCLUDED.status,
                                      reason = EXCLUDED.reason,
                                      reported_at = EXCLUDED.reported_at,
                                      source = EXCLUDED.source
                        """
                    ),
                    {
                        "sp": sport_code,
                        "tid": team_id,
                        "p": player[:100],
                        "st": severity
                        if severity in {"out", "doubtful", "questionable", "probable", "active"}
                        else "questionable",
                        "rsn": impact,
                        "src": f"ner:{source}"[:50],
                    },
                )
                n += 1
        except Exception as exc:
            logger.debug("news_pipeline.persist_injury_fail", error=str(exc)[:80])
    if n:
        logger.info("news_pipeline.injuries_persisted", n=n, sport=sport_code)
    return n


def _dedupe_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedupe noticias por título similar (>85%) usando rapidfuzz.

    Muchas fuentes (ESPN, Marca, BBC) publican la misma historia con minor
    variations. Evita procesar NER duplicado → 33% menos LLM calls.
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return entries

    unique: list[dict[str, Any]] = []
    titles_seen: list[str] = []
    for e in entries:
        title = (e.get("title") or "").strip().lower()
        if not title:
            unique.append(e)
            continue
        is_dup = any(fuzz.ratio(title, t) > 85 for t in titles_seen)
        if not is_dup:
            unique.append(e)
            titles_seen.append(title)
    return unique


async def _cache_ner_hit(url: str, valkey_client: Any) -> bool:
    """Check if URL already procesed today (24h TTL). Cache key: ner:url:sha256(url)."""
    if valkey_client is None or not url:
        return False
    try:
        import hashlib

        key = f"ner:url:{hashlib.sha256(url.encode()).hexdigest()[:32]}"
        return await valkey_client.exists(key) > 0
    except Exception:
        return False


async def _cache_ner_set(url: str, valkey_client: Any) -> None:
    if valkey_client is None or not url:
        return
    try:
        import hashlib

        key = f"ner:url:{hashlib.sha256(url.encode()).hexdigest()[:32]}"
        await valkey_client.setex(key, 86400, "1")  # 24h TTL
    except Exception:
        pass


async def run_news_ingest_pipeline() -> dict[str, int]:
    """Flow completo. Paralelizado con semáforo + dedupe + cache URL.

    Optimizaciones:
    - `asyncio.Semaphore(10)` → 10 NER concurrentes (DeepSeek tier free ~60 req/min).
    - Dedupe por título similar >85% → 33% menos calls.
    - Cache URL SHA256 con TTL 24h en Valkey → skip noticias ya procesadas hoy.
    - Truncate content ya existía (3000 chars para NER, 4000 para embedding).
    """
    entries = await consolidate_sources()
    if not entries:
        return {"total": 0, "processed": 0, "skipped": 0}

    # Sport focus filter ANTES del dedup — barato (no LLM) y reduce el
    # universo de URLs a procesar. Noticias de deportes off (NHL, NFL, tennis,
    # boxing, mma) salen aquí. Mantenemos noticias sin `sports` declarado.
    original_count = len(entries)
    entries = _filter_by_sport_focus(entries)

    # Dedupe por título similar
    pre_dedupe = len(entries)
    entries = _dedupe_entries(entries)
    logger.info(
        "news_pipeline.deduped",
        before=original_count,
        after_focus=pre_dedupe,
        after_dedupe=len(entries),
    )

    # Valkey cache (opcional — si falla, continúa sin cache)
    valkey_client: Any = None
    try:
        import redis.asyncio as _redis

        from apuestas.config import get_settings as _gs_v

        valkey_url = _gs_v().valkey.valkey_url
        valkey_client = await _redis.from_url(str(valkey_url))
    except Exception as exc:
        logger.info("news_pipeline.valkey_unavailable", error=str(exc)[:80])

    # Pre-filtros baratos (sin LLM) — orden por costo ascendente:
    #   1. Cache URL (Valkey GET)         → URLs ya procesadas
    #   2. Source blocklist (set lookup)  → ESPN/BBC/Marca con waste >40%
    #   3. Min content length             → headlines truncados sin contexto
    #   4. Active team keywords           → contenido sin team relevante
    keywords = await _load_active_team_keywords()
    logger.info("news_pipeline.keywords_loaded", n=len(keywords))

    to_process: list[dict[str, Any]] = []
    cached = 0
    skip_source = 0
    skip_short = 0
    skip_no_team = 0
    for e in entries:
        url = e.get("url") or ""
        if await _cache_ner_hit(url, valkey_client):
            cached += 1
            continue
        src = (e.get("source") or "").lower()
        if src in _NER_SKIP_SOURCES:
            skip_source += 1
            continue
        content = e.get("content") or ""
        title = e.get("title") or ""
        if len(content) + len(title) < _NER_MIN_CONTENT_LEN:
            skip_short += 1
            continue
        if not _is_relevant_to_active_teams(e, keywords):
            skip_no_team += 1
            continue
        to_process.append(e)

    logger.info(
        "news_pipeline.prefilter_stats",
        cached=cached,
        skip_source=skip_source,
        skip_short=skip_short,
        skip_no_team=skip_no_team,
        to_process=len(to_process),
    )

    if not to_process:
        return {
            "total": original_count,
            "processed": 0,
            "skipped": 0,
            "cached": cached,
            "skip_source": skip_source,
            "skip_short": skip_short,
            "skip_no_team": skip_no_team,
        }

    # Respeta LLM_BACKEND
    from apuestas.config import get_settings as _gs

    backend = _gs().llm.llm_backend
    if backend == "deepseek":
        from apuestas.llm.deepseek_client import DeepSeekClient

        llm_cls: Any = DeepSeekClient
    else:
        llm_cls = LlamaClient

    # Semáforo para limitar concurrencia (DeepSeek tier free ~60 req/min)
    sem = asyncio.Semaphore(10)
    processed = 0
    skipped = 0

    async with llm_cls() as llm, EmbedClient() as embed:

        async def _process_one(entry: dict[str, Any]) -> bool:
            async with sem:
                try:
                    article_id, _ner = await process_article(entry, llm=llm, embed=embed)
                    if article_id is not None:
                        await _cache_ner_set(entry.get("url") or "", valkey_client)
                        return True
                    return False
                except Exception as exc:
                    logger.warning(
                        "news_pipeline.process_fail",
                        url=entry.get("url"),
                        error=str(exc)[:120],
                    )
                    return False

        results = await asyncio.gather(
            *[_process_one(e) for e in to_process], return_exceptions=True
        )
        for r in results:
            if r is True:
                processed += 1
            else:
                skipped += 1

    if valkey_client is not None:
        try:
            await valkey_client.aclose()
        except Exception:
            pass

    logger.info(
        "news_pipeline.done",
        total=original_count,
        deduped=len(entries),
        cached=cached,
        processed=processed,
        skipped=skipped,
    )
    return {
        "total": original_count,
        "processed": processed,
        "skipped": skipped,
        "cached": cached,
    }


if __name__ == "__main__":
    import asyncio as _asyncio

    result = _asyncio.run(run_news_ingest_pipeline())
    print(f"✅ News pipeline: {result}")
