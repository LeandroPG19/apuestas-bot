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


async def resolve_team_ids(team_names: list[str]) -> list[int]:
    """Busca team_ids por similaridad trigram. Devuelve [] si ninguno."""
    if not team_names:
        return []
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT DISTINCT id FROM teams
                WHERE name IN (SELECT unnest(:names))
                   OR similarity(name, ANY(:names)) > 0.55
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
                "content": full_text[:3000],  # trunc por ctx window
                "lang": entry.get("lang", "en"),
                "source": entry.get("source", "unknown"),
            },
        )
    except Exception as exc:
        logger.warning("news_pipeline.ner_failed", url=entry.get("url"), error=str(exc))
        return None, None

    assert isinstance(ner_result, NERExtraction)

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

    # INSERT news_articles
    vec_str = "[" + ",".join(f"{x:.6f}" for x in embedding) + "]" if embedding else None
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                INSERT INTO news_articles
                  (source, url, title, content, lang, published_at,
                   sports, teams_mentioned, players_mentioned, sentiment_score, embedding)
                VALUES
                  (:source, :url, :title, :content, :lang, :published_at,
                   :sports, :teams_mentioned, :players_mentioned, :sentiment_score,
                   CASE WHEN :embedding IS NULL THEN NULL ELSE (:embedding)::vector END)
                ON CONFLICT (url) DO UPDATE
                  SET title = EXCLUDED.title,
                      content = EXCLUDED.content,
                      sentiment_score = EXCLUDED.sentiment_score,
                      teams_mentioned = EXCLUDED.teams_mentioned,
                      players_mentioned = EXCLUDED.players_mentioned,
                      embedding = EXCLUDED.embedding
                RETURNING id
                """
            ),
            {
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
                "embedding": vec_str,
            },
        )
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

    return article_id, ner_result


async def run_news_ingest_pipeline() -> dict[str, int]:
    """Flow completo. Llamable desde Prefect o directamente."""
    entries = await consolidate_sources()
    if not entries:
        return {"total": 0, "processed": 0, "skipped": 0}

    processed = 0
    skipped = 0
    # Respeta LLM_BACKEND (deepseek cloud vs llama local)
    from apuestas.config import get_settings as _gs

    backend = _gs().llm.llm_backend
    if backend == "deepseek":
        from apuestas.llm.deepseek_client import DeepSeekClient

        llm_cls: Any = DeepSeekClient
    else:
        llm_cls = LlamaClient

    async with llm_cls() as llm, EmbedClient() as embed:
        for entry in entries:
            article_id, ner = await process_article(entry, llm=llm, embed=embed)
            if article_id is not None:
                processed += 1
            else:
                skipped += 1

    logger.info("news_pipeline.done", total=len(entries), processed=processed, skipped=skipped)
    return {"total": len(entries), "processed": processed, "skipped": skipped}
