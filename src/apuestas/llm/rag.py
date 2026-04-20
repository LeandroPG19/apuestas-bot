"""RAG retrieval con pgvector HNSW + hybrid search (dense + BM25 + RRF).

Estrategia:
1. Query expansion opcional (Qwen genera 2-3 queries alternas).
2. Dense retrieval vía BGE-M3 embedding + pgvector HNSW cosine.
3. BM25 via PostgreSQL FTS (to_tsvector con 'spanish' + unaccent).
4. Reciprocal Rank Fusion (RRF) para combinar dense + sparse.
5. Reranker opcional con BGE-reranker-v2-m3 (Fase 9-10).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.llm.embed import EmbedClient
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class RAGHit:
    id: int
    source: str
    url: str | None
    title: str | None
    content: str
    published_at: datetime | None
    score_dense: float | None
    score_sparse: float | None
    score_rrf: float
    teams_mentioned: list[int]
    players_mentioned: list[int]


class RAGRetriever:
    """Retrieval multi-estrategia sobre news_articles + player_news."""

    def __init__(self, *, embed_client: EmbedClient | None = None) -> None:
        self.embed_client = embed_client

    async def dense_search(
        self,
        query: str,
        *,
        top_k: int = 50,
        sports: list[str] | None = None,
        team_ids: list[int] | None = None,
        since: datetime | None = None,
    ) -> list[RAGHit]:
        """Similaridad coseno sobre pgvector HNSW."""
        since = since or (datetime.now(tz=UTC) - timedelta(days=7))

        if self.embed_client is None:
            msg = "RAGRetriever.dense_search requiere embed_client"
            raise RuntimeError(msg)

        vec = await self.embed_client.embed_one(query)
        vec_str = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

        params: dict[str, object] = {
            "vec": vec_str,
            "since": since,
            "top_k": top_k,
        }
        filters = ["published_at >= :since"]
        if sports:
            filters.append("sports && :sports")
            params["sports"] = sports
        if team_ids:
            filters.append("teams_mentioned && :team_ids")
            params["team_ids"] = team_ids

        where_clause = " AND ".join(filters)

        async with session_scope() as session:
            result = await session.execute(
                text(
                    f"""
                    SELECT id, source, url, title, content, published_at,
                           teams_mentioned, players_mentioned,
                           1 - (embedding <=> (:vec)::vector) AS score
                    FROM news_articles
                    WHERE {where_clause}
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> (:vec)::vector
                    LIMIT :top_k
                    """
                ),
                params,
            )
            rows = result.all()

        return [
            RAGHit(
                id=r.id,
                source=r.source,
                url=r.url,
                title=r.title,
                content=r.content or "",
                published_at=r.published_at,
                score_dense=float(r.score),
                score_sparse=None,
                score_rrf=float(r.score),
                teams_mentioned=list(r.teams_mentioned or []),
                players_mentioned=list(r.players_mentioned or []),
            )
            for r in rows
        ]

    async def sparse_search(
        self,
        query: str,
        *,
        top_k: int = 50,
        sports: list[str] | None = None,
        team_ids: list[int] | None = None,
        since: datetime | None = None,
    ) -> list[RAGHit]:
        """BM25-like via PostgreSQL FTS con índice idx_news_fts."""
        since = since or (datetime.now(tz=UTC) - timedelta(days=7))
        params: dict[str, object] = {"q": query, "since": since, "top_k": top_k}

        filters = ["published_at >= :since"]
        if sports:
            filters.append("sports && :sports")
            params["sports"] = sports
        if team_ids:
            filters.append("teams_mentioned && :team_ids")
            params["team_ids"] = team_ids
        where_clause = " AND ".join(filters)

        async with session_scope() as session:
            result = await session.execute(
                text(
                    f"""
                    SELECT id, source, url, title, content, published_at,
                           teams_mentioned, players_mentioned,
                           ts_rank_cd(
                             to_tsvector('spanish', unaccent(coalesce(title,'') || ' ' || coalesce(content,''))),
                             websearch_to_tsquery('spanish', unaccent(:q))
                           ) AS score
                    FROM news_articles
                    WHERE {where_clause}
                      AND to_tsvector('spanish', unaccent(coalesce(title,'') || ' ' || coalesce(content,'')))
                          @@ websearch_to_tsquery('spanish', unaccent(:q))
                    ORDER BY score DESC
                    LIMIT :top_k
                    """
                ),
                params,
            )
            rows = result.all()

        return [
            RAGHit(
                id=r.id,
                source=r.source,
                url=r.url,
                title=r.title,
                content=r.content or "",
                published_at=r.published_at,
                score_dense=None,
                score_sparse=float(r.score),
                score_rrf=float(r.score),
                teams_mentioned=list(r.teams_mentioned or []),
                players_mentioned=list(r.players_mentioned or []),
            )
            for r in rows
        ]

    async def hybrid_search(
        self,
        query: str,
        *,
        top_k: int = 10,
        pool_size: int = 50,
        rrf_k: int = 60,
        sports: list[str] | None = None,
        team_ids: list[int] | None = None,
        since: datetime | None = None,
    ) -> list[RAGHit]:
        """Combina dense + sparse con Reciprocal Rank Fusion.

        RRF: score(d) = Σ 1/(k + rank_i(d))
        Referencia: Cormack et al. 2009.
        """
        dense = await self.dense_search(
            query, top_k=pool_size, sports=sports, team_ids=team_ids, since=since
        )
        sparse = await self.sparse_search(
            query, top_k=pool_size, sports=sports, team_ids=team_ids, since=since
        )

        scores: dict[int, float] = {}
        hits_by_id: dict[int, RAGHit] = {}

        for rank, hit in enumerate(dense, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (rrf_k + rank)
            hits_by_id[hit.id] = hit

        for rank, hit in enumerate(sparse, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (rrf_k + rank)
            if hit.id not in hits_by_id:
                hits_by_id[hit.id] = hit

        # Ordenar por RRF score y devolver top_k con scores finales
        ordered_ids = sorted(scores, key=lambda i: scores[i], reverse=True)[:top_k]
        final: list[RAGHit] = []
        for i in ordered_ids:
            h = hits_by_id[i]
            final.append(
                RAGHit(
                    id=h.id,
                    source=h.source,
                    url=h.url,
                    title=h.title,
                    content=h.content,
                    published_at=h.published_at,
                    score_dense=h.score_dense,
                    score_sparse=h.score_sparse,
                    score_rrf=scores[i],
                    teams_mentioned=h.teams_mentioned,
                    players_mentioned=h.players_mentioned,
                )
            )
        return final

    @staticmethod
    def format_snippets(hits: list[RAGHit], *, max_chars: int = 400) -> str:
        """Formatea hits para inyección en prompt LLM."""
        lines: list[str] = []
        for i, h in enumerate(hits, 1):
            title = h.title or "(sin título)"
            ts = h.published_at.strftime("%Y-%m-%d %H:%M") if h.published_at else "s/f"
            content = (h.content or "")[:max_chars]
            if len(h.content or "") > max_chars:
                content += "..."
            lines.append(f"[{i}] [{ts}] {h.source} — {title}\n{content}")
        return "\n\n".join(lines)
