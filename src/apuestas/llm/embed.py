"""Cliente TEI (text-embeddings-inference) con cache persistente sha256.

BGE-M3 INT8 en GPU RTX 4050 (~600 MB VRAM).
Cache en tabla `embeddings_cache` para evitar re-embed de contenido repetido
(noticias idénticas scrapeadas de múltiples fuentes, etc).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import httpx
import stamina
from sqlalchemy import text

from apuestas.config import get_settings
from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

logger = get_logger(__name__)


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class EmbedError(Exception):
    """Error genérico del servicio de embeddings."""


class EmbedClient:
    """Cliente async para TEI BGE-M3 con cache integrado."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        dim: int | None = None,
        timeout: float = 30.0,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.llm.tei_url).rstrip("/")
        self.model = model or settings.llm.embed_model
        self.dim = dim or settings.llm.embed_dim
        self._client: httpx.AsyncClient | None = None
        self._timeout = timeout

    async def __aenter__(self) -> EmbedClient:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self._timeout,
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            msg = "EmbedClient usado fuera de async context"
            raise RuntimeError(msg)
        return self._client

    async def health(self) -> bool:
        try:
            resp = await self.client.get("/health", timeout=5.0)
        except httpx.HTTPError:
            return False
        return resp.status_code == 200

    # Cache in-memory: si TEI host no resuelve (ConnectError DNS) durante esta
    # sesión, cae a sentence-transformers CPU local en vez de fallar.
    _tei_unreachable: bool = False
    _st_model: Any = None  # sentence-transformers lazy-loaded
    _st_unavailable: bool = False  # circuit breaker: si ST también falla, usa zeros
    _st_load_lock: Any = None  # asyncio.Lock lazy-init para serializar carga

    @stamina.retry(
        on=(httpx.ReadTimeout, httpx.RemoteProtocolError),
        attempts=2,
        wait_initial=0.3,
        wait_max=1.0,
    )
    async def _embed_raw(self, inputs: Sequence[str]) -> list[list[float]]:
        """Llamada a TEI; si TEI offline, degrada a sentence-transformers CPU.

        Orden:
        1. TEI (GPU, preferido)
        2. sentence-transformers CPU local con el mismo modelo BAAI/bge-m3
           (~500 MB RAM, 200-500 ms/batch)
        """
        if EmbedClient._tei_unreachable:
            return await self._embed_local(list(inputs))
        try:
            resp = await self.client.post("/embed", json={"inputs": list(inputs)})
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            logger.info("embed.tei_unreachable_fallback_cpu", error=str(exc)[:80])
            EmbedClient._tei_unreachable = True
            return await self._embed_local(list(inputs))
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list) or len(data) != len(inputs):
            msg = f"TEI returned unexpected shape: len={len(data)} vs expected {len(inputs)}"
            raise EmbedError(msg)
        return data

    async def _embed_local(self, inputs: list[str]) -> list[list[float]]:
        """Fallback CPU con sentence-transformers (BAAI/bge-m3). Lazy-load.

        Circuit breaker: si carga O encode falla, flipea `_st_unavailable=True`
        y el resto de la sesión devuelve zeros inmediatamente (evita reintentos
        lentos que bloquean el catchup entero cuando CUDA OOM o disco corrupto).
        """
        import asyncio as _asyncio

        # Circuit breaker ya activo → zeros directos
        if EmbedClient._st_unavailable:
            return [[0.0] * 1024 for _ in inputs]

        if EmbedClient._st_model is None:
            # Serializar el cold-start: 6 calls concurrentes pegando al
            # _load_model causaban race condition (meta tensor error en 5/6).
            if EmbedClient._st_load_lock is None:
                EmbedClient._st_load_lock = _asyncio.Lock()
            async with EmbedClient._st_load_lock:
                if EmbedClient._st_unavailable:
                    return [[0.0] * 1024 for _ in inputs]
                if EmbedClient._st_model is None:
                    try:
                        from sentence_transformers import (
                            SentenceTransformer,  # type: ignore[import-untyped]
                        )
                    except ImportError:
                        logger.warning("embed.sentence_transformers_missing")
                        EmbedClient._st_unavailable = True
                        return [[0.0] * 1024 for _ in inputs]
                    logger.info("embed.loading_st_cpu", model="BAAI/bge-m3")

                    def _load_model() -> Any:
                        # low_cpu_mem_usage=False evita el "meta tensor" error de
                        # torch 2.x con modelos lazy init (BAAI/bge-m3 usa
                        # XLM-RoBERTa base). El meta init se queda en meta device
                        # y `model.to('cpu')` falla pidiendo `to_empty()`.
                        return SentenceTransformer(
                            "BAAI/bge-m3",
                            device="cpu",
                            trust_remote_code=True,
                            model_kwargs={"low_cpu_mem_usage": False},
                        )

                    try:
                        EmbedClient._st_model = await _asyncio.to_thread(_load_model)
                    except Exception as exc:
                        logger.warning("embed.st_load_failed_fallback_zeros", error=str(exc)[:120])
                        EmbedClient._st_unavailable = True
                        return [[0.0] * 1024 for _ in inputs]

        def _encode() -> list[list[float]]:
            arr = EmbedClient._st_model.encode(inputs, show_progress_bar=False)
            return arr.tolist() if hasattr(arr, "tolist") else [list(v) for v in arr]

        try:
            return await _asyncio.to_thread(_encode)
        except Exception as exc:
            logger.warning("embed.st_encode_failed_fallback_zeros", error=str(exc)[:120])
            EmbedClient._st_unavailable = True
            return [[0.0] * 1024 for _ in inputs]

    async def embed(
        self,
        contents: str | Iterable[str],
        *,
        use_cache: bool = True,
    ) -> list[list[float]]:
        """Devuelve embeddings para uno o varios textos.

        Strategy:
        1. Hashear cada input.
        2. Consultar `embeddings_cache` por hashes conocidos.
        3. Llamar TEI solo con los faltantes.
        4. Persistir nuevos en cache.
        5. Combinar respetando orden original.
        """
        if isinstance(contents, str):
            contents_list = [contents]
        else:
            contents_list = list(contents)

        if not contents_list:
            return []

        hashes = [_sha256(c) for c in contents_list]

        if not use_cache:
            return await self._embed_raw(contents_list)

        cached = await self._fetch_cached(hashes)

        # Indices faltantes
        missing_idx = [i for i, h in enumerate(hashes) if h not in cached]
        new_vectors: dict[str, list[float]] = {}

        if missing_idx:
            missing_contents = [contents_list[i] for i in missing_idx]
            fresh_vectors = await self._embed_raw(missing_contents)
            for i, vec in zip(missing_idx, fresh_vectors, strict=True):
                new_vectors[hashes[i]] = vec

            await self._store_cached(new_vectors)

        # Combinar
        combined: list[list[float]] = []
        for h in hashes:
            if h in cached:
                combined.append(cached[h])
            else:
                combined.append(new_vectors[h])
        return combined

    async def embed_one(self, content: str, *, use_cache: bool = True) -> list[float]:
        vectors = await self.embed(content, use_cache=use_cache)
        return vectors[0]

    # ─── Cache helpers ───────────────────────────────────────────────────

    async def _fetch_cached(self, hashes: Sequence[str]) -> dict[str, list[float]]:
        """Consulta batch a embeddings_cache."""
        if not hashes:
            return {}
        try:
            async with session_scope() as session:
                result = await session.execute(
                    text(
                        """
                        UPDATE embeddings_cache
                        SET hits = hits + 1, last_used_at = NOW()
                        WHERE content_hash = ANY(:hashes) AND model = :model
                        RETURNING content_hash, embedding::text
                        """
                    ),
                    {"hashes": list(hashes), "model": self.model},
                )
                rows = result.all()
        except Exception as exc:
            logger.debug("embed.cache.fetch_failed", error=str(exc))
            return {}

        cached: dict[str, list[float]] = {}
        for content_hash, emb_text in rows:
            # pgvector devuelve string "[0.1,0.2,...]" cuando se hace ::text
            if emb_text:
                try:
                    vec = [float(x) for x in emb_text.strip("[]").split(",")]
                    if len(vec) == self.dim:
                        cached[content_hash] = vec
                except (ValueError, AttributeError):  # fmt: skip
                    continue
        return cached

    async def _store_cached(self, items: dict[str, list[float]]) -> None:
        """Persiste nuevos embeddings."""
        if not items:
            return
        try:
            async with session_scope() as session:
                for content_hash, vec in items.items():
                    vec_str = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
                    await session.execute(
                        text(
                            """
                            INSERT INTO embeddings_cache
                              (content_hash, model, embedding, hits)
                            VALUES
                              (:content_hash, :model, (:embedding)::vector, 0)
                            ON CONFLICT (content_hash) DO NOTHING
                            """
                        ),
                        {
                            "content_hash": content_hash,
                            "model": self.model,
                            "embedding": vec_str,
                        },
                    )
        except Exception as exc:
            logger.debug("embed.cache.store_failed", error=str(exc))
