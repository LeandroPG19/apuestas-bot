"""Ingesta feeds RSS deportivos — ESPN, Marca, Record, Mediotiempo, BBC Sport.

Cadencia 30 min. Dedupe por URL. Almacena en news_articles y dispara
pipeline de embedding + NER en worker-ml.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import feedparser
import httpx
import yaml

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "news_sources.yaml"


def _load_config() -> dict[str, Any]:
    with _CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


async def _fetch_feed_async(url: str, timeout: float = 15.0) -> str | None:
    """Descarga el XML del feed en async; parseo se hace en to_thread."""
    async with httpx.AsyncClient(
        timeout=timeout,
        headers={"User-Agent": "apuestas-bot/0.1 (+news_rss)"},
        follow_redirects=True,
    ) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text
        except httpx.HTTPError as exc:
            logger.warning("rss.fetch_failed", url=url, error=str(exc))
            return None


async def parse_feed(source_entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Un feed → lista de entries normalizadas."""
    url = source_entry["url"]
    xml = await _fetch_feed_async(url)
    if xml is None:
        return []

    parsed = await asyncio.to_thread(feedparser.parse, xml)
    entries: list[dict[str, Any]] = []
    for item in parsed.entries:
        published = None
        if hasattr(item, "published_parsed") and item.published_parsed:
            try:
                published = datetime(*item.published_parsed[:6], tzinfo=UTC)
            except (ValueError, TypeError):  # fmt: skip
                published = None

        content = getattr(item, "summary", "") or getattr(item, "description", "")
        entries.append(
            {
                "source": _domain_from_url(url),
                "url": getattr(item, "link", None),
                "title": getattr(item, "title", None),
                "content": content,
                "lang": source_entry.get("lang", "en"),
                "published_at": published,
                "sports": source_entry.get("sports", []),
                "teams_mentioned": [],
                "players_mentioned": [],
            }
        )

    logger.info("rss.parsed", url=url, count=len(entries))
    return entries


def _domain_from_url(url: str) -> str:
    from urllib.parse import urlparse

    return urlparse(url).netloc.replace("www.", "")


async def ingest_all_rss() -> list[dict[str, Any]]:
    """Consolida todos los feeds configurados."""
    config = _load_config()
    all_entries: list[dict[str, Any]] = []
    tasks = []
    for _category, feeds in config.get("rss_feeds", {}).items():
        for feed in feeds:
            tasks.append(parse_feed(feed))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):
            logger.warning("rss.feed_exception", error=str(r))
            continue
        all_entries.extend(r)

    # Dedupe por URL
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for e in all_entries:
        url = e.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        unique.append(e)

    logger.info("rss.total_unique", count=len(unique))
    return unique


def content_hash(entry: dict[str, Any]) -> str:
    """Hash para dedupe en embeddings_cache."""
    data = (entry.get("title", "") + entry.get("content", "")).encode()
    return hashlib.sha256(data).hexdigest()
