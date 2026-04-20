"""Ingesta Reddit (asyncpraw) + Bluesky (atproto).

Reddit: 100 QPM gratis con OAuth. Bluesky: sin rate-limit documentado
para reads públicos.

Estrategia:
- Solo posts con > min_upvotes por sub
- Filtros de stickies/megathreads
- Comentarios top cuando post es "megathread de partido"
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from apuestas.config import get_settings
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "news_sources.yaml"


def _load_config() -> dict[str, Any]:
    with _CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


async def fetch_reddit_sub(
    sub_name: str,
    *,
    min_upvotes: int = 20,
    limit: int = 50,
    since: datetime | None = None,
) -> list[dict[str, Any]]:
    """Fetch top posts de un sub.

    Estrategia:
    1. Si hay credenciales OAuth → usa `asyncpraw` (60 req/min, más datos).
    2. Si NO hay credenciales → endpoint público `reddit.com/r/<sub>/new.json`
       (10 req/min sin auth, suficiente para noticias). User-Agent compliant
       según Reddit API rules.

    Así el bot funciona sin config adicional; OAuth es un upgrade opcional.
    """
    settings = get_settings()
    client_id = (
        settings.apis.reddit_client_id.get_secret_value()
        if settings.apis.reddit_client_id
        else None
    )
    client_secret = (
        settings.apis.reddit_client_secret.get_secret_value()
        if settings.apis.reddit_client_secret
        else None
    )
    since = since or (datetime.now(tz=UTC) - timedelta(hours=24))

    if client_id and client_secret:
        return await _fetch_reddit_oauth(
            sub_name,
            client_id=client_id,
            client_secret=client_secret,
            user_agent=settings.apis.reddit_user_agent,
            min_upvotes=min_upvotes,
            limit=limit,
            since=since,
        )
    return await _fetch_reddit_public(
        sub_name,
        user_agent=settings.apis.reddit_user_agent,
        min_upvotes=min_upvotes,
        limit=limit,
        since=since,
    )


async def _fetch_reddit_oauth(
    sub_name: str,
    *,
    client_id: str,
    client_secret: str,
    user_agent: str,
    min_upvotes: int,
    limit: int,
    since: datetime,
) -> list[dict[str, Any]]:
    import asyncpraw  # type: ignore[import-untyped]

    reddit = asyncpraw.Reddit(
        client_id=client_id, client_secret=client_secret, user_agent=user_agent
    )
    posts: list[dict[str, Any]] = []
    try:
        sub = await reddit.subreddit(sub_name)
        async for submission in sub.new(limit=limit):
            if submission.stickied or submission.score < min_upvotes:
                continue
            created = datetime.fromtimestamp(submission.created_utc, tz=UTC)
            if created < since:
                continue
            posts.append(
                {
                    "source": f"reddit:/r/{sub_name}",
                    "url": f"https://reddit.com{submission.permalink}",
                    "title": submission.title,
                    "content": submission.selftext or "",
                    "lang": "en",
                    "published_at": created,
                    "sports": [],
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                }
            )
    finally:
        await reddit.close()
    logger.info("reddit.fetched_oauth", sub=sub_name, count=len(posts))
    return posts


async def _fetch_reddit_public(
    sub_name: str,
    *,
    user_agent: str,
    min_upvotes: int,  # ignorado en RSS (no hay score)
    limit: int,
    since: datetime,
) -> list[dict[str, Any]]:
    """RSS feed oficial de Reddit — más tolerante que JSON scraping.

    Desde 2024 Reddit bloquea 403 en JSON sin OAuth, pero el RSS oficial
    (`/r/<sub>/new.rss`) sigue accesible. Trade-off: no trae `score` ni
    `num_comments`, así que `min_upvotes` se ignora aquí (filtro de ruido
    se hace después por el pipeline de noticias via keywords).
    """
    import re
    import xml.etree.ElementTree as ET

    import httpx

    ua = user_agent or "linux:apuestas-bot:0.1 (by /u/anon)"
    if "apuestas" not in ua.lower():
        ua = f"linux:apuestas-bot:0.1 ({ua})"

    url = f"https://www.reddit.com/r/{sub_name}/new.rss"
    headers = {
        "User-Agent": ua,
        "Accept": "application/rss+xml, application/atom+xml, */*",
    }

    posts: list[dict[str, Any]] = []
    _ = min_upvotes
    try:
        async with httpx.AsyncClient(
            timeout=15.0, follow_redirects=True, headers=headers
        ) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                logger.warning(
                    "reddit.public_http",
                    sub=sub_name,
                    status=resp.status_code,
                )
                return []
            xml_text = resp.text
    except Exception as exc:
        logger.warning("reddit.public_fetch_fail", sub=sub_name, error=str(exc))
        return []

    # Atom feed parser
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("reddit.public_parse_fail", sub=sub_name, error=str(exc))
        return []

    entries = root.findall("atom:entry", ns)[:limit]
    for entry in entries:
        title_el = entry.find("atom:title", ns)
        link_el = entry.find("atom:link", ns)
        updated_el = entry.find("atom:updated", ns)
        content_el = entry.find("atom:content", ns)
        author_el = entry.find("atom:author/atom:name", ns)

        title = (title_el.text or "").strip() if title_el is not None else ""
        url_p = link_el.get("href") if link_el is not None else ""
        updated_str = (updated_el.text or "") if updated_el is not None else ""
        try:
            created = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        if created < since:
            continue

        # El content viene en HTML — extraer texto plano
        raw_html = (content_el.text or "") if content_el is not None else ""
        # Strip HTML tags básicamente
        text_content = re.sub(r"<[^>]+>", " ", raw_html)
        text_content = re.sub(r"\s+", " ", text_content).strip()[:2000]

        posts.append(
            {
                "source": f"reddit:/r/{sub_name}",
                "url": url_p,
                "title": title[:300],
                "content": text_content,
                "lang": "en",
                "published_at": created,
                "sports": [],
                "score": 0,  # RSS no expone score
                "num_comments": 0,
                "author": author_el.text if author_el is not None else None,
            }
        )

    logger.info("reddit.fetched_public_rss", sub=sub_name, count=len(posts))
    return posts


async def fetch_all_reddit() -> list[dict[str, Any]]:
    config = _load_config()
    import asyncio

    tasks = [
        fetch_reddit_sub(s["name"], min_upvotes=s.get("min_upvotes", 20))
        for s in config.get("reddit_subs", [])
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    all_posts: list[dict[str, Any]] = []
    for r, sub in zip(results, config.get("reddit_subs", []), strict=False):
        if isinstance(r, Exception):
            logger.warning("reddit.sub_error", sub=sub.get("name"), error=str(r))
            continue
        # Anotar sports del sub
        for p in r:
            p["sports"] = sub.get("sports", [])
            all_posts.append(p)
    return all_posts


async def fetch_bluesky_author(handle: str, *, limit: int = 50) -> list[dict[str, Any]]:
    """Bluesky posts recientes de un author (atproto)."""
    from atproto import AsyncClient  # type: ignore[import-untyped]

    client = AsyncClient()
    try:
        # Resolver handle → DID
        profile = await client.get_profile(actor=handle)
        feed = await client.get_author_feed(actor=profile.did, limit=limit)
    except Exception as exc:
        logger.warning("bluesky.fetch_failed", handle=handle, error=str(exc))
        return []

    posts: list[dict[str, Any]] = []
    for item in feed.feed:
        post = item.post
        record = post.record
        published = datetime.fromisoformat(record.created_at.replace("Z", "+00:00"))
        posts.append(
            {
                "source": f"bluesky:{handle}",
                "url": f"https://bsky.app/profile/{handle}/post/{post.uri.split('/')[-1]}",
                "title": None,
                "content": record.text,
                "lang": "en",
                "published_at": published,
                "sports": [],
            }
        )
    return posts


async def fetch_all_bluesky() -> list[dict[str, Any]]:
    import asyncio

    config = _load_config()
    tasks = [fetch_bluesky_author(f["handle"]) for f in config.get("bluesky_feeds", [])]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    all_posts: list[dict[str, Any]] = []
    for r, feed in zip(results, config.get("bluesky_feeds", []), strict=False):
        if isinstance(r, Exception):
            logger.warning("bluesky.author_error", handle=feed.get("handle"), error=str(r))
            continue
        for p in r:
            p["sports"] = feed.get("sports", [])
            all_posts.append(p)
    return all_posts
