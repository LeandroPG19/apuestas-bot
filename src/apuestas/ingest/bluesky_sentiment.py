"""Bluesky sentiment — reemplaza Twitter/X API paga ($100/mes).

Bluesky / atproto es gratis y open. Scrapeamos feeds de beat writers
verificados + hashtag-based queries (#NBA, #PremierLeague, #LigaMX).

Sentiment score por post: -1 (muy negativo) a +1 (muy positivo).
Usamos VADER rule-based (gratis, rápido, sin GPU) + keywords deportivas
específicas. No requiere modelo LLM para esto.

Posts se asocian a matches via:
- teams_mentioned (nombre fuzzy match contra `teams`)
- players_mentioned (nombre fuzzy match contra `players`)
- sports_mentioned (hashtags o keywords)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


SPORT_HASHTAGS: dict[str, list[str]] = {
    "nba": ["#NBA", "#NBABets", "#NBAPlayoffs", "#BasketballTwitter"],
    "nfl": ["#NFL", "#NFLTwitter", "#NFLBets"],
    "mlb": ["#MLB", "#MLBTwitter", "#Baseball"],
    "soccer": ["#LigaMX", "#PremierLeague", "#ChampionsLeague", "#LaLiga"],
    "boxing": ["#Boxing", "#Canelo", "#BoxingTwitter"],
    "nhl": ["#NHL", "#Hockey"],
    "tennis": ["#Tennis", "#ATP", "#WTA", "#GrandSlam"],
}


# Léxico deportivo simple para augmentar VADER
POSITIVE_SPORT_KEYWORDS = frozenset(
    {
        "healthy",
        "cleared",
        "returns",
        "streak",
        "dominant",
        "clutch",
        "upset the",
        "comeback",
        "momentum",
        "firing",
        "elite",
    }
)
NEGATIVE_SPORT_KEYWORDS = frozenset(
    {
        "out",
        "injured",
        "doubtful",
        "suspended",
        "slump",
        "blowout loss",
        "struggles",
        "ejected",
        "benched",
        "trade request",
    }
)


def simple_sentiment(text_blob: str) -> float:
    """Rule-based sentiment. -1 a +1.

    Intenta VADER si está instalado, si no usa léxico deportivo.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        score = analyzer.polarity_scores(text_blob)
        return float(score.get("compound", 0.0))
    except ImportError:
        low = text_blob.lower()
        pos = sum(1 for k in POSITIVE_SPORT_KEYWORDS if k in low)
        neg = sum(1 for k in NEGATIVE_SPORT_KEYWORDS if k in low)
        if pos + neg == 0:
            return 0.0
        return (pos - neg) / (pos + neg)


async def fetch_bluesky_feed(
    *, handle: str | None = None, query: str | None = None, limit: int = 25
) -> list[dict[str, Any]]:
    """Fetch posts de Bluesky sin auth.

    - `handle`: feed público de un usuario (ej. "shamsmania.bsky.social")
    - `query`: búsqueda de texto (ej. "#NBA injury")

    Retorna lista de dicts con keys: uri, author, content, published_at, likes, reposts.
    """
    import httpx

    posts: list[dict[str, Any]] = []
    async with httpx.AsyncClient(
        timeout=15,
        headers={"User-Agent": "apuestas-bot/0.1", "Accept": "application/json"},
    ) as c:
        try:
            if handle:
                url = "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed"
                params = {"actor": handle, "limit": str(limit)}
            elif query:
                url = "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts"
                params = {"q": query, "limit": str(limit)}
            else:
                return []
            r = await c.get(url, params=params)
            if r.status_code != 200:
                logger.debug(
                    "bluesky.http",
                    status=r.status_code,
                    handle=handle,
                    query=query,
                )
                return []
            data = r.json()
        except Exception as exc:
            logger.warning("bluesky.fetch_fail", error=str(exc))
            return []

    feed = data.get("feed") or data.get("posts") or []
    for item in feed:
        post = item.get("post") or item
        record = post.get("record", {})
        text_content = record.get("text", "")
        if not text_content:
            continue
        try:
            ts = datetime.fromisoformat(record.get("createdAt", "").replace("Z", "+00:00"))
        except ValueError:
            ts = datetime.now(tz=UTC)
        author = post.get("author", {})
        posts.append(
            {
                "post_uri": post.get("uri", ""),
                "author_handle": author.get("handle", ""),
                "author_followers": author.get("followersCount", 0),
                "content": text_content[:1000],
                "published_at": ts,
                "likes": post.get("likeCount", 0),
                "reposts": post.get("repostCount", 0),
            }
        )
    return posts


async def persist_posts(posts: list[dict[str, Any]], *, sport_code: str | None = None) -> int:
    """Persiste posts + calcula sentiment_score."""
    if not posts:
        return 0
    inserted = 0
    async with session_scope() as s:
        for p in posts:
            sentiment = simple_sentiment(p["content"])
            sports_arr = [sport_code] if sport_code else []
            try:
                await s.execute(
                    text(
                        """
                        INSERT INTO bluesky_posts
                            (post_uri, author_handle, author_followers,
                             content, published_at, likes, reposts,
                             sports_mentioned, sentiment_score)
                        VALUES (:uri, :ah, :af, :c, :pa, :lk, :rp,
                                CAST(:sm AS text[]), :sent)
                        ON CONFLICT (post_uri) DO NOTHING
                        """
                    ),
                    {
                        "uri": p["post_uri"],
                        "ah": p["author_handle"],
                        "af": p.get("author_followers", 0),
                        "c": p["content"],
                        "pa": p["published_at"],
                        "lk": p.get("likes", 0),
                        "rp": p.get("reposts", 0),
                        "sm": sports_arr,
                        "sent": round(sentiment, 4),
                    },
                )
                inserted += 1
            except Exception as exc:
                logger.debug("bluesky.persist_fail", error=str(exc))
    return inserted


async def ingest_sport_feed(sport_code: str, *, limit_per_handle: int = 20) -> int:
    """Ingiere feeds de beat writers verificados del sport + queries hashtag."""
    from apuestas.ingest.injury_feed import BEAT_WRITERS

    total = 0
    for handle in BEAT_WRITERS.get(sport_code, []):
        posts = await fetch_bluesky_feed(handle=handle, limit=limit_per_handle)
        total += await persist_posts(posts, sport_code=sport_code)

    for tag in SPORT_HASHTAGS.get(sport_code, [])[:2]:
        posts = await fetch_bluesky_feed(query=tag, limit=limit_per_handle)
        total += await persist_posts(posts, sport_code=sport_code)

    logger.info("bluesky.sport_ingested", sport=sport_code, posts=total)
    return total
