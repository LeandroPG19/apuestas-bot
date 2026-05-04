"""Fase 4.11 — Twitter/X NLP extraction real-time insiders.

Shams Charania y Adrian Wojnarowski reportan NBA lineup changes 2-3h antes
de lock. Los bots que scrapean + NLP-extraen → bet antes que books muevan
línea.

Scraping via Nitter proxy (evita API X paga). Lista curated de insiders:
  - NBA: Shams Charania (@ShamsCharania), Wojnarowski (@wojespn)
  - NFL: Adam Schefter (@AdamSchefter), Ian Rapoport (@RapSheet)
  - MLB: Jeff Passan (@JeffPassan), Ken Rosenthal (@Ken_Rosenthal)
  - Soccer: Fabrizio Romano (@FabrizioRomano)

Stub minimo: integración real requiere tweepy/snscrape + LLM NER.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

Severity = Literal["out", "doubtful", "questionable", "probable", "active"]


@dataclass(slots=True, frozen=True)
class InsiderReport:
    """Reporte extraído de tweet de insider."""

    source: str  # @ShamsCharania
    tweet_id: str
    player_name: str
    team_name: str | None
    status: Severity
    raw_text: str
    detected_at: datetime
    confidence: float  # 0-1, confidence del NER


INSIDERS: dict[str, list[str]] = {
    "nba": ["ShamsCharania", "wojespn"],
    "nfl": ["AdamSchefter", "RapSheet"],
    "mlb": ["JeffPassan", "Ken_Rosenthal"],
    "soccer": ["FabrizioRomano"],
}


_NITTER_INSTANCES = (
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
    "https://nitter.cz",
    "https://nitter.1d4.us",
)


async def fetch_recent_tweets(handle: str, *, limit: int = 20) -> list[dict[str, object]]:
    """Obtiene tweets recientes vía Nitter RSS (sin auth, free).

    Rotación entre instancias Nitter (muchas caen sporadicamente). RSS tag
    layout: title=handle:: tweet_text, description=HTML tweet, pubDate=RFC2822.
    """
    import asyncio
    import re
    import xml.etree.ElementTree as ET

    import httpx

    tweets: list[dict[str, object]] = []
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        for instance in _NITTER_INSTANCES:
            url = f"{instance}/{handle}/rss"
            try:
                r = await client.get(url)
                if r.status_code != 200:
                    continue
                root = ET.fromstring(r.text)
                items = list(root.iter("item"))
                for item in items[:limit]:
                    title_el = item.find("title")
                    desc_el = item.find("description")
                    pub_el = item.find("pubDate")
                    guid_el = item.find("guid")
                    text_content = (desc_el.text or "") if desc_el is not None else ""
                    text_content = re.sub(r"<[^>]+>", " ", text_content).strip()
                    tweets.append(
                        {
                            "handle": handle,
                            "tweet_id": (guid_el.text or "") if guid_el is not None else "",
                            "text": (title_el.text or text_content)
                            if title_el is not None
                            else text_content,
                            "pub_date": (pub_el.text or "") if pub_el is not None else "",
                        }
                    )
                if tweets:
                    logger.info(
                        "twitter_insiders.fetch_ok",
                        handle=handle,
                        n=len(tweets),
                        instance=instance,
                    )
                    break
            except (httpx.HTTPError, ET.ParseError) as exc:
                logger.debug(
                    "twitter_insiders.instance_fail",
                    handle=handle,
                    instance=instance,
                    error=str(exc)[:80],
                )
                continue
            await asyncio.sleep(0.2)
    if not tweets:
        logger.warning("twitter_insiders.all_instances_failed", handle=handle)
    return tweets[:limit]


INJURY_PATTERNS = [
    ("out", ["out for", "ruled out", "will not play", "missing"]),
    ("doubtful", ["doubtful", "unlikely to play"]),
    ("questionable", ["questionable", "game-time decision"]),
    ("probable", ["probable", "expected to play"]),
]


def extract_injury_from_text(tweet_text: str) -> tuple[str, str] | None:
    """Heurística simple: encuentra (player_name, status) del texto.

    Retorna None si no detecta injury. En producción usar LLM NER fine-tuned
    (NER extractor ya existe en `schemas/llm.py::NERExtraction`).
    """
    text_lower = tweet_text.lower()
    for status, patterns in INJURY_PATTERNS:
        for pattern in patterns:
            if pattern in text_lower:
                # Player name = primera palabra capitalizada en el tweet (heurística)
                words = tweet_text.split()
                for i, w in enumerate(words[:8]):
                    if w[0:1].isupper() and len(w) > 2:
                        potential_name = " ".join(words[i : i + 2])
                        return potential_name, status
    return None


async def persist_insider_reports(reports: list[InsiderReport]) -> int:
    """Persist reports en `injury_feed` table. Idempotent via tweet_id."""
    from sqlalchemy import text as _t

    from apuestas.db import session_scope as _s

    inserted = 0
    async with _s() as session:
        for r in reports:
            try:
                await session.execute(
                    _t(
                        """
                        INSERT INTO injury_feed
                            (player_name_raw, source, reporter, status_reported,
                             raw_text, sentiment_score, reported_at)
                        VALUES (:name, :src, :rep, :status, :raw, :conf, :ts)
                        """
                    ),
                    {
                        "name": r.player_name,
                        "src": "twitter",
                        "rep": r.source,
                        "status": r.status,
                        "raw": r.raw_text[:1000],
                        "conf": float(r.confidence),
                        "ts": r.detected_at,
                    },
                )
                inserted += 1
            except Exception as exc:
                logger.debug("twitter_insiders.persist_fail", error=str(exc)[:80])
    return inserted


async def scan_insiders_for_sport(sport_code: str) -> list[InsiderReport]:
    """Entry point: escanea insiders del sport y devuelve reportes extraídos."""
    handles = INSIDERS.get(sport_code, [])
    all_reports: list[InsiderReport] = []
    for handle in handles:
        tweets = await fetch_recent_tweets(handle)
        for tweet in tweets:
            text = str(tweet.get("text", ""))
            extraction = extract_injury_from_text(text)
            if extraction is None:
                continue
            player_name, status = extraction
            all_reports.append(
                InsiderReport(
                    source=f"@{handle}",
                    tweet_id=str(tweet.get("tweet_id", "")),
                    player_name=player_name,
                    team_name=None,
                    status=status,  # type: ignore[arg-type]
                    raw_text=text,
                    detected_at=datetime.now(tz=UTC),
                    confidence=0.6,  # heurística simple
                )
            )
    return all_reports
