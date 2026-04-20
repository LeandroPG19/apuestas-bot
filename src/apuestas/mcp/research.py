"""Helpers de alto nivel para cuba-search (§8.5).

Mapea cuba_search/cuba_research/cuba_scrape/cuba_validate/etc a operaciones
del bot:
- deep_brief(event_id): research consolidado top 3 picks
- validate_injury_claim
- scrape_boxrec (fighter)
- search_spanish_news(query) Liga MX / boxeo mexicano
- scrape_fallback_caliente si camoufox falla
"""

from __future__ import annotations

from typing import Any

from apuestas.mcp.client import MCPClient
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def deep_brief(
    *,
    topic: str,
    depth: str = "deep",
    max_sources: int = 10,
) -> dict[str, Any] | None:
    """§8.5: brief consolidado multi-fuente para un evento de alta prioridad.

    `depth=deep` activa el pipeline search→scrape→validate→compress.
    Se invoca solo para los top 3 picks con mayor EV (ahorra cuota).
    """
    client = MCPClient.get()
    return await client.call(
        "search",
        "cuba_research",
        {"topic": topic, "depth": depth, "max_sources": max_sources},
    )


async def validate_claim(
    *,
    claim: str,
    min_sources: int = 3,
) -> dict[str, Any] | None:
    """Cross-source validation de una aserción (ej. 'Player X is out')."""
    client = MCPClient.get()
    return await client.call(
        "search",
        "cuba_validate",
        {"claim": claim, "min_sources": min_sources},
    )


async def search_spanish_news(
    *,
    query: str,
    window_days: int = 7,
    sites: list[str] | None = None,
) -> dict[str, Any] | None:
    """Busca noticias Liga MX / boxeo mexicano en medios ES (Marca, Record, Mediotiempo)."""
    client = MCPClient.get()
    args: dict[str, Any] = {
        "query": query,
        "time_window_days": window_days,
        "lang": "es",
    }
    if sites:
        args["allowed_domains"] = sites
    return await client.call("search", "cuba_search", args)


async def scrape_url(*, url: str, extract_fields: list[str] | None = None) -> dict[str, Any] | None:
    """Scraping dirigido con evasión Cloudflare incluida."""
    client = MCPClient.get()
    args: dict[str, Any] = {"url": url}
    if extract_fields:
        args["extract_fields"] = extract_fields
    return await client.call("search", "cuba_scrape", args)


async def scrape_boxrec_fighter(*, fighter_slug: str) -> dict[str, Any] | None:
    """§25.2: BoxRec sin API; scraping vía cuba-search.

    Extrae: récord W-L-KO%, reach, altura, stance, edad, fights_last_year,
    ranking, weight_class, próximos oponentes.
    """
    url = f"https://boxrec.com/en/proboxer/{fighter_slug}"
    result = await scrape_url(
        url=url,
        extract_fields=[
            "record",
            "ranking",
            "weight_class",
            "reach",
            "height",
            "stance",
            "age",
            "last_fight_date",
            "fights_last_year",
        ],
    )
    logger.info("mcp.boxrec_scraped", fighter=fighter_slug, has_data=result is not None)
    return result


async def scrape_caliente_fallback(*, url: str) -> dict[str, Any] | None:
    """§22/§9: fallback cuando camoufox falla por Cloudflare agresivo."""
    return await scrape_url(
        url=url,
        extract_fields=["event_title", "home", "away", "markets", "odds"],
    )


async def extract_entities(*, content: str, schema: dict[str, Any]) -> dict[str, Any] | None:
    """Extracción estructurada de texto (lineups leaks, sparring reports)."""
    client = MCPClient.get()
    return await client.call(
        "search",
        "cuba_extract",
        {"content": content, "schema": schema},
    )


async def crawl_transfer_markt(*, team_slug: str, depth: int = 2) -> dict[str, Any] | None:
    """§16.1 capa 6: scrape transfermarkt para fichajes recientes fútbol."""
    url = f"https://www.transfermarkt.com/{team_slug}/transfers"
    client = MCPClient.get()
    return await client.call(
        "search",
        "cuba_crawl",
        {"start_url": url, "depth": depth, "same_domain": True},
    )


async def docs_lookup(*, library: str, query: str) -> dict[str, Any] | None:
    """Acceso a docs técnicas (complemento a Context7)."""
    client = MCPClient.get()
    return await client.call(
        "search",
        "cuba_docs",
        {"library": library, "query": query},
    )
