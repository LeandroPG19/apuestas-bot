"""Ingesta Boxeo — BoxRec scraping (vía cuba-search MCP) + ESPN schedule.

Boxeo es el deporte con peor ecosistema de APIs. Estrategia:
1. ESPN schedule gratis vía endpoint no oficial (site.api.espn.com)
2. BoxRec scraping delegado a cuba-search MCP (mejor evasión Cloudflare)
3. Reportes cualitativos → LLM extraction
4. Tapology como fallback para MMA/KBO relacionado
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import polars as pl

from apuestas.ingest.http_base import BaseAPIClient
from apuestas.obs.logging import get_logger
from apuestas.validators.schemas import validate_fixtures

logger = get_logger(__name__)


class ESPNBoxingClient(BaseAPIClient):
    """Cliente ESPN no oficial para schedule de boxeo.

    Endpoints descubiertos y verificados:
    - /apis/site/v2/sports/boxing/boxing/scoreboard
    - /apis/site/v2/sports/boxing/boxing/events/{event_id}
    """

    base_url = "https://site.api.espn.com"
    source_name = "espn_boxing"
    rate_limit = (30, 60.0)

    async def fetch_scoreboard(
        self, *, dates: str | None = None, limit: int = 100
    ) -> dict[str, Any]:
        """dates en formato YYYYMMDD-YYYYMMDD."""
        params: dict[str, Any] = {"limit": limit}
        if dates:
            params["dates"] = dates
        return await self.get("/apis/site/v2/sports/boxing/boxing/scoreboard", params=params)

    async def fetch_event(self, event_id: str) -> dict[str, Any]:
        return await self.get(f"/apis/site/v2/sports/boxing/boxing/events/{event_id}")


def espn_boxing_to_fixtures(raw: dict[str, Any]) -> pl.DataFrame:
    """Aplana ESPN scoreboard al schema universal."""
    events = raw.get("events", [])
    if not events:
        return _empty_fixtures_df()

    status_map = {
        "STATUS_SCHEDULED": "scheduled",
        "STATUS_IN_PROGRESS": "live",
        "STATUS_FINAL": "finished",
        "STATUS_CANCELED": "cancelled",
        "STATUS_POSTPONED": "postponed",
    }

    rows: list[dict[str, Any]] = []
    for ev in events:
        competitions = ev.get("competitions", [])
        if not competitions:
            continue
        comp = competitions[0]
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        # En boxeo: home/away se toma del orden (main event suele ser [0] vs [1])
        comp_a = competitors[0]
        comp_b = competitors[1]

        rows.append(
            {
                "external_id": str(ev.get("id", "")),
                "sport_code": "boxing",
                "home_team_external_id": str(comp_a.get("id", "")),
                "away_team_external_id": str(comp_b.get("id", "")),
                "start_time": ev.get("date"),
                "status": status_map.get(
                    ev.get("status", {}).get("type", {}).get("name", ""), "scheduled"
                ),
                "league_external_id": "boxing_espn",
                "season": str(datetime.fromisoformat(ev["date"].replace("Z", "+00:00")).year)
                if ev.get("date")
                else None,
            }
        )

    if not rows:
        return _empty_fixtures_df()
    return pl.DataFrame(rows).with_columns(
        pl.col("start_time").str.to_datetime(time_zone="UTC", strict=False).alias("start_time")
    )


def _empty_fixtures_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "external_id": pl.Utf8,
            "sport_code": pl.Utf8,
            "home_team_external_id": pl.Utf8,
            "away_team_external_id": pl.Utf8,
            "start_time": pl.Datetime(time_zone="UTC"),
            "status": pl.Utf8,
            "league_external_id": pl.Utf8,
            "season": pl.Utf8,
        }
    )


async def ingest_boxing_upcoming(*, days_ahead: int = 30) -> pl.DataFrame:
    """Próximos combates ESPN en ventana de days_ahead días."""
    today = datetime.now(tz=__import__("datetime").UTC).strftime("%Y%m%d")
    end = (
        datetime.now(tz=__import__("datetime").UTC).replace(hour=23, minute=59)
        + __import__("datetime").timedelta(days=days_ahead)
    ).strftime("%Y%m%d")

    client = ESPNBoxingClient()
    async with client.session():
        raw = await client.fetch_scoreboard(dates=f"{today}-{end}")

    df = espn_boxing_to_fixtures(raw)
    if df.height == 0:
        logger.info("boxing.no_upcoming", days=days_ahead)
        return df
    validated = validate_fixtures(df)
    logger.info("boxing.upcoming_ingested", rows=validated.height)
    return validated


async def fetch_boxrec_fighter(fighter_slug: str) -> dict[str, Any]:
    """Delegado a cuba-search MCP (Fase 9-10).

    Placeholder: la función se implementará cuando se integre el cliente MCP
    al flow deep_analysis. Por ahora retorna estructura vacía.
    """
    logger.warning(
        "boxing.boxrec.not_implemented",
        fighter=fighter_slug,
        msg="Se resolverá en Fase 9-10 con cuba-search MCP",
    )
    return {
        "fighter": fighter_slug,
        "record": None,
        "ranking": None,
        "weight_class": None,
        "last_fight_date": None,
        "fights_last_year": None,
    }
