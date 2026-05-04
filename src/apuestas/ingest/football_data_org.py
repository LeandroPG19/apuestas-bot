"""Cliente football-data.org — alternativa GRATIS a API-Football.

Rate limit free tier: 10 req/min = 14,400/día. Cobertura free:
PL (Premier), PD (LaLiga), BL1 (Bundesliga), SA (Serie A), FL1 (Ligue 1),
CL (Champions), BSA (Brasileirão), DED (Eredivisie), PPL (Primeira), ELC (Championship).

NO cubre en free tier: Liga MX, MLS. Esos se cubren con Sofascore unofficial
(`apuestas.ingest.sofascore`) o con `sync_odds_api_scores` (fuzzy matching).

Auth: header `X-Auth-Token` con key gratuita (sin tarjeta) en football-data.org/client/register.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from apuestas.config import get_settings
from apuestas.ingest.http_base import BaseAPIClient
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


# Sport_code interno → competition code de football-data.org
COMPETITION_CODE_MAP: dict[str, str] = {
    "soccer_epl": "PL",
    "soccer_laliga": "PD",
    "soccer_bundesliga": "BL1",
    "soccer_seriea": "SA",
    "soccer_ligue1": "FL1",
    "soccer_ucl": "CL",
    "soccer_eredivisie": "DED",
    "soccer_championship": "ELC",
    "soccer_brasileirao": "BSA",
    "soccer_primeira": "PPL",
}

# Status FDO → status interno
FINAL_STATUSES = {"FINISHED", "AWARDED"}


class FootballDataOrgClient(BaseAPIClient):
    base_url = "https://api.football-data.org/v4"
    source_name = "football_data_org"
    # Free tier: 10 req/min. Bajado a 6/min para evitar bursts que el limiter
    # local no detecta (la API mide su propio bucket; cada 429 dispara stamina
    # retry x4 → 4 warnings extras por cada miss). 6/min = 1 req/10s da margen.
    rate_limit = (6, 60.0)

    def __init__(self, *, api_key: str | None = None) -> None:
        settings = get_settings()
        key = api_key or (
            settings.apis.football_data_org_key.get_secret_value()
            if settings.apis.football_data_org_key
            else None
        )
        if not key:
            msg = "FOOTBALL_DATA_ORG_KEY requerida"
            raise ValueError(msg)
        super().__init__(api_key=key)
        self._key = key

    def _default_headers(self) -> dict[str, str]:
        return {
            "X-Auth-Token": self._key,
            "User-Agent": f"apuestas-bot/0.1 (+{self.source_name})",
        }

    async def list_competitions(self) -> list[dict[str, Any]]:
        data = await self.get("/competitions")
        return list(data.get("competitions", []))

    async def fetch_matches(
        self,
        competition_code: str,
        *,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """GET /competitions/{code}/matches.

        Args:
            competition_code: 'PL', 'PD', 'BL1', ...
            date_from/date_to: ventana temporal (formato YYYY-MM-DD).
            status: 'SCHEDULED' | 'LIVE' | 'IN_PLAY' | 'FINISHED' | ...
        """
        params: dict[str, Any] = {}
        if date_from:
            params["dateFrom"] = date_from.strftime("%Y-%m-%d")
        if date_to:
            params["dateTo"] = date_to.strftime("%Y-%m-%d")
        if status:
            params["status"] = status
        data = await self.get(f"/competitions/{competition_code}/matches", params=params)
        return list(data.get("matches", []))

    async def fetch_match(self, match_id: int) -> dict[str, Any]:
        return await self.get(f"/matches/{match_id}")

    async def fetch_teams(self, competition_code: str) -> list[dict[str, Any]]:
        data = await self.get(f"/competitions/{competition_code}/teams")
        return list(data.get("teams", []))
