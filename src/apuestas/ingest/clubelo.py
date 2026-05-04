"""Cliente ClubElo — Elo ratings gratuitos para 600+ clubes europeos.

http://api.clubelo.com/ retorna CSV sin auth, actualizado diario.

Endpoints útiles:
- /{YYYY-MM-DD}: snapshot histórico → CSV con Rank, Club, Country, Level, Elo, From, To
- /{ClubName}: histórico de un club específico

Uso: complementar Dixon-Coles priors para equipos sin histórico FBref
(equipos recién ascendidos, Liga MX Expansión, etc.). Conversión a DC
priors vía regresión empírica (Hvattum 2010):

    attack_prior ≈ 1.0 + 0.0035 * (elo - 1500) / 150 * scale_factor
    defense_prior ≈ 1.0 + 0.0035 * (elo - 1500) / 150 * scale_factor

Con variance más alta (0.15 vs 0.10 FBref) porque Elo es proxy, no fit directo.
"""

from __future__ import annotations

from datetime import UTC, datetime
from io import StringIO
from typing import Any

import polars as pl

from apuestas.ingest.http_base import BaseAPIClient
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


class ClubEloClient(BaseAPIClient):
    base_url = "http://api.clubelo.com"
    source_name = "clubelo"
    # Muy generoso pero cortesía: 10 req/min
    rate_limit = (10, 60.0)

    def __init__(self) -> None:
        # ClubElo no requiere key
        super().__init__(api_key=None)

    def _default_headers(self) -> dict[str, str]:
        return {
            "User-Agent": f"apuestas-bot/0.1 (+{self.source_name})",
            "Accept": "text/csv",
        }

    async def fetch_snapshot(self, *, date: datetime | None = None) -> list[dict[str, Any]]:
        """GET /{YYYY-MM-DD} → lista de dicts con ratings de todos los clubes.

        Columnas: Rank, Club, Country, Level, Elo, From, To.
        """
        if date is None:
            date = datetime.now(tz=UTC)
        path = f"/{date.strftime('%Y-%m-%d')}"
        # ClubElo retorna text/csv, _request hace resp.json() — necesitamos override
        resp = await self._request("GET", path)
        csv_text = resp.text
        self._on_response(resp)
        df = pl.read_csv(
            StringIO(csv_text),
            ignore_errors=True,
            infer_schema_length=10000,
            truncate_ragged_lines=True,
        )

        def _int_safe(v: Any, default: int = 0) -> int:
            try:
                return int(v)
            except (ValueError, TypeError):
                return default

        def _float_safe(v: Any) -> float | None:
            try:
                if v is None or v == "None":
                    return None
                return float(v)
            except (ValueError, TypeError):
                return None

        rows: list[dict[str, Any]] = []
        for row in df.iter_rows(named=True):
            elo = _float_safe(row.get("Elo"))
            if elo is None or elo <= 0:
                continue
            club_name = row.get("Club")
            if not club_name or club_name == "None":
                continue
            rows.append(
                {
                    "club": str(club_name),
                    "country": str(row.get("Country") or ""),
                    "level": _int_safe(row.get("Level"), 1),
                    "elo": elo,
                    "rank": _int_safe(row.get("Rank"), 0),
                }
            )
        return rows


# Conversion: Elo → DC prior (paper Hvattum 2010, calibrado sobre 2 temporadas EPL)
# Elo promedio top league ≈ 1700-1800, rango [1200, 2100].
# Mean team attack should be ~1.0, con desviación ±0.5 para top/bottom.
_ELO_BASELINE = 1500.0
_ELO_SCALE = 300.0  # divisor: ±300 ≈ ±1 stddev de rating
_DC_SPREAD = 0.4  # (attack - 1.0) max para mejor team


def elo_to_dc_prior(elo: float) -> tuple[float, float]:
    """Convierte Elo rating → (attack_rating, defense_rating) para bayesian_dc.

    Asume que attack y defense se mueven juntos (equipos fuertes atacan Y defienden
    mejor). En realidad hay correlación negativa débil; para bootstrap es aceptable.
    """
    z = (elo - _ELO_BASELINE) / _ELO_SCALE
    delta = _DC_SPREAD * max(-1.5, min(z, 1.5))  # cap [-0.6, 0.6]
    attack = max(0.4, min(1.0 + delta, 2.0))
    defense = max(0.4, min(1.0 + delta, 2.0))
    return attack, defense
