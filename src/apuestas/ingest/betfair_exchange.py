"""Betfair Exchange — odds de exchange (fair prices, bajo vig) — §15.4 del plan.

Betfair Exchange es el mercado peer-to-peer donde los precios son los más
cercanos a la probabilidad real porque hay traders profesionales matcheando
ofertas. La "delayed API key" es gratis (lag variable 1-180s) y suficiente
para picks pre-match.

Requiere cuenta Betfair (depósito £10 activación única) + crear una App Key en
https://apps.betfair.com/visualisers/api-ng-account-operations/ con "Delayed"
seleccionado.

Credenciales esperadas en .env:
    BETFAIR_APP_KEY        (app key delayed, obligatorio)
    BETFAIR_USERNAME       (email cuenta)
    BETFAIR_PASSWORD       (pass)
    BETFAIR_CERT_PATH      (opcional — si tienes client cert para non-interactive)

Si faltan credenciales → módulo entero fail-soft: funciones retornan listas
vacías sin error. Así el bot sigue arrancando sin Betfair configurado.

Integración: cada event + market se mapea a nuestro schema (sport_code,
bookmaker='betfair', outcome, odds_decimal). Se guarda en odds_history con
bookmaker='betfair' para de-vigging como benchmark sharp.

Referencias:
- betfairlightweight: https://github.com/liampauling/betfair
- API NG endpoints: https://docs.developer.betfair.com/
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

# Event type IDs en Betfair (estables, verificados 2026)
BETFAIR_EVENT_TYPE_IDS: dict[str, int] = {
    "soccer": 1,
    "tennis": 2,
    "nba": 7522,  # Basketball genérico 7522 filtra por competition
    "mlb": 7511,
    "nfl": 6423,  # American football
    "nhl": 7524,  # Ice hockey
    "boxing": 6,
    "mma": 26420387,
}

# Type IDs de mercados straight
STRAIGHT_MARKET_TYPES = {
    "MATCH_ODDS": "h2h",
    "MONEY_LINE": "h2h",
    "OVER_UNDER_25": "totals",
    "TOTAL_POINTS": "totals",
    "HANDICAP": "spreads",
    "ASIAN_HANDICAP": "spreads",
}


@dataclass(slots=True, frozen=True)
class BetfairOdds:
    market_id: str
    event_id: str
    sport_code: str
    home: str
    away: str
    start_time: datetime
    market: str
    outcome: str
    back_price: float  # decimal (mejor oferta back)
    lay_price: float | None = None
    total_matched: float = 0.0


def _credentials_available() -> bool:
    return bool(
        os.environ.get("BETFAIR_APP_KEY")
        and os.environ.get("BETFAIR_USERNAME")
        and os.environ.get("BETFAIR_PASSWORD")
    )


class BetfairExchangeClient:
    """Wrapper de betfairlightweight con fail-soft si no hay creds."""

    def __init__(self) -> None:
        self._tr: Any = None  # betfairlightweight.APIClient
        self._logged_in = False

    async def login(self) -> bool:
        """Login no-interactive con cert opcional, interactive sin cert.

        Returns: True si éxito, False si credenciales ausentes o falla.
        """
        if not _credentials_available():
            logger.info("betfair.creds_missing", hint="set BETFAIR_APP_KEY + USERNAME + PASSWORD")
            return False
        try:
            import betfairlightweight as bfl  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("betfair.lib_missing", hint="pip install betfairlightweight")
            return False

        try:
            cert_path = os.environ.get("BETFAIR_CERT_PATH")
            kwargs: dict[str, Any] = {
                "username": os.environ["BETFAIR_USERNAME"],
                "password": os.environ["BETFAIR_PASSWORD"],
                "app_key": os.environ["BETFAIR_APP_KEY"],
            }
            if cert_path and Path(cert_path).exists():
                kwargs["certs"] = cert_path

            self._tr = bfl.APIClient(**kwargs)

            # cert=non-interactive login, else interactive session
            if cert_path and Path(cert_path).exists():
                self._tr.login()
            else:
                self._tr.login_interactive()

            self._logged_in = True
            logger.info("betfair.login_ok")
        except Exception as exc:
            logger.warning("betfair.login_failed", error=str(exc)[:120])
            return False
        return True

    def logout(self) -> None:
        if self._logged_in and self._tr is not None:
            try:
                self._tr.logout()
            except Exception:
                pass
        self._logged_in = False

    async def fetch_events(self, sport_code: str, hours_ahead: int = 48) -> list[BetfairOdds]:
        """Descarga markets principales de un deporte en ventana hours_ahead."""
        if not self._logged_in:
            ok = await self.login()
            if not ok:
                return []

        event_type_id = BETFAIR_EVENT_TYPE_IDS.get(sport_code)
        if event_type_id is None:
            logger.info("betfair.unknown_sport", sport=sport_code)
            return []

        try:
            from betfairlightweight import filters  # type: ignore[import-untyped]

            now = datetime.now(tz=UTC)
            market_filter = filters.market_filter(
                event_type_ids=[str(event_type_id)],
                market_type_codes=list(STRAIGHT_MARKET_TYPES.keys()),
                market_start_time={
                    "from": now.isoformat(),
                    "to": (now + timedelta(hours=hours_ahead)).isoformat(),
                },
            )

            # List events + markets (llamadas bloqueantes de bfl — envolver en to_thread)
            import asyncio as _asyncio

            catalogue = await _asyncio.to_thread(
                self._tr.betting.list_market_catalogue,
                filter=market_filter,
                market_projection=["EVENT", "MARKET_START_TIME", "RUNNER_DESCRIPTION"],
                max_results=50,
                sort="FIRST_TO_START",
            )

            # Para cada market, obtener best prices
            market_ids = [m.market_id for m in catalogue][:20]
            if not market_ids:
                return []

            book_filter = filters.price_projection(
                price_data=["EX_BEST_OFFERS"],
                ex_best_offers_overrides={"bestPricesDepth": 1},
            )
            books = await _asyncio.to_thread(
                self._tr.betting.list_market_book,
                market_ids=market_ids,
                price_projection=book_filter,
            )
        except Exception as exc:
            logger.warning("betfair.fetch_failed", sport=sport_code, error=str(exc)[:120])
            return []

        out: list[BetfairOdds] = []
        books_by_mid = {b.market_id: b for b in books}
        for cat in catalogue:
            book = books_by_mid.get(cat.market_id)
            if not book:
                continue
            event = cat.event
            market = STRAIGHT_MARKET_TYPES.get(cat.market_name or "", "h2h")
            runners = cat.runners or []
            for i, runner in enumerate(runners):
                rb = next((r for r in book.runners if r.selection_id == runner.selection_id), None)
                if rb is None or not rb.ex.available_to_back:
                    continue
                best_back = rb.ex.available_to_back[0].price
                best_lay = rb.ex.available_to_lay[0].price if rb.ex.available_to_lay else None
                # outcome heurística: runner[0] = home en la mayoría
                if market == "h2h":
                    outcome = "home" if i == 0 else ("draw" if i == 2 else "away")
                elif market == "totals":
                    outcome = "over" if "over" in (runner.runner_name or "").lower() else "under"
                else:
                    outcome = "home" if i == 0 else "away"
                out.append(
                    BetfairOdds(
                        market_id=cat.market_id,
                        event_id=str(event.id),
                        sport_code=sport_code,
                        home=runners[0].runner_name if len(runners) else "",
                        away=runners[1].runner_name if len(runners) > 1 else "",
                        start_time=cat.market_start_time or datetime.now(tz=UTC),
                        market=market,
                        outcome=outcome,
                        back_price=float(best_back),
                        lay_price=float(best_lay) if best_lay else None,
                        total_matched=float(book.total_matched or 0),
                    )
                )
        logger.info("betfair.fetched", sport=sport_code, rows=len(out))
        return out


async def ingest(sport_codes: list[str], hours_ahead: int = 48) -> list[BetfairOdds]:
    """Entry point: descarga + retorna odds. Fail-soft sin creds."""
    if not _credentials_available():
        return []

    client = BetfairExchangeClient()
    try:
        all_odds: list[BetfairOdds] = []
        for sport in sport_codes:
            odds = await client.fetch_events(sport, hours_ahead=hours_ahead)
            all_odds.extend(odds)
        return all_odds
    finally:
        client.logout()
