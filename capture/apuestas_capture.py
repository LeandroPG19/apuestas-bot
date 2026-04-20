"""Capture job ligero — CLV tracking independiente del stack Docker.

Standalone (~30 MB RAM). Corre vía systemd --user timer cada 5 min.
Consulta bets pendientes ya terminadas, busca closing line en odds_history
(preferentemente Pinnacle/Circa/Betfair), computa CLV y actualiza bets.

Idempotente: solo escribe closing_line si es NULL.
Usa The Odds API como fallback si odds_history no tiene el cierre.

Uso:
    uv run python capture/apuestas_capture.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import asyncpg
import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

SHARP_BOOKS = ("pinnacle", "circa", "bookmaker", "betfair")


def _read_env() -> dict[str, str]:
    env_path = ROOT / ".env"
    result: dict[str, str] = {}
    if not env_path.exists():
        return result
    for line in env_path.read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            result[key.strip()] = val.strip().strip('"').strip("'")
    return result


async def _find_closing_in_db(
    conn: asyncpg.Connection,
    *,
    match_id: int,
    market: str,
    outcome: str,
    line: float | None,
    start_time: datetime,
    window_minutes: int = 5,
) -> tuple[float, str] | None:
    """Busca odds en odds_history cercanas al start_time, preferencia sharp books."""
    window_start = start_time - timedelta(minutes=window_minutes)
    row = await conn.fetchrow(
        """
        SELECT bookmaker, odds
        FROM odds_history
        WHERE match_id = $1
          AND market = $2
          AND outcome = $3
          AND (line = $4 OR ($4 IS NULL AND line IS NULL))
          AND ts BETWEEN $5 AND $6
        ORDER BY
          CASE WHEN bookmaker = 'pinnacle' THEN 0
               WHEN bookmaker = 'circa' THEN 1
               WHEN bookmaker = 'betfair' THEN 2
               WHEN bookmaker = 'bookmaker' THEN 3
               ELSE 10 END ASC,
          ts DESC
        LIMIT 1
        """,
        match_id,
        market,
        outcome,
        line,
        window_start,
        start_time,
    )
    if row is None:
        return None
    return float(row["odds"]), str(row["bookmaker"])


async def _fallback_odds_api(
    client: httpx.AsyncClient,
    *,
    api_key: str,
    external_match_id: str,
    market: str,
) -> float | None:
    """Fallback: The Odds API historical (requiere tier paid). Retorna None en free."""
    # The Odds API historical requiere /historical endpoint ($30+/mes).
    # En free tier retornamos None silenciosamente.
    return None


async def _update_bet_clv(
    conn: asyncpg.Connection,
    *,
    bet_id: int,
    odds_placed: float,
    closing_odds: float,
) -> None:
    """Update idempotente si closing_line aún es NULL."""
    clv = odds_placed / closing_odds - 1.0 if closing_odds > 1.0 else 0.0
    await conn.execute(
        """
        UPDATE bets
        SET closing_line = $2, clv = $3
        WHERE id = $1 AND closing_line IS NULL
        """,
        bet_id,
        closing_odds,
        clv,
    )


async def main() -> int:
    env = {**_read_env(), **os.environ}
    user = env.get("POSTGRES_USER", "apuestas")
    pwd = env.get("POSTGRES_PASSWORD", "")
    host = env.get("POSTGRES_HOST_LOCALHOST", "localhost")
    port = env.get("POSTGRES_HOST_PORT", "5433")
    db = env.get("POSTGRES_DB", "apuestas")
    odds_key = env.get("THE_ODDS_API_KEY", "")

    dsn = f"postgresql://{user}:{pwd}@{host}:{port}/{db}"
    try:
        conn = await asyncpg.connect(dsn, timeout=10)
    except OSError as exc:
        print(f"⚠ Postgres no accesible ({exc}); stack apagado?")
        return 0

    stats = {"checked": 0, "reconciled_db": 0, "reconciled_api": 0, "no_data": 0}

    try:
        cutoff = datetime.now(tz=UTC) - timedelta(hours=2)
        rows = await conn.fetch(
            """
            SELECT b.id, b.match_id, b.market, b.outcome, b.line, b.odds_placed,
                   m.external_id, m.start_time
            FROM bets b JOIN matches m ON m.id = b.match_id
            WHERE b.status = 'pending'
              AND b.closing_line IS NULL
              AND m.start_time < $1
              AND m.start_time > NOW() - INTERVAL '14 days'
            LIMIT 50
            """,
            cutoff,
        )

        if not rows:
            return 0

        print(f"▶ {len(rows)} bets pendientes con closing pendiente")

        async with httpx.AsyncClient(timeout=10) as client:
            for row in rows:
                stats["checked"] += 1
                bet_id = int(row["id"])
                match_id = int(row["match_id"])
                market = str(row["market"])
                outcome = str(row["outcome"])
                line = float(row["line"]) if row["line"] is not None else None
                odds_placed = float(row["odds_placed"])
                start_time = row["start_time"]

                # Intentar odds_history primero
                result = await _find_closing_in_db(
                    conn,
                    match_id=match_id,
                    market=market,
                    outcome=outcome,
                    line=line,
                    start_time=start_time,
                )
                if result is not None:
                    closing_odds, source = result
                    await _update_bet_clv(
                        conn,
                        bet_id=bet_id,
                        odds_placed=odds_placed,
                        closing_odds=closing_odds,
                    )
                    stats["reconciled_db"] += 1
                    continue

                # Fallback Odds API si paid
                if odds_key:
                    api_odds = await _fallback_odds_api(
                        client,
                        api_key=odds_key,
                        external_match_id=str(row["external_id"]),
                        market=market,
                    )
                    if api_odds is not None:
                        await _update_bet_clv(
                            conn,
                            bet_id=bet_id,
                            odds_placed=odds_placed,
                            closing_odds=api_odds,
                        )
                        stats["reconciled_api"] += 1
                        continue

                stats["no_data"] += 1

        print(
            f"✓ CLV capture done: checked={stats['checked']} "
            f"db={stats['reconciled_db']} api={stats['reconciled_api']} "
            f"no_data={stats['no_data']}"
        )
    finally:
        await conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
