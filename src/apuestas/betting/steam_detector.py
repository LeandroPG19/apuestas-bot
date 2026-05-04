"""Steam move detector — técnica #1 de Billy Walters.

Un "steam move" es cuando múltiples sportsbooks mueven una línea de forma
coordinada en poco tiempo, señal de que dinero sharp (Pinnacle/Circa)
está entrando.

Algoritmo:
1. Para cada mercado (match + outcome), toma snapshots de line_movement
   de últimos N minutos por bookmaker.
2. Calcula delta_pct por book.
3. Si ≥3 books movieron ≥3% en misma dirección dentro de ventana 10min → steam.
4. Si Pinnacle lidera (movió primero) → confianza alta.
5. Inserta en steam_moves y (opcional) NOTIFY.

Uso:
    await detect_steam_moves(window_minutes=15, min_books=3, min_pct=0.03)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

SHARP_BOOKS = frozenset({"pinnacle", "circa", "bookmaker"})


@dataclass(slots=True)
class SteamCandidate:
    match_id: int
    market: str
    outcome: str
    direction: str
    magnitude_pct: float
    n_books: int
    pinnacle_led: bool
    books_moved: list[str]


async def capture_line_snapshot(
    *,
    match_id: int,
    bookmaker: str,
    market: str,
    outcome: str,
    odds: float,
    line: float | None = None,
) -> None:
    """Persiste snapshot actual para alimentar al detector."""
    async with session_scope() as s:
        # Leer odds previa del mismo libro para calcular volume_indicator
        r = await s.execute(
            text(
                """
                SELECT odds FROM line_movement_snapshots
                WHERE match_id = :m AND bookmaker = :b AND market = :mk AND outcome = :o
                ORDER BY ts DESC LIMIT 1
                """
            ),
            {"m": match_id, "b": bookmaker, "mk": market, "o": outcome},
        )
        prev = r.first()
        vol_pct = 0.0
        if prev and float(prev.odds) > 0:
            vol_pct = (odds - float(prev.odds)) / float(prev.odds)

        await s.execute(
            text(
                """
                INSERT INTO line_movement_snapshots
                    (ts, match_id, bookmaker, market, outcome, odds, line, volume_indicator)
                VALUES (NOW(), :m, :b, :mk, :o, :odds, :ln, :v)
                ON CONFLICT DO NOTHING
                """
            ),
            {
                "m": match_id,
                "b": bookmaker,
                "mk": market,
                "o": outcome,
                "odds": odds,
                "ln": line,
                "v": vol_pct,
            },
        )


async def detect_steam_moves(
    *,
    window_minutes: int = 360,
    min_books: int = 2,
    min_pct: float = 0.03,
) -> list[SteamCandidate]:
    """Escanea movimientos recientes en odds_history; detecta steam en progreso.

    Estrategia: para cada (match, market, outcome, bookmaker), compara la odds
    más reciente vs la más antigua dentro de la ventana. Si ≥3 books movieron
    en la misma dirección con magnitud ≥3%, es steam.

    Antes leía de `line_movement_snapshots` (que estaba vacía — nunca se pobló).
    Ahora lee directo de `odds_history` (1.8M rows reales).
    """
    now = datetime.now(tz=UTC)
    since = now - timedelta(minutes=window_minutes)

    async with session_scope() as s:
        r = await s.execute(
            text(
                """
                WITH bookended AS (
                    SELECT match_id, market, outcome, bookmaker, odds, ts,
                           ROW_NUMBER() OVER (
                               PARTITION BY match_id, market, outcome, bookmaker
                               ORDER BY ts ASC
                           ) AS rn_first,
                           ROW_NUMBER() OVER (
                               PARTITION BY match_id, market, outcome, bookmaker
                               ORDER BY ts DESC
                           ) AS rn_last
                    FROM odds_history
                    WHERE ts >= :since
                ),
                pairs AS (
                    SELECT
                        f.match_id, f.market, f.outcome, f.bookmaker,
                        f.odds AS first_odds, f.ts AS first_ts,
                        l.odds AS last_odds
                    FROM bookended f
                    JOIN bookended l ON l.match_id = f.match_id
                        AND l.market = f.market AND l.outcome = f.outcome
                        AND l.bookmaker = f.bookmaker
                        AND l.rn_last = 1
                    WHERE f.rn_first = 1 AND f.odds != l.odds
                )
                SELECT match_id, market, outcome, bookmaker,
                       first_odds, last_odds, first_ts
                FROM pairs
                """
            ),
            {"since": since},
        )
        rows = r.all()

    # Agrupar por (match, market, outcome) y contar books que se movieron
    grouped: dict[tuple[int, str, str], list[dict]] = {}
    for row in rows:
        delta_pct = (float(row.last_odds) - float(row.first_odds)) / float(row.first_odds)
        if abs(delta_pct) < min_pct:
            continue
        key = (int(row.match_id), str(row.market), str(row.outcome))
        grouped.setdefault(key, []).append(
            {
                "bookmaker": row.bookmaker,
                "delta_pct": delta_pct,
                "first_ts": row.first_ts,
            }
        )

    candidates: list[SteamCandidate] = []
    for (match_id, market, outcome), moves in grouped.items():
        if len(moves) < min_books:
            continue

        # Todos en la misma dirección
        up = [m for m in moves if m["delta_pct"] > 0]
        down = [m for m in moves if m["delta_pct"] < 0]
        direction_books = up if len(up) >= min_books else (down if len(down) >= min_books else None)
        if not direction_books:
            continue

        direction = "up" if direction_books is up else "down"
        # Clamp a 9.9999 — schema steam_moves.magnitude_pct es numeric(5,4).
        # Sin clamp, casos extremos (odds 1.01 → 5.0 = delta_pct=4.0; promedio
        # ponderado podría disparar a 4.5+) sobrepasan el rango y causan
        # NumericValueOutOfRangeError. 9.99 = 999% es magnitud absurda en odds.
        magnitude = min(
            9.9999,
            sum(abs(m["delta_pct"]) for m in direction_books) / len(direction_books),
        )

        # ¿Pinnacle lideró? (se movió antes que los demás)
        direction_books.sort(key=lambda m: m["first_ts"])
        first_book = direction_books[0]["bookmaker"]
        pinnacle_led = first_book in SHARP_BOOKS

        candidate = SteamCandidate(
            match_id=match_id,
            market=market,
            outcome=outcome,
            direction=direction,
            magnitude_pct=magnitude,
            n_books=len(direction_books),
            pinnacle_led=pinnacle_led,
            books_moved=[m["bookmaker"] for m in direction_books],
        )
        candidates.append(candidate)

        # Persistir steam
        async with session_scope() as s:
            await s.execute(
                text(
                    """
                    INSERT INTO steam_moves
                        (detected_at, match_id, market, outcome, direction,
                         magnitude_pct, n_books_moved, pinnacle_leading,
                         books_involved, window_minutes)
                    VALUES
                        (NOW(), :m, :mk, :o, :d, :mag, :n, :pin,
                         CAST(:books AS jsonb), :w)
                    """
                ),
                {
                    "m": match_id,
                    "mk": market,
                    "o": outcome,
                    "d": direction,
                    "mag": round(magnitude, 4),
                    "n": len(direction_books),
                    "pin": pinnacle_led,
                    "books": _json.dumps(candidate.books_moved),
                    "w": window_minutes,
                },
            )

    if candidates:
        logger.info(
            "steam_detector.found",
            count=len(candidates),
            pinnacle_led_count=sum(1 for c in candidates if c.pinnacle_led),
        )
    return candidates


import json as _json
