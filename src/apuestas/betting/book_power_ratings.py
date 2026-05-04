"""Book power ratings — Sprint 11 Fase D.

¿Qué casa miente más en cuál deporte? Calcula el edge medio en basis points
(bps) vs Pinnacle de-vigged por (bookmaker, league) sobre últimos 90 días.

Uso en `detector.py::line_shopping`: priorizar libros con histórico de líneas
débiles (mean_edge_bps > 0) en el deporte específico. Ejemplo:

- `(caliente, liga_mx)`: +50 bps promedio → excellent soft book MX
- `(draftkings, nba)`: +30 bps en props → target para NBA props
- `(pinnacle, *)`: 0 bps por construcción → benchmark, no apostar
- `(pointsbet, *)`: +5 bps → muy bien calibrado, evitar

Rating se recalcula diariamente mediante job Prefect.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class BookEdgeProfile:
    bookmaker: str
    league: str
    sport_code: str
    mean_edge_bps: float
    std_edge_bps: float
    n_samples: int
    last_updated: datetime


_RATINGS_CACHE: dict[tuple[str, str], BookEdgeProfile] = {}


async def compute_book_power_ratings(
    session,  # type: ignore[no-untyped-def]
    *,
    lookback_days: int = 90,
    min_samples: int = 50,
) -> dict[tuple[str, str], BookEdgeProfile]:
    """Computa edge bps promedio por (bookmaker, league).

    Fórmula por pick:
        edge_bps = (1/pinnacle_fair_odds - 1/book_odds) * 10000

    Donde `pinnacle_fair_odds` es la odds de-vigged del mismo market+outcome
    al momento. `book_odds` es el precio ofrecido por el libro soft.
    Positivo = libro paga más (edge a favor del jugador).
    """
    from sqlalchemy import text

    try:
        rows = (
            await session.execute(
                text(
                    """
                    WITH pinn AS (
                        SELECT oh.match_id, oh.market, oh.outcome,
                               AVG(oh.odds) AS pinn_odds
                        FROM odds_history oh
                        JOIN matches m ON m.id = oh.match_id
                        WHERE oh.bookmaker = 'pinnacle'
                          AND oh.ts >= NOW() - MAKE_INTERVAL(days => :lookback)
                          AND m.status = 'finished'
                        GROUP BY oh.match_id, oh.market, oh.outcome
                    )
                    SELECT oh.bookmaker,
                           COALESCE(l.name, m.sport_code) AS league,
                           m.sport_code,
                           AVG((1.0/pinn.pinn_odds - 1.0/oh.odds) * 10000) AS mean_edge_bps,
                           STDDEV_POP((1.0/pinn.pinn_odds - 1.0/oh.odds) * 10000) AS std_bps,
                           COUNT(*) AS n
                    FROM odds_history oh
                    JOIN pinn
                      ON pinn.match_id = oh.match_id
                     AND pinn.market   = oh.market
                     AND pinn.outcome  = oh.outcome
                    JOIN matches m ON m.id = oh.match_id
                    LEFT JOIN leagues l ON l.id = m.league_id
                    WHERE oh.bookmaker <> 'pinnacle'
                      AND oh.ts >= NOW() - MAKE_INTERVAL(days => :lookback)
                      AND oh.odds BETWEEN 1.2 AND 10.0
                      AND pinn.pinn_odds BETWEEN 1.2 AND 10.0
                    GROUP BY oh.bookmaker, league, m.sport_code
                    HAVING COUNT(*) >= :min_n
                    ORDER BY mean_edge_bps DESC
                    """
                ),
                {"lookback": lookback_days, "min_n": min_samples},
            )
        ).fetchall()
    except Exception as exc:
        logger.warning("book_power.compute_fail", error=str(exc)[:100])
        return {}

    out: dict[tuple[str, str], BookEdgeProfile] = {}
    now = datetime.now(UTC)
    for r in rows:
        bookmaker, league, sport, mean_bps, std_bps, n = r
        key = (str(bookmaker).lower(), str(league).lower())
        out[key] = BookEdgeProfile(
            bookmaker=str(bookmaker).lower(),
            league=str(league).lower(),
            sport_code=str(sport).lower(),
            mean_edge_bps=float(mean_bps or 0.0),
            std_edge_bps=float(std_bps or 0.0),
            n_samples=int(n or 0),
            last_updated=now,
        )
    logger.info(
        "book_power.computed",
        n_profiles=len(out),
        top=sorted(out.items(), key=lambda kv: -kv[1].mean_edge_bps)[:3],
    )
    # Actualizar cache global
    _RATINGS_CACHE.clear()
    _RATINGS_CACHE.update(out)
    return out


def _load_from_file_if_empty() -> None:
    """Lazy-load desde `artifacts/book_power/latest.json` si cache memoria vacío."""
    if _RATINGS_CACHE:
        return
    try:
        import json
        from pathlib import Path

        path = Path(__file__).resolve().parents[3] / "artifacts" / "book_power" / "latest.json"
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        for key_str, v in data.items():
            bk, lg = key_str.split("|", 1)
            _RATINGS_CACHE[(bk, lg)] = BookEdgeProfile(
                bookmaker=v["bookmaker"],
                league=v["league"],
                sport_code=v["sport_code"],
                mean_edge_bps=float(v["mean_edge_bps"]),
                std_edge_bps=float(v["std_edge_bps"]),
                n_samples=int(v["n_samples"]),
                last_updated=datetime.fromisoformat(v["last_updated"]),
            )
        logger.info("book_power.loaded_from_file", n=len(_RATINGS_CACHE))
    except Exception as exc:
        logger.debug("book_power.file_load_fail", error=str(exc)[:80])


def get_cached_edge(bookmaker: str, league: str) -> float:
    """Devuelve mean_edge_bps en cache (0 si no hay data). Lazy-load archivo si vacío."""
    _load_from_file_if_empty()
    key = (bookmaker.lower(), league.lower())
    profile = _RATINGS_CACHE.get(key)
    return profile.mean_edge_bps if profile else 0.0


def rank_books_for(
    league: str, sport_code: str, *, min_edge_bps: float = 10.0
) -> list[BookEdgeProfile]:
    """Lista de books ordenados por edge positivo para (league, sport)."""
    candidates = [
        p
        for (book, lg), p in _RATINGS_CACHE.items()
        if lg == league.lower()
        and p.sport_code == sport_code.lower()
        and p.mean_edge_bps >= min_edge_bps
    ]
    return sorted(candidates, key=lambda p: -p.mean_edge_bps)


__all__ = [
    "BookEdgeProfile",
    "compute_book_power_ratings",
    "get_cached_edge",
    "rank_books_for",
]
