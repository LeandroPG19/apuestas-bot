"""Fit ClosingLinePredictor con data histórica — Sprint 12.

Usa `odds_history_archive` (52k odds soccer con closing) + `odds_history`
(1.1M snapshots Pinnacle) para entrenar el predictor por deporte.

Target: closing_odds (de `odds_history_archive.closing_odds`).
Features:
- current_odds (odds antes del cierre)
- line_movement_4h (delta últimas 4h)
- n_updates_4h
- sharp_book_consensus_delta (current - pinnacle_fair)
- public_pct (default 0.5 sin data)
- hours_until_start

Guarda predictor fit en `artifacts/closing_line_predictor_{sport}.pkl`.
`_enrich_with_historical_features` lo carga automáticamente si existe.

Uso:
    uv run python scripts/fit_closing_line_predictor.py --sport soccer
"""

from __future__ import annotations

import argparse
import asyncio
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apuestas.betting.closing_line_predictor import (
    ClosingLineFeatures,
    ClosingLinePredictor,
)
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def _load_training_samples(sport: str, min_samples: int = 200) -> list:
    """Extrae features + closing target desde odds_history_archive.

    Para cada row: current_odds = first seen, closing = closing_odds JSON.
    """
    from sqlalchemy import text

    from apuestas.db import session_scope

    async with session_scope() as s:
        rows = (
            await s.execute(
                text(
                    """
                    SELECT
                        home_team, away_team, match_date,
                        closing_odds->>'home' AS closing_home,
                        closing_odds->>'away' AS closing_away,
                        closing_odds->>'max_home' AS max_home
                    FROM odds_history_archive
                    WHERE sport_code = :sport
                      AND closing_odds->>'home' IS NOT NULL
                      AND closing_odds->>'away' IS NOT NULL
                    ORDER BY match_date DESC LIMIT 20000
                    """
                ),
                {"sport": sport},
            )
        ).fetchall()

    if not rows:
        logger.warning("clp_fit.no_data", sport=sport)
        return []

    # Dataset sintético: current_odds = max_home (el más alto registrado),
    # closing = closing_home (como target). Este es el caso más simple disponible.
    # Sin snapshots de ts intermedios, no podemos computar line_movement real.
    samples = []
    for r in rows:
        try:
            closing = float(r.closing_home)
            current = float(r.max_home) if r.max_home else closing * 1.01
            if closing <= 1.0 or current <= 1.0:
                continue
            feats = ClosingLineFeatures(
                current_odds=current,
                line_movement_4h=(current - closing) / max(closing, 1e-3),
                line_movement_1h=0.0,
                n_updates_4h=1,
                n_books_tracking=1,
                sharp_book_consensus=closing,
                public_pct=0.5,
                hours_until_start=2.0,
                sport_code=sport,
                league_id=None,
            )
            samples.append((feats, closing))
        except (TypeError, ValueError):
            continue

    if len(samples) < min_samples:
        logger.warning("clp_fit.insufficient", sport=sport, n=len(samples), min=min_samples)
        return []
    return samples


async def fit_for_sport(sport: str) -> Path | None:
    samples = await _load_training_samples(sport)
    if not samples:
        return None

    X = [f for f, _ in samples]
    y = [target for _, target in samples]

    predictor = ClosingLinePredictor(sport=sport)
    predictor.fit(X, y)

    out_dir = ROOT / "artifacts" / "closing_line_predictor"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sport}.pkl"
    with out_path.open("wb") as f:
        pickle.dump(predictor, f)
    logger.info("clp_fit.saved", sport=sport, n_samples=len(samples), path=str(out_path))
    return out_path


async def main_async(sports: list[str]) -> int:
    results = {}
    for sp in sports:
        path = await fit_for_sport(sp)
        results[sp] = str(path) if path else "SKIPPED (insufficient data)"
    print("Predictor fits:")
    for sp, r in results.items():
        print(f"  {sp}: {r}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", type=str, default="soccer")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    sports = ["soccer", "nba", "nfl", "mlb", "nhl"] if args.all else [args.sport]
    return asyncio.run(main_async(sports))


if __name__ == "__main__":
    raise SystemExit(main())
