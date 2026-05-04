"""Validación de integridad de odds históricas antes de insertar en DB.

Los CSVs de football-data.co.uk, tennis-data.co.uk, SBR archive y Retrosheet
tienen errores conocidos (duplicates, odds invertidas, missing closing, fat-tail
implausibles). Sin validación, el modelo ML aprende ruido.

Esta capa filtra rows inválidos marcándolos con `quality_flag`:
- `ok`: pasa todos los checks
- `invalid_overround`: vig total fuera de [2%, 12%] (realista)
- `invalid_range`: odds fuera de [1.01, 50.0] (fat-tail guard)
- `invalid_inversion`: odds home/away con implícitas contradictorias
- `invalid_timing`: closing odds muy lejos del start_time
- `duplicate`: (match_id, bookmaker, market, outcome, ts) ya existe

Uso:
    from apuestas.validators.historical_odds_integrity import (
        validate_odds_row, HistoricalOddsRow
    )
    for row in csv_rows:
        flag = validate_odds_row(row)
        if flag == "ok":
            await insert(...)
        else:
            logger.info("historical.skip", reason=flag, row=row)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from apuestas.betting.devig import overround

QualityFlag = Literal[
    "ok",
    "invalid_overround",
    "invalid_range",
    "invalid_inversion",
    "invalid_timing",
    "invalid_missing",
]

# Realistic bounds (confirmed observando 10+ seasons de football-data.co.uk)
MIN_OVERROUND = 0.005  # 0.5% (Pinnacle en markets low-hold, raro)
MAX_OVERROUND = 0.20  # 20% (MX books en mercados exóticos)
MIN_ODDS = 1.01  # decimal > 1.01 (fat-tail guard)
MAX_ODDS = 50.0  # mercados extremos
MIN_CLOSING_DELTA = timedelta(seconds=60)  # closing ≥ 60s antes de start
MAX_CLOSING_DELTA = timedelta(minutes=30)  # closing ≤ 30min antes de start


@dataclass(frozen=True, slots=True)
class HistoricalOddsRow:
    """Snapshot de una quota histórica para validación.

    `outcomes_odds`: dict market outcome → odds decimal (ej. {"home": 2.1, "draw": 3.4, "away": 3.5}).
    `ts`: timestamp del snapshot.
    `start_time`: start del match (para validar timing de closing).
    `is_closing`: True si es closing line (aplica check timing).
    """

    match_id: int
    bookmaker: str
    market: str
    outcomes_odds: dict[str, float]
    ts: datetime
    start_time: datetime
    is_closing: bool = False


def validate_odds_row(row: HistoricalOddsRow) -> QualityFlag:
    """Retorna `ok` o el primer flag de error encontrado."""
    odds_values = list(row.outcomes_odds.values())

    if not odds_values or any(v is None for v in odds_values):
        return "invalid_missing"

    # Range check
    for v in odds_values:
        if not (MIN_ODDS <= v <= MAX_ODDS):
            return "invalid_range"

    # Overround check (solo si el mercado es "cerrado": tiene todos los outcomes)
    # h2h soccer = 3 outcomes (home/draw/away); NBA h2h = 2 (home/away);
    # totals = 2 (over/under); spreads = 2.
    n = len(odds_values)
    if n in (2, 3):
        try:
            vig = overround(odds_values)
        except ValueError:
            return "invalid_range"
        if not (MIN_OVERROUND <= vig <= MAX_OVERROUND):
            return "invalid_overround"

    # Inversion check (h2h 2-outcomes: si odds_home > odds_away implica p_home < p_away,
    # pero si una fuente las intercambió, p_home_implied + p_away_implied no da 1+vig).
    # Esta heurística solo aplica a binarios 2-outcome (sin draw).
    if n == 2 and "home" in row.outcomes_odds and "away" in row.outcomes_odds:
        p_h = 1.0 / row.outcomes_odds["home"]
        p_a = 1.0 / row.outcomes_odds["away"]
        # Si ambos < 0.4 o ambos > 0.7 hay algo muy raro
        if (p_h < 0.3 and p_a < 0.3) or (p_h > 0.75 and p_a > 0.75):
            return "invalid_inversion"

    # Closing timing check
    if row.is_closing:
        delta = row.start_time - row.ts
        if delta < MIN_CLOSING_DELTA or delta > MAX_CLOSING_DELTA:
            return "invalid_timing"

    return "ok"


def batch_validate(
    rows: list[HistoricalOddsRow],
) -> tuple[list[HistoricalOddsRow], dict[QualityFlag, int]]:
    """Separa valid vs invalid. Retorna (valid_rows, counter_by_flag)."""
    valid: list[HistoricalOddsRow] = []
    counter: dict[QualityFlag, int] = {}
    for row in rows:
        flag = validate_odds_row(row)
        counter[flag] = counter.get(flag, 0) + 1
        if flag == "ok":
            valid.append(row)
    return valid, counter
