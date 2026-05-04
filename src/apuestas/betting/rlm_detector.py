"""Fase 4.1 — Reverse Line Movement (RLM) detector.

La señal sharp más robusta en NBA/NFL: cuando **línea se mueve CONTRA el % público**
(público 80% Lakers pero línea mueve hacia Celtics) → sharp money firme en el
lado contrarian. Complementa el contrarian signal de Fase 2.2.

Combina:
  - `odds_history` últimos 24h → line_movement por outcome
  - `public_betting_snapshots` últimos 4h → % público
  - Divergencia `line_direction` vs `public_direction` = RLM signal

Output: `rlm_signal ∈ {none, weak, strong}` + `contrarian_outcome`.

Uso en detector: feature adicional que bump Kelly bonus ×1.2 si RLM strong.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class RLMSignal:
    match_id: int
    market: str
    contrarian_outcome: str  # outcome que el sharp money está respaldando
    strength: str  # "none" | "weak" | "strong"
    line_move_pct: float  # fraccional (+0.03 = 3% line moved)
    public_pct_on_other_side: float  # 0-1
    detected_at: datetime


async def detect_rlm(
    match_id: int,
    market: str = "h2h",
    *,
    public_threshold: float = 0.65,
    line_move_threshold: float = 0.02,
) -> RLMSignal | None:
    """Detecta RLM para un match+market.

    Criterio strong:
      - Público apostando ≥ 65% en un outcome
      - Línea moviéndose hacia el otro outcome ≥ 2%
    """
    # Get public betting % latest
    async with session_scope() as session:
        pub_row = (
            await session.execute(
                text(
                    """
                    SELECT outcome, pct_bets, pct_money
                    FROM public_betting_snapshots
                    WHERE match_id = :mid
                      AND market = :mk
                      AND captured_at > now() - interval '4 hours'
                    ORDER BY captured_at DESC
                    LIMIT 2
                    """
                ),
                {"mid": match_id, "mk": market},
            )
        ).all()

        if len(pub_row) < 2:
            return None

        # Por convención: el outcome con mayor pct_bets = público
        sorted_pub = sorted(pub_row, key=lambda r: float(r.pct_bets or 0), reverse=True)
        public_outcome = sorted_pub[0].outcome
        public_pct = float(sorted_pub[0].pct_bets or 0)

        if public_pct < public_threshold:
            return None  # público no está claramente pesado en un lado

        # El outcome opuesto es el potencial "sharp side"
        contrarian_outcome = sorted_pub[1].outcome

        # Line movement: compara odds openings vs closing o last
        line_row = (
            await session.execute(
                text(
                    """
                    SELECT outcome,
                        (SELECT odds FROM odds_history
                         WHERE match_id = :mid AND market = :mk
                           AND outcome = :oc
                           AND bookmaker IN ('pinnacle', 'bet365', 'consensus')
                         ORDER BY ts ASC LIMIT 1) AS opening,
                        (SELECT odds FROM odds_history
                         WHERE match_id = :mid AND market = :mk
                           AND outcome = :oc
                           AND bookmaker IN ('pinnacle', 'bet365', 'consensus')
                         ORDER BY ts DESC LIMIT 1) AS current
                    """
                ),
                {"mid": match_id, "mk": market, "oc": contrarian_outcome},
            )
        ).first()

        if line_row is None or line_row.opening is None or line_row.current is None:
            return None

        opening = float(line_row.opening)
        current = float(line_row.current)
        # Si odds BAJAN en el contrarian → su prob IMPLÍCITA sube → línea mueve hacia allá
        line_move_pct = (opening - current) / opening

        # Strong: línea mueve ≥ 2% hacia contrarian + público pesado al otro lado
        if line_move_pct >= line_move_threshold:
            strength = "strong" if line_move_pct >= 0.04 and public_pct >= 0.75 else "weak"
            logger.info(
                "rlm.detected",
                match_id=match_id,
                market=market,
                strength=strength,
                contrarian=contrarian_outcome,
                public_pct=public_pct,
                line_move_pct=line_move_pct,
            )
            return RLMSignal(
                match_id=match_id,
                market=market,
                contrarian_outcome=contrarian_outcome,
                strength=strength,
                line_move_pct=line_move_pct,
                public_pct_on_other_side=public_pct,
                detected_at=datetime.now(tz=UTC),
            )

    return None


async def scan_rlm_upcoming(
    hours_ahead: int = 48,
    *,
    public_threshold: float = 0.65,
) -> list[RLMSignal]:
    """Escanea todos los matches próximos N horas para detectar RLM."""
    since = datetime.now(tz=UTC)
    until = since + timedelta(hours=hours_ahead)

    async with session_scope() as session:
        match_ids = (
            await session.execute(
                text(
                    """
                    SELECT DISTINCT m.id
                    FROM matches m
                    JOIN public_betting_snapshots pbs ON pbs.match_id = m.id
                    WHERE m.start_time BETWEEN :since AND :until
                      AND m.status = 'scheduled'
                      AND pbs.captured_at > now() - interval '4 hours'
                    """
                ),
                {"since": since, "until": until},
            )
        ).all()

    signals: list[RLMSignal] = []
    for row in match_ids:
        signal = await detect_rlm(row.id, public_threshold=public_threshold)
        if signal is not None:
            signals.append(signal)

    logger.info("rlm.scan_complete", n_matches=len(match_ids), n_signals=len(signals))
    return signals
