"""Portfolio allocation — múltiples picks +EV simultáneos con correlación.

§17.2: full Kelly individual sobrestima cuando hay N picks correlacionados
(mismo evento, misma liga, mismo día). Este módulo:

1. Toma lista de ValueBet candidatos.
2. Aplica correlation_aware_kelly (SLSQP) con daily_cap 15%.
3. Respeta psychological stop-loss + bot_state.paused.
4. Produce asignación final: stake reducido + razón del ajuste.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import text

from apuestas.betting.detector import ValueBet
from apuestas.config import get_settings
from apuestas.db import session_scope
from apuestas.obs.logging import get_logger
from apuestas.risk.kelly import KellyBet, correlation_aware_kelly

logger = get_logger(__name__)


@dataclass(slots=True)
class PortfolioPick:
    bet: ValueBet
    individual_kelly_pct: float
    portfolio_kelly_pct: float
    final_stake_units: float
    adjustment_reason: str | None = None


@dataclass(slots=True)
class PortfolioConfig:
    daily_cap_pct: float = 0.15
    cap_per_bet_pct: float = 0.05
    kelly_fraction: float = 0.25
    stop_loss_pct: float = 0.30  # pausa si bankroll < initial × (1 - stop_loss)
    reduced_fraction_drawdown: float = 0.20  # si DD > 20%, usar ⅛ Kelly


async def is_bot_paused() -> tuple[bool, str | None]:
    """Consulta bot_state.paused (manual o psychological stop-loss)."""
    async with session_scope() as session:
        result = await session.execute(text("SELECT value FROM bot_state WHERE key = 'paused'"))
        row = result.first()
    if row is None:
        return False, None
    value = row.value or {}
    return bool(value.get("paused", False)), value.get("reason")


async def get_current_bankroll() -> float:
    """Último valor del bankroll desde bankroll_history (paper o real)."""
    s = get_settings()
    is_paper = s.apuestas_paper_trading
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT bankroll_units
                FROM bankroll_history
                WHERE is_paper = :paper
                ORDER BY ts DESC
                LIMIT 1
                """
            ),
            {"paper": is_paper},
        )
        row = result.first()
    if row is None:
        return float(s.betting.default_bankroll_units)
    return float(row.bankroll_units)


async def compute_recent_drawdown(*, days: int = 30) -> float:
    """Drawdown desde el peak de últimos N días. 0 si bankroll en/por encima del peak."""
    s = get_settings()
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT MAX(bankroll_units) AS peak,
                       (ARRAY_AGG(bankroll_units ORDER BY ts DESC))[1] AS current
                FROM bankroll_history
                WHERE is_paper = :paper
                  AND ts >= NOW() - (:days || ' days')::interval
                """
            ),
            {"paper": s.apuestas_paper_trading, "days": days},
        )
        row = result.first()
    if row is None or row.peak is None or row.peak == 0:
        return 0.0
    peak = float(row.peak)
    current = float(row.current)
    return max(0.0, (peak - current) / peak)


def _effective_kelly_fraction(drawdown: float, cfg: PortfolioConfig) -> float:
    """Reduce Kelly a ⅛ si DD > reduced_fraction_drawdown."""
    if drawdown > cfg.reduced_fraction_drawdown:
        return cfg.kelly_fraction / 2.0  # ej 0.25 → 0.125
    return cfg.kelly_fraction


async def allocate_portfolio(
    candidates: list[ValueBet],
    *,
    cfg: PortfolioConfig | None = None,
    bankroll_override: float | None = None,
) -> list[PortfolioPick]:
    """Dado N candidatos +EV, devuelve asignación correlation-aware.

    Reglas:
    - Bot paused → vacío.
    - Bankroll bajo stop_loss → vacío + alerta.
    - DD > threshold → Kelly reducido.
    """
    cfg = cfg or PortfolioConfig()
    picks = [c for c in candidates if c.is_bet]
    if not picks:
        return []

    paused, reason = await is_bot_paused()
    if paused:
        logger.warning("portfolio.paused", reason=reason, skipped=len(picks))
        return [
            PortfolioPick(
                bet=p,
                individual_kelly_pct=p.kelly_fraction_pct,
                portfolio_kelly_pct=0.0,
                final_stake_units=0.0,
                adjustment_reason=f"paused:{reason or 'unknown'}",
            )
            for p in picks
        ]

    bankroll = bankroll_override if bankroll_override is not None else await get_current_bankroll()
    initial = get_settings().betting.default_bankroll_units
    if bankroll <= initial * (1 - cfg.stop_loss_pct):
        logger.error(
            "portfolio.stop_loss_hit",
            bankroll=bankroll,
            initial=initial,
            threshold=initial * (1 - cfg.stop_loss_pct),
        )
        return [
            PortfolioPick(
                bet=p,
                individual_kelly_pct=p.kelly_fraction_pct,
                portfolio_kelly_pct=0.0,
                final_stake_units=0.0,
                adjustment_reason="stop_loss_triggered",
            )
            for p in picks
        ]

    drawdown = await compute_recent_drawdown()
    fraction = _effective_kelly_fraction(drawdown, cfg)
    if fraction < cfg.kelly_fraction:
        logger.warning(
            "portfolio.kelly_reduced",
            drawdown=drawdown,
            new_fraction=fraction,
            threshold=cfg.reduced_fraction_drawdown,
        )

    kelly_bets = [
        KellyBet(
            p=p.p_blended,
            odds=p.odds,
            event_id=p.event_id,
            market=p.market,
        )
        for p in picks
    ]

    stakes = correlation_aware_kelly(
        kelly_bets,
        fraction=fraction,
        cap_per_bet=cfg.cap_per_bet_pct,
        daily_cap=cfg.daily_cap_pct,
    )

    result: list[PortfolioPick] = []
    for p, portfolio_pct in zip(picks, stakes, strict=True):
        indiv_pct = p.kelly_fraction_pct
        reason: str | None = None
        if fraction < cfg.kelly_fraction:
            reason = f"drawdown_reduction_{drawdown:.1%}"
        elif portfolio_pct < indiv_pct * 0.95:
            reason = "correlation_penalty"

        result.append(
            PortfolioPick(
                bet=p,
                individual_kelly_pct=indiv_pct,
                portfolio_kelly_pct=portfolio_pct,
                final_stake_units=portfolio_pct * bankroll,
                adjustment_reason=reason,
            )
        )

    logger.info(
        "portfolio.allocated",
        n_picks=len(result),
        total_stake_pct=sum(stakes),
        kelly_fraction=fraction,
        drawdown=drawdown,
        bankroll=bankroll,
    )
    return result


async def pause_bot(
    *,
    reason: str,
    triggered_by: str = "auto",
) -> None:
    """§17.12: psychological stop-loss automático. Se activa/desactiva via bot_state."""
    import json as _json

    payload = {
        "paused": True,
        "reason": reason,
        "paused_at": datetime.now(tz=UTC).isoformat(),
        "triggered_by": triggered_by,
    }
    async with session_scope() as session:
        await session.execute(
            text(
                """
                INSERT INTO bot_state (key, value, updated_at)
                VALUES ('paused', CAST(:value AS jsonb), NOW())
                ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value, updated_at = NOW()
                """
            ),
            {"value": _json.dumps(payload)},
        )
    logger.warning("portfolio.pause_bot", reason=reason, triggered_by=triggered_by)


async def resume_bot() -> None:
    async with session_scope() as session:
        await session.execute(
            text(
                """
                UPDATE bot_state
                SET value = jsonb_build_object(
                    'paused', false,
                    'reason', null,
                    'paused_at', null
                ), updated_at = NOW()
                WHERE key = 'paused'
                """
            )
        )
    logger.info("portfolio.resume_bot")
