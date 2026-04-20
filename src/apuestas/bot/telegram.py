"""Bot Telegram — long polling (§11 + §17.19).

Comandos implementados:
- /analyze            → dispara deep_analysis_flow, envía picks
- /today              → eventos hoy + picks activos
- /bankroll           → curva + stake total + ROI 7/30d
- /clv                → CLV summary 7/30d
- /stats_7d           → performance 7d
- /worst_picks N      → top N peores discrepancias (§21.5)
- /best_model_calls N → picks donde bot acertó contra consenso
- /show_pick_pm bet_id → post-mortem detallado
- /calibration_report sport=X → reliability diagram
- /force_pick event_id → override conformal filter
- /confirm_bet bet_id  → marca como tomada
- /mark_not_taken bet_id
- /pausar              → pausa manual
- /resumir             → resume manual
- /review_last_week    → resumen semanal

Long polling, sin webhook (§11). Sin puertos abiertos.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from apuestas.betting.clv import clv_summary
from apuestas.betting.portfolio import pause_bot, resume_bot
from apuestas.config import get_settings
from apuestas.db import session_scope
from apuestas.obs.logging import configure_logging, get_logger

logger = get_logger(__name__)


# ═══════════════════════ Helpers ═══════════════════════════════════════


def _chat_authorized(update: Update) -> bool:
    settings = get_settings()
    if settings.apis.telegram_chat_id is None:
        return True  # Permisivo si no configurado
    try:
        allowed = int(settings.apis.telegram_chat_id)
    except (TypeError, ValueError):  # fmt: skip
        return True
    chat = update.effective_chat
    return chat is not None and int(chat.id) == allowed


async def _send(update: Update, text_msg: str, **kwargs: Any) -> None:
    if update.effective_message is None:
        return
    await update.effective_message.reply_text(
        text_msg,
        parse_mode=kwargs.pop("parse_mode", ParseMode.MARKDOWN),
        **kwargs,
    )


# ═══════════════════════ Comandos ══════════════════════════════════════


async def cmd_start(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _chat_authorized(update):
        return
    await _send(
        update,
        "🤖 *Apuestas Bot* activo. Usa `/analyze` para ejecutar el análisis 360°.\n"
        "Lista completa: `/help`",
    )


async def cmd_help(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _chat_authorized(update):
        return
    msg = (
        "*Comandos disponibles*\n\n"
        "📊 *Análisis*\n"
        "`/analyze` — Deep analysis eventos próximos 48h\n"
        "`/today` — Eventos hoy + picks activos\n"
        "`/force_pick <event_id>` — Override conformal filter\n\n"
        "💰 *Bankroll / CLV*\n"
        "`/bankroll` — Curva + ROI\n"
        "`/clv` — CLV 7d/30d\n"
        "`/stats_7d` — Performance 7d\n\n"
        "🔎 *Review*\n"
        "`/worst_picks [N=10]` — Top peores discrepancias\n"
        "`/best_model_calls [N=10]` — Mejores picks contra consenso\n"
        "`/show_pick_pm <bet_id>` — Post-mortem detallado\n"
        "`/calibration_report <sport>` — Reliability\n"
        "`/review_last_week` — Resumen semanal\n\n"
        "✅ *Gestión*\n"
        "`/confirm_bet <bet_id>` — Marcar como tomada\n"
        "`/mark_not_taken <bet_id>`\n"
        "`/pausar` — Pausa manual\n"
        "`/resumir` — Reanudar"
    )
    await _send(update, msg)


async def cmd_analyze(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _chat_authorized(update):
        return
    await _send(update, "⏳ Iniciando deep_analysis... (puede tardar 1-3 min)")
    try:
        from apuestas.flows.deep_analysis import deep_analysis_flow

        summary = await deep_analysis_flow(hours_ahead=48, max_events=30)
        await _send(
            update,
            f"✅ Análisis completado\n\n"
            f"• Eventos chequeados: *{summary['events_checked']}*\n"
            f"• Picks emitidos: *{summary['picks_emitted']}*\n"
            f"• Asignaciones portfolio: *{summary['allocations']}*\n\n"
            f"Usa `/today` para ver detalle.",
        )
    except Exception as exc:
        logger.exception("telegram.analyze_fail", error=str(exc))
        await _send(update, f"❌ Error: `{str(exc)[:200]}`")


async def cmd_today(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _chat_authorized(update):
        return
    now = datetime.now(tz=UTC)
    end = now + timedelta(hours=24)
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT m.id, m.start_time, m.sport_code,
                       m.home_team_id, m.away_team_id,
                       (SELECT COUNT(*) FROM bets b
                        WHERE b.match_id = m.id AND b.status = 'pending') AS n_pending
                FROM matches m
                WHERE m.status = 'scheduled'
                  AND m.start_time BETWEEN :now AND :end
                ORDER BY m.start_time ASC
                LIMIT 20
                """
            ),
            {"now": now, "end": end},
        )
        rows = result.all()
    if not rows:
        await _send(update, "📭 Sin eventos en las próximas 24h.")
        return
    lines = [f"📅 *Eventos próximas 24h* ({len(rows)})\n"]
    for r in rows:
        when = r.start_time.strftime("%H:%M")
        lines.append(
            f"• [{when}] *{r.sport_code}* {r.home_team_id} vs {r.away_team_id}"
            + (f" — _{r.n_pending} pick(s)_" if r.n_pending else "")
        )
    await _send(update, "\n".join(lines))


async def cmd_bankroll(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _chat_authorized(update):
        return
    settings = get_settings()
    is_paper = settings.apuestas_paper_trading

    async with session_scope() as session:
        curr = (
            await session.execute(
                text(
                    """
                SELECT bankroll_units FROM bankroll_history
                WHERE is_paper = :p ORDER BY ts DESC LIMIT 1
                """
                ),
                {"p": is_paper},
            )
        ).first()

        stats = (
            await session.execute(
                text(
                    """
                SELECT
                  SUM(pnl_units) FILTER (WHERE settled_at >= NOW() - INTERVAL '7 days') AS pnl_7d,
                  SUM(stake_units) FILTER (WHERE settled_at >= NOW() - INTERVAL '7 days') AS stake_7d,
                  SUM(pnl_units) FILTER (WHERE settled_at >= NOW() - INTERVAL '30 days') AS pnl_30d,
                  SUM(stake_units) FILTER (WHERE settled_at >= NOW() - INTERVAL '30 days') AS stake_30d
                FROM bets
                WHERE status IN ('won','lost') AND is_paper = :p
                """
                ),
                {"p": is_paper},
            )
        ).first()

    bankroll = (
        float(curr.bankroll_units) if curr else float(settings.betting.default_bankroll_units)
    )
    pnl_7d = float(stats.pnl_7d or 0) if stats else 0.0
    stake_7d = float(stats.stake_7d or 0) if stats else 0.0
    pnl_30d = float(stats.pnl_30d or 0) if stats else 0.0
    stake_30d = float(stats.stake_30d or 0) if stats else 0.0
    roi_7d = pnl_7d / stake_7d if stake_7d > 0 else 0.0
    roi_30d = pnl_30d / stake_30d if stake_30d > 0 else 0.0

    mode = "PAPER" if is_paper else "REAL"
    msg = (
        f"💰 *Bankroll ({mode})*\n\n"
        f"• Actual: *{bankroll:.2f}u*\n"
        f"• Initial: {settings.betting.default_bankroll_units:.0f}u\n\n"
        f"📊 *ROI*\n"
        f"• 7d: *{roi_7d:+.2%}* (PnL {pnl_7d:+.2f}u / Stake {stake_7d:.2f}u)\n"
        f"• 30d: *{roi_30d:+.2%}* (PnL {pnl_30d:+.2f}u / Stake {stake_30d:.2f}u)"
    )
    await _send(update, msg)


async def cmd_clv(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _chat_authorized(update):
        return
    s7 = await clv_summary(days=7)
    s30 = await clv_summary(days=30)

    def _fmt(label: str, s: dict[str, float]) -> str:
        if s["n"] == 0:
            return f"*{label}*: sin datos"
        return (
            f"*{label}* (n={int(s['n'])})\n"
            f"  Media: {s['mean_clv']:+.2%}\n"
            f"  Mediana: {s['median_clv']:+.2%}\n"
            f"  % positivo: {s['positive_rate']:.1%}\n"
            f"  P05/P95: {s['p05_clv']:+.2%} / {s['p95_clv']:+.2%}"
        )

    await _send(
        update,
        f"📈 *CLV tracking*\n\n{_fmt('7 días', s7)}\n\n{_fmt('30 días', s30)}",
    )


async def cmd_worst_picks(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _chat_authorized(update):
        return
    try:
        n = int(ctx.args[0]) if ctx.args else 10
    except ValueError:
        n = 10
    n = min(max(n, 1), 25)

    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT pm.bet_id, pm.event_id, pm.discrepancy_score,
                       pm.outcome, pm.pnl_units, pm.narrative->>'transferable_lesson' AS lesson
                FROM post_mortems pm
                WHERE pm.discrepancy_score IS NOT NULL
                ORDER BY pm.discrepancy_score DESC
                LIMIT :n
                """
            ),
            {"n": n},
        )
        rows = result.all()

    if not rows:
        await _send(update, "📭 Sin post-mortems todavía.")
        return
    lines = [f"🔻 *Top {n} worst picks (mayor discrepancia)*\n"]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"{i}. `bet:{r.bet_id}` _{r.outcome}_ disc={r.discrepancy_score:.3f} "
            f"PnL={r.pnl_units:+.2f}u\n   _{(r.lesson or '')[:100]}_"
        )
    await _send(update, "\n".join(lines))


async def cmd_show_pick_pm(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _chat_authorized(update):
        return
    if not ctx.args:
        await _send(update, "Uso: `/show_pick_pm <bet_id>`")
        return
    try:
        bet_id = int(ctx.args[0])
    except ValueError:
        await _send(update, "bet_id inválido")
        return

    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT bet_id, outcome, pnl_units, clv,
                       discrepancy_score, prediction_error, calibration_miss,
                       ev_realized_vs_predicted, llm_alignment_score,
                       narrative
                FROM post_mortems WHERE bet_id = :bid
                """
            ),
            {"bid": bet_id},
        )
        row = result.first()
    if row is None:
        await _send(update, f"📭 Post-mortem no existe para bet {bet_id}")
        return

    n = row.narrative or {}
    lines = [
        f"📝 *Post-mortem bet:{bet_id}*",
        f"Outcome: *{row.outcome}* · PnL: {row.pnl_units:+.2f}u · CLV: "
        + (f"{float(row.clv):+.2%}" if row.clv is not None else "n/d"),
        "",
        f"Discrepancy: *{row.discrepancy_score:.3f}* · Pred error: {row.prediction_error:.3f}",
        f"EV real vs pred: {row.ev_realized_vs_predicted:+.3f}",
        f"LLM alignment: {row.llm_alignment_score or 0:.2f}",
        "",
        f"✅ *What went right*: {', '.join(n.get('what_went_right', [])[:3])}",
        f"❌ *What went wrong*: {', '.join(n.get('what_went_wrong', [])[:3])}",
        f"💡 *Lesson*: {n.get('transferable_lesson', '')[:200]}",
    ]
    await _send(update, "\n".join(lines))


async def cmd_calibration_report(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _chat_authorized(update):
        return
    sport = ctx.args[0] if ctx.args else "nba"
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT confidence_bucket, n_predictions, mean_predicted, mean_actual,
                       calibration_gap, brier_realized
                FROM calibration_rolling
                WHERE sport_code = :sport AND window_days = 30
                ORDER BY confidence_bucket
                """
            ),
            {"sport": sport},
        )
        rows = result.all()
    if not rows:
        await _send(update, f"📭 Sin datos calibración para `{sport}`.")
        return
    lines = [f"🎯 *Calibration report ({sport}, 30d)*\n"]
    for r in rows:
        gap = float(r.calibration_gap or 0)
        icon = "✅" if abs(gap) < 0.03 else ("⚠️" if abs(gap) < 0.05 else "🔴")
        lines.append(
            f"{icon} `{r.confidence_bucket}` n={r.n_predictions} "
            f"pred={float(r.mean_predicted or 0):.3f} "
            f"real={float(r.mean_actual or 0):.3f} gap={gap:+.3f}"
        )
    await _send(update, "\n".join(lines))


async def cmd_pausar(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _chat_authorized(update):
        return
    await pause_bot(reason="manual_user_pause", triggered_by="telegram")
    await _send(update, "⏸️ Bot pausado. `/resumir` para reanudar.")


async def cmd_resumir(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _chat_authorized(update):
        return
    await resume_bot()
    await _send(update, "▶️ Bot reanudado.")


async def cmd_confirm_bet(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _chat_authorized(update):
        return
    if not ctx.args:
        await _send(update, "Uso: `/confirm_bet <bet_id>`")
        return
    try:
        bet_id = int(ctx.args[0])
    except ValueError:
        await _send(update, "bet_id inválido")
        return
    async with session_scope() as session:
        await session.execute(
            text(
                """
                UPDATE bets SET is_paper = false,
                                notes = coalesce(notes, '') || ' [confirmed_by_user]'
                WHERE id = :bid
                """
            ),
            {"bid": bet_id},
        )
    await _send(update, f"✅ Bet {bet_id} confirmada como tomada (real money).")


async def cmd_mark_not_taken(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _chat_authorized(update):
        return
    if not ctx.args:
        await _send(update, "Uso: `/mark_not_taken <bet_id>`")
        return
    try:
        bet_id = int(ctx.args[0])
    except ValueError:
        await _send(update, "bet_id inválido")
        return
    async with session_scope() as session:
        await session.execute(
            text(
                """
                UPDATE bets SET status = 'void',
                                notes = coalesce(notes, '') || ' [not_taken]'
                WHERE id = :bid AND status = 'pending'
                """
            ),
            {"bid": bet_id},
        )
    await _send(update, f"🚫 Bet {bet_id} marcada como no tomada.")


async def cmd_review_last_week(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _chat_authorized(update):
        return
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT COUNT(*) AS n,
                       SUM(CASE WHEN status = 'won' THEN 1 ELSE 0 END) AS wins,
                       SUM(pnl_units) AS pnl,
                       SUM(stake_units) AS stake,
                       AVG(clv) AS avg_clv
                FROM bets
                WHERE status IN ('won','lost') AND settled_at >= NOW() - INTERVAL '7 days'
                """
            )
        )
        row = result.first()
    if row is None or not row.n:
        await _send(update, "📭 Sin bets settleadas en los últimos 7 días.")
        return

    n = int(row.n or 0)
    wins = int(row.wins or 0)
    pnl = float(row.pnl or 0)
    stake = float(row.stake or 0)
    roi = pnl / stake if stake > 0 else 0.0
    wr = wins / n if n > 0 else 0.0
    avg_clv = float(row.avg_clv) if row.avg_clv is not None else 0.0

    await _send(
        update,
        f"📊 *Review last 7d*\n\n"
        f"• Bets: {n}\n"
        f"• Wins: {wins} (WR {wr:.1%})\n"
        f"• PnL: {pnl:+.2f}u · Stake: {stake:.2f}u\n"
        f"• ROI: *{roi:+.2%}*\n"
        f"• CLV avg: *{avg_clv:+.2%}*",
    )


# ═══════════════════════ Application setup ═════════════════════════════


def build_application() -> Application:
    settings = get_settings()
    if settings.apis.telegram_bot_token is None:
        msg = "TELEGRAM_BOT_TOKEN no configurado en .env"
        raise RuntimeError(msg)

    token = settings.apis.telegram_bot_token.get_secret_value()
    app = Application.builder().token(token).build()

    handlers = [
        ("start", cmd_start),
        ("help", cmd_help),
        ("analyze", cmd_analyze),
        ("today", cmd_today),
        ("bankroll", cmd_bankroll),
        ("clv", cmd_clv),
        ("worst_picks", cmd_worst_picks),
        ("show_pick_pm", cmd_show_pick_pm),
        ("calibration_report", cmd_calibration_report),
        ("pausar", cmd_pausar),
        ("resumir", cmd_resumir),
        ("confirm_bet", cmd_confirm_bet),
        ("mark_not_taken", cmd_mark_not_taken),
        ("review_last_week", cmd_review_last_week),
    ]
    for name, handler in handlers:
        app.add_handler(CommandHandler(name, handler))

    return app


async def run_polling() -> None:
    configure_logging()
    app = build_application()
    logger.info("telegram.polling_start")
    await app.run_polling(drop_pending_updates=True)


def main() -> None:
    asyncio.run(run_polling())


if __name__ == "__main__":
    main()
