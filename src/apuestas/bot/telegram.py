"""Bot Telegram — long polling (§11 + §17.19).

Comandos principales (nombres amigables en español):
- /analyze              → analizar eventos ahora → envía picks
- /today                → eventos hoy + picks activos
- /historial [7d|30d|all] → picks que el bot ya emitió
- /bankroll             → saldo + rendimiento + ventaja vs cierre
- /deposit USD|MXN <monto> → depositar al saldo
- /moneda USD|MXN       → cambiar moneda primaria
- /fx                   → tipo de cambio USD↔MXN actual
- /ventaja_cierre       → antes CLV (edge vs closing line)
- /stats_7d             → rendimiento últimos 7 días
- /peores_picks N       → top N peores resultados
- /analisis_pick <id>   → antes post-mortem de un pick
- /precision_modelo sport=X → antes reliability diagram
- /confirmar <bet_id>   → marca como tomada
- /no_tomada <bet_id>
- /pausar / /resumir    → control manual

Long polling, sin webhook (§11). Sin puertos abiertos.
"""

from __future__ import annotations

import warnings as _warnings

# Silenciar FutureWarnings de sklearn que saturan los logs sin valor accionable
# (force_all_finite renamed → ensure_all_finite en 1.6, deprecation cosmética).
# Lo mismo para Pydantic v1 compat warning (Prefect lo emite en cada flow run).
_warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
_warnings.filterwarnings("ignore", category=UserWarning, module="prefect.flows")
_warnings.filterwarnings("ignore", category=UserWarning, module="pydantic_settings")

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text
from telegram import (
    BotCommand,
    BotCommandScopeAllPrivateChats,
    BotCommandScopeChat,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    MenuButtonCommands,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    Defaults,
    MessageHandler,
    filters,
)

from apuestas.bot.control import pause_bot, resume_bot
from apuestas.config import get_settings
from apuestas.db import session_scope
from apuestas.obs.logging import configure_logging, get_logger

logger = get_logger(__name__)

# Texto devuelto por handlers retirados tras el pivote a "detector puro".
# Sprint 2 reemplaza estos stubs por equivalentes sin dinero o los elimina.
_LEGAL_DISCLAIMER = (
    "⚠️ <b>Disclaimer legal:</b> este bot es una herramienta educativa "
    "y de investigación. Los picks son detecciones de valor esperado "
    "positivo basadas en modelos probabilísticos; <b>no son consejo "
    "financiero</b>. Apuesta sólo lo que puedas perder, respeta los "
    "términos de cada casa y cumple la ley de tu jurisdicción."
)

_RETIRED_MSG = (
    "ℹ️ Este comando fue retirado al pasar el bot a modo <b>detector puro</b>.\n"
    "El bot ya no gestiona saldo/stake/PnL — solo emite alertas de valor.\n\n"
    "Comandos activos: /picks  /today  /historial  /estado  /region."
)


# ═══════════════════════ Helpers ═══════════════════════════════════════


async def send_admin_alert(message: str, *, max_retries: int = 4) -> bool:
    """Envía alerta al chat_id admin vía Bot API directo (sin Application running).

    Usado por sistemas (Odds API credit tracker, drift monitor) para notificar
    estados críticos sin necesitar el bot polling activo.

    Con rate limiting implícito (via httpx single-request) + retry en 429/RetryAfter
    respetando el header `retry_after` de Telegram.
    """
    import asyncio

    from apuestas.config import get_settings

    settings = get_settings()
    tok = settings.apis.telegram_bot_token
    chat_id = settings.apis.telegram_chat_id
    if not tok or not chat_id:
        return False

    import httpx

    url = f"https://api.telegram.org/bot{tok.get_secret_value()}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.post(url, json=payload)
            if r.status_code == 200:
                return True
            # Rate limit: respeta retry_after del response
            if r.status_code == 429:
                try:
                    retry_after = int(r.json().get("parameters", {}).get("retry_after", 5))
                except Exception:
                    retry_after = 5
                wait = min(retry_after + 1, 60)
                await asyncio.sleep(wait)
                continue
            # 5xx → backoff exponencial
            if r.status_code >= 500:
                await asyncio.sleep(min(2**attempt, 30))
                continue
            # 4xx non-429 → permanente, no retry
            return False
        except Exception:
            await asyncio.sleep(min(2**attempt, 30))
    return False


def _chat_authorized(update: Update) -> bool:
    """Fail-closed: si no hay chat_id configurado, rechaza.

    Previamente era permisivo en dev (retornaba True sin config), lo cual
    en prod podía permitir cualquier chat. Ahora fail-closed siempre.
    """
    settings = get_settings()
    if settings.apis.telegram_chat_id is None:
        logger.warning("telegram.unauthorized_no_config")
        return False
    try:
        allowed = int(settings.apis.telegram_chat_id)
    except (TypeError, ValueError):  # fmt: skip
        logger.error("telegram.invalid_chat_id_config")
        return False
    chat = update.effective_chat
    if chat is None:
        logger.warning("telegram.unauthorized_no_chat")
        return False
    actual_id = int(chat.id)
    if actual_id != allowed:
        # Visibilidad: bot vivo pero rechazando todos los mensajes silente.
        # Antes este path no loggeaba → diagnóstico imposible.
        logger.warning(
            "telegram.unauthorized_chat",
            actual_chat_id=actual_id,
            allowed_chat_id=allowed,
            chat_type=chat.type if chat else None,
        )
        return False
    return True


_HTML_ESCAPE = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}
_DIV = "━━━━━━━━━━━━━━━━━━━━"
_DIV_SOFT = "─ ─ ─ ─ ─ ─ ─ ─ ─ ─"


def _escape_md(s: str) -> str:
    """Escape HTML para Telegram (parse_mode=HTML).

    Mantiene el nombre histórico <code>_escape_md</code> para no romper callers; ahora
    devuelve texto seguro para HTML (escapa <code>&</code>, <code><`, `></code>).
    """
    if not s:
        return ""
    return "".join(_HTML_ESCAPE.get(c, c) for c in s)


def _sem(value: float, *, thr_good: float = 0.03, thr_bad: float = 0.0) -> str:
    """Semáforo visual para métricas: 🟢🟡🔴 + ▲▶▼."""
    if value >= thr_good:
        return "🟢"
    if value >= thr_bad:
        return "🟡"
    return "🔴"


def _arrow(value: float) -> str:
    if value > 0:
        return "▲"
    if value < 0:
        return "▼"
    return "▶"


def _bar(pct: float, width: int = 10) -> str:
    """Barra ASCII <code>████████░░ 80%</code>."""
    pct = max(0.0, min(1.0, pct))
    filled = round(pct * width)
    return "█" * filled + "░" * (width - filled) + f" {pct:.0%}"


async def _typing(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Indicador 'typing…' antes de respuestas lentas."""
    chat = update.effective_chat
    if chat is not None:
        try:
            await ctx.bot.send_chat_action(chat.id, ChatAction.TYPING)
        except Exception:  # fmt: skip
            pass


async def _send(
    update: Update,
    text_msg: str,
    *,
    reply_markup: Any = None,
    **kwargs: Any,
) -> None:
    """Envío HTML con keyboard opcional. Por defecto añade quick actions."""
    if update.effective_message is None:
        return
    await update.effective_message.reply_text(
        text_msg,
        parse_mode=kwargs.pop("parse_mode", ParseMode.HTML),
        disable_web_page_preview=True,
        reply_markup=reply_markup if reply_markup is not None else _quick_actions(),
        **kwargs,
    )


# ═══════════════════════ Keyboards (UX pro) ═════════════════════════════


def _main_keyboard() -> ReplyKeyboardMarkup:
    """Teclado persistente minimalista — 4 acciones primarias.

    UX: el historial es la 4ª opción para que el usuario vea los picks que
    el bot ya emitió y analizó, incluso si perdió las notificaciones.
    """
    return ReplyKeyboardMarkup(
        [
            [KeyboardButton("🎯 Picks"), KeyboardButton("📜 Historial")],
            [KeyboardButton("📊 Estado"), KeyboardButton("⚙️ Más")],
        ],
        resize_keyboard=True,
        is_persistent=True,
        input_field_placeholder="💡 Toca un botón o escribe /help",
    )


def _kb_header(label: str) -> list[InlineKeyboardButton]:
    """Fila-header visual no clickable (callback noop)."""
    return [InlineKeyboardButton(f"─── {label} ───", callback_data="noop")]


def _main_inline() -> InlineKeyboardMarkup:
    """Menú principal inline, dividido en secciones con headers visuales."""
    btn = InlineKeyboardButton
    return InlineKeyboardMarkup(
        [
            _kb_header("ANÁLISIS"),
            [
                btn("🎯 Analizar 48h", callback_data="cmd:analyze"),
                btn("📅 Eventos hoy", callback_data="cmd:today"),
            ],
            _kb_header("PERFORMANCE"),
            [
                btn("💰 Bankroll", callback_data="cmd:bankroll"),
                btn("📊 CLV 7/30d", callback_data="cmd:clv"),
            ],
            [
                btn("📈 Stats 7d", callback_data="cmd:review_last_week"),
                btn("🔻 Peores", callback_data="cmd:worst_picks"),
            ],
            _kb_header("CONTROL"),
            [
                btn("⏸ Pausar", callback_data="cmd:pausar"),
                btn("▶️ Resumir", callback_data="cmd:resumir"),
            ],
            [btn("❓ Ayuda completa", callback_data="cmd:help")],
        ]
    )


def _quick_actions() -> InlineKeyboardMarkup:
    """Quick actions pequeñas añadidas a cada respuesta — navegación contextual."""
    btn = InlineKeyboardButton
    return InlineKeyboardMarkup(
        [
            [
                btn("🔄 Actualizar", callback_data="cmd:today"),
                btn("💰 Bankroll", callback_data="cmd:bankroll"),
                btn("🏠 Menú", callback_data="cmd:start"),
            ]
        ]
    )


def _pick_inline(bet_id: int) -> InlineKeyboardMarkup:
    """Botones de cada pick: tomar · descartar · análisis detallado."""
    btn = InlineKeyboardButton
    return InlineKeyboardMarkup(
        [
            [
                btn("✅ Tomé este pick", callback_data=f"bet:confirm:{bet_id}"),
                btn("🚫 No tomada", callback_data=f"bet:skip:{bet_id}"),
            ],
            [
                btn("🔍 Ver análisis", callback_data=f"bet:pm:{bet_id}"),
                btn("🏆 Ventaja cierre", callback_data="cmd:clv"),
            ],
        ]
    )


# ═══════════════════════ Comandos ══════════════════════════════════════


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _chat_authorized(update):
        return
    if update.effective_message is None:
        return
    await _typing(update, ctx)
    welcome = (
        "<b>🤖 Apuestas CU</b>\n"
        "<i>Bot automático de value betting multi-deporte</i>\n"
        f"<code>{_DIV}</code>\n\n"
        "<b>👋 ¿Cómo funciono?</b>\n\n"
        "1️⃣ Cada 6h analizo <b>todos los partidos</b> de NBA, NFL, MLB, NHL, "
        "Soccer EU, Tenis, Boxeo y MMA próximos 48h.\n\n"
        "2️⃣ Comparo odds entre <b>78+ books</b> (Pinnacle, DraftKings, "
        "FanDuel, Caliente, Codere, etc.) con mi modelo ML + de-vig Shin.\n\n"
        "3️⃣ Cuando encuentro <b>EV positivo</b> (expected value > +1%), "
        "te mando el pick <u>aquí al chat</u> con todos los detalles.\n\n"
        "4️⃣ Tocas <b>✅ Tomé</b> o <b>🚫 No tomada</b> — yo trackeo "
        "CLV + resultado automático.\n\n"
        "<b>🎯 Usa estos 4 botones:</b>\n"
        "• <b>🎯 Picks</b> → ver picks activos del día\n"
        "• <b>💰 Mi cuenta</b> → saldo (USD + MXN) y rendimiento\n"
        "• <b>📜 Historial</b> → picks que ya emití\n"
        "• <b>⚙️ Más</b> → análisis, estadísticas, control\n\n"
        f"<code>{_DIV_SOFT}</code>\n"
        "<i>Escribe /help para guía detallada.</i>\n\n" + _LEGAL_DISCLAIMER
    )
    await update.effective_message.reply_html(
        welcome,
        reply_markup=_main_keyboard(),
        disable_web_page_preview=True,
    )


async def cmd_glosario(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Glosario de términos técnicos en lenguaje claro."""
    if not _chat_authorized(update):
        return
    msg = (
        "📚 <b>Glosario — términos del bot</b>\n"
        f"<code>{_DIV}</code>\n\n"
        "🎯 <b>EV (Valor Esperado)</b>\n"
        "Si EV = +3%, por cada $100 apostados ganas $3 en promedio a largo plazo.\n\n"
        "⭐ <b>Estrellas de confianza</b>\n"
        "⭐ Muy baja (EV &lt;1%) · ⭐⭐ Baja · ⭐⭐⭐ Media · "
        "⭐⭐⭐⭐ Alta · ⭐⭐⭐⭐⭐ Muy alta (EV ≥5%)\n\n"
        "💵 <b>Stake Kelly ¼</b>\n"
        "Cuánto apostar por pick. Fórmula conservadora (1/4 del Kelly óptimo). "
        "Cap 5% del bankroll total.\n\n"
        "🏆 <b>Ventaja vs cierre (CLV)</b>\n"
        "Si tu odds (2.10) era mejor que la de cierre Pinnacle (2.00), "
        "tienes CLV+. &gt;52% de picks con CLV+ = skill real.\n\n"
        "⚡ <b>Steam move</b>\n"
        "Cuando los apostadores profesionales mueven la línea de Pinnacle "
        "rápido, los books soft tardan 5-15min en ajustar. Apostar antes = edge.\n\n"
        "🧠 <b>Probabilidad fair</b>\n"
        "Lo que la apuesta &quot;debería&quot; valer sin el margen del book. "
        "Pinnacle quitando su hold con método Shin.\n\n"
        "📊 <b>Shin devig</b>\n"
        "Algoritmo que quita el margen (vig) de las odds para obtener la "
        "probabilidad real que el book cree que tiene el evento.\n\n"
        "🔥 <b>Underdog con valor / Favorito con valor</b>\n"
        "Underdog: probabilidad &lt;40% pero odds paga más de lo justo.\n"
        "Favorito: probabilidad &gt;55% y el book lo infrapaga.\n\n"
        "⚠️ <b>EV negativo</b>\n"
        "Perder a largo plazo. El bot NUNCA emite picks con EV negativo.\n\n"
        "<i>Usa /guia &lt;termino&gt; para explicación profunda.</i>"
    )
    await _send(update, msg)


async def cmd_guia(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Explicación profunda de un término específico. Uso: /guia ev"""
    if not _chat_authorized(update):
        return
    if not ctx.args:
        await _send(
            update,
            "Uso: <code>/guia &lt;termino&gt;</code>\n"
            "Términos disponibles: ev, kelly, clv, steam, devig, "
            "shin, conformal, overround, hold, underdog, favorito",
        )
        return
    term = ctx.args[0].lower().strip()
    guides = {
        "ev": (
            "📊 <b>Valor Esperado (EV)</b>\n\n"
            "Si EV = +3%, por cada $100 apostados ganas $3 en promedio a la "
            "larga (cientos de picks).\n\n"
            "<b>Fórmula</b>: EV = (p_real × odds_book) - 1\n\n"
            "Ejemplo: Lakers p_real 58%, DraftKings paga 1.90.\n"
            "EV = 0.58 × 1.90 - 1 = +10.2% (excelente pick)\n\n"
            "<b>Umbrales pros</b>: +1-2% = marginal, +3-5% = sólido, +5%+ = oro."
        ),
        "kelly": (
            "💵 <b>Kelly Criterion (¼)</b>\n\n"
            "Fórmula óptima para apostar sin arruinarse.\n\n"
            "<b>Full Kelly</b>: stake = (p × odds - 1) / (odds - 1)\n"
            "<b>Kelly ¼</b>: stake = Full Kelly / 4 (conservador)\n"
            "<b>Cap</b>: máximo 5% del bankroll por pick.\n\n"
            "Por qué Kelly ¼: Full Kelly es agresivo y vulnerable a varianza. "
            "¼ es el sweet spot entre crecimiento y drawdown (estándar pros)."
        ),
        "clv": (
            "🏆 <b>Closing Line Value (CLV)</b>\n\n"
            "Mide si tu odds era mejor que la de cierre Pinnacle.\n\n"
            "Ejemplo: apuestas Lakers @ 2.10 ahora.\n"
            "Partido cierra Lakers @ 2.00 (línea baja).\n"
            "CLV = 2.10/2.00 - 1 = +5% (anticipaste el mercado sharp)\n\n"
            "<b>Indicador de skill real</b>: >52% picks con CLV+ en 30+ muestras "
            "= gana a largo plazo (Buchdahl threshold)."
        ),
        "steam": (
            "⚡ <b>Steam Move</b>\n\n"
            "Movimiento fuerte de línea en Pinnacle causado por apostadores "
            "profesionales (sharps).\n\n"
            "Ejemplo: Pinnacle Lakers 1.85 → 1.72 en 10 min.\n"
            "Los books soft (BetMGM, FD) tardan 5-15 min en ajustar.\n\n"
            "<b>Oportunidad</b>: apostar Lakers @ 1.80 en BetMGM antes del "
            "ajuste = +5% edge garantizado."
        ),
        "devig": (
            "📊 <b>De-vig (quitar el margen)</b>\n\n"
            "Los books suman ~5% de 'vig' (margen) a sus odds. De-vigging "
            "quita ese margen para obtener la probabilidad <i>real</i> "
            "que el book cree que tiene el evento.\n\n"
            "<b>Método Shin</b>: asume insider trading + imperfect competition. "
            "Matemática compleja pero más precisa que multiplicative simple.\n\n"
            "Pinnacle de-vigged = fair probability gold standard."
        ),
        "shin": (
            "🧮 <b>Shin de-vig</b>\n\n"
            "Método matemático para quitar margen de odds, asumiendo que hay "
            "insider traders que conocen el resultado.\n\n"
            "Fórmula: resuelve iterativamente z tal que "
            "sum((z²+4(1-z)p_i²)^0.5 - z) / (2(1-z)) = 1\n\n"
            "Más robusto que multiplicative cuando hay favoritos extremos."
        ),
        "conformal": (
            "🎯 <b>Conformal Prediction</b>\n\n"
            "Método para obtener intervalos de confianza alrededor de una "
            "predicción. El bot genera p_lower, p_upper (rango).\n\n"
            "Solo emite pick si <b>p_lower > implied_prob_book + margen</b>.\n\n"
            "Evita apostar cuando modelo tiene alta varianza (no está seguro)."
        ),
        "overround": (
            "📉 <b>Overround (hold)</b>\n\n"
            "Suma de inverses de odds - 1. Es el margen del book.\n\n"
            "Ejemplo: home 1.95, away 1.95 → 1/1.95 + 1/1.95 = 1.026 → hold 2.6%\n\n"
            "Pros solo juegan mercados con hold &lt; 3%. Caliente MX tiene hold "
            "8-10% → imposible EV+ excepto en mispricings grandes."
        ),
        "hold": (
            "📉 <b>Hold</b>\n\n"
            "Sinónimo de overround. Margen que se queda el book.\n\n"
            "Pinnacle: 2-3% (sharp book, competitivo).\n"
            "DraftKings: 4-5%.\n"
            "Caliente MX: 8-10% (no competitivo).\n\n"
            "Apostar con hold alto = pérdida esperada."
        ),
        "underdog": (
            "🌶 <b>Underdog</b>\n\n"
            "Equipo con &lt;50% probabilidad de ganar.\n\n"
            "<b>Underdog con valor</b>: book paga más de lo justo.\n"
            "Ej: Magic p 22% vs Pistons. Pinnacle paga 4.55.\n"
            "Fair sería 4.48. EV +1.47%.\n\n"
            "Los pros apuestan underdogs con valor = pagan más por cada win."
        ),
        "favorito": (
            "⭐ <b>Favorito</b>\n\n"
            "Equipo con &gt;50% probabilidad de ganar.\n\n"
            "<b>Favorito con valor</b>: book infrapaga relativo al modelo.\n"
            "Ej: Lakers p 65%. Pinnacle paga 1.60.\n"
            "Fair sería 1.54. EV +4%.\n\n"
            "Si tienes /auto_on only_favorites, solo recibes este tipo."
        ),
    }
    msg = guides.get(term)
    if msg is None:
        await _send(
            update,
            f"❌ Término <code>{_escape_md(term)}</code> no encontrado.\n\n"
            "Disponibles: ev, kelly, clv, steam, devig, shin, "
            "conformal, overround, hold, underdog, favorito",
        )
        return
    await _send(update, msg)


async def cmd_simular(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Simula ganancias esperadas con stake X por pick. Uso: /simular 5"""
    if not _chat_authorized(update):
        return
    try:
        stake_per_pick = float(ctx.args[0]) if ctx.args else 5.0
    except (ValueError, IndexError):  # fmt: skip
        stake_per_pick = 5.0
    stake_per_pick = max(0.5, min(stake_per_pick, 100.0))  # clamp $0.50-$100

    # Estimaciones basadas en target pro
    picks_per_day = 10  # medio
    avg_ev = 0.025  # 2.5% EV promedio
    days = 30
    monthly_picks = picks_per_day * days
    monthly_stake = monthly_picks * stake_per_pick
    expected_profit = monthly_stake * avg_ev
    # Rango según varianza
    low_profit = expected_profit * 0.3  # mal mes
    high_profit = expected_profit * 2.0  # buen mes

    msg = (
        f"🔮 <b>Simulación mensual</b>\n"
        f"<code>{_DIV}</code>\n\n"
        f"Apostando <b>${stake_per_pick:.2f}</b> por pick:\n\n"
        f"📊 <b>Volumen mensual</b>:\n"
        f"  • Picks esperados: ~{monthly_picks}\n"
        f"  • Total apostado: ~<b>${monthly_stake:.2f}</b>\n"
        f"  • EV promedio: 2.5%\n\n"
        f"💰 <b>Ganancia esperada</b>:\n"
        f"  • Escenario medio: <b>+${expected_profit:.2f}</b>/mes\n"
        f"  • Mal mes (varianza): +${low_profit:.2f}\n"
        f"  • Buen mes: +${high_profit:.2f}\n\n"
        f"⚠️ <i>Son estimados matemáticos. Hay varianza a corto plazo. "
        f"Confía en el proceso a 100+ picks.</i>\n\n"
        f"<i>Prueba con diferentes stakes: /simular 10 · /simular 20</i>"
    )
    await _send(update, msg)


async def cmd_mi_primer_pick(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Walkthrough del primer pick paso a paso."""
    if not _chat_authorized(update):
        return
    msg = (
        "🎓 <b>Tu primer pick — guía paso a paso</b>\n"
        f"<code>{_DIV}</code>\n\n"
        "<b>PASO 1 — Recibes el mensaje</b>\n"
        "Te llega al chat algo como:\n"
        "<blockquote>\n"
        "🏀 Pick #X · EV +2.5%\n"
        "✅ APOSTAR A: Lakers\n"
        "🎯 Probabilidad: 58%\n"
        "🏦 DraftKings @ 1.90\n"
        "💵 Stake: 1.5u (0.75% bankroll)\n"
        "</blockquote>\n\n"
        "<b>PASO 2 — Interpretar el mensaje</b>\n"
        "• <b>EV +2.5%</b>: ganas $2.50 por cada $100 a largo plazo\n"
        "• <b>Lakers 58%</b>: gana 58 de 100 veces\n"
        "• <b>1.5u</b>: unidades (1u = 1% de tu bankroll)\n"
        "• Con bankroll $200, 1.5u = $3.00\n\n"
        "<b>PASO 3 — Apostar</b>\n"
        "  1. Abre la app del book mencionado (ej. DraftKings)\n"
        "  2. Busca el partido: <code>Lakers vs Warriors</code>\n"
        "  3. Selecciona <b>Moneyline</b> → Lakers\n"
        "  4. Apuesta el stake recomendado ($3)\n"
        "  5. Confirma la apuesta\n\n"
        "<b>PASO 4 — Notificar al bot</b>\n"
        "Toca el botón <b>✅ Tomé este pick</b> en el mensaje.\n"
        "Esto hace 3 cosas:\n"
        "  ✓ Marca el pick como confirmado\n"
        "  ✓ Guarda tus odds reales (para CLV)\n"
        "  ✓ Ajusta bankroll cuando termine el partido\n\n"
        "<b>PASO 5 — Esperar resultado</b>\n"
        "El bot auto-detecta cuando termina y calcula:\n"
        "  ✓ Ganancia/pérdida\n"
        "  ✓ CLV (tu odds vs odds de cierre)\n"
        "  ✓ Actualiza tu saldo\n\n"
        "<b>REGLAS DE ORO</b>:\n"
        "• Nunca apuestes más del stake sugerido\n"
        "• No persigas pérdidas\n"
        "• Confía en 100+ picks, no en 1-5\n"
        "• Si tienes dudas: /glosario o /guia &lt;termino&gt;\n\n"
        "<i>¿Listo? Usa /picks para ver picks activos.</i>"
    )
    await _send(update, msg)


async def cmd_como_funciona(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Tutorial de 5 minutos sobre cómo usar el bot."""
    if not _chat_authorized(update):
        return
    msg = (
        "🎓 <b>¿Cómo funciona el bot? — Tutorial 5 min</b>\n"
        f"<code>{_DIV}</code>\n\n"
        "<b>PASO 1 — Qué hago</b>\n"
        "Analizo 60+ ligas en 10+ deportes cada 30 min. Comparo odds entre "
        "115+ casas de apuestas para encontrar VALOR matemático.\n\n"
        "<b>PASO 2 — Qué es un pick</b>\n"
        "Una apuesta donde gané matemáticamente a largo plazo. Ejemplo:\n"
        "  · Lakers ganará 60% (dice el modelo)\n"
        "  · DraftKings paga como si fuera 55%\n"
        "  · Diferencia 5% = tu ganancia esperada\n\n"
        "<b>PASO 3 — Recibir picks</b>\n"
        "Cuando detecto valor, te mando mensaje al chat/grupo con:\n"
        "  ✅ Equipo a apostar\n"
        "  🏦 Casa de apuestas y odds\n"
        "  💵 Cuánto apostar (Kelly ¼)\n"
        "  ⭐ Nivel de confianza (1-5)\n\n"
        "<b>PASO 4 — Apostar</b>\n"
        "Abres la app del book mencionado, apuestas el stake recomendado, "
        "tocas ✅ en el mensaje para que trackee resultado.\n\n"
        "<b>PASO 5 — Seguir resultado</b>\n"
        "Cuando termina el partido, actualizo tu saldo automático y "
        "calculo tu Ventaja vs cierre (CLV) — indicador de skill real.\n\n"
        f"<code>{_DIV_SOFT}</code>\n"
        "<b>⚠️ REGLAS DE ORO</b>\n"
        "1. Nunca apostar más del stake recomendado.\n"
        "2. NO tomarlo como &quot;picks seguros&quot; — a corto plazo hay varianza.\n"
        "3. Confía en EL PROCESO: 60% wins a largo plazo = rentable.\n"
        "4. Si tienes mala racha: el bot pausa solo si CLV+ rate &lt;52% en 30 picks.\n\n"
        "<i>Comandos clave: /picks /bankroll /historial /estado</i>\n"
        "<i>Para términos técnicos: /glosario</i>"
    )
    await _send(update, msg)


async def cmd_menu(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Submenú contextual con todas las acciones — se invoca desde '⚙️ Más'."""
    if not _chat_authorized(update):
        return
    if update.effective_message is None:
        return
    btn = InlineKeyboardButton
    kb = InlineKeyboardMarkup(
        [
            [btn("🎯 Analizar ahora", callback_data="cmd:analyze")],
            [
                btn("📅 Eventos próximas 48h", callback_data="cmd:today"),
                btn("📜 Historial picks", callback_data="cmd:historial"),
            ],
            [btn("📈 Resumen semana", callback_data="cmd:review_last_week")],
            [
                btn("🏆 Ventaja vs cierre", callback_data="cmd:clv"),
                btn("🎯 Precisión modelo", callback_data="cmd:calibration_report"),
            ],
            [
                btn("🔻 Peores picks", callback_data="cmd:worst_picks"),
                btn("💱 Tipo de cambio", callback_data="cmd:fx"),
            ],
            [btn("🌎 Región (VPN)", callback_data="cmd:region")],
            [
                btn("⏸ Pausar bot", callback_data="cmd:pausar"),
                btn("▶️ Reanudar bot", callback_data="cmd:resumir"),
            ],
            [btn("❓ Guía completa", callback_data="cmd:help")],
        ]
    )
    msg = (
        "<b>⚙️ Menú</b>\n"
        "<i>Funciones avanzadas. Para el día a día usa los botones del teclado.</i>\n\n"
        "<blockquote expandable>"
        "<b>🎯 Analizar ahora</b>\n"
        "Busca picks <u>en este momento</u> sin esperar el ciclo automático.\n\n"
        "<b>📜 Historial de picks</b>\n"
        "Todos los picks que el bot ya emitió (pendientes + cerrados).\n\n"
        "<b>📈 Resumen semana</b>\n"
        "Rendimiento, picks ganados/perdidos, ventaja promedio.\n\n"
        "<b>🏆 Ventaja vs cierre</b>\n"
        "Tu odds comparada con la odds de cierre de Pinnacle. "
        "Si >52% de tus picks tuvieron ventaja ⇒ edge real (skill).\n\n"
        "<b>🎯 Precisión del modelo</b>\n"
        "¿Cuando el modelo dice 60%, gana 60% de las veces? "
        "Mide qué tan bien calibrado está.\n\n"
        "<b>🔻 Peores picks</b>\n"
        "Los picks con mayor diferencia entre lo pronosticado y lo real — "
        "aprendemos de errores.\n\n"
        "<b>💱 Tipo de cambio</b>\n"
        "USD↔MXN actual para los cálculos del saldo.\n\n"
        "<b>🌎 Región (VPN)</b>\n"
        "Re-detecta IP y ajusta books activos (MX ↔ US)."
        "</blockquote>"
    )
    await update.effective_message.reply_html(msg, reply_markup=kb)


async def cmd_help(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _chat_authorized(update):
        return
    msg = (
        "<b>📖 Guía de uso</b>\n"
        f"<code>{_DIV}</code>\n\n"
        "<b>🟢 Uso diario (4 botones)</b>\n\n"
        "<b>🎯 Picks</b> → picks activos de hoy\n"
        "<i>Partidos con valor esperado positivo detectado.</i>\n\n"
        "<b>💰 Mi cuenta</b> → saldo + rendimiento\n"
        "<i>Saldo en USD y MXN, total consolidado, rendimiento 7d/30d.</i>\n\n"
        "<b>📜 Historial</b> → picks ya emitidos\n"
        "<i>Todos los picks del bot: pendientes, ganados, perdidos.</i>\n\n"
        "<b>⚙️ Más</b> → submenú de funciones avanzadas\n\n"
        f"<code>{_DIV_SOFT}</code>\n"
        "<b>🟡 Cuando recibes un pick en el chat</b>\n\n"
        "Aparece con esta estructura:\n"
        "<blockquote>"
        "🏀 <b>Equipo A vs Equipo B</b>\n"
        "📊 Mercado · Apuesta · Odds\n"
        "💵 Apuesta recomendada (unidades + % del saldo)\n"
        "📈 Valor esperado · Probabilidad · Ventaja\n"
        "🧠 Análisis del modelo"
        "</blockquote>\n\n"
        "Tocas uno de 3 botones:\n"
        "• <b>✅ Tomé</b> → registra tu apuesta y sigue resultado\n"
        "• <b>🚫 No tomada</b> → descarta, no afecta saldo\n"
        "• <b>🔍 Ver análisis</b> → (tras partido) qué salió bien/mal\n\n"
        f"<code>{_DIV_SOFT}</code>\n"
        "<b>💰 Saldo multi-moneda</b>\n\n"
        "El bot soporta USD + MXN a la vez:\n"
        "• <code>/deposit USD 500</code> → depositar dólares\n"
        "• <code>/deposit MXN 8000</code> → depositar pesos\n"
        "• <code>/moneda USD</code> o <code>/moneda MXN</code> → moneda primaria\n"
        "• <code>/fx</code> → tipo de cambio actual\n"
        "El total se consolida en USD internamente para el cálculo del stake.\n\n"
        f"<code>{_DIV_SOFT}</code>\n"
        "<b>🔵 Funciones avanzadas (⚙️ Más)</b>\n\n"
        "• <b>🎯 Analizar ahora</b>: busca picks sin esperar ciclo automático\n"
        "• <b>🏆 Ventaja vs cierre</b>: qué tan buena era tu odds vs la odds de cierre (&gt;52% = skill)\n"
        "• <b>📈 Resumen semana</b>: rendimiento de los últimos 7 días\n"
        "• <b>🔻 Peores picks</b>: aprende de los errores\n"
        "• <b>🎯 Precisión modelo</b>: qué tan bien acierta el modelo\n"
        "• <b>💱 Tipo de cambio</b>: USD↔MXN actual\n"
        "• <b>🌎 Región</b>: re-detecta VPN MX/US\n"
        "• <b>⏸ Pausar</b>: el bot deja de emitir picks\n\n"
        f"<code>{_DIV_SOFT}</code>\n"
        "<b>🧠 Conceptos clave</b>\n\n"
        "<b>Valor esperado (EV)</b>: si EV = +3%, por cada $100 apostados "
        "ganas en promedio $3 a la larga. <u>&gt;1% = emito pick</u>.\n\n"
        "<b>Tamaño recomendado (Kelly ¼)</b>: cuánto apostar por pick. "
        "Fórmula conservadora (1/4 de la Kelly óptima). Cap 5% del saldo.\n\n"
        '<b>Ventaja vs cierre (antes "CLV")</b>: si tu odds (2.10) era mejor '
        "que la odds de cierre de Pinnacle (2.00), ganaste valor. "
        "&gt;52% de picks positivos ⇒ edge real.\n\n"
        "<b>Mejor odds entre books (line shopping)</b>: tomo la mejor "
        "odds disponible entre 78+ casas de apuestas.\n\n"
        "<b>Referencia Pinnacle</b>: cuando no tengo modelo propio para un "
        "deporte, uso la odds de Pinnacle quitándole el margen como referencia.\n\n"
        "<i>Dudas: pregúntame escribiendo aquí.</i>"
    )
    if update.effective_message is not None:
        await update.effective_message.reply_html(
            msg,
            reply_markup=_main_keyboard(),
            disable_web_page_preview=True,
        )


async def cmd_analyze(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _chat_authorized(update):
        return
    await _send(update, "⏳ Iniciando deep_analysis... (puede tardar 1-3 min)")
    try:
        from apuestas.flows.deep_analysis import deep_analysis_flow

        summary = await deep_analysis_flow.fn(hours_ahead=48, max_events=300)
        await _send(
            update,
            f"✅ Análisis completado\n\n"
            f"• Eventos chequeados: <b>{summary['events_checked']}</b>\n"
            f"• Picks emitidos: <b>{summary['picks_emitted']}</b>\n"
            f"• Asignaciones portfolio: <b>{summary['allocations']}</b>\n\n"
            f"Usa <code>/today</code> para ver detalle.",
        )
    except Exception as exc:
        logger.exception("telegram.analyze_fail", error=str(exc))
        await _send(
            update,
            f"❌ Error: <code>{type(exc).__name__}</code> (ver logs para detalle)",
        )


async def cmd_analizar(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Análisis profundo on-demand de UN partido específico.

    Uso:
        /analizar PSG vs Bayern
        /analizar Boca vs Cruzeiro
        /analizar 116617              # match_id directo

    Diferente a /analyze (deep_analysis batch sobre TODOS los matches), este
    comando lanza el agente `analyze_single_match` que ensambla múltiples
    señales (modelo production + Dixon-Coles + Pinnacle de-vig + Polymarket +
    LLM cualitativo) y devuelve un reporte explicable con los picks de mayor
    EV. Toma 30-60s por partido.
    """
    if not _chat_authorized(update):
        return
    if not ctx.args:
        await _send(
            update,
            "📋 <b>Uso del comando /analizar</b>\n\n"
            "Ejemplos:\n"
            "• <code>/analizar PSG vs Bayern</code>\n"
            "• <code>/analizar Boca vs Cruzeiro</code>\n"
            "• <code>/analizar 116617</code>\n"
            "• <code>/analizar PSG vs Bayern totals</code>\n\n"
            "Acepta nombre de equipos (con 'vs', 'v', 'x' o ' - ') "
            "o el id numérico del match. Último argumento opcional: market "
            "(<code>h2h</code> default, <code>totals</code>, <code>spreads</code>, "
            "<code>runline</code>, <code>btts</code>).",
        )
        return

    # B5: parsear market opcional como último arg conocido
    args = list(ctx.args)
    valid_markets = {"h2h", "totals", "spreads", "runline", "btts"}
    market = "h2h"
    if len(args) >= 2 and args[-1].lower() in valid_markets:
        market = args.pop().lower()
    query_str = " ".join(args).strip()

    # B8: rate limit por chat_id
    user_key: str | None = None
    try:
        chat = update.effective_chat
        user_key = str(chat.id) if chat else None
    except Exception:
        user_key = None

    await _send(
        update,
        f"🔬 <b>Análisis profundo on-demand</b>\n\n"
        f"Partido: <code>{_escape_md(query_str)}</code>\n"
        f"Mercado: <code>{market}</code>\n"
        f"Ensamblando señales (modelo + DC + Pinnacle + Polymarket + LLM + RAG)...\n"
        f"<i>Tiempo esperado 30-60s.</i>",
    )
    try:
        from apuestas.agents.match_analyzer import analyze_single_match

        report = await analyze_single_match(query_str, market=market, user_key=user_key)
        if report is None:
            # Puede ser rate-limited o no encontrado. Diagnosticamos.
            from apuestas.agents.match_analyzer import _RATE_LIMIT_MAX, _RATE_LIMIT_TS

            if user_key and len(_RATE_LIMIT_TS.get(user_key, [])) >= _RATE_LIMIT_MAX:
                await _send(
                    update,
                    f"⏸️ Rate limit alcanzado: máximo {_RATE_LIMIT_MAX} análisis/hora. "
                    "Espera unos minutos antes de pedir otro.",
                )
                return
            tips = await _build_match_not_found_diagnosis(query_str)
            await _send(
                update,
                f"❌ No encontré el partido <code>{_escape_md(query_str)}</code> "
                f"en la base de datos.\n\n{tips}",
            )
            return
        msg = _format_match_analysis_report(report)
        await _send(update, msg)
    except Exception as exc:
        logger.exception("telegram.analizar_fail", error=str(exc))
        await _send(
            update,
            f"❌ Error analizando: <code>{_escape_md(type(exc).__name__)}</code>\n"
            f"Ver logs para detalle.",
        )


async def _build_match_not_found_diagnosis(query_str: str) -> str:
    """Diagnóstico best-effort cuando `resolve_match` retorna None.

    Intenta detectar:
      1. Falta `vs`/separator en la query.
      2. Existen teams con nombre similar (trigram) pero ningún match scheduled.
      3. Existen matches con esos teams pero ya finished o cancelados.
    """
    tips: list[str] = []
    raw = query_str.strip().lower()
    if not any(sep in raw for sep in (" vs ", " v ", " - ", " x ", " contra ", " versus ")):
        tips.append("<b>Falta separador.</b> Usa <code>Equipo vs Equipo</code> o pasa el match id.")
    try:
        from sqlalchemy import text as _text

        from apuestas.db import session_scope as _ss

        async with _ss() as s:
            r = await s.execute(
                _text(
                    """
                    SELECT name FROM teams
                    WHERE lower(name) % :q OR similarity(lower(name), :q) > 0.4
                    ORDER BY similarity(lower(name), :q) DESC
                    LIMIT 5
                    """
                ),
                {"q": raw},
            )
            similar = [row.name for row in r.all()]
        if similar:
            tips.append(
                "<b>Teams similares encontrados:</b> "
                + ", ".join(f"<code>{_escape_md(t)}</code>" for t in similar)
                + ". Reintenta con el nombre exacto o usa el match id."
            )
    except Exception:
        pass
    if not tips:
        tips.append(
            "Verifica que el partido esté en las próximas 7 días y que ambos teams "
            "estén en la base. También puedes pasar el id numérico: "
            "<code>/analizar 116618</code>."
        )
    return "\n".join(tips)


def _format_match_analysis_report(report: Any) -> str:
    """Construye el mensaje Telegram para el reporte del agente."""
    sport_emoji = {
        "soccer": "⚽",
        "epl": "⚽",
        "laliga": "⚽",
        "liga_mx": "⚽",
        "bundesliga": "⚽",
        "seriea": "⚽",
        "ligue1": "⚽",
        "mlb": "⚾",
        "nba": "🏀",
        "nfl": "🏈",
        "nhl": "🏒",
        "tennis": "🎾",
    }.get((report.sport_code or "").lower(), "🎯")

    start_str = ""
    if report.start_time:
        from datetime import timedelta as _td

        try:
            local = report.start_time + _td(hours=-6)
            start_str = local.strftime("%a %d %b · %H:%M")
        except Exception:
            start_str = ""

    league_str = report.league_name or "(liga sin asignar)"
    home_e = _escape_md(report.home_name)
    away_e = _escape_md(report.away_name)

    # ── Header ──
    parts: list[str] = [
        f"{sport_emoji} <b>Análisis profundo</b>",
        f"<b>{home_e}</b> vs <b>{away_e}</b>",
        f"🏆 {_escape_md(league_str)}",
    ]
    if start_str:
        parts.append(f"🕐 {start_str} (MX)")
    parts.append("")

    # B9: si la query era ambigua (Real, Inter, etc.) y hay candidatos cercanos,
    # listar los alternativos para que el user sepa que puede haberse elegido
    # el partido equivocado.
    candidates = list(getattr(report, "ambiguous_candidates", []) or [])
    if len(candidates) >= 2:
        parts.append(
            "⚠️ <b>Query ambigua</b> — elegí este partido por mejor similitud, "
            "pero hay otros candidatos:"
        )
        for c in candidates[1:4]:  # primero ya es el elegido
            cn_h = _escape_md(str(c.get("home_name") or ""))
            cn_a = _escape_md(str(c.get("away_name") or ""))
            ln = _escape_md(str(c.get("league_name") or ""))
            parts.append(f"  • {cn_h} vs {cn_a} ({ln})")
        parts.append("Si querías otro, repite con nombres más específicos.")
        parts.append("")

    # ── Mercado analizado (B5) ──
    market_lbl = getattr(report, "market", "h2h")
    parts.append(f"🎯 Mercado: <code>{_escape_md(market_lbl)}</code>")
    parts.append("")

    # ── Probabilidades fusionadas + bandas conformales (B4) ──
    if report.fused_probs:
        parts.append("🎲 <b>Probabilidades fusionadas</b>")
        bands = getattr(report, "fused_bands", {}) or {}
        for outcome, p in sorted(report.fused_probs.items(), key=lambda x: x[1], reverse=True):
            label = {
                "home": report.home_name,
                "away": report.away_name,
                "draw": "Empate",
                "over": "Over",
                "under": "Under",
            }.get(outcome, outcome)
            band = bands.get(outcome) if isinstance(bands, dict) else None
            if band:
                parts.append(
                    f"  • {_escape_md(label)}: <b>{p * 100:.1f}%</b> "
                    f"<i>[{band[0] * 100:.1f}-{band[1] * 100:.1f}%]</i>"
                )
            else:
                parts.append(f"  • {_escape_md(label)}: <b>{p * 100:.1f}%</b>")
        parts.append("")

    # ── Señales usadas ──
    # Solo señales que aportaron probs al fusion. Las "placeholder" (LLM en
    # totals/spreads) no contaminan probs pero igual aportan reasoning chain
    # y se reportan en sección dedicada al final.
    active_signals = [s for s in report.signals_used if s.probs and s.weight > 0]
    if active_signals:
        parts.append(f"📊 <b>Señales usadas ({len(active_signals)})</b>")
        for s in active_signals:
            conf_pct = s.confidence * 100
            parts.append(f"  • <code>{_escape_md(s.name)}</code> <i>(conf {conf_pct:.0f}%)</i>")
        parts.append("")

    if report.skipped_signals:
        parts.append(
            f"⚪ <i>Señales no disponibles: "
            f"{', '.join(_escape_md(s.split(':')[0]) for s in report.skipped_signals)}</i>"
        )
        parts.append("")

    # ── Picks recomendados ──
    if report.picks:
        parts.append(f"💡 <b>Picks recomendados ({len(report.picks)})</b>")
        for i, p in enumerate(report.picks, 1):
            conf_emoji = {
                "high": "🟢",
                "medium": "🟡",
                "low": "🟠",
                "stale": "⚠️",
            }.get(p.confidence, "⚪")
            outcome_label = {
                "home": report.home_name,
                "away": report.away_name,
                "draw": "Empate",
                "over": "Over",
                "under": "Under",
            }.get(p.outcome, p.outcome)
            line_str = f" {p.line}" if p.line is not None else ""
            parts.append(
                f"\n{i}. {conf_emoji} <b>{_escape_md(outcome_label)}</b>{line_str} "
                f"@ {p.odds:.2f} ({_escape_md(p.book)})"
            )
            parts.append(
                f"   EV <b>{p.ev * 100:+.2f}%</b> · edge {p.edge * 100:+.2f}% · "
                f"prob fair {p.p_fused * 100:.1f}%"
            )
            # B4 conformal band
            if getattr(p, "p_low", None) is not None and getattr(p, "p_high", None) is not None:
                parts.append(f"   <i>banda fair: {p.p_low * 100:.1f}-{p.p_high * 100:.1f}%</i>")
            # B6 anticipated CLV
            if getattr(p, "anticipated_clv", None) is not None:
                clv_emoji = "↗️" if p.anticipated_clv > 0 else "↘️"
                parts.append(
                    f"   {clv_emoji} CLV anticipado: <b>{p.anticipated_clv * 100:+.2f}%</b>"
                )
            # B6 book power edge
            if getattr(p, "book_edge_bps", None):
                parts.append(f"   📈 book edge histórico: <b>{p.book_edge_bps:+.0f} bps</b>")
            # B6 ¼ Kelly hint
            if getattr(p, "kelly_quarter_pct", None):
                parts.append(f"   💰 ¼ Kelly: <b>{p.kelly_quarter_pct:.2f}%</b> de tu bankroll")
            # Stale warning (EV anormal por soft book stale vs Pinnacle fresh)
            stale = getattr(p, "stale_warning", None)
            if stale:
                parts.append(f"   ⚠️ <i>{_escape_md(stale)}</i>")
            parts.append(f"   <i>{_escape_md(p.reasoning)}</i>")
        parts.append("")
    else:
        parts.append("💤 <b>Sin picks de valor</b>")
        # Diagnóstico explícito por outcome (B2 honest skip reasons)
        skip_reasons = getattr(report, "skip_reasons", {}) or {}
        global_reason = skip_reasons.get("__all__")
        reason_labels = {
            "only_sharp_derivative_no_independent_model": (
                "Solo señales sharp-derivativas (Pinnacle / Catchall / "
                "Polymarket / LLM prior). Sin modelo independiente (Bayesian xG, "
                "Dixon-Coles real, sklearn), cualquier edge es ruido de Pinnacle. "
                "Guarda anti-pattern activa."
            ),
            "no_odds_available": "No hay odds en la ventana usada.",
            "no_fused_probs": "Todas las señales fallaron — no se pudo computar fair.",
        }
        if global_reason and global_reason in reason_labels:
            parts.append(f"<i>Razón: {_escape_md(reason_labels[global_reason])}</i>")
        else:
            parts.append(
                "<i>El mercado está priced eficiente o las señales no son "
                "suficientemente convergentes para emitir un pick con edge real.</i>"
            )
        # Detalle por outcome cuando aplique (no mostrar __all__)
        per_outcome = {k: v for k, v in skip_reasons.items() if k != "__all__"}
        if per_outcome:
            parts.append("")
            parts.append("<i>Por outcome:</i>")
            label_map = {
                "home": report.home_name,
                "away": report.away_name,
                "draw": "Empate",
                "over": "Over",
                "under": "Under",
            }
            for oc, reason in per_outcome.items():
                lbl = label_map.get(oc, oc)
                parts.append(f"  • {_escape_md(str(lbl))}: <code>{_escape_md(reason)}</code>")
        parts.append("")

    # ── Picks ya emitidos por el detector batch (B2 fix UX) ──
    existing_picks = getattr(report, "existing_picks", None) or []
    if existing_picks:
        from datetime import timedelta as _td

        parts.append(f"📌 <b>Picks ya activos para este partido ({len(existing_picks)})</b>")
        for ep in existing_picks:
            outcome_label = {
                "home": report.home_name,
                "away": report.away_name,
                "draw": "Empate",
                "over": "Over",
                "under": "Under",
            }.get(ep.outcome, ep.outcome)
            line_str = f" {ep.line}" if ep.line is not None else ""
            try:
                placed_local = ep.placed_at + _td(hours=-6)
                placed_str = placed_local.strftime("%d %b %H:%M")
            except Exception:
                placed_str = ""
            status_emoji = {
                "won": "✅",
                "lost": "❌",
                "push": "↩️",
            }.get(ep.outcome_result or "", "⏳")
            sharp_str = (
                f" · sharp {ep.p_consensus_sharp * 100:.1f}%"
                if ep.p_consensus_sharp is not None
                else ""
            )
            parts.append(
                f"  {status_emoji} #{ep.pick_id} <b>{_escape_md(ep.market)}</b> "
                f"{_escape_md(str(outcome_label))}{line_str} @ {ep.odds_placed:.2f} "
                f"({_escape_md(ep.book)}) · emit {placed_str}{sharp_str}"
            )
        parts.append(
            "<i>Estos picks fueron emitidos por el bot batch. El análisis de arriba "
            "es una segunda opinión on-demand con señales actuales — pueden discrepar.</i>"
        )
        parts.append("")

    # ── LLM reasoning chain (B8) ──
    llm_reasoning = getattr(report, "llm_reasoning", None)
    if llm_reasoning:
        parts.append("🧠 <b>Razonamiento LLM</b>")
        summary = llm_reasoning.get("summary") or ""
        if summary:
            parts.append(f"  <i>{_escape_md(str(summary)[:300])}</i>")
        factors = llm_reasoning.get("key_factors") or []
        if factors:
            parts.append("  <b>Factores clave:</b>")
            for f in factors[:5] if isinstance(factors, list) else []:
                parts.append(f"    • {_escape_md(str(f)[:120])}")
        risks = llm_reasoning.get("risks") or []
        if risks:
            parts.append("  <b>Riesgos:</b>")
            for r in risks[:3] if isinstance(risks, list) else []:
                parts.append(f"    ⚠ {_escape_md(str(r)[:120])}")
        rag_n = llm_reasoning.get("rag_snippets_chars", 0)
        if rag_n:
            parts.append(f"  <i>📰 RAG: {rag_n} chars de noticias inyectados</i>")
        parts.append("")

    # ── Disclaimer odds desactualizadas ──
    odds_warn = getattr(report, "odds_freshness_warning", None)
    if odds_warn:
        parts.append(f"⚠️ <i>{_escape_md(odds_warn)}</i>")
        parts.append("")

    # ── Footer ──
    parts.append(
        f"<i>⏱ {report.duration_s:.1f}s · "
        f"{len(report.signals_used)}/"
        f"{len(report.signals_used) + len(report.skipped_signals)} señales</i>"
    )
    parts.append("\n<i>ℹ️ Información con fines educativos. No es consejo financiero.</i>")
    return "\n".join(parts)


async def cmd_picks(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Picks activos del bot (pendientes, aún por resolverse).

    Se vincula al botón principal "🎯 Picks": muestra los picks que el bot
    ya emitió y están esperando resultado, con nombres de equipos claros.
    """
    if not _chat_authorized(update):
        return
    settings = get_settings()
    is_paper = settings.apuestas_paper_trading

    now = datetime.now(tz=UTC)
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT b.id, b.placed_at, b.stake_units, b.odds_placed AS odds,
                       b.outcome, b.bookmaker, b.market,
                       COALESCE(b.currency, 'USD') AS currency,
                       m.sport_code, m.start_time,
                       COALESCE(th.name, 'Equipo ' || m.home_team_id::text) AS home_name,
                       COALESCE(ta.name, 'Equipo ' || m.away_team_id::text) AS away_name,
                       EXTRACT(EPOCH FROM (m.start_time - NOW()))::int AS seconds_to_start
                FROM bets b
                LEFT JOIN matches m ON m.id = b.match_id
                LEFT JOIN teams th ON th.id = m.home_team_id
                LEFT JOIN teams ta ON ta.id = m.away_team_id
                WHERE b.is_paper = :p
                  AND b.status = 'pending'
                  AND (m.start_time IS NULL OR m.start_time > NOW())
                ORDER BY m.start_time ASC NULLS LAST, b.placed_at DESC
                LIMIT 15
                """
            ),
            {"p": is_paper},
        )
        rows = result.all()

    if not rows:
        await _send(
            update,
            "📭 <b>No hay picks activos disponibles</b>\n\n"
            "<i>Los picks se muestran aquí hasta que empieza el partido. "
            "Cuando un partido empieza, el pick pasa a <code>/historial</code>.</i>\n\n"
            "Opciones:\n"
            "• <code>/analyze</code> → forzar búsqueda ahora\n"
            "• <code>/historial</code> → picks ya cerrados o expirados\n"
            "• <code>/eventos</code> → próximos partidos",
        )
        return

    sport_emoji = {
        "nba": "🏀", "mlb": "⚾", "nfl": "🏈", "nhl": "🏒",
        "soccer": "⚽", "laliga": "🇪🇸", "epl": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "liga_mx": "🇲🇽",
        "tennis": "🎾", "boxing": "🥊", "mma": "🥋",
    }  # fmt: skip

    def _time_until(seconds: int | None) -> str:
        if seconds is None:
            return "hora por confirmar"
        if seconds < 60:
            return "⚡ empieza ya"
        if seconds < 3600:
            return f"⏰ en {seconds // 60} min"
        if seconds < 86400:
            h = seconds // 3600
            m = (seconds % 3600) // 60
            return f"⏰ en {h}h {m}min"
        d = seconds // 86400
        h = (seconds % 86400) // 3600
        return f"📅 en {d}d {h}h"

    lines = [
        f"🎯 <b>Picks activos</b>  <i>· {len(rows)} disponibles para apostar</i>",
        f"<code>{_DIV}</code>",
        "<i>Estos picks aún están a tiempo (partido no empezado).</i>\n",
    ]
    for r in rows:
        emoji = sport_emoji.get(str(r.sport_code or "").lower(), "🎯")
        if r.start_time is not None:
            local = r.start_time + timedelta(hours=-6)
            when = local.strftime("%a %d %b · %H:%M")
        else:
            when = "—"
        time_tag = _time_until(int(r.seconds_to_start) if r.seconds_to_start is not None else None)
        home = _escape_md(str(r.home_name))
        away = _escape_md(str(r.away_name))
        outcome = _escape_md(str(r.outcome or ""))
        bookmaker = _escape_md(str(r.bookmaker or ""))
        currency = str(r.currency or "USD")
        lines.append(
            f"{emoji} <b>Pick #{r.id}</b>  <code>{when}</code>  <i>{time_tag}</i>\n"
            f"  <b>{home}</b> vs <b>{away}</b>\n"
            f"  📊 {outcome}  @ {bookmaker} <code>{float(r.odds or 0):.2f}</code>\n"
            f"  💵 {float(r.stake_units or 0):.2f}u {currency}\n"
        )
    lines.append(
        f"<code>{_DIV_SOFT}</code>\n"
        "<i>Usa <code>/confirmar &lt;id&gt;</code> si ya apostaste o "
        "<code>/no_tomada &lt;id&gt;</code> para descartar. "
        "<code>/historial</code> muestra todos los picks.</i>"
    )
    await _send(update, "\n".join(lines))


async def cmd_today(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Eventos próximas 48h agrupados por deporte (máx 4 por deporte).

    Horizonte alineado con el detector (48h adelante). Los picks pueden
    aparecer en cualquier momento dentro de esta ventana: no hay regla
    de "24h antes"; se emiten cuando detecta EV positivo.
    """
    if not _chat_authorized(update):
        return
    now = datetime.now(tz=UTC)
    end = now + timedelta(hours=48)
    async with session_scope() as session:
        # Filtro común: excluye prop-markets fantasma creados por Pinnacle
        # ("Odd vs Even", "(Corners)", "(Games)", "(Cards)", "1st Half", etc.)
        base_where = """
            status = 'scheduled'
            AND start_time BETWEEN :now AND :end
            AND home_team_id NOT IN (
                SELECT id FROM teams WHERE
                    name LIKE '%(Corners)%' OR name LIKE '%(Games)%'
                    OR name LIKE '%(Cards)%' OR name LIKE '%(Bookings)%'
                    OR name LIKE 'Odd' OR name LIKE 'Even'
                    OR name LIKE '1st Half%' OR name LIKE '2nd Half%'
                    OR name LIKE 'Period %' OR name LIKE 'Inning %'
            )
            AND away_team_id NOT IN (
                SELECT id FROM teams WHERE
                    name LIKE '%(Corners)%' OR name LIKE '%(Games)%'
                    OR name LIKE '%(Cards)%' OR name LIKE '%(Bookings)%'
                    OR name LIKE 'Odd' OR name LIKE 'Even'
                    OR name LIKE '1st Half%' OR name LIKE '2nd Half%'
                    OR name LIKE 'Period %' OR name LIKE 'Inning %'
            )
        """

        counts_result = await session.execute(
            text(
                f"""
                SELECT sport_code, COUNT(DISTINCT (LEAST(home_team_id, away_team_id),
                                                   GREATEST(home_team_id, away_team_id),
                                                   DATE_TRUNC('hour', start_time))) AS n
                FROM matches
                WHERE {base_where}
                GROUP BY sport_code
                ORDER BY n DESC
                """
            ),
            {"now": now, "end": end},
        )
        sport_counts = [(r.sport_code, int(r.n)) for r in counts_result.all()]

        result = await session.execute(
            text(
                f"""
                WITH dedup AS (
                    SELECT DISTINCT ON (
                        m.sport_code,
                        LEAST(m.home_team_id, m.away_team_id),
                        GREATEST(m.home_team_id, m.away_team_id),
                        DATE_TRUNC('hour', m.start_time)
                    )
                        m.id, m.start_time, m.sport_code,
                        m.home_team_id, m.away_team_id
                    FROM matches m
                    WHERE {base_where.replace("status", "m.status").replace("start_time", "m.start_time").replace("home_team_id", "m.home_team_id").replace("away_team_id", "m.away_team_id")}
                    ORDER BY m.sport_code, LEAST(m.home_team_id, m.away_team_id),
                             GREATEST(m.home_team_id, m.away_team_id),
                             DATE_TRUNC('hour', m.start_time), m.id
                ),
                ranked AS (
                    SELECT d.id, d.start_time, d.sport_code,
                           COALESCE(th.name, 'Equipo ' || d.home_team_id::text) AS home_name,
                           COALESCE(ta.name, 'Equipo ' || d.away_team_id::text) AS away_name,
                           (SELECT COUNT(*) FROM bets b
                            WHERE b.match_id = d.id AND b.status = 'pending') AS n_pending,
                           ROW_NUMBER() OVER (PARTITION BY d.sport_code ORDER BY d.start_time ASC) AS rn
                    FROM dedup d
                    LEFT JOIN teams th ON th.id = d.home_team_id
                    LEFT JOIN teams ta ON ta.id = d.away_team_id
                )
                SELECT * FROM ranked
                WHERE rn <= 4
                ORDER BY
                    CASE sport_code
                        WHEN 'nba' THEN 1 WHEN 'nfl' THEN 2 WHEN 'mlb' THEN 3
                        WHEN 'nhl' THEN 4 WHEN 'soccer' THEN 5
                        WHEN 'laliga' THEN 6 WHEN 'epl' THEN 7 WHEN 'liga_mx' THEN 8
                        WHEN 'boxing' THEN 9 WHEN 'mma' THEN 10
                        WHEN 'tennis' THEN 11 ELSE 12
                    END,
                    start_time ASC
                """
            ),
            {"now": now, "end": end},
        )
        rows = result.all()
    if not rows:
        await _send(update, "📭 Sin eventos en las próximas 48h.")
        return

    sport_emoji = {
        "nba": "🏀", "mlb": "⚾", "nfl": "🏈", "nhl": "🏒",
        "soccer": "⚽", "laliga": "🇪🇸", "epl": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "liga_mx": "🇲🇽",
        "tennis": "🎾", "boxing": "🥊", "mma": "🥋",
    }  # fmt: skip
    sport_label = {
        "nba": "NBA", "mlb": "MLB", "nfl": "NFL", "nhl": "NHL",
        "soccer": "Fútbol", "laliga": "LaLiga", "epl": "Premier League",
        "liga_mx": "Liga MX", "tennis": "Tenis", "boxing": "Boxeo", "mma": "MMA",
    }  # fmt: skip
    total = sum(n for _, n in sport_counts)
    lines = [
        f"📅 <b>Eventos próximas 48h</b>  <i>· {total} partidos en total</i>",
        f"<code>{_DIV}</code>",
    ]

    # Agrupado por deporte
    current_sport: str | None = None
    for r in rows:
        sport = str(r.sport_code or "").lower()
        if sport != current_sport:
            emoji = sport_emoji.get(sport, "🎯")
            label = sport_label.get(sport, sport.upper())
            total_sport = next((n for s, n in sport_counts if s == sport), 0)
            more = f"  <i>(+{total_sport - 4} más)</i>" if total_sport > 4 else ""
            lines.append(f"\n{emoji} <b>{label}</b>{more}")
            current_sport = sport
        local = r.start_time + timedelta(hours=-6)
        when = local.strftime("%a %H:%M")
        home = _escape_md(str(r.home_name))
        away = _escape_md(str(r.away_name))
        pick_tag = "  🎯" if r.n_pending else ""
        lines.append(f"  <code>{when}</code>  {home} vs {away}{pick_tag}")
    lines.append(
        f"\n<code>{_DIV_SOFT}</code>\n"
        "<i>Muestro máx. 4 partidos por deporte para balance. "
        "Los picks aparecen en el chat cuando detecto valor (EV+).</i>"
    )
    await _send(update, "\n".join(lines))


async def cmd_bankroll(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """(Retirado en pivote detector puro)."""
    if not _chat_authorized(update):
        return
    await _send(update, _RETIRED_MSG)


async def cmd_deposit(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """(Retirado en pivote detector puro)."""
    if not _chat_authorized(update):
        return
    await _send(update, _RETIRED_MSG)


async def cmd_moneda(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """(Retirado en pivote detector puro)."""
    if not _chat_authorized(update):
        return
    await _send(update, _RETIRED_MSG)


async def cmd_fx(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """(Retirado en pivote detector puro)."""
    if not _chat_authorized(update):
        return
    await _send(update, _RETIRED_MSG)


async def cmd_historial(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Historial de alertas emitidas (sin PnL — solo resultado won/lost/pending).

    Uso:
        /historial       → últimos 7 días (default)
        /historial 30d   → últimos 30 días
        /historial all   → todos
    """
    if not _chat_authorized(update):
        return
    filt = (ctx.args[0].lower() if ctx.args else "7d").strip()
    if filt not in ("7d", "30d", "all"):
        filt = "7d"
    interval_sql = {
        "7d": "AND pa.placed_at >= NOW() - INTERVAL '7 days'",
        "30d": "AND pa.placed_at >= NOW() - INTERVAL '30 days'",
        "all": "",
    }[filt]

    async with session_scope() as session:
        result = await session.execute(
            text(
                f"""
                SELECT pa.id, pa.placed_at, pa.outcome_result, pa.status,
                       pa.odds_placed AS odds, pa.outcome, pa.bookmaker,
                       pa.market, pa.upgrade_count, pa.best_odds_seen,
                       m.home_team_id, m.away_team_id, m.sport_code, m.start_time
                FROM pick_alerts pa
                LEFT JOIN matches m ON m.id = pa.match_id
                WHERE 1=1 {interval_sql}
                ORDER BY pa.placed_at DESC
                LIMIT 15
                """
            )
        )
        rows = result.all()

    if not rows:
        label = {"7d": "últimos 7 días", "30d": "últimos 30 días", "all": "histórico"}[filt]
        await _send(
            update,
            f"📭 Sin alertas en {label}.\n\n"
            "<i>Las alertas aparecen cuando el bot detecta valor esperado positivo.</i>",
        )
        return

    status_emoji = {
        None: "⏳ Abierta",
        "pending": "⏳ Abierta",
        "won": "✅ Acertada",
        "lost": "❌ Fallada",
        "void": "🚫 Anulada",
        "halfwon": "½✅ Media",
        "halflost": "½❌ Media",
        "expired": "⌛ Expirada",
    }
    sport_emoji = {
        "nba": "🏀",
        "mlb": "⚾",
        "nfl": "🏈",
        "nhl": "🏒",
        "soccer": "⚽",
        "tennis": "🎾",
        "boxing": "🥊",
        "mma": "🥋",
    }

    label = {"7d": "7 días", "30d": "30 días", "all": "histórico completo"}[filt]
    lines = [
        f"📜 <b>Historial de alertas</b>  <i>· {label}</i>",
        f"<code>{_DIV}</code>",
    ]
    for r in rows:
        emoji = sport_emoji.get(str(r.sport_code or "").lower(), "🎯")
        stat = status_emoji.get(r.outcome_result, "⏳ Abierta")
        when = r.start_time.strftime("%d %b %H:%M") if r.start_time else "—"
        home = _escape_md(str(r.home_team_id or "?"))
        away = _escape_md(str(r.away_team_id or "?"))
        outcome = _escape_md(str(r.outcome or ""))
        bookmaker = _escape_md(str(r.bookmaker or ""))
        odds = float(r.odds or 0)
        best = float(r.best_odds_seen or r.odds or 0)
        up_str = f"  <i>↑{int(r.upgrade_count)}x (mejor {best:.2f})</i>" if r.upgrade_count else ""
        lines.append(
            f"\n{emoji} <b>#{r.id}</b>  <code>{when}</code>  {stat}\n"
            f"  {home} vs {away}\n"
            f"  {outcome} @ {bookmaker} <code>{odds:.2f}</code>{up_str}"
        )
    lines.append(
        f"\n<code>{_DIV_SOFT}</code>\n"
        "<i>Filtros:</i> <code>/historial 7d</code> · "
        "<code>/historial 30d</code> · <code>/historial all</code>"
    )
    await _send(update, "\n".join(lines))


async def cmd_clv(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """(Retirado en pivote detector puro)."""
    if not _chat_authorized(update):
        return
    await _send(update, _RETIRED_MSG)


async def cmd_worst_picks(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """(Retirado — depende de post_mortems/PnL. Sprint 2 reintroducirá versión basada en Brier por pick.)"""
    if not _chat_authorized(update):
        return
    await _send(update, _RETIRED_MSG)


async def cmd_show_pick_pm(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """(Retirado — Sprint 2 sustituirá con /explain usando SHAP top-5.)"""
    if not _chat_authorized(update):
        return
    await _send(update, _RETIRED_MSG)


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
        await _send(update, f"📭 Sin datos de precisión del modelo para <code>{sport}</code>.")
        return
    lines = [
        f"🎯 <b>Precisión del modelo</b>  <i>· {sport.upper()}, 30 días</i>",
        f"<code>{_DIV_SOFT}</code>",
        "<i>Compara qué tan seguido acierta el modelo vs lo que prometió.</i>\n",
    ]
    for r in rows:
        gap = float(r.calibration_gap or 0)
        icon = "✅" if abs(gap) < 0.03 else ("⚠️" if abs(gap) < 0.05 else "🔴")
        lines.append(
            f"{icon} <code>{r.confidence_bucket}</code> n={r.n_predictions}  "
            f"previsto={float(r.mean_predicted or 0):.1%}  "
            f"real={float(r.mean_actual or 0):.1%}  "
            f"gap={gap:+.3f}"
        )
    await _send(update, "\n".join(lines))


async def cmd_explain(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """/explain <pick_id> — muestra SHAP top-5 de un pick emitido (Sprint 8).

    Lee `pick_alerts.shap_top5` JSONB; si aún no está poblado, avisa al
    usuario que Sprint 5b/futuro wire lo calculará en el emit.
    """
    if not _chat_authorized(update):
        return
    if not ctx.args:
        await _send(update, "Uso: <code>/explain &lt;pick_id&gt;</code>")
        return
    try:
        pick_id = int(ctx.args[0])
    except ValueError:
        await _send(update, "❌ pick_id inválido.")
        return

    async with session_scope() as session:
        row = (
            await session.execute(
                text(
                    """
                    SELECT pa.id, pa.market, pa.outcome, pa.odds_placed,
                           pa.bookmaker, pa.shap_top5,
                           p.probability AS p_model, p.ev,
                           p.model_name, p.model_version,
                           ht.name AS home, at.name AS away
                    FROM pick_alerts pa
                    LEFT JOIN predictions p ON p.id = pa.prediction_id
                    JOIN matches m ON m.id = pa.match_id
                    JOIN teams ht ON ht.id = m.home_team_id
                    JOIN teams at ON at.id = m.away_team_id
                    WHERE pa.id = :id
                    """
                ),
                {"id": pick_id},
            )
        ).first()

    if row is None:
        await _send(update, f"📭 Pick #{pick_id} no encontrado.")
        return

    from apuestas.ml.shap_explain import format_shap_top5_markdown

    shap_top5 = row.shap_top5 or []
    # JSONB puede llegar como lista o string según driver; defensivo
    if isinstance(shap_top5, str):
        import json as _json

        try:
            shap_top5 = _json.loads(shap_top5)
        except Exception:
            shap_top5 = []

    header = (
        f"🔍 <b>Explicación Pick #{pick_id}</b>\n"
        f"{_escape_md(row.home)} vs {_escape_md(row.away)}\n"
        f"{row.market}/{row.outcome} @ {row.bookmaker} "
        f"<code>{float(row.odds_placed):.2f}</code>\n"
        f"Modelo: <code>{row.model_name or '?'} {row.model_version or ''}</code>\n"
        f"p_modelo: <b>{float(row.p_model or 0) * 100:.1f}%</b> · "
        f"EV: <b>{float(row.ev or 0) * 100:+.2f}%</b>\n"
        f"<code>{_DIV_SOFT}</code>\n"
    )
    body = format_shap_top5_markdown(shap_top5)
    footer = (
        "\n\n<i>📈 = feature empuja la predicción ARRIBA · 📉 = empuja ABAJO</i>"
        if shap_top5
        else "\n\n<i>SHAP no poblado aún — los próximos picks emitidos lo incluirán.</i>"
    )
    await _send(update, header + body + footer)


async def cmd_calibration(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """KPIs primarios 30d por deporte (Sprint 4d).

    Muestra Brier/BSS/ECE/hit_rate sobre pick_alerts resueltas. Marca
    PASS/FAIL según MVP thresholds (plan §7.2).
    """
    if not _chat_authorized(update):
        return
    import numpy as np

    from apuestas.ml.metrics import compute_metrics

    sport_filter = ctx.args[0].lower() if ctx.args else None

    async with session_scope() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT m.sport_code,
                           CASE WHEN pa.outcome_result = 'won' THEN 1 ELSE 0 END AS y,
                           p.probability AS p_model,
                           pa.odds_placed
                    FROM pick_alerts pa
                    LEFT JOIN predictions p ON p.id = pa.prediction_id
                    JOIN matches m ON m.id = pa.match_id
                    WHERE pa.outcome_result IN ('won', 'lost')
                      AND pa.result_settled_at >= NOW() - INTERVAL '30 days'
                    """
                )
            )
        ).all()

    if not rows:
        await _send(
            update,
            "📭 Sin alertas resueltas en últimos 30 días para computar KPIs.",
        )
        return

    by_sport: dict[str, list[Any]] = {}
    for r in rows:
        if sport_filter and str(r.sport_code).lower() != sport_filter:
            continue
        by_sport.setdefault(str(r.sport_code or "?"), []).append(r)

    if not by_sport:
        await _send(update, f"📭 Sin datos para <code>{sport_filter or '-'}</code>.")
        return

    kpi_brier_cap = {"nba": 0.22, "nfl": 0.23}
    lines = [
        "🎯 <b>KPIs de calidad</b> <i>· últimos 30 días</i>",
        f"<code>{_DIV_SOFT}</code>",
        "<i>Objetivos:</i> Brier ≤ 0.22 · BSS ≥ +0.03 · ECE ≤ 0.05 · HR−impl ≥ +2pp",
        "",
    ]
    for sport, sport_rows in sorted(by_sport.items()):
        y = np.array([int(r.y) for r in sport_rows])
        p = np.array([float(r.p_model) if r.p_model is not None else 0.5 for r in sport_rows])
        odds_arr = np.array([float(r.odds_placed) for r in sport_rows if r.odds_placed is not None])
        avg_odds = float(odds_arr[odds_arr > 1.0].mean()) if (odds_arr > 1.0).any() else None
        m = compute_metrics(y, p, avg_odds=avg_odds)
        brier_cap = kpi_brier_cap.get(sport.lower(), 0.24)
        passes = (
            m.brier <= brier_cap
            and m.brier_skill_score >= 0.03
            and m.ece <= 0.05
            and m.hit_rate_minus_implied >= 0.02
        )
        status = "✅ PASS" if passes else "❌ FAIL"
        sport_up = _escape_md(sport.upper())
        lines.append(
            f"<b>{sport_up}</b> ({m.n} picks) {status}\n"
            f"  Brier <code>{m.brier:.4f}</code> · "
            f"BSS <code>{m.brier_skill_score:+.4f}</code> · "
            f"ECE <code>{m.ece:.4f}</code>\n"
            f"  Hit rate <code>{m.hit_rate:.1%}</code> "
            f"(vs implícita {m.implied_rate:.1%} → "
            f"<code>{m.hit_rate_minus_implied:+.1%}</code>)\n"
        )
    await _send(update, "\n".join(lines))


async def cmd_pausar(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _chat_authorized(update):
        return
    await pause_bot(reason="manual_user_pause", triggered_by="telegram")
    await _send(update, "⏸️ Bot pausado. <code>/resumir</code> para reanudar.")


async def cmd_resumir(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _chat_authorized(update):
        return
    await resume_bot()
    await _send(update, "▶️ Bot reanudado.")


_AUTO_TASK_KEY = "auto_analysis_task"
_AUTO_INTERVAL_MIN = 360  # primer ciclo inmediato, luego cada 6 horas

# El Task se guarda como variable de módulo (NO en ctx.bot_data) porque
# PicklePersistence intenta serializar bot_data cada 5s y Task no es picklable.
# Solo flag `auto_running` + metadata se guarda en bot_data para sobrevivir restart.
_auto_task_ref: dict[str, Any] = {"task": None}


async def _auto_analysis_loop(interval_min: int, chat_id: int, token: str) -> None:
    """Loop continuo COMPLETO con todos los enrichers + idempotente.

    Pipeline por ciclo (todo idempotente, sin duplicados):
      1. Cleanup (pre-ingesta) — cancelar matches huérfanos sin odds
      2. Ingesta paralela:
         a. catchup_flow (odds Pinnacle+Kambi+OddsAPI)
         b. injury_nlp_ingest (NBA + MLB ESPN, ON CONFLICT UPDATE)
         c. capture_weather_forecasts (Open-Meteo, skip si <3h fresh)
      3. Detector: deep_analysis_flow.fn() (gating multivector)
      4. Post-detección paralela:
         a. capture_closing_lines (CLV pre-kickoff, ON CONFLICT)
         b. steam_detector (steam_moves)
      5. Cada 3 ciclos: refresh book_power_ratings + sofascore_injuries
         (Camoufox lento ~8min, pero retorna value real)
      6. Match summary unificado (si flag activado)

    Cancelable vía task.cancel(). Notifica al chat en cada ciclo.
    """
    import asyncio

    from telegram import Bot

    bot = Bot(token=token)
    cycle = 0
    while True:
        cycle += 1
        start = datetime.now(tz=UTC)
        try:
            from apuestas.flows.alert_cleanup import cancel_orphan_matches
            from apuestas.flows.catchup import catchup_flow
            from apuestas.flows.deep_analysis import deep_analysis_flow

            logger.info("auto_loop.cycle_start", cycle=cycle, interval_min=interval_min)
            await bot.send_message(
                chat_id=chat_id,
                text=(
                    f"🔎 <b>Ciclo #{cycle}</b> iniciando...\n"
                    f"<i>1/4 · Cleanup huérfanos + ingesta paralela "
                    f"(odds + injuries + weather)</i>"
                ),
                parse_mode=ParseMode.HTML,
            )

            # Heartbeat: cada 2 min avisa que sigue vivo (no colgado)
            async def _heartbeat(phase: str, start_ts: datetime, cycle_num: int = cycle) -> None:
                while True:
                    await asyncio.sleep(120)
                    mins = int((datetime.now(tz=UTC) - start_ts).total_seconds() / 60)
                    try:
                        await bot.send_message(
                            chat_id=chat_id,
                            text=(
                                f"⏳ <i>Ciclo #{cycle_num} · {phase} · {mins} min corriendo...</i>"
                            ),
                            parse_mode=ParseMode.HTML,
                        )
                    except Exception:  # fmt: skip
                        pass

            # ───── PASO 1: Cleanup huérfanos (idempotente, rápido <1s) ─────
            try:
                n_orphans = await cancel_orphan_matches()
                if n_orphans > 0:
                    logger.info("auto_loop.orphans_cleaned", cycle=cycle, n=n_orphans)
            except Exception as exc_clean:
                logger.warning("auto_loop.cleanup_fail", error=str(exc_clean)[:120])

            # ───── PASO 2: Ingesta paralela (catchup + injuries + weather) ─────
            logger.info("auto_loop.ingest_start", cycle=cycle)
            hb = asyncio.create_task(_heartbeat("ingesta paralela", start))

            async def _safe_catchup() -> dict[str, Any]:
                try:
                    return await catchup_flow.fn() or {}
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning("auto_loop.catchup_fail", error=str(e)[:120])
                    return {}

            # NOTA: deep_analysis_flow.fn() debe invocarse con skip_catchup=True
            # porque PASO 2 ya ejecutó catchup_flow. Sin ese flag, deep_analysis
            # vuelve a llamar catchup_flow internamente (deep_analysis.py:808),
            # duplicando el gasto de OddsAPI cada ciclo.

            async def _safe_injuries() -> int:
                """ESPN NBA+MLB + ESPN/API-Football Soccer injuries.

                Soccer: ESPN endpoint devuelve injuries=[] consistentemente,
                así que API-Football paid tier (existe `API_FOOTBALL_KEY` en
                .env) es la fuente real. Ambas fuentes idempotentes vía
                ON CONFLICT UPDATE.
                """
                try:
                    from apuestas.ingest.injury_nlp_ingest import (
                        fetch_api_football_soccer_injuries,
                        fetch_espn_mlb_injuries,
                        fetch_espn_nba_injuries,
                        fetch_espn_soccer_injuries,
                        persist_injuries,
                        persist_injuries_mlb,
                        persist_injuries_soccer,
                    )

                    nba_inj = await fetch_espn_nba_injuries()
                    n_nba = await persist_injuries(nba_inj)
                    mlb_inj = await fetch_espn_mlb_injuries()
                    n_mlb = await persist_injuries_mlb(mlb_inj)
                    soccer_inj_espn = await fetch_espn_soccer_injuries()
                    n_soccer_espn = await persist_injuries_soccer(soccer_inj_espn)
                    soccer_inj_af = await fetch_api_football_soccer_injuries()
                    n_soccer_af = await persist_injuries_soccer(soccer_inj_af)
                    return n_nba + n_mlb + n_soccer_espn + n_soccer_af
                except Exception as e:
                    logger.warning("auto_loop.injuries_fail", error=str(e)[:120])
                    return 0

            async def _safe_weather() -> dict[str, Any]:
                """Weather forecasts Open-Meteo (idempotente vía NOT EXISTS <3h)."""
                try:
                    from apuestas.flows.capture_weather_forecasts import (
                        capture_weather_forecasts_flow,
                    )

                    return await capture_weather_forecasts_flow.fn() or {}
                except Exception as e:
                    logger.warning("auto_loop.weather_fail", error=str(e)[:120])
                    return {}

            try:
                catchup_res, n_inj, weather_res = await asyncio.gather(
                    _safe_catchup(),
                    _safe_injuries(),
                    _safe_weather(),
                    return_exceptions=False,
                )
            finally:
                hb.cancel()
                try:
                    await hb
                except (asyncio.CancelledError, Exception):  # fmt: skip
                    pass
            logger.info(
                "auto_loop.ingest_done",
                cycle=cycle,
                injuries=n_inj,
                weather=weather_res.get("captured", 0) if isinstance(weather_res, dict) else 0,
            )

            await bot.send_message(
                chat_id=chat_id,
                text=(
                    f"🔎 <b>Ciclo #{cycle}</b> · 2/4 · Detector multivector "
                    f"(ML + sharp + RAG + injuries + weather + steam)..."
                ),
                parse_mode=ParseMode.HTML,
            )

            # ───── PASO 3: Detector multivector (gating gates injuries+weather) ─────
            logger.info("auto_loop.analysis_start", cycle=cycle)
            analysis_start = datetime.now(tz=UTC)
            hb = asyncio.create_task(_heartbeat("análisis multivector", analysis_start))
            try:
                # .fn() ejecuta sin Prefect runtime (evita "Failed to reach API at prefect:4200")
                # hours_ahead=48 (no 168/7 días) — el detector debe enfocarse en
                # partidos apostables HOY+mañana. Más allá la línea aún se mueve mucho.
                # skip_catchup=True: el PASO 2 ya ejecutó catchup_flow; sin este
                # flag se duplica el gasto de OddsAPI cada ciclo.
                summary = await deep_analysis_flow.fn(
                    hours_ahead=48, max_events=300, skip_catchup=True
                )
            finally:
                hb.cancel()
                try:
                    await hb
                except (asyncio.CancelledError, Exception):  # fmt: skip
                    pass
            logger.info(
                "auto_loop.analysis_done",
                cycle=cycle,
                picks_emitted=summary.get("alerts_new", 0),
                events_checked=summary.get("events_checked", 0),
            )

            await bot.send_message(
                chat_id=chat_id,
                text=(
                    f"🔎 <b>Ciclo #{cycle}</b> · 3/4 · Post-análisis "
                    f"(closing lines CLV + steam moves)..."
                ),
                parse_mode=ParseMode.HTML,
            )

            # ───── PASO 4: Post-detección paralelo (CLV + steam) ─────
            async def _safe_clv() -> dict[str, Any]:
                """CLV closing lines snapshot (idempotente ON CONFLICT)."""
                try:
                    from apuestas.flows.capture_closing_lines import (
                        capture_closing_lines_flow,
                    )

                    count = await capture_closing_lines_flow.fn()
                    return {"snapshots_captured": int(count or 0)}
                except Exception as e:
                    logger.warning("auto_loop.clv_fail", error=str(e)[:120])
                    return {}

            async def _safe_steam() -> int:
                """Steam moves (lee odds_history, escribe steam_moves)."""
                try:
                    from apuestas.betting.steam_detector import detect_steam_moves

                    steams = await detect_steam_moves()
                    return len(steams) if steams else 0
                except Exception as e:
                    logger.warning("auto_loop.steam_fail", error=str(e)[:120])
                    return 0

            clv_res, n_steam = await asyncio.gather(
                _safe_clv(),
                _safe_steam(),
                return_exceptions=False,
            )
            logger.info(
                "auto_loop.post_analysis_done",
                cycle=cycle,
                clv_captured=clv_res.get("captured", 0) if isinstance(clv_res, dict) else 0,
                steams=n_steam,
            )

            # ───── PASO 5: Heavy enrichers cada 3 ciclos (sofascore + book_power) ─────
            if cycle % 3 == 1:  # ciclo 1, 4, 7, ...
                logger.info("auto_loop.heavy_enrichers_start", cycle=cycle)

                async def _safe_book_power() -> int:
                    try:
                        from apuestas.scripts.refresh_book_power_ratings import (
                            refresh_book_power_ratings,
                        )

                        await refresh_book_power_ratings()
                        return 1
                    except Exception as e:
                        logger.debug("auto_loop.book_power_fail", error=str(e)[:120])
                        return 0

                async def _safe_sofascore_inj() -> dict[str, Any]:
                    """Sofascore soccer injuries via Camoufox (idempotente UPSERT).
                    Lento (~8min). Solo cada 3 ciclos.
                    """
                    try:
                        from apuestas.ingest.sofascore_injuries import (
                            ingest_soccer_injuries,
                        )

                        return await ingest_soccer_injuries() or {}
                    except Exception as e:
                        logger.warning("auto_loop.sofascore_inj_fail", error=str(e)[:120])
                        return {}

                # En background — no bloquea el resto del ciclo
                asyncio.create_task(_safe_book_power())
                asyncio.create_task(_safe_sofascore_inj())

            # ───── PASO 6: Match summary unificado (opt-in via env) ─────
            import os as _os

            if _os.environ.get("APUESTAS_UNIFIED_MESSAGES", "false").lower() == "true":
                try:
                    from apuestas.bot.match_summary import get_builder, reset_builder

                    flush_stats = await get_builder().flush_all()
                    logger.info("auto_loop.match_summary_done", cycle=cycle, **flush_stats)
                    reset_builder()
                except Exception as exc_ms:
                    logger.warning("auto_loop.match_summary_fail", error=str(exc_ms)[:120])

            # ───── Notificación final del ciclo ─────
            elapsed = (datetime.now(tz=UTC) - start).total_seconds()
            picks = summary.get("alerts_new", 0)
            events = summary.get("events_checked", 0)
            mood = "🎯" if picks > 0 else "💤"
            await bot.send_message(
                chat_id=chat_id,
                text=(
                    f"✅ <b>Ciclo #{cycle} completado</b> {mood}\n"
                    f"  • Eventos analizados: <b>{events}</b>\n"
                    f"  • Picks nuevos: <b>{picks}</b>\n"
                    f"  • Steam moves: <b>{n_steam}</b>\n"
                    f"  • Injuries refresh: <b>{n_inj}</b>\n"
                    f"  • Weather forecasts: <b>"
                    f"{weather_res.get('captured', 0) if isinstance(weather_res, dict) else 0}</b>\n"
                    f"  • Duración: {elapsed:.0f}s\n\n"
                    + (
                        f"<i>Próximo ciclo en {interval_min} min.</i>"
                        if picks > 0
                        else f"<i>Sin EV+ esta vuelta (mercados eficientes). "
                        f"Próximo ciclo en {interval_min} min.</i>"
                    )
                ),
                parse_mode=ParseMode.HTML,
            )
        except asyncio.CancelledError:
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text="⏹️ <b>Análisis automático DETENIDO</b>",
                    parse_mode=ParseMode.HTML,
                )
            except Exception:  # fmt: skip
                pass
            raise
        except Exception as exc:
            logger.exception("auto_loop.fail", cycle=cycle, error=str(exc))
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=(
                        f"⚠️ <b>Error en ciclo #{cycle}</b>\n"
                        f"<code>{type(exc).__name__}: {str(exc)[:150]}</code>\n\n"
                        f"<i>Continúo en {interval_min} min.</i>"
                    ),
                    parse_mode=ParseMode.HTML,
                )
            except Exception:  # fmt: skip
                pass
        # Sleep entre ciclos; interrumpe inmediatamente si es cancelada
        await asyncio.sleep(interval_min * 60)


async def cmd_auto_on(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Inicia análisis automático CONTINUO hasta /auto_off o reinicio del bot."""
    if not _chat_authorized(update):
        return
    import asyncio

    # Parse opcional del intervalo: /auto_on 15 → cada 15 min
    # Por defecto: primer ciclo inmediato + cada 6 horas (360 min)
    interval = _AUTO_INTERVAL_MIN
    if ctx.args:
        try:
            interval = max(5, min(1440, int(ctx.args[0])))  # max 24h
        except ValueError:
            pass

    existing = _auto_task_ref.get("task")
    if existing is not None and not existing.done():
        await _send(
            update,
            "⚠️ Ya hay un análisis automático corriendo.\n"
            "Usa <code>/auto_off</code> para detenerlo primero.",
        )
        return

    settings = get_settings()
    token = settings.apis.telegram_bot_token
    chat_id = settings.apis.telegram_chat_id
    if token is None or chat_id is None:
        await _send(update, "❌ Bot no configurado (falta token o chat_id).")
        return

    task = asyncio.create_task(
        _auto_analysis_loop(interval, int(chat_id), token.get_secret_value())
    )
    # Task en var módulo (no picklable en bot_data), solo metadata en bot_data.
    _auto_task_ref["task"] = task
    ctx.bot_data["auto_running"] = True
    ctx.bot_data["auto_started_at"] = datetime.now(tz=UTC).isoformat()
    ctx.bot_data["auto_interval"] = interval

    interval_label = f"{interval // 60}h" if interval >= 60 else f"{interval} min"
    await _send(
        update,
        f"🔁 <b>Análisis automático INICIADO (modo OPTIMIZADO)</b>\n\n"
        f"  • Primer ciclo: <b>INMEDIATO</b>\n"
        f"  • Luego cada: <b>{interval_label}</b>\n\n"
        f"  <b>Pipeline por ciclo (idempotente, sin duplicados):</b>\n"
        f"    1️⃣ <i>Cleanup</i> · cancela matches huérfanos sin odds\n"
        f"    2️⃣ <i>Ingesta paralela</i>:\n"
        f"        • Catchup odds (Pinnacle+Kambi+OddsAPI 78 books)\n"
        f"        • Injuries ESPN NBA+MLB (UPSERT idempotente)\n"
        f"        • Weather Open-Meteo (skip si fresh &lt;3h)\n"
        f"    3️⃣ <i>Detector multivector</i> · ML+sharp+RAG+venue+inj+weather\n"
        f"    4️⃣ <i>Post-análisis paralelo</i>:\n"
        f"        • Arbitrage scanner (ROI garantizado)\n"
        f"        • Closing lines CLV (idempotente ON CONFLICT)\n"
        f"        • Steam moves (lee odds_history)\n"
        f"    5️⃣ <i>Cada 3 ciclos · enrichers pesados (background)</i>:\n"
        f"        • Sofascore soccer injuries (Camoufox)\n"
        f"        • Book power ratings refresh (90d edge)\n"
        f"    6️⃣ <i>Notificación consolidada</i> con métricas\n\n"
        f"<i>Detener: <code>/auto_off</code></i>\n"
        f"<i>Cambiar intervalo: <code>/auto_on &lt;min&gt;</code> (5-1440)</i>",
    )


async def cmd_auto_off(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Detiene el análisis automático continuo."""
    if not _chat_authorized(update):
        return
    task = _auto_task_ref.get("task")
    if task is None or task.done():
        await _send(update, "ℹ️ El análisis automático no está corriendo.")
        return
    task.cancel()
    _auto_task_ref["task"] = None
    ctx.bot_data["auto_running"] = False
    started = ctx.bot_data.get("auto_started_at", "")
    await _send(
        update,
        "⏹️ <b>Análisis automático DETENIDO</b>\n\n"
        + (f"<i>Estuvo activo desde {started}</i>" if started else ""),
    )


async def cmd_estado(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Estado en tiempo real: ¿qué está haciendo el bot AHORA?

    Muestra: bot activo, última búsqueda, próxima búsqueda programada,
    picks pendientes, picks emitidos hoy, errores recientes, auto-loop.
    """
    if not _chat_authorized(update):
        return
    import subprocess

    settings = get_settings()
    is_paper = settings.apuestas_paper_trading

    # Estado systemd del bot + timer auto-analyze
    import asyncio as _asyncio

    def _svc_active_sync(name: str) -> bool:
        try:
            r = subprocess.run(
                ["systemctl", "--user", "is-active", name],
                capture_output=True,
                text=True,
                timeout=3,
            )
            return r.stdout.strip() == "active"
        except Exception:
            return False

    async def _svc_active(name: str) -> bool:
        return await _asyncio.to_thread(_svc_active_sync, name)

    bot_up = await _svc_active("apuestas-telegram.service")
    timer_up = await _svc_active("apuestas-analyze.timer")

    # Próximo análisis programado
    def _next_run_sync() -> str:
        try:
            r = subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "list-timers",
                    "apuestas-analyze.timer",
                    "--no-pager",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
            for line in r.stdout.splitlines():
                if "apuestas-analyze.timer" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        return " ".join(parts[:4])
        except Exception:
            pass
        return "—"

    next_run = await _asyncio.to_thread(_next_run_sync)

    # Datos DB
    async with session_scope() as session:
        row = (
            await session.execute(
                text(
                    """
                    SELECT
                      (SELECT COUNT(*) FROM bets
                       WHERE is_paper = :p AND status = 'pending') AS pending,
                      (SELECT COUNT(*) FROM bets
                       WHERE is_paper = :p AND placed_at >= NOW() - INTERVAL '24 hours')
                        AS emitidos_24h,
                      (SELECT MAX(placed_at) FROM bets
                       WHERE is_paper = :p) AS last_pick,
                      (SELECT COUNT(*) FROM odds_history
                       WHERE ts >= NOW() - INTERVAL '2 hours') AS odds_2h,
                      (SELECT COUNT(*) FROM matches
                       WHERE status = 'scheduled'
                         AND start_time BETWEEN NOW() AND NOW() + INTERVAL '48 hours')
                        AS matches_48h
                    """
                ),
                {"p": is_paper},
            )
        ).first()

    pending = int(row.pending or 0) if row else 0
    emitidos_24h = int(row.emitidos_24h or 0) if row else 0
    matches_48h = int(row.matches_48h or 0) if row else 0
    odds_2h = int(row.odds_2h or 0) if row else 0
    last_pick_str = "—"
    if row and row.last_pick is not None:
        delta = datetime.now(tz=UTC) - row.last_pick
        mins = int(delta.total_seconds() / 60)
        if mins < 60:
            last_pick_str = f"hace {mins} min"
        elif mins < 1440:
            last_pick_str = f"hace {mins // 60}h {mins % 60}min"
        else:
            last_pick_str = f"hace {mins // 1440}d"

    bot_icon = "🟢" if bot_up else "🔴"
    timer_icon = "🟢" if timer_up else "🟡"
    odds_icon = "🟢" if odds_2h > 100 else "🟡" if odds_2h > 0 else "🔴"

    # Auto-loop interno (el comando /auto_on)
    auto_task = _auto_task_ref.get("task")
    auto_running = auto_task is not None and not auto_task.done()
    auto_interval = ctx.bot_data.get("auto_interval", _AUTO_INTERVAL_MIN)
    auto_icon = "🟢" if auto_running else "🟡"
    auto_line = (
        f"cada {auto_interval} min (activo) · detener con /auto_off"
        if auto_running
        else "detenido · activar con /auto_on"
    )

    msg = (
        f"🤖 <b>Estado del bot</b>\n"
        f"<code>{_DIV}</code>\n\n"
        f"{bot_icon} <b>Bot Telegram:</b> {'activo' if bot_up else 'INACTIVO'}\n"
        f"{auto_icon} <b>Análisis automático continuo:</b>\n"
        f"    {auto_line}\n"
        f"{timer_icon} <b>Timer 6h (systemd):</b> "
        f"{'activo' if timer_up else 'inactivo'}\n"
        f"📅 <b>Próxima búsqueda programada:</b> <code>{next_run}</code>\n\n"
        f"<code>{_DIV_SOFT}</code>\n"
        f"<b>📊 Actividad</b>\n"
        f"  • Partidos próximos 48h: <b>{matches_48h}</b>\n"
        f"  {odds_icon} Odds recibidas últimas 2h: <b>{odds_2h}</b>\n"
        f"  🎯 Picks emitidos últimas 24h: <b>{emitidos_24h}</b>\n"
        f"  ⏳ Picks pendientes: <b>{pending}</b>\n"
        f"  🕐 Último pick: {last_pick_str}\n\n"
        f"<code>{_DIV_SOFT}</code>\n"
        f"<i>Comandos útiles:</i>\n"
        f"  <code>/auto_on</code> — análisis continuo cada 30 min\n"
        f"  <code>/auto_on 15</code> — personaliza intervalo (5-360 min)\n"
        f"  <code>/auto_off</code> — detener análisis continuo\n"
        f"  <code>/analyze</code> — búsqueda única ahora"
    )
    await _send(update, msg)


async def cmd_region(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Re-detecta región actual (IP → MX/US) y ajusta flags books. Útil tras
    conectar/desconectar ProtonVPN."""
    if not _chat_authorized(update):
        return
    from apuestas.region import auto_configure_region

    try:
        result = await auto_configure_region()
        country = result.get("country") or "?"
        region = result.get("region", "OTHER")
        flags = result.get("flags_applied", {}) or {}
        vpn = flags.get("APUESTAS_US_VPN_ACTIVE") == "true"
        emoji = "🇺🇸" if region == "US" else ("🇲🇽" if region == "MX" else "🌍")
        msg_lines = [
            f"{emoji} <b>Región detectada:</b> <code>{country}</code>",
            f"  • VPN US: {'🟢 activa' if vpn else '🔴 inactiva'}",
            f"  • US books (DK/FD/MGM): {'🟢 habilitados' if vpn else '🔴 deshabilitados'}",
            "  • MX books (Caliente/Codere/Winpot): 🟢 siempre activos",
            "  • Offshore (Pinnacle/BetUS): 🟢 siempre activos",
            "",
            "ℹ️ Flags .env actualizados. Reinicia bot si activaste VPN para que surta efecto completo.",
        ]
        await _send(update, "\n".join(msg_lines))
    except Exception as exc:
        await _send(update, f"⚠️ Detección falló: <code>{str(exc)[:80]}</code>")


async def cmd_confirm_bet(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """(Retirado — el bot ya no gestiona apuestas reales)."""
    if not _chat_authorized(update):
        return
    await _send(update, _RETIRED_MSG)


async def cmd_mark_not_taken(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """(Retirado — el bot ya no gestiona apuestas reales)."""
    if not _chat_authorized(update):
        return
    await _send(update, _RETIRED_MSG)


async def cmd_review_last_week(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Precisión del bot últimos 7 días (hit_rate sobre alertas resueltas)."""
    if not _chat_authorized(update):
        return
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT COUNT(*) AS n,
                       SUM(CASE WHEN outcome_result = 'won' THEN 1 ELSE 0 END) AS wins,
                       SUM(CASE WHEN outcome_result = 'lost' THEN 1 ELSE 0 END) AS losses,
                       AVG(odds_placed) AS avg_odds
                FROM pick_alerts
                WHERE outcome_result IN ('won','lost')
                  AND result_settled_at >= NOW() - INTERVAL '7 days'
                """
            )
        )
        row = result.first()
    if row is None or not row.n:
        await _send(update, "📭 Sin alertas resueltas en los últimos 7 días.")
        return

    n = int(row.n or 0)
    wins = int(row.wins or 0)
    losses = int(row.losses or 0)
    wr = wins / (wins + losses) if (wins + losses) > 0 else 0.0
    avg_odds = float(row.avg_odds or 0)
    implied = 1.0 / avg_odds if avg_odds > 1.0 else 0.0

    await _send(
        update,
        f"📊 <b>Precisión últimos 7 días</b>\n"
        f"<code>{_DIV_SOFT}</code>\n\n"
        f"• Alertas resueltas: <b>{n}</b>\n"
        f"• Aciertos: {wins}  ·  Fallos: {losses}\n"
        f"• Hit rate: <b>{wr:.1%}</b>\n"
        f"• Implícita media de las odds: {implied:.1%}\n"
        f"• Skill vs mercado: <b>{(wr - implied):+.1%}</b>\n\n"
        "<i>Sprint 5 añade Brier Score + BSS + ECE aquí.</i>",
    )


# ═══════════════════════ Auto-notify (Gap #11) ═════════════════════════


async def send_pick_to_telegram(
    bet_id: int,
    detail: dict[str, Any],
) -> bool:
    """Envía un pick recién emitido al chat configurado (one-shot, sin polling).

    Retorna True si envió con éxito, False si skip (config ausente / error).
    Idempotente: si <code>bets.notification_sent_at IS NOT NULL</code>, no reenvía.
    """
    settings = get_settings()
    token = settings.apis.telegram_bot_token
    chat_id = settings.apis.telegram_chat_id
    if token is None or chat_id is None:
        return False

    async with session_scope() as session:
        row = (
            await session.execute(
                text("SELECT notification_sent_at FROM pick_alerts WHERE id = :bid"),
                {"bid": bet_id},
            )
        ).first()
    if row is None or row.notification_sent_at is not None:
        return False

    home = _escape_md(str(detail.get("home", "")))
    away = _escape_md(str(detail.get("away", "")))
    market = _escape_md(str(detail.get("market", "")))
    outcome = _escape_md(str(detail.get("outcome", "")))
    bookmaker = _escape_md(str(detail.get("bookmaker", "")))
    odds = float(detail.get("odds", 0) or 0)
    stake = float(detail.get("stake_units", 0) or 0)
    ev_raw = float(detail.get("ev_pct", 0) or 0)
    ev_pct = ev_raw * 100
    kelly_pct = float(detail.get("kelly_frac", 0) or 0) * 100
    reason = _escape_md(str(detail.get("reason", ""))[:280])
    sport = _escape_md(str(detail.get("sport", "")))

    line = detail.get("line")
    line_str = f" @ {line}" if line is not None else ""
    sport_emoji = {
        "nba": "🏀",
        "mlb": "⚾",
        "nfl": "🏈",
        "nhl": "🏒",
        "soccer": "⚽",
        "tennis": "🎾",
        "boxing": "🥊",
        "mma": "🥋",
    }.get(str(detail.get("sport", "")).lower(), "🎯")

    sem = _sem(ev_raw)
    arr = _arrow(ev_raw)

    # Metadata adicional (opcional, todo graceful si faltan)
    start_time = detail.get("start_time")
    start_str = ""
    if start_time is not None:
        try:
            from datetime import datetime as _dt

            if isinstance(start_time, str):
                dt = _dt.fromisoformat(start_time.replace("Z", "+00:00"))
            elif isinstance(start_time, _dt):
                dt = start_time
            else:
                dt = None
            if dt is not None:
                # Zona horaria MX (UTC-6). Naive-format
                from datetime import timedelta as _td

                local = dt + _td(hours=-6)
                start_str = local.strftime("%a %d %b · %H:%M")
        except Exception:
            start_str = ""

    # Probabilidades
    p_model = detail.get("p_model")
    p_pinnacle = detail.get("p_pinnacle_fair")
    implied = detail.get("implied_prob")
    p_line = ""
    if p_model is not None and p_pinnacle is not None:
        p_line = f"🎲 Probabilidad fair: <b>{float(p_pinnacle) * 100:.1f}%</b> (Pinnacle de-vig)\n"
    elif p_pinnacle is not None:
        p_line = f"🎲 Probabilidad fair: <b>{float(p_pinnacle) * 100:.1f}%</b>\n"

    # Best odds vs book implied (cuánto edge gana el apostador)
    edge_pct = ""
    if implied is not None and p_pinnacle is not None:
        edge_raw = float(p_pinnacle) - float(implied)
        if edge_raw > 0:
            edge_pct = f"⚔️ Ventaja vs book: <b>+{edge_raw * 100:.2f}pp</b>\n"

    # Favorito/underdog label
    role = ""
    if p_pinnacle is not None:
        p = float(p_pinnacle)
        if p > 0.55:
            role = "⭐ <i>Favorito</i>"
        elif p < 0.40:
            role = "🌶 <i>Underdog con valor</i>"
        else:
            role = "⚖ <i>Pickem</i>"

    # Alternativas (si se enviaron en detail)
    alt_line = ""
    alternatives = detail.get("alternatives") or []
    if alternatives:
        alts = ", ".join(f"{a['book']} {a['odds']:.2f}" for a in alternatives[:3])
        alt_line = f"🔁 <i>También disponible:</i> {alts}\n"

    # Sprint 4c — bloque regional MX/US/Global (si detail lo trae).
    # Formato: diccionario `regional` con subkeys `mx`, `us`, `global` cada uno
    # con `{book, odds}`. El ingester `emit_alerts` puebla opcionalmente vía
    # apuestas.betting.regional.compare_regions.
    regional_line = ""
    regional = detail.get("regional") or {}
    if isinstance(regional, dict) and regional:
        parts: list[str] = []
        for flag, key in (("🇲🇽 MX", "mx"), ("🇺🇸 US", "us"), ("🌍 Global", "global")):
            entry = regional.get(key)
            if isinstance(entry, dict):
                book = _escape_md(str(entry.get("book", "")))
                odds_val = entry.get("odds")
                try:
                    odds_fmt = f"{float(odds_val):.2f}" if odds_val is not None else "?"
                except TypeError, ValueError:
                    odds_fmt = "?"
                if book and odds_val is not None:
                    parts.append(f"{flag}: {book} @ <code>{odds_fmt}</code>")
        if parts:
            regional_line = "🌐 <i>Mejor por región</i>: " + " · ".join(parts) + "\n"

    # Weather hint (Sprint 4b)
    weather_line = ""
    ws = detail.get("weather_summary")
    if ws:
        hint = detail.get("weather_hint")
        suffix = f"  <i>(hint ×{float(hint):.2f})</i>" if hint is not None else ""
        weather_line = f"🌦 <i>Clima</i>: {_escape_md(str(ws))}{suffix}\n"

    # ───────── UX v4: muestra SIEMPRE los 3 outcomes (home/draw/away) ─────────
    # Evita confusión: el lector ve quién es favorito real del partido ANTES
    # de ver qué outcome apostamos. Importante cuando apostamos al underdog
    # o al empate (puede parecer que apostamos al favorito si no se lee bien).
    p_pick = None
    if p_pinnacle is not None:
        p_pick = float(p_pinnacle) * 100

    # Determinar nombre del equipo apostado vs contrario
    outcome_lower = str(detail.get("outcome", "")).lower()
    if outcome_lower in ("home", "local"):
        team_pick = home
    elif outcome_lower in ("away", "visitor", "visitante"):
        team_pick = away
    elif outcome_lower == "draw":
        team_pick = "Empate"
    else:
        team_pick = outcome

    # Consulta 3-way odds (h2h) desde odds_history para construir el market
    # overview. Si no encontramos 3 outcomes o es un market no-h2h, fallback
    # al formato antiguo (solo team_pick + otro).
    market_overview = ""
    try:
        match_id = detail.get("match_id") or detail.get("event_id")
        if match_id is not None and str(detail.get("market", "")).lower() in (
            "h2h",
            "moneyline",
            "1x2",
        ):
            async with session_scope() as _s:
                _rows = (
                    await _s.execute(
                        text(
                            """
                            SELECT outcome, AVG(odds) AS avg_odds
                            FROM odds_history
                            WHERE match_id = :mid AND market = 'h2h'
                              AND bookmaker = 'pinnacle'
                              AND ts > NOW() - INTERVAL '6 hours'
                            GROUP BY outcome
                            """
                        ),
                        {"mid": int(match_id)},
                    )
                ).all()
            _probs = {}
            _total_vig = 0.0
            for _r in _rows:
                if float(_r.avg_odds) > 1.0:
                    _implied = 1.0 / float(_r.avg_odds)
                    _probs[_r.outcome] = _implied
                    _total_vig += _implied
            if _probs and _total_vig > 0:
                _probs = {k: v / _total_vig for k, v in _probs.items()}  # de-vig prop
                _p_home = _probs.get("home", 0) * 100
                _p_draw = _probs.get("draw", 0) * 100
                _p_away = _probs.get("away", 0) * 100
                _star_home = " ⭐" if _p_home == max(_p_home, _p_draw, _p_away) else ""
                _star_draw = (
                    " ⭐" if _p_draw > 0 and _p_draw == max(_p_home, _p_draw, _p_away) else ""
                )
                _star_away = " ⭐" if _p_away == max(_p_home, _p_draw, _p_away) else ""
                _mark_home = " 🎯" if outcome_lower in ("home", "local") else ""
                _mark_draw = " 🎯" if outcome_lower == "draw" else ""
                _mark_away = " 🎯" if outcome_lower in ("away", "visitor", "visitante") else ""
                _parts = [
                    f"   🏠 <b>{home}</b>: {_p_home:.1f}%{_star_home}{_mark_home}",
                ]
                if _p_draw > 0:
                    _parts.append(f"   🤝 Empate: {_p_draw:.1f}%{_star_draw}{_mark_draw}")
                _parts.append(f"   ✈ <b>{away}</b>: {_p_away:.1f}%{_star_away}{_mark_away}")
                market_overview = (
                    "📊 <b>Probabilidades del partido (mercado):</b>\n"
                    + "\n".join(_parts)
                    + "\n   <i>⭐ = favorito real · 🎯 = nuestro pick</i>\n"
                )
    except Exception:
        market_overview = ""

    # Favorito vs underdog (rol del OUTCOME apostado)
    if p_pick is not None:
        if p_pick > 55:
            pick_role = "⭐ Favorito del modelo"
        elif p_pick < 40:
            pick_role = "🌶 Underdog con valor (pago alto por outcome poco probable)"
        else:
            pick_role = "⚖ Pickem (pareja, cualquiera puede ganar)"
        prob_block = (
            market_overview
            + "\n🎯 <b>Nuestro pick:</b>\n"
            + f"   Apostamos a <b>{team_pick}</b> — el modelo le da <b>{p_pick:.1f}%</b>\n"
            + f"   {pick_role}\n"
        )
    else:
        prob_block = market_overview

    # Confianza multi-componente (Sprint 2). La fórmula centralizada vive en
    # apuestas.bot.confidence.classify_confidence. Pesos suman 1.0 exacto.
    from apuestas.bot.confidence import classify_confidence, fetch_rolling_ece

    p_blended_raw = detail.get("p_blended") or detail.get("p_pinnacle_fair") or 0.5
    try:
        pb_val = float(p_blended_raw)
    except TypeError, ValueError:
        pb_val = 0.5
    try:
        pl_val = float(detail["p_lower"]) if detail.get("p_lower") is not None else None
    except TypeError, ValueError:
        pl_val = None
    try:
        pu_val = float(detail["p_upper"]) if detail.get("p_upper") is not None else None
    except TypeError, ValueError:
        pu_val = None

    # ECE rolling 30d del deporte (si hay datos en calibration_rolling).
    sport_for_ece = str(detail.get("sport") or "").lower() or None
    try:
        async with session_scope() as _s:
            ece_val = await fetch_rolling_ece(_s, sport_for_ece)
    except Exception:
        ece_val = 0.05

    # Soft tags (Sprint 4: odds_spike los poblará).
    soft_tags = frozenset(detail.get("soft_tags") or [])

    conf = classify_confidence(
        ev_raw=ev_raw,
        p_blended=pb_val,
        p_lower=pl_val,
        p_upper=pu_val,
        rolling_ece_30d=ece_val,
        market_consensus_delta=float(detail.get("market_consensus_delta") or 0.0),
        soft_tags=soft_tags,
    )
    conf_stars = conf.stars
    conf_label = conf.label
    conf_score = conf.score
    # Alias local para preservar referencias al nombre `p_blended` más abajo.
    p_blended = pb_val
    p_low = pl_val
    p_up = pu_val

    # Explicación simple del EV
    ev_explain = f"Por cada $100 apostados, ganas en promedio <b>${ev_pct:.2f}</b> a largo plazo."

    # Bloque de incertidumbre conformal (si disponible)
    uncertainty = ""
    if p_low is not None and p_up is not None:
        try:
            uncertainty = (
                f"🎯 <b>Intervalo de probabilidad:</b> "
                f"[{float(p_low) * 100:.1f}% – {float(p_up) * 100:.1f}%]\n"
            )
        except TypeError, ValueError:
            pass

    msg = (
        f"<b>{sport_emoji} Alerta #{bet_id}</b>\n"
        f"<code>{_DIV}</code>\n\n"
        f"<b>{home}</b> <i>vs</i> <b>{away}</b>\n"
        f"<i>{sport.upper()}" + (f" · 🕐 {start_str}" if start_str else "") + "</i>\n"
        f"<code>{_DIV_SOFT}</code>\n\n"
        f"✅ <b>Valor detectado en:</b> <b>{_escape_md(team_pick)}</b> "
        f"({outcome}{line_str})\n"
        f"🏦 <b>Mejor casa:</b> {bookmaker} @ <code>{odds:.2f}</code>\n"
        f"{alt_line}"
        f"{regional_line}"
        f"{weather_line}"
        f"\n"
        f"{prob_block}"
        f"{uncertainty}"
        f"\n"
        f"📊 <b>Valor esperado (EV):</b> <b>+{ev_pct:.2f}%</b>\n"
        f"   <i>{ev_explain}</i>\n\n"
        f"{conf_stars} <b>Confianza:</b> {conf_label}"
        f"  <i>(score {conf_score:.2f})</i>\n\n"
        f"<blockquote expandable>"
        f"<b>🧠 ¿Por qué esta alerta?</b>\n"
        + (
            reason
            if reason
            else (
                f"El book {bookmaker} paga {odds:.2f} por {_escape_md(team_pick)}, "
                f"pero el mercado de referencia (Pinnacle de-vigged) asigna "
                f"probabilidad <b>{p_blended * 100:.1f}%</b> → odds justa ≈ {1 / p_blended:.2f}. "
                f"Como {bookmaker} paga MÁS de lo justo, hay valor matemático "
                f"positivo aunque {_escape_md(team_pick)} sea "
                f"{'favorito' if p_blended > 0.5 else 'underdog'}. "
                f"No garantiza que gane este partido — informa que a largo plazo "
                f"apostar así tiene retorno positivo."
            )
        )
        + "</blockquote>\n\n"
        "<i>ℹ️ Información con fines educativos. No es consejo financiero. "
        "Tu criterio y tu riesgo.</i>"
    )

    try:
        from telegram import Bot

        from apuestas.bot.telegram_ratelimit import send_with_ratelimit

        bot = Bot(token=token.get_secret_value())
        # 1) Mensaje al chat personal (con botones interactivos) — rate limited
        ok_main = await send_with_ratelimit(
            bot,
            chat_id=int(chat_id),
            text=msg,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
            reply_markup=_pick_inline(bet_id),
        )
        if not ok_main:
            logger.warning("telegram.pick_notify_main_fail", bet_id=bet_id)
        # 2) Broadcast al canal (sin botones) — rate limited + retry auto en 429
        channel_id = settings.apis.telegram_channel_id
        if channel_id:
            target: str | int = int(channel_id) if channel_id.lstrip("-").isdigit() else channel_id
            ok_ch = await send_with_ratelimit(
                bot,
                chat_id=target,
                text=msg,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
            if ok_ch:
                logger.info("telegram.pick_broadcast_channel", bet_id=bet_id)
            else:
                logger.warning(
                    "telegram.channel_broadcast_fail_final",
                    bet_id=bet_id,
                    channel=channel_id,
                )
    except Exception as exc:
        logger.warning("telegram.pick_notify_fail", bet_id=bet_id, error=str(exc)[:120])
        return False

    async with session_scope() as session:
        await session.execute(
            text(
                "UPDATE pick_alerts SET notification_sent_at = NOW() "
                "WHERE id = :bid AND notification_sent_at IS NULL"
            ),
            {"bid": bet_id},
        )
    logger.info("telegram.pick_notified", bet_id=bet_id)
    return True


# ═══════════════════════ Application setup ═════════════════════════════


def build_application() -> Application[Any, Any, Any, Any, Any, Any]:
    settings = get_settings()
    if settings.apis.telegram_bot_token is None:
        msg = "TELEGRAM_BOT_TOKEN no configurado en .env"
        raise RuntimeError(msg)

    token = settings.apis.telegram_bot_token.get_secret_value()
    # Defaults: HTML parse_mode global
    defaults = Defaults(
        parse_mode=ParseMode.HTML,
    )

    # Persistencia en disco: sobrevive restarts. Guarda user_data, chat_data,
    # bot_data, callback_data y conversation states en un pickle que se
    # flushea tras cada update.
    from pathlib import Path

    from telegram.ext import PicklePersistence

    persistence_dir = Path(__file__).resolve().parents[3] / "logs"
    persistence_dir.mkdir(parents=True, exist_ok=True)
    persistence = PicklePersistence(
        filepath=persistence_dir / "telegram_state.pickle",
        update_interval=5,
    )

    app = (
        Application.builder()
        .token(token)
        .defaults(defaults)
        .persistence(persistence)
        .post_init(_post_init)
        .build()
    )

    handlers = [
        ("start", cmd_start),
        ("help", cmd_help),
        ("glosario", cmd_glosario),  # NUEVO: términos técnicos
        ("como_funciona", cmd_como_funciona),  # NUEVO: tutorial 5min
        ("guia", cmd_guia),  # NUEVO: explicación específica
        ("simular", cmd_simular),  # NUEVO: proyección mensual
        ("mi_primer_pick", cmd_mi_primer_pick),  # NUEVO: walkthrough
        ("analyze", cmd_analyze),
        ("analizar", cmd_analizar),  # NUEVO: agente on-demand multi-señal por partido
        ("auto_on", cmd_auto_on),
        ("auto_off", cmd_auto_off),
        ("estado", cmd_estado),
        ("picks", cmd_picks),
        ("today", cmd_today),
        ("eventos", cmd_today),  # alias amigable para lista de partidos próximos
        ("bankroll", cmd_bankroll),
        ("deposit", cmd_deposit),
        ("moneda", cmd_moneda),
        ("fx", cmd_fx),
        ("historial", cmd_historial),
        ("calibration_report", cmd_calibration_report),
        ("precision_modelo", cmd_calibration_report),  # alias amigable
        ("calibration", cmd_calibration),  # Sprint 4d: KPIs primarios (Brier/BSS/ECE)
        ("calidad", cmd_calibration),  # alias amigable
        ("explain", cmd_explain),  # Sprint 8: SHAP top-5 por pick
        ("explicar", cmd_explain),  # alias amigable
        ("pausar", cmd_pausar),
        ("resumir", cmd_resumir),
        ("review_last_week", cmd_review_last_week),
        ("resumen_semana", cmd_review_last_week),  # alias amigable
        ("region", cmd_region),
        ("menu", cmd_menu),
        # Retirados — respondidos con _RETIRED_MSG para pedagogía del pivote.
        # Sprint 2 los elimina del registro definitivamente si no hay uso.
        ("bankroll", cmd_bankroll),
        ("deposit", cmd_deposit),
        ("moneda", cmd_moneda),
        ("fx", cmd_fx),
        ("clv", cmd_clv),
        ("ventaja_cierre", cmd_clv),
        ("worst_picks", cmd_worst_picks),
        ("peores_picks", cmd_worst_picks),
        ("show_pick_pm", cmd_show_pick_pm),
        ("analisis_pick", cmd_show_pick_pm),
        ("confirm_bet", cmd_confirm_bet),
        ("confirmar", cmd_confirm_bet),
        ("mark_not_taken", cmd_mark_not_taken),
        ("no_tomada", cmd_mark_not_taken),
    ]
    for name, handler in handlers:
        app.add_handler(CommandHandler(name, handler))

    # Inline buttons (tras /start, /help) + picks con ✅/🚫/🔍
    app.add_handler(CallbackQueryHandler(on_callback))
    # Teclado persistente: textos "🎯 Analizar", "💰 Bankroll", etc.
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text_button))

    # Error handler: captura excepciones sin contaminar logs con TypeError de
    # structlog al serializar tracebacks tipo list (PTB 22 + structlog 25 bug).
    async def _error_handler(update: object, context: Any) -> None:
        err = context.error
        try:
            logger.warning(
                "telegram.handler_error",
                error_type=type(err).__name__,
                error=str(err)[:200],
            )
        except Exception:
            pass

    app.add_error_handler(_error_handler)

    return app


async def _post_init(app: Application[Any, Any, Any, Any, Any, Any]) -> None:
    """Configuración one-shot al arrancar: comandos (ES/EN), menu button, branding."""
    settings = get_settings()
    chat_id_raw = settings.apis.telegram_chat_id

    commands_es = [
        BotCommand("start", "🏠 Menú principal"),
        BotCommand("help", "❓ Ayuda completa"),
        BotCommand("analyze", "🎯 Analizar partidos ahora"),
        BotCommand("analizar", "🔬 Análisis profundo de un partido"),
        BotCommand("auto_on", "🔁 Iniciar análisis continuo"),
        BotCommand("auto_off", "⏹️ Detener análisis continuo"),
        BotCommand("estado", "📊 Estado del bot (qué está haciendo)"),
        BotCommand("picks", "🎯 Picks activos (pendientes)"),
        BotCommand("eventos", "📅 Eventos próximas 48h"),
        BotCommand("historial", "📜 Alertas emitidas (7d/30d/all)"),
        BotCommand("resumen_semana", "📈 Precisión últimos 7 días"),
        BotCommand("precision_modelo", "🎯 Precisión/calibración del modelo"),
        BotCommand("calibration", "📊 KPIs Brier/BSS/ECE últimos 30d"),
        BotCommand("explain", "🔍 SHAP top-5 de un pick (ej. /explain 112)"),
        BotCommand("pausar", "⏸ Pausar bot"),
        BotCommand("resumir", "▶️ Reanudar bot"),
    ]

    try:
        await app.bot.set_my_commands(
            commands_es,
            scope=BotCommandScopeAllPrivateChats(),
            language_code="es",
        )
        # Scope admin → idéntico por ahora pero deja lugar para comandos privados
        if chat_id_raw is not None:
            try:
                admin_id = int(chat_id_raw)
                await app.bot.set_my_commands(
                    commands_es,
                    scope=BotCommandScopeChat(admin_id),
                )
            except (TypeError, ValueError):  # fmt: skip
                pass

        # Menú botón naranja → abre lista de comandos (nativo)
        await app.bot.set_chat_menu_button(menu_button=MenuButtonCommands())

        # Branding: descripción larga + corta (se ven al abrir el bot)
        await app.bot.set_my_description(
            "Bot profesional multi-deporte MX+US.\n"
            "NBA · MLB · NFL · Fútbol · Tenis · NHL · Boxeo.\n\n"
            "Detecta value bets con EV≥3% vs consensus Shin-devigged. "
            "Kelly ¼, cap 5%, stop-loss 30%. CLV tracking vs Pinnacle closing.",
            language_code="es",
        )
        await app.bot.set_my_short_description(
            "Picks calibrados MX+US con EV≥3%, Kelly ¼ y notificaciones automáticas.",
            language_code="es",
        )
        logger.info("telegram.branding_applied")
    except Exception as exc:
        logger.warning("telegram.post_init_partial", error=str(exc)[:120])


# ═══════════════════════ Callback + text button handlers ════════════════


_TEXT_BUTTON_MAP: dict[str, Any] = {
    # Teclado actual post-pivote (4 botones sin "Mi cuenta")
    "🎯 Picks": "cmd_picks",
    "📜 Historial": "cmd_historial",
    "📊 Estado": "cmd_estado",
    "⚙️ Más": "cmd_menu",
    # Compat con teclados anteriores (legacy — redirige al helper central)
    "🎯 Analizar": "cmd_analyze",
    "📅 Hoy": "cmd_today",
    "📈 Stats 7d": "cmd_review_last_week",
    "⏸ Pausar": "cmd_pausar",
    "▶️ Resumir": "cmd_resumir",
    "❓ Ayuda": "cmd_help",
    "🏠 Menú": "cmd_start",
}


async def on_text_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Resuelve textos del ReplyKeyboard al comando correspondiente."""
    if not _chat_authorized(update):
        return
    msg = update.effective_message
    if msg is None or msg.text is None:
        return
    fn_name = _TEXT_BUTTON_MAP.get(msg.text.strip())
    if fn_name is None:
        # No es botón → ignora silenciosamente (no responde a texto libre)
        return
    fn = globals().get(fn_name)
    if fn is not None:
        await fn(update, ctx)


async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Procesa clicks en botones inline.

    Formato:
    - <code>cmd:<name></code> → invoca el command handler sin args.
    - <code>bet:confirm:<id></code> / <code>bet:skip:<id></code> / <code>bet:pm:<id></code> → acción sobre pick.
    """
    if not _chat_authorized(update):
        return
    query = update.callback_query
    if query is None or query.data is None:
        return
    await query.answer()
    data = query.data

    if data.startswith("cmd:"):
        cmd = data.split(":", 1)[1]
        fn = globals().get(f"cmd_{cmd}")
        if fn is None:
            await query.edit_message_text("⚠️ Comando desconocido")
            return
        # Inyecta args vacíos y dispara
        ctx.args = []
        await fn(update, ctx)
        return

    if data.startswith("bet:"):
        parts = data.split(":")
        if len(parts) != 3:
            return
        action, bet_id_str = parts[1], parts[2]
        try:
            bet_id = int(bet_id_str)
        except ValueError:
            return
        if action == "confirm":
            async with session_scope() as session:
                await session.execute(
                    text(
                        """
                        UPDATE bets SET is_paper = false,
                                        notes = coalesce(notes,'') || ' [confirmed_by_user]'
                        WHERE id = :bid
                        """
                    ),
                    {"bid": bet_id},
                )
            if query.message is not None:
                await query.edit_message_reply_markup(
                    InlineKeyboardMarkup(
                        [[InlineKeyboardButton(f"✅ Bet #{bet_id} tomada", callback_data="noop")]]
                    )
                )
        elif action == "skip":
            async with session_scope() as session:
                await session.execute(
                    text(
                        """
                        UPDATE bets SET status = 'void',
                                        notes = coalesce(notes,'') || ' [not_taken]'
                        WHERE id = :bid AND status = 'pending'
                        """
                    ),
                    {"bid": bet_id},
                )
            if query.message is not None:
                await query.edit_message_reply_markup(
                    InlineKeyboardMarkup(
                        [
                            [
                                InlineKeyboardButton(
                                    f"🚫 Bet #{bet_id} descartada", callback_data="noop"
                                )
                            ]
                        ]
                    )
                )
        elif action == "pm":
            ctx.args = [str(bet_id)]
            await cmd_show_pick_pm(update, ctx)


def main() -> None:
    """Arranca el bot en modo long-polling.

    <code>app.run_polling()</code> gestiona su propio event loop — no se debe envolver
    en <code>asyncio.run()</code> (eso rompe con <code>Cannot close a running event loop</code>).
    """
    configure_logging()

    # Auto-detect región (MX/US) al arrancar → ajusta flags .env para books
    # según VPN activa o no. Idempotente.
    try:
        import asyncio as _asyncio

        from apuestas.region import auto_configure_region

        result = _asyncio.run(auto_configure_region())
        logger.info(
            "telegram.region_configured",
            country=result.get("country"),
            region=result.get("region"),
        )
    except Exception as exc:
        logger.warning("telegram.region_detect_fail", error=str(exc)[:120])

    app = build_application()
    logger.info("telegram.polling_start")
    # drop_pending_updates=True para evitar consumir backlog de mensajes
    # acumulados durante un Conflict (otra instancia del bot que estaba
    # robando getUpdates). Con backlog limpio garantizamos que el primer
    # mensaje del usuario después del start lo procesemos NOSOTROS sin
    # carrera. Habilitar `False` solo si confirmas que NO hay otra instancia
    # con el mismo TELEGRAM_BOT_TOKEN.
    import os as _os

    drop_pending = _os.environ.get("APUESTAS_TELEGRAM_DROP_PENDING", "true").lower() == "true"
    app.run_polling(drop_pending_updates=drop_pending)


if __name__ == "__main__":
    main()
