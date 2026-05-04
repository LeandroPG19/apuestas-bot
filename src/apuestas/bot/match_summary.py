"""Mensaje unificado por partido — agrupa pick + arb + steam + props.

En lugar de enviar N mensajes separados (1 pick, 1 arb, 1 steam move, 3 props)
al chat Telegram, se AGRUPA TODO en UN solo mensaje por partido.

Flujo:
1. Durante el ciclo de análisis, cada detector empuja a `MatchSummaryBuilder`
2. Al final del ciclo, `flush_all()` envía un mensaje consolidado por match

Estructura del mensaje:
    🏀 [sport] home vs away · 🕐 hora
    ━━━━━━━━━━━━━━━
    🎯 MEJORES PICKS
      • pick 1 con EV, stars, book
      • pick 2 ...
    💎 ARBITRAJE (si hay)
      • profit garantizado X%
    ⚡ STEAM MOVES (si hay)
      • Pinnacle movió X% en Y min
    📊 PROPS DETECTADOS (corners, shots, player)
      • corners > 5.5 @ bet365 EV +3.2%
    🔁 ALTERNATIVAS
      • top 3 líneas alternas

Dedup: el mismo match no se envía 2x en la misma sesión (usa match_id + cycle_id).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import text

from apuestas.config import get_settings
from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

_HTML_ESCAPE = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}
_DIV = "━━━━━━━━━━━━━━━━━━━━"
_DIV_SOFT = "─ ─ ─ ─ ─ ─ ─ ─ ─ ─"
_SPORT_EMOJI = {
    "nba": "🏀",
    "wnba": "🏀",
    "mlb": "⚾",
    "nfl": "🏈",
    "ncaaf": "🏈",
    "nhl": "🏒",
    "soccer": "⚽",
    "laliga": "⚽",
    "epl": "⚽",
    "liga_mx": "⚽",
    "soccer_ucl": "⚽",
    "tennis": "🎾",
    "boxing": "🥊",
    "mma": "🥋",
    "cricket_ipl": "🏏",
    "euroleague": "🏀",
}


def _escape_md(s: str) -> str:
    if not s:
        return ""
    return "".join(_HTML_ESCAPE.get(c, c) for c in s)


def _confidence_stars(ev: float) -> tuple[str, str]:
    if ev >= 0.05:
        return ("⭐⭐⭐⭐⭐", "Muy alta")
    if ev >= 0.03:
        return ("⭐⭐⭐⭐", "Alta")
    if ev >= 0.02:
        return ("⭐⭐⭐", "Media")
    if ev >= 0.01:
        return ("⭐⭐", "Baja")
    return ("⭐", "Muy baja")


@dataclass(slots=True)
class PickEntry:
    bet_id: int
    market: str  # h2h | spreads | totals | team_totals
    outcome: str
    team_name: str  # nombre del equipo apostado (ya resuelto)
    bookmaker: str
    odds: float
    p_fair: float
    ev: float
    stake_units: float
    kelly_pct: float


@dataclass(slots=True)
class ArbEntry:
    profit_pct: float
    legs: list[tuple[str, str, float]]  # (outcome, book, odds)
    stakes_per_book: dict[str, float]
    market: str


@dataclass(slots=True)
class SteamEntry:
    outcome: str
    delta_pp: float  # puntos porcentuales
    pinnacle_before: float
    pinnacle_now: float
    books_behind: list[tuple[str, float]]


@dataclass(slots=True)
class PropEntry:
    prop_type: str  # corners, shots, HR, points, etc.
    team_or_player: str
    line: float
    model_prob: float  # P(over line)
    bookmaker: str
    odds: float
    ev: float


@dataclass(slots=True)
class MatchSummary:
    match_id: int
    home: str
    away: str
    sport: str
    start_time: datetime | None
    picks: list[PickEntry] = field(default_factory=list)
    arbs: list[ArbEntry] = field(default_factory=list)
    steams: list[SteamEntry] = field(default_factory=list)
    props: list[PropEntry] = field(default_factory=list)
    alternatives: list[tuple[str, float]] = field(default_factory=list)  # (book, odds)


class MatchSummaryBuilder:
    """Agrega picks/arbs/steams/props por match durante un ciclo y envía UN
    mensaje consolidado por match al final."""

    def __init__(self) -> None:
        self._summaries: dict[int, MatchSummary] = {}

    async def add_pick(
        self,
        bet_id: int,
        match_id: int,
        home: str,
        away: str,
        sport: str,
        start_time: datetime | None,
        **kwargs: Any,
    ) -> None:
        s = self._get_or_create(match_id, home, away, sport, start_time)
        s.picks.append(PickEntry(bet_id=bet_id, **kwargs))

    async def add_arb(
        self,
        match_id: int,
        home: str,
        away: str,
        sport: str,
        start_time: datetime | None,
        **kwargs: Any,
    ) -> None:
        s = self._get_or_create(match_id, home, away, sport, start_time)
        s.arbs.append(ArbEntry(**kwargs))

    async def add_steam(
        self,
        match_id: int,
        home: str,
        away: str,
        sport: str,
        start_time: datetime | None,
        **kwargs: Any,
    ) -> None:
        s = self._get_or_create(match_id, home, away, sport, start_time)
        s.steams.append(SteamEntry(**kwargs))

    async def add_prop(
        self,
        match_id: int,
        home: str,
        away: str,
        sport: str,
        start_time: datetime | None,
        **kwargs: Any,
    ) -> None:
        s = self._get_or_create(match_id, home, away, sport, start_time)
        s.props.append(PropEntry(**kwargs))

    def _get_or_create(
        self, mid: int, home: str, away: str, sport: str, start: datetime | None
    ) -> MatchSummary:
        if mid not in self._summaries:
            self._summaries[mid] = MatchSummary(
                match_id=mid,
                home=home,
                away=away,
                sport=sport,
                start_time=start,
            )
        return self._summaries[mid]

    async def flush_all(self) -> dict[str, int]:
        """Envía mensajes consolidados a Telegram + canal. Retorna stats."""
        sent = 0
        skipped = 0
        for mid, summary in self._summaries.items():
            if not (summary.picks or summary.arbs or summary.steams or summary.props):
                skipped += 1
                continue
            if await _send_summary(summary):
                sent += 1
        logger.info(
            "match_summary.flush",
            total_matches=len(self._summaries),
            sent=sent,
            skipped=skipped,
        )
        return {"total": len(self._summaries), "sent": sent, "skipped": skipped}


def _format_summary(s: MatchSummary) -> str:
    """Construye mensaje HTML consolidado."""
    emoji = _SPORT_EMOJI.get(s.sport.lower(), "🎯")
    home = _escape_md(s.home)
    away = _escape_md(s.away)
    sport_label = s.sport.upper().replace("_", " ")

    start_str = ""
    if s.start_time:
        local = s.start_time + timedelta(hours=-6)  # MX tz UTC-6
        start_str = local.strftime("%a %d %b · %H:%M")

    lines = [
        f"{emoji} <b>{home}</b> <i>vs</i> <b>{away}</b>",
        f"<i>{sport_label}" + (f" · 🕐 {start_str}" if start_str else "") + "</i>",
        f"<code>{_DIV}</code>",
    ]

    # Sección 1: MEJORES PICKS
    if s.picks:
        lines.append("\n🎯 <b>MEJORES PICKS</b>")
        # Ordenar por EV descendente
        for pk in sorted(s.picks, key=lambda p: -p.ev):
            stars, conf_label = _confidence_stars(pk.ev)
            lines.append(
                f"  {stars} <b>{_escape_md(pk.team_name)}</b> <i>({pk.market}·{pk.outcome})</i>"
            )
            lines.append(
                f"     🏦 {_escape_md(pk.bookmaker)} @ <code>{pk.odds:.2f}</code>  "
                f"· 🎯 {pk.p_fair * 100:.1f}% ganar  "
                f"· 📊 EV <b>+{pk.ev * 100:.2f}%</b>"
            )
            lines.append(
                f"     💵 {pk.stake_units:.2f}u "
                f"<i>({pk.kelly_pct:.2f}% bankroll · {conf_label})</i>"
            )

    # Sección 2: ARBITRAJE
    if s.arbs:
        lines.append("\n💎 <b>ARBITRAJE DETECTADO</b>")
        for arb in s.arbs:
            lines.append(
                f"  🟢 Profit garantizado: <b>+{arb.profit_pct * 100:.2f}%</b> "
                f"<i>({arb.market})</i>"
            )
            for leg in arb.legs:
                outcome, book, odds = leg
                stake_pct = arb.stakes_per_book.get(book, 0) * 100
                lines.append(
                    f"     • <b>{_escape_md(book)}</b> → {outcome} "
                    f"@ <code>{odds:.2f}</code> "
                    f"<i>(stake {stake_pct:.1f}%)</i>"
                )

    # Sección 3: STEAM MOVES
    if s.steams:
        lines.append("\n⚡ <b>STEAM MOVE</b>")
        for st in s.steams:
            direction_icon = "📉" if st.delta_pp > 0 else "📈"
            lines.append(
                f"  {direction_icon} <b>{st.outcome}</b>: Pinnacle "
                f"<code>{st.pinnacle_before:.2f} → {st.pinnacle_now:.2f}</code> "
                f"({st.delta_pp:+.1f}pp)"
            )
            if st.books_behind:
                behind_str = ", ".join(f"{_escape_md(b)} @ {o:.2f}" for b, o in st.books_behind[:3])
                lines.append(f"     📌 Books aún con odds vieja: <i>{behind_str}</i>")

    # Sección 4: PROPS DETECTADOS
    if s.props:
        lines.append("\n📊 <b>PROPS DETECTADOS</b>")
        for prop in sorted(s.props, key=lambda p: -p.ev):
            stars, _ = _confidence_stars(prop.ev)
            lines.append(
                f"  {stars} <b>{_escape_md(prop.team_or_player)}</b> {prop.prop_type} > {prop.line}"
            )
            lines.append(
                f"     🏦 {_escape_md(prop.bookmaker)} @ <code>{prop.odds:.2f}</code>  "
                f"· 🎯 {prop.model_prob * 100:.1f}% prob  "
                f"· EV <b>+{prop.ev * 100:.2f}%</b>"
            )

    # Sección 5: ALTERNATIVAS (si hay)
    if s.alternatives:
        lines.append(f"\n<code>{_DIV_SOFT}</code>")
        lines.append("🔁 <i>Otras líneas disponibles:</i>")
        alts_str = ", ".join(f"{_escape_md(b)} {o:.2f}" for b, o in s.alternatives[:5])
        lines.append(f"   {alts_str}")

    lines.append(f"\n<code>{_DIV_SOFT}</code>")
    lines.append("<i>👇 Toca ✅/🚫 en cada pick individual</i>")

    return "\n".join(lines)


async def _send_summary(s: MatchSummary) -> bool:
    """Envía mensaje consolidado al chat privado + canal."""
    settings = get_settings()
    token = settings.apis.telegram_bot_token
    chat_id = settings.apis.telegram_chat_id
    if token is None or chat_id is None:
        return False

    # Dedup: si ya se envió este match_id en últimos 10 min, skip
    async with session_scope() as session:
        row = (
            await session.execute(
                text(
                    "SELECT updated_at FROM bot_state "
                    "WHERE key = :k AND updated_at > NOW() - INTERVAL '10 minutes'"
                ),
                {"k": f"match_summary_sent_{s.match_id}"},
            )
        ).first()
    if row is not None:
        logger.debug("match_summary.dedup_skip", match_id=s.match_id)
        return False

    msg = _format_summary(s)

    try:
        from telegram import Bot
        from telegram.constants import ParseMode

        bot = Bot(token=token.get_secret_value())
        await bot.send_message(
            chat_id=int(chat_id),
            text=msg,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
        channel_id = settings.apis.telegram_channel_id
        if channel_id:
            try:
                target: str | int = (
                    int(channel_id) if channel_id.lstrip("-").isdigit() else channel_id
                )
                await bot.send_message(
                    chat_id=target,
                    text=msg,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )
            except Exception as exc_ch:
                logger.warning(
                    "match_summary.channel_broadcast_fail",
                    error=str(exc_ch)[:100],
                )
    except Exception as exc:
        logger.warning("match_summary.send_fail", error=str(exc)[:120])
        return False

    # Marcar como enviado para dedup
    async with session_scope() as session:
        await session.execute(
            text(
                """
                INSERT INTO bot_state (key, value, updated_at)
                VALUES (:k, 'sent', NOW())
                ON CONFLICT (key) DO UPDATE SET updated_at = NOW()
                """
            ),
            {"k": f"match_summary_sent_{s.match_id}"},
        )

    logger.info(
        "match_summary.sent",
        match_id=s.match_id,
        picks=len(s.picks),
        arbs=len(s.arbs),
        steams=len(s.steams),
        props=len(s.props),
    )
    return True


# Singleton accesible desde todo el flow
_global_builder: MatchSummaryBuilder | None = None


def get_builder() -> MatchSummaryBuilder:
    """Retorna instancia global usada durante un ciclo auto_loop."""
    global _global_builder
    if _global_builder is None:
        _global_builder = MatchSummaryBuilder()
    return _global_builder


def reset_builder() -> None:
    """Resetea builder al final del ciclo."""
    global _global_builder
    _global_builder = None
