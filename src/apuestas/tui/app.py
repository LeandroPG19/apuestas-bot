"""TUI principal — Apuestas Bot con UX profesional e inductiva.

Diseño:
- Header: título + indicadores live (● BD / ● LLM / ● Memoria) + reloj.
- Sidebar izquierdo: ayuda contextual por tab (toggle con [h]).
- Main: tabs con contenido educativo + empty-states con CTAs claros.
- Footer: status bar con última acción + siguiente refresh + '?' para ayuda.

Atajos globales:
  ? / F1   → Overlay de ayuda completo
  Ctrl+P   → Command palette (buscar acción)
  h        → Toggle sidebar ayuda
  d/p/b/m/c/e → saltar a tab
  a        → Analizar próximos 48h
  r        → Refresh tab actual
  P        → Pausar/reanudar
  q        → Salir
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.align import Align
from rich.panel import Panel
from rich.text import Text
from sqlalchemy import text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import DiscoveryHit, Hit, Hits, Provider
from textual.containers import Container, Grid, Horizontal, VerticalScroll
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Static,
    TabbedContent,
    TabPane,
)

from apuestas.db import session_scope
from apuestas.obs.logging import configure_logging, get_logger

logger = get_logger(__name__)


# ════════════════════════ Widgets base ═══════════════════════════════════


class SystemStatus(Static):
    """Barra superior con indicadores live: BD · LLM · Memoria · créditos API."""

    bd_ok: reactive[bool] = reactive(False)
    llm_ok: reactive[bool] = reactive(False)
    mem_ok: reactive[bool] = reactive(False)
    api_credits: reactive[int] = reactive(-1)
    last_refresh: reactive[str] = reactive("-")

    def render(self) -> Panel:
        def dot(ok: bool) -> str:
            return "[green]●[/]" if ok else "[red]●[/]"

        cred_str = f"[cyan]{self.api_credits}[/]" if self.api_credits >= 0 else "[dim]?[/]"
        body = Text.from_markup(
            f" {dot(self.bd_ok)} BD   "
            f"{dot(self.llm_ok)} LLM (DeepSeek)   "
            f"{dot(self.mem_ok)} Memoria   "
            f"[dim]|[/]   "
            f"🔑 {cred_str} créditos OddsAPI   "
            f"[dim]|[/]   "
            f"⟳ {self.last_refresh}"
        )
        return Panel(
            body,
            title="[b]🎯 Apuestas Bot[/]",
            title_align="left",
            border_style="cyan",
            padding=(0, 1),
        )


class MetricCard(Static):
    """Tarjeta con métrica + delta + explicación corta (educativo)."""

    def __init__(
        self,
        *,
        label: str,
        icon: str,
        value: str = "—",
        delta: str = "",
        hint: str = "",
        color: str = "cyan",
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self._label = label
        self._icon = icon
        self._value = value
        self._delta = delta
        self._hint = hint
        self._color = color

    def update_metric(
        self, *, value: str, delta: str = "", hint: str = "", color: str | None = None
    ) -> None:
        self._value = value
        self._delta = delta
        if hint:
            self._hint = hint
        if color:
            self._color = color
        self.refresh()

    def render(self) -> Panel:
        t = Text()
        t.append(f"\n  {self._icon}  ", style="bold")
        t.append(self._value, style=f"bold {self._color}")
        if self._delta:
            t.append(f"\n  {self._delta}", style="dim")
        if self._hint:
            t.append(f"\n\n  {self._hint}", style="italic dim")
        t.append("\n")
        return Panel(t, title=f"[b]{self._label}[/]", border_style=self._color, padding=(0, 1))


class WelcomeCard(Static):
    """Welcome card con checklist de onboarding (solo primera vez)."""

    def render(self) -> Panel:
        body = Text()
        body.append("\n  👋 ", style="bold yellow")
        body.append("Bienvenido al Bot de Apuestas\n", style="bold yellow")
        body.append(
            "\n  El bot no aposta solo — te recomienda picks con edge, tú decides si los tomas.\n",
            style="dim italic",
        )
        body.append("\n  📋 Primeros pasos:\n", style="bold")
        body.append(
            "\n    [1] Presiona ",
            style="white",
        )
        body.append(" A ", style="bold black on green")
        body.append(
            "  para ingestar eventos próximos 48h + odds + análisis DeepSeek\n",
            style="white",
        )
        body.append("    [2] Revisa el listado de picks que aparecerá abajo\n", style="white")
        body.append("    [3] Cuando un match termine, corre ", style="white")
        body.append("apuestas settle", style="bold cyan on black")
        body.append("  en otra terminal\n", style="white")
        body.append(
            "    [4] El Dashboard te mostrará PnL, CLV y ROI automáticamente\n",
            style="white",
        )
        body.append("\n  💡 Tip: ", style="bold magenta")
        body.append("presiona ", style="dim")
        body.append(" ? ", style="bold black on yellow")
        body.append(" en cualquier momento para ver todos los atajos.\n", style="dim")
        return Panel(
            body,
            title="[b green]Primeros pasos[/]",
            border_style="green",
            padding=(1, 2),
        )


class HelpSidebar(Static):
    """Panel lateral con ayuda contextual por tab."""

    current_tab: reactive[str] = reactive("dash")

    _help_map: dict[str, tuple[str, list[tuple[str, str]]]] = {
        "dash": (
            "Panel de control principal.",
            [
                ("Ver", "bankroll, ROI, CLV, picks activos"),
                ("a", "disparar análisis 48h completo"),
                ("r", "refrescar datos"),
                ("P", "pausar/reanudar bot"),
                ("", ""),
                ("CLV", "Closing Line Value — si tu odds es mejor que"),
                ("", "la del cierre, vas por buen camino aunque pierdas"),
                ("ROI", "return-on-investment últimos 7 días"),
            ],
        ),
        "pm": (
            "Post-mortems de bets liquidadas.",
            [
                ("Ordenado", "por discrepancia (más arriba = peor análisis)"),
                ("Discrepancy", "cuánto se alejó la predicción de la realidad"),
                ("Lección", "narrativa LLM de qué aprender"),
                ("r", "refrescar"),
            ],
        ),
        "bankroll": (
            "Evolución del capital.",
            [
                ("Curva", "últimos 60 días (plotext ASCII)"),
                ("Bets", "últimas 20 con su PnL"),
                ("Verde", "bet ganada · Rojo: perdida · Dim: void"),
                ("r", "refrescar"),
            ],
        ),
        "models": (
            "Modelos ML en producción.",
            [
                ("Stage", "shadow (pruebas) / production / archived"),
                ("Drift", "ok / warning / critical (PSI + CBPE)"),
                ("", "retrain manual: make retrain SPORT=nba"),
                ("r", "refrescar"),
            ],
        ),
        "calibration": (
            "¿El modelo es confiable?",
            [
                ("Gap", "predicho vs real (0 = perfecto)"),
                ("<0.03", "calibrado · <0.05 aceptable · >0.05 revisar"),
                ("Bucket", "rango de probabilidad predicha"),
                ("r", "refrescar"),
            ],
        ),
        "memory": (
            "Memoria persistente (cuba-memorys).",
            [
                ("Status", "conectado/offline"),
                ("Qué hace", "inyecta contexto histórico al LLM"),
                ("", "→ menos alucinaciones en análisis"),
                ("x", "scan de contradicciones"),
                ("z", "analizar gaps de memoria"),
                ("r", "refrescar"),
            ],
        ),
        "regional": (
            "Line shopping MX vs US.",
            [
                ("Qué muestra", "mejor casa por cada pick activo"),
                ("MX EV", "EV neto en mejor MX (ajustado límite)"),
                ("US EV", "EV neto en mejor US"),
                ("Recom.", "cuál región conviene más"),
                ("", "SEGOB = legal MX · EEUU legal por estado"),
                ("r", "refrescar"),
            ],
        ),
        "llm": (
            "Costo + latencia DeepSeek V3.2.",
            [
                ("Calls", "total de llamadas al LLM"),
                ("Costo", "USD acumulado ($0.27/M in, $1.10/M out)"),
                ("Latency p95", "<3s ideal · 3-8s ok · >8s warning"),
                ("Tabla", "últimas 30 con task_kind + tokens"),
                ("r", "refrescar"),
            ],
        ),
        "logs": (
            "Tail en vivo de structlog.",
            [
                ("Buffer", "últimos 500 eventos en memoria"),
                ("Autorefresh", "cada 2 segundos"),
                ("i", "filtrar INFO+"),
                ("w", "filtrar WARNING+"),
                ("e", "filtrar ERROR+"),
                ("c", "limpiar buffer"),
                ("r", "refrescar ahora"),
            ],
        ),
        "setup": (
            "Panel de control con botones.",
            [
                ("Servicios", "activar/detener worker + backup"),
                ("Integraciones", "Telegram + Reddit wizards"),
                ("Mantenim.", "backup ahora · cache · test APIs"),
                ("Estado", "live dashboard de salud sistema"),
                ("Tab/Enter", "navegar y activar botones"),
                ("r", "refrescar estados"),
            ],
        ),
    }

    def render(self) -> Panel:
        title, rows = self._help_map.get(self.current_tab, ("Ayuda no disponible", []))
        t = Text()
        t.append("\n  ", style="")
        t.append(title, style="bold cyan")
        t.append("\n\n")
        for key, desc in rows:
            if not key and not desc:
                t.append("\n")
                continue
            if key:
                t.append(f"  {key:<12}", style="bold green")
                t.append(f"{desc}\n", style="white")
            else:
                t.append(f"              {desc}\n", style="dim")
        t.append("\n\n  ", style="")
        t.append("Toggle con ", style="dim italic")
        t.append("h", style="bold cyan")
        t.append(" · Ayuda global con ", style="dim italic")
        t.append("?", style="bold cyan")
        t.append("\n", style="")
        return Panel(t, title="[b]💡 Guía rápida[/]", border_style="magenta", padding=(0, 1))


class EmptyState(Static):
    """Empty state educativo con CTA claro."""

    def __init__(
        self, *, icon: str, title: str, description: str, cta: str = "", **kw: Any
    ) -> None:
        super().__init__(**kw)
        self._icon = icon
        self._title = title
        self._desc = description
        self._cta = cta

    def render(self) -> Panel:
        t = Text()
        t.append(f"\n       {self._icon}\n\n", style="bold yellow")
        t.append(f"       {self._title}\n\n", style="bold white")
        for line in self._desc.split("\n"):
            t.append(f"       {line}\n", style="dim")
        if self._cta:
            t.append(f"\n       → {self._cta}\n", style="bold green")
        t.append("\n")
        return Panel(Align.center(t), border_style="yellow")


# ════════════════════════ Help overlay modal ══════════════════════════════


class HelpScreen(ModalScreen[None]):
    """Overlay con todos los atajos. Se abre con ? / F1."""

    BINDINGS = [Binding("escape", "dismiss", "Cerrar"), Binding("q", "dismiss", "Cerrar")]

    def compose(self) -> ComposeResult:
        body = Text()
        body.append("\n")
        sections = [
            (
                "Navegación (tabs)",
                [
                    ("d", "Dashboard"),
                    ("p", "Post-mortems"),
                    ("b", "Bankroll"),
                    ("m", "Models ML"),
                    ("c", "Calibración"),
                    ("e", "Memoria cuba"),
                    ("g", "Regional MX/US"),
                    ("l", "LLM consumo"),
                    ("L", "Logs live"),
                    ("s", "Setup + botones"),
                ],
            ),
            (
                "Acciones",
                [
                    ("a", "Analizar próximos 48h (ingesta + LLM)"),
                    ("r", "Refrescar tab actual"),
                    ("Ctrl+R", "Refrescar TODOS los tabs"),
                    ("P", "Pausar / Reanudar el bot (con confirm)"),
                    ("h", "Mostrar / ocultar panel de ayuda lateral"),
                    ("Enter", "Ver detalle completo del pick seleccionado"),
                ],
            ),
            (
                "Sistema",
                [
                    ("?", "Este menú de ayuda"),
                    ("F1", "Abrir este menú (alternativa)"),
                    ("Ctrl+P", "Command palette (buscar comandos)"),
                    ("t", "Tutorial interactivo 7 pasos"),
                    ("q", "Salir (guarda sesión en cuba-memorys)"),
                ],
            ),
            (
                "Conceptos clave",
                [
                    ("CLV+", "Cierra mejor que la línea = positive expectation"),
                    ("Kelly ¼", "Stake = ¼ × edge × bankroll · cap 5%"),
                    ("EV ≥ 3%", "Threshold mínimo para emitir pick"),
                    ("Shin devig", "De-vigging robusto del overround del book"),
                    ("Regional", "MX (SEGOB) vs US (estatal) line shopping"),
                ],
            ),
        ]
        for title, rows in sections:
            body.append(f"  {title}\n", style="bold cyan")
            body.append(f"  {'─' * 60}\n", style="dim")
            for key, desc in rows:
                body.append(f"    {key:<12}", style="bold green")
                body.append(f"{desc}\n", style="white")
            body.append("\n")
        body.append("  Presiona ESC o q para cerrar este overlay\n", style="dim italic")
        yield Container(
            Static(
                Panel(
                    body,
                    title="[b]📖 Ayuda — Apuestas Bot TUI[/]",
                    border_style="cyan",
                    padding=(1, 2),
                )
            ),
            id="help_box",
        )

    def action_dismiss(self, result: None = None) -> None:
        self.app.pop_screen()


class ConfirmScreen(ModalScreen[bool]):
    """Modal sí/no genérico. Retorna True/False al App.push_screen_wait."""

    BINDINGS = [
        Binding("y", "confirm", "Sí"),
        Binding("Y", "confirm", "Sí"),
        Binding("n", "cancel", "No"),
        Binding("N", "cancel", "No"),
        Binding("escape", "cancel", "Cancelar"),
    ]

    def __init__(self, *, title: str, message: str, danger: bool = False) -> None:
        super().__init__()
        self._title = title
        self._message = message
        self._danger = danger

    def compose(self) -> ComposeResult:
        body = Text()
        body.append(f"\n  {self._message}\n\n", style="white")
        body.append("  ", style="")
        body.append(" Y ", style="bold black on green")
        body.append("  Sí        ", style="green")
        body.append(" N ", style="bold black on red")
        body.append("  No / Cancelar\n", style="red")
        color = "red" if self._danger else "yellow"
        yield Container(
            Static(
                Panel(
                    body,
                    title=f"[b]⚠  {self._title}[/]",
                    border_style=color,
                    padding=(1, 2),
                )
            ),
            id="confirm_box",
        )

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class TutorialScreen(ModalScreen[None]):
    """Wizard interactivo multi-paso — primer uso o tecla t."""

    BINDINGS = [
        Binding("escape", "dismiss", "Cerrar"),
        Binding("q", "dismiss", "Cerrar"),
        Binding("right", "next_step", "Siguiente"),
        Binding("n", "next_step", "Siguiente"),
        Binding("space", "next_step", "Siguiente"),
        Binding("left", "prev_step", "Anterior"),
        Binding("b", "prev_step", "Anterior"),
    ]

    STEPS = [
        (
            "1 · ¿Qué es este bot?",
            "🎯",
            "Un asistente de apuestas deportivas. NO apuesta solo.\n\n"
            "• Ingiere fixtures y odds en vivo (football-data.org, The Odds API)\n"
            "• De-vigging Shin para estimar probabilidad justa\n"
            "• DeepSeek V3.2 + memoria cuba-memorys para análisis sin alucinaciones\n"
            "• Kelly ¼ con cap 5% · EV ≥ 3% · conformal CI 90% (MAPIE)\n"
            "• Regional line-shopping MX (SEGOB) vs US (estatal)\n\n"
            "Tú ves la recomendación y decides si la tomas. Presiona → para seguir.",
        ),
        (
            "2 · Dashboard: tu centro de control",
            "📊",
            "Al abrir verás 4 tarjetas grandes:\n\n"
            "• 💰 Bankroll: capital virtual actual\n"
            "• 📊 ROI 7d: retorno últimos 7 días\n"
            "• 📈 CLV 7d: Closing Line Value (sube aún si pierdes)\n"
            "• 🎯 Picks activos: posiciones abiertas\n\n"
            "Debajo: próximos eventos 48h y picks pending.\n"
            "Presiona [ A ] para disparar el análisis completo.",
        ),
        (
            "3 · Cómo nace un pick",
            "🧠",
            "Al presionar [ A ] el bot ejecuta en ~1 minuto:\n\n"
            "1. Ingest fixtures football-data + odds consensus\n"
            "2. De-vigging Shin sobre los bookmakers\n"
            "3. Cálculo EV + Kelly ¼ por outcome\n"
            "4. Si EV ≥ 3% y conformal CI soporta → pick candidato\n"
            "5. DeepSeek analiza con contexto cuba-memorys\n"
            "6. Si todo OK, se persiste en BD como paper bet pending\n\n"
            "Sin edge real, el bot NO apuesta. Eso es bueno.",
        ),
        (
            "4 · Ver el análisis completo",
            "🔍",
            "En la tabla 'Picks activos' selecciona una fila con ↑↓\n"
            "y presiona [ Enter ].\n\n"
            "Se abre un overlay con TODO lo que el bot sabe:\n"
            "• 🎲 Recomendación: equipo + odds + stake Kelly\n"
            "• 🏪 Dónde apostar: mejor MX vs US con edge por casa\n"
            "• 📊 Posibilidades: P modelo + conformal 90% CI\n"
            "• 🧠 Narrativa LLM: home/away/matchup\n"
            "• 📐 En qué se basó: Shin devig + SHAP top-5 + memoria\n"
            "• ⚠ Warnings: flags del LLM + validaciones\n\n"
            "ESC para volver al Dashboard.",
        ),
        (
            "5 · Tabs especializados",
            "🗂️",
            "Usa estas teclas para saltar rápido:\n\n"
            "  [d] Dashboard           [p] Post-mortems (lecciones)\n"
            "  [b] Bankroll curva      [m] Models (ML + drift)\n"
            "  [c] Calibración         [e] Memoria cuba\n"
            "  [g] Regional MX/US      [l] LLM (costo + latencia)\n"
            "  [L] Logs live           [h] Toggle sidebar ayuda\n\n"
            "En cada tab:  [ r ]  refresca datos.",
        ),
        (
            "6 · Cierre del loop",
            "♻️",
            "Cuando un match termine:\n\n"
            "• 'apuestas settle' (o trigger auto) liquida tus bets\n"
            "• Se genera post-mortem con discrepancia + lección LLM\n"
            "• Bankroll se actualiza con PnL real\n"
            "• Calibración rolling se recalcula\n"
            "• cuba-memorys registra outcome → próximo análisis más sabio\n\n"
            "Este loop es lo que hace al bot mejorar con cada sesión.",
        ),
        (
            "7 · Atajos globales",
            "⌨️",
            "  [ ? ]  / F1     Menú de ayuda completo\n"
            "  [ Ctrl+P ]      Command palette (Textual)\n"
            "  [ t ]           Este tutorial\n"
            "  [ h ]           Toggle sidebar ayuda\n"
            "  [ A ]           Disparar análisis\n"
            "  [ P ]           Pausar/reanudar (con confirm)\n"
            "  [ Ctrl+R ]      Refrescar TODOS los tabs\n"
            "  [ q ]           Salir (guarda jornada)\n\n"
            "¡Listo! Presiona [ q ] o ESC para cerrar este tutorial.",
        ),
    ]

    step: reactive[int] = reactive(0)

    def compose(self) -> ComposeResult:
        yield Container(Static(id="tutorial_body"), id="tutorial_box")

    def on_mount(self) -> None:
        self._render_step()

    def watch_step(self, _old: int, _new: int) -> None:
        self._render_step()

    def _render_step(self) -> None:
        i = max(0, min(self.step, len(self.STEPS) - 1))
        title, icon, body_text = self.STEPS[i]
        t = Text()
        t.append(f"\n  {icon}  ", style="bold yellow")
        t.append(title, style="bold white")
        t.append(f"\n\n  {body_text}\n\n", style="white")
        t.append(f"  Paso {i + 1} de {len(self.STEPS)}\n", style="dim")
        t.append("\n  ")
        if i > 0:
            t.append(" ← ", style="bold black on cyan")
            t.append("  anterior   ", style="cyan")
        t.append(" → ", style="bold black on green")
        t.append("  siguiente   ", style="green")
        t.append(" ESC ", style="bold black on red")
        t.append("  cerrar\n", style="red")
        body = self.query_one("#tutorial_body", Static)
        body.update(
            Panel(
                t,
                title="[b]📚 Tutorial — Apuestas Bot[/]",
                border_style="yellow",
                padding=(0, 2),
            )
        )

    def action_next_step(self) -> None:
        if self.step < len(self.STEPS) - 1:
            self.step += 1
        else:
            self.app.pop_screen()

    def action_prev_step(self) -> None:
        if self.step > 0:
            self.step -= 1

    def action_dismiss(self, result: None = None) -> None:
        self.app.pop_screen()


# ═══════════════ Pantalla de detalle de un pick (Enter) ══════════════════


async def _fetch_pick_detail(bet_id: int) -> dict[str, Any] | None:
    """Trae todo lo que se sabe de una alerta: predicción, LLM, SHAP, odds.

    Post-pivote: lee `pick_alerts` en vez de `bets`. Campos retirados
    (stake_units, clv, pnl_units, is_paper) se omiten; la UI superior los
    mostrará como n/d si faltan.
    """
    async with session_scope() as s:
        row = (
            await s.execute(
                text(
                    """
                    SELECT
                        pa.id AS bet_id, pa.bookmaker, pa.market, pa.outcome,
                        pa.line, pa.odds_placed, pa.status, pa.placed_at,
                        pa.best_odds_seen, pa.best_odds_book,
                        pa.upgrade_count, pa.outcome_result,
                        p.id AS pred_id, p.probability AS p_model,
                        p.p_lower, p.p_upper, p.ev,
                        p.best_odds, p.best_bookmaker,
                        p.model_name, p.model_version,
                        p.shap_top5, p.llm_analysis, p.features_snapshot,
                        m.id AS match_id, m.sport_code, m.start_time,
                        m.home_score, m.away_score, m.status AS match_status,
                        h.name AS home_name, h.external_id AS home_ext,
                        a.name AS away_name, a.external_id AS away_ext,
                        l.name AS league_name
                    FROM pick_alerts pa
                    LEFT JOIN predictions p ON p.id = pa.prediction_id
                    JOIN matches m ON m.id = pa.match_id
                    JOIN teams h ON h.id = m.home_team_id
                    JOIN teams a ON a.id = m.away_team_id
                    LEFT JOIN leagues l ON l.id = m.league_id
                    WHERE pa.id = :bid
                    """
                ),
                {"bid": bet_id},
            )
        ).first()
        if not row:
            return None
        detail = dict(row._mapping)

        # Odds del mismo market+line del pick (evita mezclar h2h con spreads/totals)
        odds_rows = (
            await s.execute(
                text(
                    """
                    SELECT market, outcome, bookmaker, odds, line, ts
                    FROM odds_history
                    WHERE match_id = :mid
                      AND market = :mk
                      AND (line = :ln OR (:ln IS NULL AND line IS NULL))
                    ORDER BY ts DESC LIMIT 40
                    """
                ),
                {
                    "mid": detail["match_id"],
                    "mk": detail.get("market"),
                    "ln": detail.get("line"),
                },
            )
        ).all()
        detail["odds_rows"] = [dict(o._mapping) for o in odds_rows]

    return detail


def _regional_recommendation(odds_rows: list[dict[str, Any]], outcome: str) -> dict[str, Any]:
    """Best MX vs US para el outcome apostado, con rationale."""
    from apuestas.betting.regional import MX_BOOKS, US_BOOKS

    matching = [o for o in odds_rows if str(o["outcome"]).lower() == outcome.lower()]
    mx_offers = [(o["bookmaker"], float(o["odds"])) for o in matching if o["bookmaker"] in MX_BOOKS]
    us_offers = [(o["bookmaker"], float(o["odds"])) for o in matching if o["bookmaker"] in US_BOOKS]
    other = [
        (o["bookmaker"], float(o["odds"]))
        for o in matching
        if o["bookmaker"] not in MX_BOOKS and o["bookmaker"] not in US_BOOKS
    ]

    best_mx = max(mx_offers, key=lambda x: x[1]) if mx_offers else None
    best_us = max(us_offers, key=lambda x: x[1]) if us_offers else None
    best_other = max(other, key=lambda x: x[1]) if other else None

    return {
        "best_mx": best_mx,
        "best_us": best_us,
        "best_other": best_other,
        "n_mx": len(mx_offers),
        "n_us": len(us_offers),
        "n_other": len(other),
    }


def _hedge_suggestion(d: dict[str, Any]) -> str | None:
    """Sugiere hedge/cash-out según movimiento de línea post-pick.

    Heurística:
    - Compara odds al momento de la bet vs odds actuales de mismo outcome.
    - Si las odds bajaron ≥5% (mercado cree MÁS en el pick) → sugiere
      hedge parcial en el lado contrario para lock-in parcial de profit.
    - Si las odds subieron ≥7% (mercado cree MENOS) → warning, considera
      revisar el pick antes del kickoff.
    """
    odds_placed = float(d.get("odds_placed") or 0)
    outcome = str(d.get("outcome") or "").lower()
    odds_rows = d.get("odds_rows") or []
    if odds_placed <= 1.0 or not odds_rows:
        return None

    current_same = [
        float(o["odds"]) for o in odds_rows if str(o.get("outcome", "")).lower() == outcome
    ]
    if not current_same:
        return None
    current_best = max(current_same)
    delta_pct = (current_best - odds_placed) / odds_placed

    lines: list[str] = []
    if delta_pct <= -0.05:
        move_pct = abs(delta_pct) * 100
        lines.append(
            f"[green]✓ LINE MOVEMENT A FAVOR:[/] odds de {odds_placed:.2f} → "
            f"{current_best:.2f} ([b]-{move_pct:.1f}%[/])."
        )
        lines.append("  El mercado cree MÁS en tu pick. Opciones:")
        lines.append("    [b]1.[/] Dejar correr y esperar resultado (máximo upside).")
        lines.append(
            f"    [b]2.[/] [cyan]Hedge parcial[/] contra-apostando el otro lado "
            f"(lock-in ~{move_pct * 0.5:.1f}% del stake)."
        )
        lines.append("    [b]3.[/] [dim]Cash-out en la casa[/] si ofrece esa opción.")
    elif delta_pct >= 0.07:
        move_pct = delta_pct * 100
        lines.append(
            f"[red]⚠ LINE MOVEMENT EN CONTRA:[/] odds de {odds_placed:.2f} → "
            f"{current_best:.2f} ([b]+{move_pct:.1f}%[/])."
        )
        lines.append("  El mercado se mueve en contra. Revisa antes del kickoff:")
        lines.append("    • ¿Hay lesión/noticia reciente no capturada por el LLM?")
        lines.append("    • ¿El análisis LLM/cuba-memorys sigue siendo válido?")
        lines.append("    • Considera no añadir stake adicional a este pick.")
    elif abs(delta_pct) >= 0.02:
        lines.append(
            f"[dim]ℹ Movimiento menor: {odds_placed:.2f} → {current_best:.2f} "
            f"({delta_pct:+.1%}). Sin acción sugerida.[/]"
        )
    else:
        return None

    return "\n".join(lines)


class PickDetailScreen(ModalScreen[None]):
    """Overlay con TODO el detalle de un pick: qué, dónde, por qué, SHAP, LLM."""

    BINDINGS = [
        Binding("escape", "dismiss", "Cerrar"),
        Binding("q", "dismiss", "Cerrar"),
    ]

    def __init__(self, bet_id: int) -> None:
        super().__init__()
        self._bet_id = bet_id

    def compose(self) -> ComposeResult:
        yield VerticalScroll(Static(id="detail_body"), id="detail_box")

    async def on_mount(self) -> None:
        body = self.query_one("#detail_body", Static)
        body.update(Panel(Text("\n  ⏳ Cargando detalle...\n"), border_style="cyan"))
        try:
            detail = await _fetch_pick_detail(self._bet_id)
        except Exception as exc:
            body.update(Panel(f"[red]Error: {exc!s}[/]", border_style="red"))
            return
        if detail is None:
            body.update(Panel(f"[red]Bet #{self._bet_id} no encontrada[/]", border_style="red"))
            return
        body.update(self._render_detail(detail))

    @staticmethod
    def _render_detail(d: dict[str, Any]) -> Panel:
        home = d.get("home_name") or "?"
        away = d.get("away_name") or "?"
        start = d.get("start_time")
        start_str = start.strftime("%a %d-%b %H:%M UTC") if start else "?"
        league = d.get("league_name") or d.get("sport_code") or "?"
        outcome = str(d.get("outcome") or "?")
        market = str(d.get("market") or "?")
        odds = float(d.get("odds_placed") or 0)
        stake = float(d.get("stake_units") or 0)
        p_model = float(d.get("p_model") or 0)
        p_lower = float(d.get("p_lower") or 0)
        p_upper = float(d.get("p_upper") or 0)
        ev = float(d.get("ev") or 0)
        kelly = float(d.get("kelly_fraction") or 0)
        book_placed = d.get("bookmaker") or "?"

        outcome_label = outcome
        if outcome.lower() == "home":
            outcome_label = f"🏠 {home}"
        elif outcome.lower() == "away":
            outcome_label = f"🛫 {away}"
        elif outcome.lower() == "draw":
            outcome_label = "🤝 Empate"

        text = Text()

        # ── Header del pick ──
        text.append("\n  🎯 Partido:  ", style="bold cyan")
        text.append(f"{home}  vs  {away}\n", style="bold white")
        text.append(f"     Liga:     {league}   ·   Kickoff: {start_str}\n", style="white")
        if d.get("match_status") == "finished":
            hs, as_ = d.get("home_score"), d.get("away_score")
            text.append(f"     Resultado final: [b yellow]{hs} - {as_}[/]\n", style="")

        # ── QUÉ APOSTAR ──
        text.append("\n  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n", style="green")
        text.append("  ┃  🎲 RECOMENDACIÓN                                ┃\n", style="bold green")
        text.append("  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n", style="green")
        text.append(f"\n     Mercado: [b]{market}[/]   ·   Pick: [b green]{outcome_label}[/]\n")
        text.append(f"     Odds:    [b]{odds:.2f}[/]   ·   Stake: [b]{stake:.2f}u[/] (Kelly ¼)\n")
        text.append(
            f"     EV:      [b {'green' if ev > 0 else 'red'}]{ev:+.3%}[/]   ·   "
            f"Kelly fraction: [b]{kelly:.2%}[/]\n"
        )

        # ── DÓNDE APOSTAR ──
        regional = _regional_recommendation(d.get("odds_rows") or [], outcome)
        text.append("\n  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n", style="cyan")
        text.append("  ┃  🏪 DÓNDE APOSTAR (line shopping regional)       ┃\n", style="bold cyan")
        text.append("  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n", style="cyan")
        if regional["best_mx"]:
            mx_book, mx_odds = regional["best_mx"]
            mx_ev = (p_model * mx_odds) - 1
            text.append("\n     🇲🇽 [b]Mejor MX:[/] ")
            text.append(f"{mx_book}", style="bold green")
            text.append(f"  @ [b]{mx_odds:.2f}[/]   ")
            text.append(f"EV [{'green' if mx_ev > 0 else 'red'}]{mx_ev:+.2%}[/]")
            text.append(f"   ({regional['n_mx']} casas MX comparadas)\n", style="dim")
        else:
            text.append("\n     🇲🇽 Sin oferta en casas MX del catálogo SEGOB.\n", style="yellow")
        if regional["best_us"]:
            us_book, us_odds = regional["best_us"]
            us_ev = (p_model * us_odds) - 1
            text.append("     🇺🇸 [b]Mejor US:[/] ")
            text.append(f"{us_book}", style="bold green")
            text.append(f"  @ [b]{us_odds:.2f}[/]   ")
            text.append(f"EV [{'green' if us_ev > 0 else 'red'}]{us_ev:+.2%}[/]")
            text.append(f"   ({regional['n_us']} casas US comparadas)\n", style="dim")
        else:
            text.append("     🇺🇸 Sin oferta en casas US reguladas.\n", style="yellow")
        if regional["best_other"]:
            ot_book, ot_odds = regional["best_other"]
            text.append(
                f"     🌐 Otras (offshore/EU): {ot_book} @ {ot_odds:.2f} "
                f"[dim]— solo referencia, no accesible desde MX[/]\n"
            )
        text.append(f"\n     📝 [b]Apostado en:[/] {book_placed} @ {odds:.2f}\n", style="")

        # ── POSIBILIDADES ──
        text.append("\n  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n", style="blue")
        text.append("  ┃  📊 POSIBILIDADES (probabilidades calculadas)     ┃\n", style="bold blue")
        text.append("  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n", style="blue")
        text.append(f"\n     P(ganar) según modelo: [b blue]{p_model:.1%}[/]\n")
        if p_lower and p_upper:
            text.append(f"     Intervalo conformal 90% (MAPIE): [{p_lower:.1%}, {p_upper:.1%}]\n")
        implied = 1.0 / odds if odds > 0 else 0
        text.append(f"     Probabilidad implícita en odds: {implied:.1%}\n")
        edge = p_model - implied
        text.append(f"     Edge bruto:  [b {'green' if edge > 0 else 'red'}]{edge:+.2%}[/]")
        text.append("   [dim](diferencia entre P modelo y P implícita)[/]\n")

        # ── ANÁLISIS LLM ──
        llm = d.get("llm_analysis") or {}
        if isinstance(llm, dict) and llm:
            text.append("\n  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n", style="magenta")
            text.append(
                "  ┃  🧠 ANÁLISIS LLM (DeepSeek V3.2)                  ┃\n", style="bold magenta"
            )
            text.append("  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n", style="magenta")
            for team_key, team_label in (
                ("home_team_analysis", f"🏠 LOCAL ({home})"),
                ("away_team_analysis", f"🛫 VISITANTE ({away})"),
            ):
                team = llm.get(team_key, {})
                if not isinstance(team, dict):
                    continue
                text.append(f"\n     [b]{team_label}[/]\n")
                for key, label in (
                    ("key_injuries", "Lesiones"),
                    ("lineup_changes", "Cambios lineup"),
                    ("recent_transfers_impact", "Fichajes impacto"),
                    ("narrative_momentum", "Momentum"),
                ):
                    val = team.get(key)
                    if isinstance(val, list) and val:
                        items = [str(x)[:60] for x in val[:3]]
                        text.append(f"       • {label}: {', '.join(items)}\n", style="white")
                    elif isinstance(val, (str, int, float)) and val:
                        text.append(f"       • {label}: {val}\n", style="white")
                rest = team.get("rest_days")
                b2b = team.get("back_to_back")
                if rest is not None:
                    text.append(f"       • Rest: {rest} días", style="white")
                    if b2b:
                        text.append(" ⚠️ back-to-back", style="yellow")
                    text.append("\n")

            mctx = llm.get("matchup_context") or {}
            if isinstance(mctx, dict) and mctx:
                text.append("\n     [b]⚔ Matchup[/]\n")
                for key, label in (
                    ("h2h_recent", "H2H reciente"),
                    ("home_advantage_estimate", "Home advantage"),
                    ("weather_if_outdoor", "Clima"),
                    ("referee_or_umpire_notes", "Árbitro"),
                ):
                    val = mctx.get(key)
                    if val:
                        text.append(f"       • {label}: {str(val)[:100]}\n", style="white")

            summary = llm.get("summary_es") or llm.get("summary")
            if summary:
                text.append(f"\n     [b italic]Resumen:[/] {str(summary)[:400]}\n", style="white")
            line_mov = llm.get("line_movement_assessment")
            if line_mov:
                text.append(f"     Line movement: [b]{line_mov}[/]\n", style="white")
            conf = llm.get("confidence_in_analysis")
            if conf:
                conf_color = {"high": "green", "medium": "yellow", "low": "red"}.get(
                    str(conf).lower(), "white"
                )
                text.append(f"     Confianza del análisis: [{conf_color}]{conf}[/]\n")

        # ── EN QUÉ SE BASÓ ──
        text.append("\n  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n", style="yellow")
        text.append(
            "  ┃  📐 EN QUÉ SE BASÓ (evidencia cuantitativa)       ┃\n", style="bold yellow"
        )
        text.append("  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n", style="yellow")
        odds_total = len(d.get("odds_rows") or [])
        text.append(f"\n     • De-vigging Shin sobre {odds_total} snapshots de odds\n")
        text.append(
            f"     • Modelo: {d.get('model_name') or 'consensus'} "
            f"v{d.get('model_version') or '?'}\n"
        )

        shap = d.get("shap_top5")
        if isinstance(shap, list) and shap:
            text.append("\n     [b]Top features SHAP (por qué el modelo dijo esto):[/]\n")
            for item in shap[:5]:
                if isinstance(item, dict):
                    feat = item.get("feature", "?")
                    value = item.get("value") or item.get("shap_value")
                    sign = "↑" if (value or 0) > 0 else "↓"
                    text.append(
                        f"       {sign} {feat}: {value:+.4f}\n"
                        if value is not None
                        else f"       • {feat}\n",
                        style="white",
                    )

        feats = d.get("features_snapshot")
        if isinstance(feats, dict) and feats:
            text.append(
                f"     [dim]• {len(feats)} features totales usadas (home/away/diff/matchup)[/]\n"
            )

        # Memoria inyectada
        memory_note = llm.get("memory_context_used") if isinstance(llm, dict) else None
        if memory_note:
            text.append("\n     [b]Memoria cuba-memorys inyectada:[/]\n")
            text.append(f"       {str(memory_note)[:300]}\n", style="dim")

        # ── FEATURES TIER A (referee bias + coaching + steam moves) ──
        feats = d.get("features_snapshot") or {}
        tier_a_keys = [
            k
            for k in (feats.keys() if isinstance(feats, dict) else [])
            if k.startswith(("ref_", "coach_", "active_steam"))
        ]
        if tier_a_keys:
            text.append(
                "\n  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n",
                style="yellow",
            )
            text.append(
                "  ┃  🎯 TIER A — Referee · Coaching · Steam moves       ┃\n",
                style="bold yellow",
            )
            text.append(
                "  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n",
                style="yellow",
            )
            ref_keys = [k for k in tier_a_keys if k.startswith("ref_")]
            coach_keys = [k for k in tier_a_keys if k.startswith("coach_")]
            steam_keys = [k for k in tier_a_keys if k.startswith("active_steam")]
            if ref_keys:
                text.append("\n     🏁 [b]Referee bias[/] (Voulgaris: ~25% del edge):\n")
                for k in ref_keys[:8]:
                    text.append(f"       • {k}: {feats[k]}\n", style="white")
            if coach_keys:
                text.append("\n     📋 [b]Coaching tendencies (clutch)[/]:\n")
                for k in coach_keys[:8]:
                    text.append(f"       • {k}: {feats[k]}\n", style="white")
            if steam_keys:
                text.append("\n     🚂 [b]Steam moves activos[/]:\n")
                for k in steam_keys:
                    text.append(f"       • {k}: {feats[k]}\n", style="cyan")

        # ── HEDGE / CASH-OUT SUGERIDO ──
        hedge_hint = _hedge_suggestion(d)
        if hedge_hint:
            text.append("\n  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n", style="cyan")
            text.append(
                "  ┃  🛡  HEDGE / CASH-OUT SUGERIDO                    ┃\n", style="bold cyan"
            )
            text.append("  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n", style="cyan")
            for line in hedge_hint.split("\n"):
                text.append(f"\n     {line}", style="white")
            text.append("\n")

        # ── WARNINGS ──
        warns: list[str] = []
        if isinstance(llm, dict):
            wf = llm.get("warning_flags")
            if isinstance(wf, list):
                warns.extend(str(w) for w in wf)
        if stake > 5.0:
            warns.append(f"Stake {stake:.2f}u es >5% (cap Kelly violado)")
        if d.get("is_paper"):
            warns.append("Esta es una PAPER bet (virtual, no real)")
        if warns:
            text.append("\n  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n", style="red")
            text.append(
                "  ┃  ⚠️  WARNINGS                                     ┃\n", style="bold red"
            )
            text.append("  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n", style="red")
            for w in warns[:6]:
                text.append(f"\n     • {str(w)[:100]}\n", style="yellow")

        text.append("\n  Presiona ESC o q para cerrar\n\n", style="dim italic")

        return Panel(
            text,
            title=f"[b]🎯 Pick #{d['bet_id']} — detalle completo[/]",
            border_style="cyan",
            padding=(0, 1),
        )

    def action_dismiss(self, result: None = None) -> None:
        self.app.pop_screen()


# ════════════════════════════ Data fetchers ══════════════════════════════


async def fetch_system_status() -> dict[str, Any]:
    """Healthcheck de BD + LLM + cuba-memorys + créditos API."""
    import os

    result = {"bd_ok": False, "llm_ok": False, "mem_ok": False, "api_credits": -1}

    try:
        async with session_scope() as s:
            await s.execute(text("SELECT 1"))
        result["bd_ok"] = True
    except Exception:
        pass

    try:
        from apuestas.mcp.client import MCPClient

        client = MCPClient.get()
        if client.is_connected("memorys"):
            result["mem_ok"] = True
    except Exception:
        pass

    # LLM status: si hay DEEPSEEK_API_KEY configurada, asumimos disponible
    if os.environ.get("DEEPSEEK_API_KEY", "").startswith("sk-"):
        result["llm_ok"] = True

    # API credits: leer de Valkey cache (último valor visto) o -1
    try:
        import redis.asyncio as aioredis

        url = os.environ.get("VALKEY_URL", "")
        if url:
            r = aioredis.from_url(url, socket_timeout=1, decode_responses=True)
            val = await r.get("odds_api_credits_remaining")
            await r.aclose()
            if val is not None:
                result["api_credits"] = int(val)
    except Exception:
        pass

    return result


async def fetch_dashboard_data() -> dict[str, Any]:
    """Estadísticas del Dashboard adaptadas al modo detector puro.

    Post-pivote 2026-04-23: sin bankroll, sin PnL, sin CLV. Usa `pick_alerts`
    y `outcome_result` para mostrar hit-rate, # alertas vivas/resueltas, y
    % positivo histórico. `bankroll`/`roi_7d`/`clv_7d` quedan como 0.0 por
    compatibilidad de layout (Sprint 2 los reemplaza por Brier/BSS/ECE).
    """
    async with session_scope() as session:
        perf_row = (
            await session.execute(
                text(
                    """
                    SELECT
                        COUNT(*) FILTER (
                            WHERE outcome_result IS NULL OR outcome_result = 'pending'
                        ) AS active_picks,
                        COUNT(*) FILTER (WHERE outcome_result = 'won') AS wins,
                        COUNT(*) FILTER (
                            WHERE outcome_result IN ('won','lost')
                        ) AS settled,
                        COUNT(*) FILTER (
                            WHERE outcome_result = 'won'
                              AND result_settled_at >= NOW() - INTERVAL '7 days'
                        ) AS wins_7d,
                        COUNT(*) FILTER (
                            WHERE outcome_result IN ('won','lost')
                              AND result_settled_at >= NOW() - INTERVAL '7 days'
                        ) AS settled_7d
                    FROM pick_alerts
                    """
                )
            )
        ).first()
        paused_row = (
            await session.execute(text("SELECT value FROM bot_state WHERE key = 'paused'"))
        ).first()
        events = (
            await session.execute(
                text(
                    """
                    SELECT m.id, m.start_time, m.sport_code,
                           h.name AS home_name, a.name AS away_name,
                           (SELECT COUNT(*) FROM odds_history o WHERE o.match_id = m.id) AS n_odds
                    FROM matches m
                    JOIN teams h ON h.id = m.home_team_id
                    JOIN teams a ON a.id = m.away_team_id
                    WHERE m.status = 'scheduled'
                      AND m.start_time BETWEEN NOW() AND NOW() + INTERVAL '48 hours'
                    ORDER BY m.start_time ASC LIMIT 15
                    """
                )
            )
        ).all()
        picks = (
            await session.execute(
                text(
                    """
                    SELECT pa.id, pa.market, pa.outcome, pa.odds_placed,
                           pa.bookmaker, p.ev,
                           h.name AS home_name, a.name AS away_name,
                           m.start_time
                    FROM pick_alerts pa
                    LEFT JOIN predictions p ON p.id = pa.prediction_id
                    JOIN matches m ON m.id = pa.match_id
                    JOIN teams h ON h.id = m.home_team_id
                    JOIN teams a ON a.id = m.away_team_id
                    WHERE pa.outcome_result IS NULL OR pa.outcome_result = 'pending'
                    ORDER BY m.start_time ASC LIMIT 15
                    """
                )
            )
        ).all()
        total_matches_row = (await session.execute(text("SELECT COUNT(*) FROM matches"))).first()

    active = int(perf_row.active_picks or 0) if perf_row else 0
    wins = int(perf_row.wins or 0) if perf_row else 0
    settled = int(perf_row.settled or 0) if perf_row else 0
    wins_7d = int(perf_row.wins_7d or 0) if perf_row else 0
    settled_7d = int(perf_row.settled_7d or 0) if perf_row else 0
    hit_rate_7d = wins_7d / settled_7d if settled_7d > 0 else 0.0
    paused = False
    if paused_row and paused_row.value:
        val = paused_row.value
        if isinstance(val, dict):
            paused = bool(val.get("paused", False))
        elif isinstance(val, str):
            paused = "true" in val.lower()
    total_matches = int(total_matches_row[0]) if total_matches_row else 0
    return {
        "bankroll": 0.0,  # retirado — placeholder para el card legacy
        "roi_7d": 0.0,
        "clv_7d": 0.0,
        "active_picks": active,
        "hit_rate": wins / settled if settled > 0 else 0.0,
        "hit_rate_7d": hit_rate_7d,
        "settled_count": settled,
        "paused": paused,
        "events": [dict(e._mapping) for e in events],
        "picks": [dict(p._mapping) for p in picks],
        "total_matches": total_matches,
        "is_first_run": total_matches == 0 and settled == 0,
    }


# ═══════════════════════════ Screens ═════════════════════════════════════


class DashboardScreen(VerticalScroll):
    """Overview: status + 4 métricas + eventos + picks (o welcome si primera vez)."""

    BINDINGS = [
        Binding("a", "analyze", "Analizar"),
        Binding("r", "refresh", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        with Grid(id="cards_grid"):
            yield MetricCard(
                label="Bankroll",
                icon="💰",
                color="cyan",
                hint="capital virtual actual",
                id="card_bankroll",
            )
            yield MetricCard(
                label="ROI 7d",
                icon="📊",
                color="green",
                hint="return on investment",
                id="card_roi",
            )
            yield MetricCard(
                label="CLV 7d",
                icon="📈",
                color="blue",
                hint="closing line value",
                id="card_clv",
            )
            yield MetricCard(
                label="Picks activos",
                icon="🎯",
                color="magenta",
                hint="posiciones abiertas",
                id="card_picks",
            )
        yield Container(id="welcome_slot")
        yield Label("[b cyan]📅 Próximos eventos — 48 h[/]", classes="section")
        yield DataTable(id="events_table", show_header=True, zebra_stripes=True, cursor_type="row")
        yield Container(id="events_empty_slot")
        yield Label("[b cyan]🎯 Picks activos (pending)[/]", classes="section")
        yield Static(
            "  [dim]Presiona[/] [b]Enter[/] [dim]sobre un pick para ver análisis completo: "
            "dónde apostar, cuánto stake, factores LLM, SHAP, line movement, hedge sugerido.[/]"
        )
        yield DataTable(id="picks_table", show_header=True, zebra_stripes=True, cursor_type="row")
        yield Container(id="picks_empty_slot")

    async def on_mount(self) -> None:
        ev_tbl = self.query_one("#events_table", DataTable)
        ev_tbl.add_columns("Hora", "Sport", "Home", "Away", "#Odds")
        pk_tbl = self.query_one("#picks_table", DataTable)
        pk_tbl.add_columns("Match", "Hora", "Mercado", "Pick", "Stake", "Odds", "Book", "EV%")
        # Mapeo row_key → bet_id para que Enter abra el detalle
        self._pick_row_to_bet: dict[Any, int] = {}
        await self.refresh_data()
        self.set_interval(60.0, self.refresh_data)

    async def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Enter sobre una fila de la tabla picks abre PickDetailScreen."""
        if event.data_table.id != "picks_table":
            return
        bet_id = self._pick_row_to_bet.get(event.row_key)
        if bet_id is not None:
            await self.app.push_screen(PickDetailScreen(bet_id))

    async def refresh_data(self) -> None:
        try:
            data = await fetch_dashboard_data()
        except Exception as exc:
            logger.warning("tui.dashboard_fail", error=str(exc))
            self.app.notify(f"⚠ Error: {exc!s:.80}", severity="error")
            return

        # Métricas
        paused_tag = " [red b]⏸ PAUSED[/]" if data["paused"] else ""
        self.query_one("#card_bankroll", MetricCard).update_metric(
            value=f"{data['bankroll']:.2f}u",
            delta=f"hit rate {data['hit_rate']:.0%} · {data['settled_count']} settled{paused_tag}",
            color="yellow" if data["paused"] else "cyan",
        )
        self.query_one("#card_roi", MetricCard).update_metric(
            value=f"{data['roi_7d']:+.2%}",
            color="green" if data["roi_7d"] >= 0 else "red",
        )
        self.query_one("#card_clv", MetricCard).update_metric(
            value=f"{data['clv_7d']:+.3%}",
            color="green" if data["clv_7d"] >= 0 else "red",
        )
        self.query_one("#card_picks", MetricCard).update_metric(
            value=str(data["active_picks"]),
            color="magenta" if data["active_picks"] > 0 else "dim",
        )

        # Welcome card si primera vez
        welcome_slot = self.query_one("#welcome_slot", Container)
        await welcome_slot.remove_children()
        if data["is_first_run"]:
            await welcome_slot.mount(WelcomeCard())

        # Tabla eventos
        ev_tbl = self.query_one("#events_table", DataTable)
        ev_empty = self.query_one("#events_empty_slot", Container)
        await ev_empty.remove_children()
        ev_tbl.clear()
        if not data["events"]:
            ev_tbl.display = False
            await ev_empty.mount(
                EmptyState(
                    icon="📭",
                    title="Sin eventos próximos 48h",
                    description=(
                        "No hay partidos ingestados todavía.\n"
                        "El pipeline trae fixtures de football-data.org + The Odds API."
                    ),
                    cta="Presiona [ A ] para ingesta + análisis automático",
                )
            )
        else:
            ev_tbl.display = True
            for e in data["events"]:
                dt = e["start_time"]
                hora = dt.strftime("%a %H:%M") if isinstance(dt, datetime) else str(dt)[:16]
                ev_tbl.add_row(
                    hora,
                    (e["sport_code"] or "?").upper(),
                    (e["home_name"] or "?")[:24],
                    (e["away_name"] or "?")[:24],
                    str(e["n_odds"] or 0),
                )

        # Tabla picks
        pk_tbl = self.query_one("#picks_table", DataTable)
        pk_empty = self.query_one("#picks_empty_slot", Container)
        await pk_empty.remove_children()
        pk_tbl.clear()
        self._pick_row_to_bet = {}
        if not data["picks"]:
            pk_tbl.display = False
            await pk_empty.mount(
                EmptyState(
                    icon="🎯",
                    title="Sin picks activos",
                    description=(
                        "El bot solo emite picks con EV ≥ 3% post de-vigging Shin.\n"
                        "Si no hay edge real en el mercado, no apuesta (eso es bueno).\n"
                        "Cuando haya pick, aparecerá aquí con stake Kelly ¼ sugerido."
                    ),
                    cta="Presiona [ A ] para escanear + emitir picks con edge",
                )
            )
        else:
            pk_tbl.display = True
            for p in data["picks"]:
                dt = p["start_time"]
                hora = dt.strftime("%d %b %H:%M") if isinstance(dt, datetime) else str(dt)[:16]
                ev_val = float(p.get("ev") or 0) * 100
                match_str = f"{(p['home_name'] or '?')[:14]} vs {(p['away_name'] or '?')[:14]}"
                row_key = pk_tbl.add_row(
                    match_str,
                    hora,
                    str(p["market"])[:10],
                    str(p["outcome"])[:12],
                    "—",  # stake retirado en pivote detector puro
                    f"{float(p['odds_placed']):.2f}",
                    (p["bookmaker"] or "?")[:10],
                    f"{ev_val:+.2f}",
                )
                self._pick_row_to_bet[row_key] = int(p["id"])
            # Hint educativo: cómo abrir detalle
            await pk_empty.mount(
                Static(
                    "[dim italic]   💡 Presiona [b][ Enter ][/b] sobre un pick para ver "
                    "el análisis completo: dónde apostar MX/US, por qué lo recomienda, "
                    "features SHAP, narrativa LLM y warnings.[/]"
                )
            )

    async def action_refresh(self) -> None:
        await self.refresh_data()

    async def action_analyze(self) -> None:
        self.app.notify(
            "⏳ Ingesta football-data + odds + análisis DeepSeek...",
            timeout=3,
            severity="information",
        )
        asyncio.create_task(self._run_analyze())

    async def _run_analyze(self) -> None:
        try:
            from apuestas.flows.deep_analysis import deep_analysis_flow

            summary = await deep_analysis_flow(hours_ahead=48, max_events=30)
            picks = summary.get("picks_emitted", 0) if isinstance(summary, dict) else 0
            events = summary.get("events_checked", 0) if isinstance(summary, dict) else 0
            self.app.notify(
                f"✅ {picks} picks emitidos ({events} eventos analizados)",
                severity="information",
                timeout=6,
            )
            await self.refresh_data()
        except Exception as exc:
            logger.exception("tui.analyze_fail", error=str(exc))
            self.app.notify(f"❌ Análisis falló: {exc!s:.100}", severity="error", timeout=10)


class PostMortemsScreen(VerticalScroll):
    BINDINGS = [Binding("r", "refresh", "Refresh")]

    def compose(self) -> ComposeResult:
        yield Label(
            "[b cyan]🔍 Post-mortems[/] — picks liquidadas ordenadas por discrepancia",
            classes="section",
        )
        yield DataTable(id="pm_table", show_header=True, zebra_stripes=True, cursor_type="row")
        yield Container(id="pm_empty_slot")

    async def on_mount(self) -> None:
        tbl = self.query_one("#pm_table", DataTable)
        tbl.add_columns("Pick", "Outcome", "Discrepancy", "Resultado", "Lección")
        await self.refresh_data()

    async def refresh_data(self) -> None:
        """Lee `pick_analysis` (ex-post_mortems sin columnas monetarias).

        Post-pivote: reemplaza PnL con outcome_result (won/lost/void/expired).
        Sprint 2 añade Brier por pick + SHAP top-5 al final de la fila.
        """
        try:
            async with session_scope() as s:
                rows = (
                    await s.execute(
                        text(
                            """
                            SELECT pa_meta.pick_alert_id, pa_meta.outcome,
                                   pa_meta.discrepancy_score,
                                   pa_meta.narrative->>'transferable_lesson' AS lesson,
                                   pa.outcome_result
                            FROM pick_analysis pa_meta
                            JOIN pick_alerts pa ON pa.id = pa_meta.pick_alert_id
                            WHERE pa_meta.discrepancy_score IS NOT NULL
                            ORDER BY pa_meta.discrepancy_score DESC NULLS LAST LIMIT 20
                            """
                        )
                    )
                ).all()
        except Exception as exc:
            self.app.notify(f"⚠ {exc!s:.80}", severity="error")
            return

        tbl = self.query_one("#pm_table", DataTable)
        empty_slot = self.query_one("#pm_empty_slot", Container)
        tbl.clear()
        await empty_slot.remove_children()
        if not rows:
            tbl.display = False
            await empty_slot.mount(
                EmptyState(
                    icon="🔬",
                    title="Sin análisis de picks todavía",
                    description=(
                        "Se generan cuando una alerta se resuelve (won/lost).\n"
                        "Cada análisis incluye discrepancia modelo/realidad + SHAP.\n"
                        "Sprint 2 añade Brier + BSS + ECE."
                    ),
                    cta="Corre: apuestas analyze + apuestas live-scores",
                )
            )
            return
        tbl.display = True
        result_color = {
            "won": "green",
            "lost": "red",
            "void": "dim",
            "halfwon": "cyan",
            "halflost": "yellow",
            "expired": "dim",
            None: "yellow",
        }
        for r in rows:
            disc = float(r.discrepancy_score or 0)
            disc_color = "green" if disc < 0.2 else "yellow" if disc < 0.5 else "red"
            res = r.outcome_result or "pending"
            rcol = result_color.get(r.outcome_result, "yellow")
            tbl.add_row(
                f"#{r.pick_alert_id}",
                str(r.outcome or "-"),
                f"[{disc_color}]{disc:.3f}[/]",
                f"[{rcol}]{res}[/]",
                (r.lesson or "(sin lección)")[:80],
            )

    async def action_refresh(self) -> None:
        await self.refresh_data()


class BankrollScreen(VerticalScroll):
    """Retirada en pivote detector puro (2026-04-23).

    Se mantiene la clase importable para no romper imports legacy; el tab no
    se monta en el TabbedContent principal. Sprint 2 la elimina totalmente.
    """

    BINDINGS: list[Binding] = []

    def compose(self) -> ComposeResult:
        yield Label(
            "[b yellow]💰 Bankroll — retirado[/]  "
            "[dim](pivote 2026-04-23: el bot ya no gestiona saldo)[/]",
            classes="section",
        )
        yield Static(
            "El bot pasó a modo [b]detector puro[/] — emite alertas de valor,\n"
            "no administra banca, stake ni PnL. Consulta /picks en Telegram\n"
            "o la pestaña Dashboard para ver alertas activas."
        )

    async def refresh_data(self) -> None:
        return

    async def action_refresh(self) -> None:
        return


class DriftScreen(VerticalScroll):
    """Modelos ML — FIX: usa schema real `model_registry_meta`."""

    BINDINGS = [Binding("r", "refresh", "Refresh")]

    def compose(self) -> ComposeResult:
        yield Label("[b cyan]🤖 Modelos ML[/] — production vs shadow + drift", classes="section")
        yield DataTable(id="models_table", show_header=True, zebra_stripes=True, cursor_type="row")
        yield Container(id="dr_empty_slot")

    async def on_mount(self) -> None:
        tbl = self.query_one("#models_table", DataTable)
        tbl.add_columns("Sport", "Modelo", "Version", "Stage", "Drift", "Promoted")
        await self.refresh_data()

    async def refresh_data(self) -> None:
        rows: list[Any] = []
        try:
            async with session_scope() as s:
                rows = (
                    await s.execute(
                        text(
                            """
                            SELECT sport_code, model_name, model_version, stage,
                                   drift_status, promoted_at
                            FROM model_registry_meta
                            ORDER BY promoted_at DESC NULLS LAST LIMIT 20
                            """
                        )
                    )
                ).all()
        except Exception as exc:
            # Fallback graceful: la tabla puede estar vacía o tener schema distinto
            logger.debug("tui.models_query_fail", error=str(exc))

        tbl = self.query_one("#models_table", DataTable)
        empty_slot = self.query_one("#dr_empty_slot", Container)
        tbl.clear()
        await empty_slot.remove_children()
        if not rows:
            tbl.display = False
            await empty_slot.mount(
                EmptyState(
                    icon="🤖",
                    title="Sin modelos registrados",
                    description=(
                        "Los modelos ML se registran al entrenar con MLflow.\n"
                        "Champion (production) emite picks · Shadow corre en paralelo.\n"
                        "Drift status: ok / warning / critical (PSI + CBPE).\n\n"
                        "Promote automático cuando shadow.CLV > champion.CLV + 0.5%."
                    ),
                    cta="Entrena el primero con: make retrain SPORT=nba",
                )
            )
            return
        tbl.display = True
        for r in rows:
            drift = str(r.drift_status or "unknown").lower()
            drift_color = {"ok": "green", "warning": "yellow", "critical": "red"}.get(drift, "dim")
            stage_color = {
                "production": "green",
                "shadow": "yellow",
                "archived": "dim",
            }.get(str(r.stage), "white")
            tbl.add_row(
                str(r.sport_code),
                str(r.model_name)[:20],
                str(r.model_version)[:10],
                f"[{stage_color}]{r.stage}[/]",
                f"[{drift_color}]{drift}[/]",
                r.promoted_at.strftime("%Y-%m-%d") if r.promoted_at else "-",
            )

    async def action_refresh(self) -> None:
        await self.refresh_data()


class CalibrationScreen(VerticalScroll):
    BINDINGS = [Binding("r", "refresh", "Refresh")]

    def compose(self) -> ComposeResult:
        yield Label(
            "[b cyan]🎯 Calibración[/] — KPIs primarios + gap predicho vs real",
            classes="section",
        )
        # Sprint 4d — tabla nueva de KPIs primarios por deporte
        yield Label("[dim]KPIs 30d por deporte (Brier ≤ 0.22 · BSS ≥ 0.03 · ECE ≤ 0.05)[/]")
        yield DataTable(id="kpi_table", show_header=True, zebra_stripes=True, cursor_type="row")
        yield Label("\n[dim]Detalle por bucket (legacy calibration_rolling)[/]")
        yield DataTable(id="cal_table", show_header=True, zebra_stripes=True, cursor_type="row")
        yield Container(id="cal_empty_slot")

    async def on_mount(self) -> None:
        kpi = self.query_one("#kpi_table", DataTable)
        kpi.add_columns("Sport", "N", "Brier", "BSS", "ECE", "Hit", "HR-impl", "Status")
        tbl = self.query_one("#cal_table", DataTable)
        tbl.add_columns("Sport", "Market", "Bucket", "N", "Predicho", "Real", "Gap")
        await self.refresh_data()

    async def _refresh_kpis(self) -> None:
        """Computa Brier/BSS/ECE/hit_rate desde pick_alerts + predictions.

        Usa apuestas.ml.metrics.compute_metrics sobre alertas resueltas de
        los últimos 30d por deporte. Marca PASS/FAIL según MVP thresholds.
        """
        import numpy as np

        from apuestas.ml.metrics import compute_metrics

        kpi = self.query_one("#kpi_table", DataTable)
        kpi.clear()

        try:
            async with session_scope() as s:
                rows = (
                    await s.execute(
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
        except Exception as exc:
            logger.debug("tui.kpi_query_fail", error=str(exc))
            return

        by_sport: dict[str, list[Any]] = {}
        for r in rows:
            by_sport.setdefault(str(r.sport_code or "?"), []).append(r)

        kpi_brier_cap = {"nba": 0.22, "nfl": 0.23}
        for sport, sport_rows in sorted(by_sport.items()):
            y = np.array([int(r.y) for r in sport_rows])
            p = np.array([float(r.p_model) if r.p_model is not None else 0.5 for r in sport_rows])
            odds_arr = np.array(
                [float(r.odds_placed) for r in sport_rows if r.odds_placed is not None]
            )
            avg_odds = float(odds_arr[odds_arr > 1.0].mean()) if (odds_arr > 1.0).any() else None
            m = compute_metrics(y, p, avg_odds=avg_odds)
            brier_cap = kpi_brier_cap.get(sport.lower(), 0.24)
            passes = (
                m.brier <= brier_cap
                and m.brier_skill_score >= 0.03
                and m.ece <= 0.05
                and m.hit_rate_minus_implied >= 0.02
            )
            status = "[green]PASS[/]" if passes else "[red]FAIL[/]"
            kpi.add_row(
                sport,
                str(m.n),
                f"{m.brier:.4f}",
                f"{m.brier_skill_score:+.4f}",
                f"{m.ece:.4f}",
                f"{m.hit_rate:.3f}",
                f"{m.hit_rate_minus_implied:+.3f}",
                status,
            )

    async def refresh_data(self) -> None:
        await self._refresh_kpis()
        rows: list[Any] = []
        try:
            async with session_scope() as s:
                rows = (
                    await s.execute(
                        text(
                            """
                            SELECT sport_code, market, confidence_bucket,
                                   n_predictions, mean_predicted, mean_actual, calibration_gap
                            FROM calibration_rolling
                            WHERE window_days = 30
                            ORDER BY ABS(calibration_gap) DESC NULLS LAST LIMIT 40
                            """
                        )
                    )
                ).all()
        except Exception as exc:
            logger.debug("tui.cal_query_fail", error=str(exc))

        tbl = self.query_one("#cal_table", DataTable)
        empty_slot = self.query_one("#cal_empty_slot", Container)
        tbl.clear()
        await empty_slot.remove_children()
        if not rows:
            tbl.display = False
            await empty_slot.mount(
                EmptyState(
                    icon="🎯",
                    title="Sin calibración aún",
                    description=(
                        "Se agrega cuando hay ≥30 predicciones settleadas por bucket.\n"
                        "Gap ideal: ±0.03 · aceptable <0.05 · alerta >0.05.\n\n"
                        "Un modelo bien calibrado dice 'p=0.60' y acierta ~60% del tiempo."
                    ),
                    cta="Acumula predicciones con: apuestas settle tras cada match",
                )
            )
            return
        tbl.display = True
        for r in rows:
            gap = float(r.calibration_gap or 0)
            gap_color = "green" if abs(gap) < 0.03 else "yellow" if abs(gap) < 0.05 else "red"
            tbl.add_row(
                str(r.sport_code),
                str(r.market),
                str(r.confidence_bucket),
                str(r.n_predictions),
                f"{float(r.mean_predicted or 0):.3f}",
                f"{float(r.mean_actual or 0):.3f}",
                f"[{gap_color}]{gap:+.3f}[/]",
            )

    async def action_refresh(self) -> None:
        await self.refresh_data()


class MemoryScreen(VerticalScroll):
    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("x", "scan_contra", "Contradicciones"),
        Binding("z", "analyze_gaps", "Gaps"),
    ]

    def compose(self) -> ComposeResult:
        yield Label("[b cyan]🧠 Memoria persistente (cuba-memorys)[/]", classes="section")
        yield Static(id="mem_status")
        yield DataTable(id="mem_table", show_header=True, zebra_stripes=True, cursor_type="row")
        yield Container(id="mem_empty_slot")

    async def on_mount(self) -> None:
        tbl = self.query_one("#mem_table", DataTable)
        tbl.add_columns("Tipo", "Entidad", "Contenido", "Score")
        await self.refresh_data()

    async def refresh_data(self) -> None:
        from apuestas.mcp import memory as mcp_memory
        from apuestas.mcp.client import MCPClient

        client = MCPClient.get()
        # Auto-reconectar si se desconectó
        if not client.is_connected("memorys"):
            try:
                await client.start()
            except Exception:
                pass

        connected = client.is_connected("memorys")
        status_lbl = self.query_one("#mem_status", Static)
        if connected:
            status_lbl.update(
                "[green]● CONECTADO[/]   "
                "[dim]memoria inyectada al LLM · reduce alucinaciones · "
                "registra decisions + outcomes[/]"
            )
        else:
            status_lbl.update(
                "[red]● OFFLINE[/]   [dim]análisis LLM correrán sin contexto histórico[/]"
            )

        tbl = self.query_one("#mem_table", DataTable)
        empty_slot = self.query_one("#mem_empty_slot", Container)
        tbl.clear()
        await empty_slot.remove_children()
        rows_added = 0
        try:
            faro = await mcp_memory.faro("apuestas picks bets outcomes recent", fmt="compact")
            if faro and isinstance(faro, dict):
                import json as _json

                for chunk in (faro.get("text_chunks") or [])[:4]:
                    try:
                        parsed = _json.loads(chunk) if isinstance(chunk, str) else chunk
                        results = parsed.get("results") if isinstance(parsed, dict) else []
                        for rr in (results or [])[:12]:
                            if not isinstance(rr, dict):
                                continue
                            tbl.add_row(
                                "faro",
                                str(rr.get("e") or rr.get("entity") or "-")[:22],
                                str(rr.get("c") or rr.get("content") or str(rr))[:80],
                                f"{float(rr.get('i') or 0):.2f}",
                            )
                            rows_added += 1
                    except Exception:
                        continue
        except Exception as exc:
            logger.debug("tui.memory.faro_fail", error=str(exc))

        if rows_added == 0:
            tbl.display = False
            await empty_slot.mount(
                EmptyState(
                    icon="🧠",
                    title="Sin memorias de apuestas aún",
                    description=(
                        "Cuando ejecutes un análisis (tecla A), cada decisión\n"
                        "se registra aquí con cuba_decreto. Al liquidar bets,\n"
                        "los resultados se añaden con cuba_eco.\n\n"
                        "Atajos útiles:\n"
                        "  X = scan contradicciones entre fuentes\n"
                        "  Z = gap analysis (entidades sin decisión)\n"
                        "  R = refrescar"
                    ),
                    cta="Dispara tu primer análisis con [ A ] en Dashboard",
                )
            )
        else:
            tbl.display = True

    async def action_refresh(self) -> None:
        await self.refresh_data()

    async def action_scan_contra(self) -> None:
        from apuestas.mcp import memory as mcp_memory

        self.app.notify("⏳ Escaneando contradicciones...", timeout=3)
        try:
            result = await mcp_memory.scan_contradictions()
            n = len((result or {}).get("text_chunks") or []) if result else 0
            self.app.notify(
                f"✅ {n} contradicciones encontradas",
                severity="information",
                timeout=4,
            )
            await self.refresh_data()
        except Exception as exc:
            self.app.notify(f"⚠ {exc!s:.80}", severity="warning")

    async def action_analyze_gaps(self) -> None:
        from apuestas.mcp import memory as mcp_memory

        self.app.notify("⏳ Analizando gaps de memoria...", timeout=3)
        try:
            result = await mcp_memory.analyze_gaps()
            n = len((result or {}).get("text_chunks") or []) if result else 0
            self.app.notify(f"✅ {n} gaps detectados", severity="information", timeout=4)
            await self.refresh_data()
        except Exception as exc:
            self.app.notify(f"⚠ {exc!s:.80}", severity="warning")


# ═══════════════════════════ Tab Regional ═════════════════════════════════


class RegionalScreen(VerticalScroll):
    """Comparativa line-shopping MX vs US para picks/eventos con odds."""

    BINDINGS = [Binding("r", "refresh", "Refresh")]

    def compose(self) -> ComposeResult:
        yield Label(
            "[b cyan]🌎 Regional MX vs US[/] — mejor casa por pick con edge neto",
            classes="section",
        )
        yield Static(
            "  [dim]🇲🇽 MX: Caliente · Strendus · Codere (SEGOB)    "
            "🇺🇸 US: DraftKings · FanDuel · BetMGM · Caesars (estatal)    "
            "Recom. = book con mayor EV neto ajustado por límites y tolerancia.[/]"
        )
        yield DataTable(
            id="regional_table",
            show_header=True,
            zebra_stripes=True,
            cursor_type="row",
        )
        yield Container(id="regional_empty_slot")

    async def on_mount(self) -> None:
        tbl = self.query_one("#regional_table", DataTable)
        tbl.add_columns(
            "Match",
            "Outcome",
            "🇲🇽 MX best",
            "MX EV%",
            "🇺🇸 US best",
            "US EV%",
            "Recom.",
        )
        await self.refresh_data()

    async def refresh_data(self) -> None:
        try:
            async with session_scope() as s:
                rows = (
                    await s.execute(
                        text(
                            """
                            SELECT pa.id AS bet_id, pa.outcome AS bet_outcome,
                                   pa.match_id, pa.bookmaker AS placed_book,
                                   pa.odds_placed, p.probability AS p_fair,
                                   h.name AS home_name, a.name AS away_name
                            FROM pick_alerts pa
                            LEFT JOIN predictions p ON p.id = pa.prediction_id
                            JOIN matches m ON m.id = pa.match_id
                            JOIN teams h ON h.id = m.home_team_id
                            JOIN teams a ON a.id = m.away_team_id
                            WHERE pa.outcome_result IS NULL
                               OR pa.outcome_result = 'pending'
                            ORDER BY m.start_time ASC
                            LIMIT 30
                            """
                        )
                    )
                ).all()
                bets_data = []
                for r in rows:
                    odds_rows = (
                        await s.execute(
                            text(
                                "SELECT bookmaker, outcome, odds FROM odds_history "
                                "WHERE match_id=:m ORDER BY ts DESC LIMIT 80"
                            ),
                            {"m": r.match_id},
                        )
                    ).all()
                    bets_data.append(
                        {
                            "bet_id": r.bet_id,
                            "outcome": r.bet_outcome,
                            "home": r.home_name,
                            "away": r.away_name,
                            "placed_book": r.placed_book,
                            "p_fair": float(r.p_fair or 0),
                            "odds_rows": [dict(o._mapping) for o in odds_rows],
                        }
                    )
        except Exception as exc:
            self.app.notify(f"⚠ {exc!s:.80}", severity="error")
            return

        tbl = self.query_one("#regional_table", DataTable)
        empty_slot = self.query_one("#regional_empty_slot", Container)
        tbl.clear()
        await empty_slot.remove_children()

        if not bets_data:
            tbl.display = False
            await empty_slot.mount(
                EmptyState(
                    icon="🌎",
                    title="Sin picks activos para comparar regional",
                    description=(
                        "Esta tabla muestra para cada pick pending la mejor\n"
                        "oferta disponible en MX (SEGOB) vs US (regulado estatal),\n"
                        "con EV recalculado según los límites típicos de cada casa."
                    ),
                    cta="Genera picks desde Dashboard con [ A ]",
                )
            )
            return

        tbl.display = True
        for bd in bets_data:
            reg = _regional_recommendation(bd["odds_rows"], bd["outcome"])
            p = bd["p_fair"] or 0.5
            mx_cell = "—"
            mx_ev_cell = "—"
            us_cell = "—"
            us_ev_cell = "—"
            rec = "—"
            mx_ev = us_ev = None
            if reg["best_mx"]:
                book, odd = reg["best_mx"]
                mx_ev = (p * odd) - 1
                mx_cell = f"{book[:10]}@{odd:.2f}"
                mx_ev_cell = f"[{'green' if mx_ev > 0 else 'red'}]{mx_ev:+.2%}[/]"
            if reg["best_us"]:
                book, odd = reg["best_us"]
                us_ev = (p * odd) - 1
                us_cell = f"{book[:10]}@{odd:.2f}"
                us_ev_cell = f"[{'green' if us_ev > 0 else 'red'}]{us_ev:+.2%}[/]"
            if mx_ev is not None and us_ev is not None:
                if abs(mx_ev - us_ev) < 0.005:
                    rec = "[yellow]tie[/]"
                elif mx_ev > us_ev:
                    rec = "[green]🇲🇽 MX[/]"
                else:
                    rec = "[green]🇺🇸 US[/]"
            elif mx_ev is not None:
                rec = "[green]🇲🇽 MX[/]"
            elif us_ev is not None:
                rec = "[green]🇺🇸 US[/]"

            match_str = f"{(bd['home'] or '?')[:12]} vs {(bd['away'] or '?')[:12]}"
            tbl.add_row(
                match_str,
                str(bd["outcome"])[:8],
                mx_cell,
                mx_ev_cell,
                us_cell,
                us_ev_cell,
                rec,
            )

    async def action_refresh(self) -> None:
        await self.refresh_data()


# ═══════════════════════════ Tab LLM calls ════════════════════════════════


class LLMScreen(VerticalScroll):
    """Visualiza consumo de LLM (tokens, costo, latencia) + totales."""

    BINDINGS = [Binding("r", "refresh", "Refresh")]

    def compose(self) -> ComposeResult:
        yield Label(
            "[b cyan]💬 LLM calls[/] — consumo DeepSeek V3.2 + latencia",
            classes="section",
        )
        with Grid(id="llm_totals_grid"):
            yield MetricCard(
                label="Calls totales",
                icon="📞",
                color="cyan",
                hint="histórico completo",
                id="llm_card_calls",
            )
            yield MetricCard(
                label="Costo total",
                icon="💵",
                color="green",
                hint="USD acumulado",
                id="llm_card_cost",
            )
            yield MetricCard(
                label="Tokens totales",
                icon="🔢",
                color="yellow",
                hint="in + out",
                id="llm_card_tokens",
            )
            yield MetricCard(
                label="Latencia p95",
                icon="⚡",
                color="magenta",
                hint="percentil 95 ms",
                id="llm_card_latency",
            )
        yield Label("[b cyan]📜 Últimas 30 llamadas[/]", classes="section")
        yield DataTable(id="llm_table", show_header=True, zebra_stripes=True, cursor_type="row")
        yield Container(id="llm_empty_slot")

    async def on_mount(self) -> None:
        tbl = self.query_one("#llm_table", DataTable)
        tbl.add_columns("Hora", "Task", "Modelo", "Tok-in", "Tok-out", "Latency", "$USD")
        await self.refresh_data()

    async def refresh_data(self) -> None:
        try:
            async with session_scope() as s:
                tot = (
                    await s.execute(
                        text(
                            """
                            SELECT COUNT(*) AS n,
                                   COALESCE(SUM(cost_usd), 0) AS cost,
                                   COALESCE(SUM(tokens_in + tokens_out), 0) AS toks,
                                   COALESCE(
                                       percentile_cont(0.95) WITHIN GROUP
                                       (ORDER BY latency_ms), 0
                                   ) AS p95
                            FROM llm_calls
                            """
                        )
                    )
                ).first()
                rows = (
                    await s.execute(
                        text(
                            """
                            SELECT ts, task_kind, model, tokens_in, tokens_out,
                                   latency_ms, cost_usd
                            FROM llm_calls ORDER BY ts DESC LIMIT 30
                            """
                        )
                    )
                ).all()
        except Exception as exc:
            self.app.notify(f"⚠ {exc!s:.80}", severity="error")
            return

        n = int(tot.n or 0) if tot else 0
        cost = float(tot.cost or 0) if tot else 0.0
        toks = int(tot.toks or 0) if tot else 0
        p95 = float(tot.p95 or 0) if tot else 0.0

        self.query_one("#llm_card_calls", MetricCard).update_metric(value=str(n))
        self.query_one("#llm_card_cost", MetricCard).update_metric(
            value=f"${cost:.6f}", color="green" if cost < 1.0 else "yellow"
        )
        self.query_one("#llm_card_tokens", MetricCard).update_metric(
            value=f"{toks:,}".replace(",", ".")
        )
        self.query_one("#llm_card_latency", MetricCard).update_metric(
            value=f"{p95:.0f}ms",
            color="green" if p95 < 3000 else "yellow" if p95 < 8000 else "red",
        )

        tbl = self.query_one("#llm_table", DataTable)
        empty_slot = self.query_one("#llm_empty_slot", Container)
        tbl.clear()
        await empty_slot.remove_children()
        if not rows:
            tbl.display = False
            await empty_slot.mount(
                EmptyState(
                    icon="💬",
                    title="Sin llamadas LLM registradas aún",
                    description=(
                        "Cada vez que corres 'apuestas analyze' o abres la TUI\n"
                        "y disparas análisis, DeepSeek se invoca y se registra\n"
                        "aquí con costo ($0.27/M input, $1.10/M output)."
                    ),
                    cta="Presiona [ A ] en Dashboard para disparar el primer análisis",
                )
            )
            return
        tbl.display = True
        for r in rows:
            lat = int(r.latency_ms or 0)
            lat_color = "green" if lat < 3000 else "yellow" if lat < 8000 else "red"
            tbl.add_row(
                r.ts.strftime("%d-%b %H:%M:%S") if r.ts else "-",
                str(r.task_kind)[:20],
                str(r.model)[:22],
                str(r.tokens_in or 0),
                str(r.tokens_out or 0),
                f"[{lat_color}]{lat}ms[/]",
                f"${float(r.cost_usd or 0):.6f}",
            )

    async def action_refresh(self) -> None:
        await self.refresh_data()


# ═══════════════════════════ Tab Logs ═════════════════════════════════════


class LogsScreen(VerticalScroll):
    """Tail en vivo de structlog desde ring buffer (log_capture_processor)."""

    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("c", "clear_buffer", "Clear buffer"),
        Binding("i", "filter_info", "Nivel: INFO+"),
        Binding("w", "filter_warning", "Nivel: WARNING+"),
        Binding("e", "filter_error", "Nivel: ERROR+"),
    ]

    level_filter: reactive[str] = reactive("INFO")

    def compose(self) -> ComposeResult:
        yield Label(
            "[b cyan]📜 Logs en vivo[/] — tail del ring buffer (últimos 500 eventos)",
            classes="section",
        )
        yield Static(id="log_filter_info")
        yield DataTable(id="logs_table", show_header=True, zebra_stripes=True, cursor_type="row")
        yield Container(id="logs_empty_slot")

    async def on_mount(self) -> None:
        tbl = self.query_one("#logs_table", DataTable)
        tbl.add_columns("Hora", "Nivel", "Logger", "Evento", "Detalles")
        await self.refresh_data()
        self.set_interval(2.0, self.refresh_data)

    async def refresh_data(self) -> None:
        from apuestas.obs.log_buffer import recent_logs

        logs = recent_logs(limit=150, level_min=self.level_filter)
        tbl = self.query_one("#logs_table", DataTable)
        empty_slot = self.query_one("#logs_empty_slot", Container)
        tbl.clear()
        await empty_slot.remove_children()

        filter_lbl = self.query_one("#log_filter_info", Static)
        filter_lbl.update(
            f"[dim]Filtro: [b {'yellow' if self.level_filter != 'INFO' else 'cyan'}]"
            f"{self.level_filter}+[/]   ·   "
            f"[b]i[/b]=INFO  [b]w[/b]=WARNING  [b]e[/b]=ERROR  [b]c[/b]=clear  [b]r[/b]=refresh   "
            f"Eventos cargados: {len(logs)}[/]"
        )

        if not logs:
            tbl.display = False
            await empty_slot.mount(
                EmptyState(
                    icon="📜",
                    title=f"Sin logs del nivel {self.level_filter}+ aún",
                    description=(
                        "Los logs se capturan en memoria durante la sesión TUI.\n"
                        "Navega por los tabs o dispara acciones (A, r, etc.) para\n"
                        "ver actividad. Autorefresh cada 2s."
                    ),
                )
            )
            return
        tbl.display = True
        level_colors = {
            "DEBUG": "dim",
            "INFO": "cyan",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red",
        }
        for e in logs[-80:]:
            ts = str(e.get("ts") or "")[:19].replace("T", " ")
            lvl = str(e.get("level", "INFO")).upper()
            color = level_colors.get(lvl, "white")
            extras = e.get("extra") or {}
            extra_str = " ".join(f"{k}={v!s:.40}" for k, v in list(extras.items())[:4])
            tbl.add_row(
                ts,
                f"[{color}]{lvl}[/]",
                str(e.get("logger") or "-")[:18],
                str(e.get("event") or "")[:60],
                extra_str[:80],
            )

    async def action_refresh(self) -> None:
        await self.refresh_data()

    async def action_clear_buffer(self) -> None:
        from apuestas.obs.log_buffer import clear_buffer

        n = clear_buffer()
        self.app.notify(f"🧹 Buffer limpiado ({n} eventos)", timeout=3)
        await self.refresh_data()

    async def action_filter_info(self) -> None:
        self.level_filter = "INFO"
        await self.refresh_data()

    async def action_filter_warning(self) -> None:
        self.level_filter = "WARNING"
        await self.refresh_data()

    async def action_filter_error(self) -> None:
        self.level_filter = "ERROR"
        await self.refresh_data()


# ═════════════════ Wizards modales con UI (botones + input) ═══════════════


class TelegramSetupWizard(ModalScreen[None]):
    """Wizard interactivo dentro de la TUI para configurar Telegram.

    Flujo:
    1. Muestra instrucciones para obtener token de @BotFather.
    2. Input widget donde pegas el token (enmascarado tipo password).
    3. Valida con getMe → muestra @username del bot.
    4. Long-poll hasta recibir mensaje del usuario → captura chat_id.
    5. Escribe .env + envía mensaje de confirmación.
    """

    BINDINGS = [Binding("escape", "dismiss", "Cerrar")]

    def compose(self) -> ComposeResult:
        yield Container(
            Static(id="tg_step_body"),
            Input(placeholder="Pega aquí el token de @BotFather", id="tg_token_input"),
            Horizontal(
                Button("✓ Validar", id="tg_validate", variant="primary"),
                Button("✗ Cancelar", id="tg_cancel", variant="default"),
            ),
            Static(id="tg_status"),
            id="tg_wizard_box",
        )

    def on_mount(self) -> None:
        self._render_intro()

    def _render_intro(self) -> None:
        body = Text()
        body.append("\n  📱  ", style="bold yellow")
        body.append("Configurar Telegram bot\n\n", style="bold white")
        body.append("  Paso 1: Crea el bot en Telegram (2 min).\n\n", style="white")
        body.append("    1. Abre Telegram y busca ", style="white")
        body.append("@BotFather", style="bold cyan")
        body.append(" (check azul)\n", style="white")
        body.append("    2. Escríbele:  ", style="white")
        body.append("/newbot", style="bold cyan on black")
        body.append("\n", style="white")
        body.append(
            '    3. Nombre: ej. "Mi Apuestas Bot"\n    4. Username: termina en "bot" '
            "(ej. mi_apuestas_bot)\n",
            style="white",
        )
        body.append(
            "    5. Te responde con un token tipo:\n"
            "       123456789:ABCdefGHIjklMNOpqrsTUVwxyz\n\n",
            style="dim",
        )
        body.append("  Paso 2: Pégalo en el campo de abajo y click [✓ Validar].\n\n", style="white")
        self.query_one("#tg_step_body", Static).update(
            Panel(body, title="[b]Wizard Telegram[/]", border_style="cyan", padding=(0, 2))
        )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "tg_cancel":
            self.app.pop_screen()
            return
        if event.button.id == "tg_validate":
            token = self.query_one("#tg_token_input", Input).value.strip()
            if not re.match(r"^\d+:[A-Za-z0-9_-]{30,}$", token):
                self.query_one("#tg_status", Static).update(
                    "[red]❌ Token inválido. Formato: <números>:<string alfanumérico>[/]"
                )
                return
            await self._run_setup(token)

    async def _run_setup(self, token: str) -> None:
        status = self.query_one("#tg_status", Static)
        import httpx

        status.update("[yellow]⏳ Validando token con Telegram...[/]")
        async with httpx.AsyncClient(timeout=10) as c:
            try:
                r = await c.get(f"https://api.telegram.org/bot{token}/getMe")
            except httpx.HTTPError as exc:
                status.update(f"[red]❌ Error red: {exc!s:.80}[/]")
                return
        if r.status_code != 200 or not r.json().get("ok"):
            status.update(f"[red]❌ Token rechazado: {r.text[:100]}[/]")
            return
        bot_info = r.json()["result"]
        bot_user = bot_info.get("username", "?")
        status.update(
            f"[green]✓ Bot @{bot_user} válido.[/]\n\n"
            f"[yellow]⏳ Abre Telegram, busca [b]@{bot_user}[/] y envíale "
            f"cualquier mensaje (ej. /start). Esperando 180s...[/]"
        )

        offset = 0
        deadline = asyncio.get_event_loop().time() + 180
        chat_id: str | None = None
        async with httpx.AsyncClient(timeout=35) as c:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    resp = await c.get(
                        f"https://api.telegram.org/bot{token}/getUpdates",
                        params={"offset": offset, "timeout": 25},
                    )
                    data = resp.json()
                    for u in data.get("result", []):
                        offset = u["update_id"] + 1
                        msg = u.get("message") or u.get("edited_message") or {}
                        cid = msg.get("chat", {}).get("id")
                        if cid is not None:
                            chat_id = str(cid)
                            break
                    if chat_id:
                        break
                except Exception as exc:
                    status.update(f"[yellow]⚠ polling: {exc!s:.60}[/]")
                await asyncio.sleep(1)

        if not chat_id:
            status.update("[red]❌ Timeout. Envía /start al bot y reintenta.[/]")
            return

        # Escribir .env
        env_path = Path(__file__).resolve().parents[3] / ".env"  # repo root
        _update_env_vars(
            env_path,
            {"TELEGRAM_BOT_TOKEN": token, "TELEGRAM_CHAT_ID": chat_id},
        )

        # Enviar mensaje de confirmación
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": (
                        "✅ *Apuestas Bot configurado*\n\n"
                        "Token + chat_id guardados.\n"
                        "Te llegarán picks con EV ≥ 3% aquí."
                    ),
                    "parse_mode": "Markdown",
                },
            )

        status.update(
            f"[green b]✅ Listo.[/] chat_id=[b]{chat_id}[/] guardado en .env\n"
            f"Revisa tu Telegram — debiste recibir un mensaje de confirmación.\n\n"
            f"[dim italic]ESC para cerrar este wizard.[/]"
        )
        self.app.notify("✅ Telegram configurado", severity="information", timeout=5)

    def action_dismiss(self, result: None = None) -> None:
        self.app.pop_screen()


def _update_env_vars(env_path: Path, updates: dict[str, str]) -> None:
    if not env_path.exists():
        env_path.write_text("# Apuestas Bot .env\n", encoding="utf-8")
    lines = env_path.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out.append(line)
            continue
        key = stripped.partition("=")[0].strip()
        if key in updates:
            out.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out.append(line)
    for k, v in updates.items():
        if k not in seen:
            out.append(f"{k}={v}")
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")


class RedditSetupWizard(ModalScreen[None]):
    """Wizard Reddit: decide entre modo público (ya) vs OAuth upgrade."""

    BINDINGS = [Binding("escape", "dismiss", "Cerrar")]

    def compose(self) -> ComposeResult:
        body = Text()
        body.append("\n  🤖  ", style="bold yellow")
        body.append("Reddit: público vs OAuth\n\n", style="bold white")
        body.append("  Modo ACTUAL:  ", style="white")
        body.append("Público (RSS) · 10 req/min · 0 setup", style="bold green")
        body.append("\n\n")
        body.append(
            "  El bot funciona sin OAuth usando feeds RSS. Es suficiente para\n"
            "  ~8 requests por sesión 'apuestas analyze'. Sin embargo, Reddit\n"
            "  bloquea 403 ocasionalmente a scrapers desde ciertas IPs.\n\n",
            style="white",
        )
        body.append("  Upgrade a OAuth (opcional):\n", style="bold cyan")
        body.append("    • 60 req/min (6x el rate)\n", style="white")
        body.append("    • Sin 403s\n", style="white")
        body.append("    • Metadata completa\n\n", style="white")
        body.append("  Cómo conseguir las keys (3 min):\n\n", style="bold white")
        body.append("    1. Ve a ", style="white")
        body.append("https://www.reddit.com/prefs/apps", style="bold cyan")
        body.append("\n", style="white")
        body.append(
            '    2. Click "create another app..." (abajo)\n'
            "    3. name: apuestas-bot  ·  type: script\n"
            "    4. redirect uri: http://localhost:8080\n"
            '    5. Click "create app"\n'
            "    6. Copia client_id (abajo del nombre) y secret\n\n",
            style="white",
        )
        body.append(
            "  Cuando los tengas, pégalos abajo y guarda.\n"
            "  Si prefieres seguir en modo público, cierra con ESC.\n\n",
            style="dim italic",
        )
        yield Container(
            Static(
                Panel(body, title="[b]Wizard Reddit[/]", border_style="magenta", padding=(0, 2))
            ),
            Input(placeholder="REDDIT_CLIENT_ID (~22 chars)", id="rd_id"),
            Input(placeholder="REDDIT_CLIENT_SECRET (~27 chars)", password=True, id="rd_secret"),
            Horizontal(
                Button("💾 Guardar y activar OAuth", id="rd_save", variant="primary"),
                Button("⏩ Mantener modo público", id="rd_skip", variant="default"),
            ),
            Static(id="rd_status"),
            id="rd_wizard_box",
        )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "rd_skip":
            self.app.pop_screen()
            return
        if event.button.id == "rd_save":
            cid = self.query_one("#rd_id", Input).value.strip()
            sec = self.query_one("#rd_secret", Input).value.strip()
            status = self.query_one("#rd_status", Static)
            if len(cid) < 10 or len(sec) < 15:
                status.update("[red]❌ Client ID o secret parecen inválidos (muy cortos)[/]")
                return
            _update_env_vars(
                Path(__file__).resolve().parents[3] / ".env",
                {"REDDIT_CLIENT_ID": cid, "REDDIT_CLIENT_SECRET": sec},
            )
            status.update(
                "[green]✅ Credenciales guardadas en .env[/]\n"
                "[dim]El próximo 'apuestas analyze' usará OAuth automáticamente.[/]"
            )
            self.app.notify("✅ Reddit OAuth activo", severity="information", timeout=4)

    def action_dismiss(self, result: None = None) -> None:
        self.app.pop_screen()


# ═══════════════════════════ Tab Setup & Control ══════════════════════════


class SetupScreen(VerticalScroll):
    """Panel de control con botones clicables.

    Reemplaza la necesidad de memorizar comandos CLI (apuestas worker on,
    apuestas telegram-setup, etc.). Todo es accesible con click o Tab+Enter.
    """

    BINDINGS = [Binding("r", "refresh", "Refresh")]

    def compose(self) -> ComposeResult:
        yield Label(
            "[b cyan]🎛  Setup & Control[/] — servicios, integraciones, mantenimiento",
            classes="section",
        )

        # ── Sección 1: Servicios automáticos ──
        yield Label(
            "[b yellow]⚙  Servicios automáticos[/]   "
            "[dim](corren en background via systemd-user)[/]",
            classes="subsection",
        )
        yield Container(
            Static(id="svc_worker_status"),
            Horizontal(
                Button("▶ Activar worker", id="btn_worker_on", variant="success"),
                Button("⏸ Detener worker", id="btn_worker_off", variant="warning"),
                Button("📜 Ver logs worker", id="btn_worker_logs", variant="default"),
            ),
            Static(id="svc_backup_status"),
            Horizontal(
                Button("▶ Activar backup diario", id="btn_backup_on", variant="success"),
                Button("⏸ Detener backup", id="btn_backup_off", variant="warning"),
                Button("💾 Backup AHORA", id="btn_backup_now", variant="primary"),
            ),
            id="svc_section",
        )

        # ── Sección 2: Integraciones (Telegram / Reddit) ──
        yield Label(
            "[b magenta]🔌  Integraciones[/]   "
            "[dim](canales de notificación y fuentes de datos)[/]",
            classes="subsection",
        )
        yield Container(
            Static(id="int_telegram_status"),
            Horizontal(
                Button("🧙 Setup Telegram", id="btn_tg_setup", variant="primary"),
                Button("🧪 Test mensaje", id="btn_tg_test", variant="default"),
                Button("▶ Start bot", id="btn_tg_start", variant="success"),
            ),
            Static(id="int_reddit_status"),
            Horizontal(
                Button("🧙 Setup Reddit OAuth", id="btn_rd_setup", variant="primary"),
                Button("🧪 Test fetch", id="btn_rd_test", variant="default"),
            ),
            id="int_section",
        )

        # ── Sección 3: Mantenimiento ──
        yield Label(
            "[b green]🧰  Mantenimiento[/]   [dim](acciones inmediatas)[/]",
            classes="subsection",
        )
        yield Container(
            Horizontal(
                Button("🧹 Limpiar HTTP cache", id="btn_cache_clear", variant="default"),
                Button("🧹 Limpiar logs buffer", id="btn_logs_clear", variant="default"),
                Button("🧪 Test APIs externas", id="btn_test_apis", variant="primary"),
                Button("🧪 Test DeepSeek LLM", id="btn_test_llm", variant="primary"),
            ),
            Horizontal(
                Button("🎯 Test Pinnacle guest", id="btn_test_pinnacle", variant="primary"),
                Button("🐴 Test Betfair Exchange", id="btn_test_betfair", variant="primary"),
                Button("🇺🇸 Test US books", id="btn_test_us_books", variant="primary"),
            ),
            id="maint_section",
        )

        # ── Sección 4: Info educativa live ──
        yield Label(
            "[b blue]📊  Estado del sistema[/]   [dim](live, refresca cada 30s)[/]",
            classes="subsection",
        )
        yield Static(id="sys_summary")
        yield Static(id="setup_feedback")

    async def on_mount(self) -> None:
        await self.refresh_data()
        self.set_interval(30.0, self.refresh_data)

    async def refresh_data(self) -> None:
        await self._update_service_states()
        await self._update_integration_states()
        await self._update_system_summary()

    async def _update_service_states(self) -> None:
        w_state = _systemctl_state("apuestas-settle-worker.service")
        w_color = "green" if w_state == "active" else "dim"
        self.query_one("#svc_worker_status", Static).update(
            f"\n  [{w_color}]●[/]  [b]Auto-settle worker[/]   "
            f"estado: [{w_color}]{w_state}[/]\n"
            f"  [dim]LISTEN PG → dispara settle_bets automático tras match finished.[/]\n"
        )
        b_state = _systemctl_state("apuestas-backup.timer")
        b_color = "green" if b_state == "active" else "dim"
        self.query_one("#svc_backup_status", Static).update(
            f"\n  [{b_color}]●[/]  [b]Backup diario (03:30 UTC)[/]   "
            f"estado: [{b_color}]{b_state}[/]\n"
            f"  [dim]pg_dump + HTTP cache + fiscal exports · retención 14 días.[/]\n"
        )

    async def _update_integration_states(self) -> None:
        import os

        tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        tg_chat = os.environ.get("TELEGRAM_CHAT_ID", "")
        tg_ok = tg_token and tg_token.startswith("123") is False and len(tg_token) > 20 and tg_chat
        tg_color = "green" if tg_ok else "red"
        tg_label = "configurado" if tg_ok else "sin configurar"
        self.query_one("#int_telegram_status", Static).update(
            f"\n  [{tg_color}]●[/]  [b]📱 Telegram bot[/]   estado: [{tg_color}]{tg_label}[/]\n"
            f"  [dim]Recibe alertas de picks, bankroll, CLV. Comandos /analyze, /pausar, /resumir.[/]\n"
        )
        rd_id = os.environ.get("REDDIT_CLIENT_ID", "")
        rd_ok = rd_id and not rd_id.startswith("your-")
        rd_color = "green" if rd_ok else "yellow"
        rd_label = "OAuth activo (60 req/min)" if rd_ok else "modo público (RSS, 10 req/min)"
        self.query_one("#int_reddit_status", Static).update(
            f"\n  [{rd_color}]●[/]  [b]🤖 Reddit[/]   estado: [{rd_color}]{rd_label}[/]\n"
            f"  [dim]Ingesta noticias /r/sportsbook /r/nba /r/LigaMX etc.[/]\n"
        )

    async def _update_system_summary(self) -> None:
        import os

        try:
            data = await fetch_system_status()
        except Exception:
            data = {"bd_ok": False, "llm_ok": False, "mem_ok": False, "api_credits": -1}

        dot = lambda ok: "[green]●[/]" if ok else "[red]●[/]"
        llm_status = (
            "DeepSeek API" if os.environ.get("LLM_BACKEND") == "deepseek" else "llama.cpp local"
        )
        credits = data["api_credits"] if data["api_credits"] >= 0 else "?"

        summary = Text()
        summary.append("\n  ")
        summary.append(Text.from_markup(f"{dot(data['bd_ok'])} PostgreSQL"))
        summary.append("    ·    ")
        summary.append(Text.from_markup(f"{dot(data['llm_ok'])} {llm_status}"))
        summary.append("    ·    ")
        summary.append(Text.from_markup(f"{dot(data['mem_ok'])} cuba-memorys"))
        summary.append("    ·    ")
        summary.append(f"🔑 {credits} créditos OddsAPI\n", style="cyan")

        # Cost LLM acumulado
        try:
            async with session_scope() as s:
                row = (
                    await s.execute(
                        text("SELECT COUNT(*) AS n, COALESCE(SUM(cost_usd),0) AS c FROM llm_calls")
                    )
                ).first()
            if row:
                summary.append(
                    f"  💬  {row.n} llamadas LLM · ${float(row.c or 0):.6f} acumulado\n",
                    style="dim",
                )
        except Exception:
            pass

        self.query_one("#sys_summary", Static).update(
            Panel(summary, border_style="blue", padding=(0, 1))
        )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        fb = self.query_one("#setup_feedback", Static)

        # — Servicios —
        if bid == "btn_worker_on":
            _systemctl("enable", "--now", "apuestas-settle-worker.service")
            fb.update("[green]✅ Worker auto-settle activo[/]")
            self.app.notify("✅ Worker iniciado", severity="information")
        elif bid == "btn_worker_off":
            _systemctl("disable", "--now", "apuestas-settle-worker.service")
            fb.update("[yellow]⏸ Worker detenido[/]")
        elif bid == "btn_worker_logs":
            self.app.notify("Abre otra terminal: journalctl --user -u apuestas-settle-worker -f")
        elif bid == "btn_backup_on":
            _systemctl("enable", "--now", "apuestas-backup.timer")
            fb.update("[green]✅ Backup diario activo (03:30 UTC)[/]")
        elif bid == "btn_backup_off":
            _systemctl("disable", "--now", "apuestas-backup.timer")
            fb.update("[yellow]⏸ Backup detenido[/]")
        elif bid == "btn_backup_now":
            fb.update("[yellow]⏳ Ejecutando backup...[/]")
            self.app.notify("⏳ Backup en curso...", timeout=3)
            import subprocess

            backup_script = Path(__file__).resolve().parents[3] / "scripts" / "backup.sh"
            proc = await asyncio.create_subprocess_exec(
                str(backup_script),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            stdout, _ = await proc.communicate()
            fb.update(f"[green]✅ Backup OK[/]\n[dim]{stdout.decode()[-400:]}[/]")

        # — Integraciones —
        elif bid == "btn_tg_setup":
            await self.app.push_screen(TelegramSetupWizard())
        elif bid == "btn_tg_test":
            await self._test_telegram(fb)
        elif bid == "btn_tg_start":
            self.app.notify("Levantar bot: docker compose up -d telegram", timeout=6)
        elif bid == "btn_rd_setup":
            await self.app.push_screen(RedditSetupWizard())
        elif bid == "btn_rd_test":
            await self._test_reddit(fb)

        # — Mantenimiento —
        elif bid == "btn_cache_clear":
            import shutil

            cache_dir = Path.home() / ".cache" / "apuestas"
            if cache_dir.exists():
                freed = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
                shutil.rmtree(cache_dir)
                fb.update(f"[green]✅ HTTP cache limpiado[/] ({freed / 1024:.1f} KB liberados)")
            else:
                fb.update("[dim]Cache ya estaba vacío[/]")
        elif bid == "btn_logs_clear":
            from apuestas.obs.log_buffer import clear_buffer

            n = clear_buffer()
            fb.update(f"[green]✅ {n} logs limpiados del buffer en memoria[/]")
        elif bid == "btn_test_apis":
            await self._test_apis(fb)
        elif bid == "btn_test_llm":
            await self._test_llm(fb)
        elif bid == "btn_test_pinnacle":
            await self._test_pinnacle(fb)
        elif bid == "btn_test_betfair":
            await self._test_betfair(fb)
        elif bid == "btn_test_us_books":
            await self._test_us_books(fb)

    async def _test_telegram(self, fb: Static) -> None:
        import os

        import httpx

        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat = os.environ.get("TELEGRAM_CHAT_ID", "")
        if not token or not chat:
            fb.update("[red]❌ Telegram no configurado. Usa [b]🧙 Setup Telegram[/].[/]")
            return
        fb.update("[yellow]⏳ Enviando test...[/]")
        async with httpx.AsyncClient(timeout=10) as c:
            try:
                r = await c.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    json={"chat_id": chat, "text": "🧪 Test desde TUI Apuestas Bot"},
                )
                if r.status_code == 200 and r.json().get("ok"):
                    fb.update("[green]✅ Mensaje enviado. Revisa tu Telegram.[/]")
                    self.app.notify("📨 Mensaje enviado", severity="information")
                else:
                    fb.update(f"[red]❌ {r.text[:100]}[/]")
            except Exception as exc:
                fb.update(f"[red]❌ {exc!s:.80}[/]")

    async def _test_reddit(self, fb: Static) -> None:
        fb.update("[yellow]⏳ Probando /r/sportsbook...[/]")
        try:
            from apuestas.ingest.reddit_social import fetch_reddit_sub

            posts = await fetch_reddit_sub("sportsbook", limit=5)
            if posts:
                titles = "\n  ".join(f"• {p['title'][:70]}" for p in posts[:3])
                fb.update(f"[green]✅ {len(posts)} posts obtenidos[/]\n  {titles}")
            else:
                fb.update(
                    "[yellow]⚠ Sin posts (Reddit bloqueó 403 en modo público).\n"
                    "Usa [b]🧙 Setup Reddit OAuth[/] para acceso estable.[/]"
                )
        except Exception as exc:
            fb.update(f"[red]❌ {exc!s:.100}[/]")

    async def _test_apis(self, fb: Static) -> None:
        import os

        import httpx

        fb.update("[yellow]⏳ Probando 4 APIs externas...[/]")
        results: list[str] = []
        async with httpx.AsyncClient(timeout=10) as c:
            for name, url, headers in [
                (
                    "football-data.org",
                    "https://api.football-data.org/v4/competitions/PL",
                    {"X-Auth-Token": os.environ.get("FOOTBALL_DATA_ORG_KEY", "")},
                ),
                (
                    "The Odds API",
                    f"https://api.the-odds-api.com/v4/sports?apiKey={os.environ.get('THE_ODDS_API_KEY', '')}",
                    {},
                ),
                (
                    "OpenWeatherMap",
                    f"https://api.openweathermap.org/data/2.5/weather?q=Mexico City&appid={os.environ.get('OPENWEATHERMAP_KEY', '')}",
                    {},
                ),
            ]:
                try:
                    r = await c.get(url, headers=headers)
                    ok = r.status_code == 200
                    color = "green" if ok else "red"
                    results.append(f"[{color}]● {name}: HTTP {r.status_code}[/]")
                except Exception as exc:
                    results.append(f"[red]● {name}: {exc!s:.50}[/]")
        fb.update("\n  " + "\n  ".join(results))

    async def _test_llm(self, fb: Static) -> None:
        fb.update("[yellow]⏳ Probando DeepSeek chat completion...[/]")
        try:
            from apuestas.llm.deepseek_client import DeepSeekClient

            async with DeepSeekClient() as llm:
                health = await llm.health()
                if not health:
                    fb.update("[red]❌ DeepSeek no respondió[/]")
                    return
            fb.update(
                "[green]✅ DeepSeek V3.2 OK[/]\n"
                "[dim]/v1/models responde 200. Listo para análisis estructurados.[/]"
            )
        except Exception as exc:
            fb.update(f"[red]❌ {exc!s:.100}[/]")

    async def _test_pinnacle(self, fb: Static) -> None:
        """Consulta guest API Pinnacle y reporta cobertura por deporte."""
        fb.update("[yellow]⏳ Consultando Pinnacle guest API (NBA+EPL+LigaMX+NHL)...[/]")
        try:
            from apuestas.ingest.pinnacle_scraper import ingest_league

            lines = ["[green]✅ Pinnacle guest (fuente sharp GRATIS):[/]"]
            for sport in ("nba", "soccer_epl", "soccer_liga_mx", "nhl"):
                try:
                    matchups, odds = await ingest_league(sport, persist=False)
                    lines.append(
                        f"  [cyan]{sport:<20}[/] {len(matchups):>3} matchups · {len(odds):>4} odds"
                    )
                except Exception as exc:
                    lines.append(f"  [red]{sport}: {exc!s:.40}[/]")
            fb.update("\n".join(lines))
        except Exception as exc:
            fb.update(f"[red]❌ {exc!s:.120}[/]")

    async def _test_betfair(self, fb: Static) -> None:
        """Verifica credenciales Betfair + intenta login."""
        from apuestas.ingest.betfair_exchange import (
            BetfairExchangeClient,
            _credentials_available,
        )

        if not _credentials_available():
            fb.update(
                "[yellow]⚠ Betfair Exchange no configurado[/]\n"
                "[dim]Para activar, añade al .env:\n"
                "  BETFAIR_APP_KEY, BETFAIR_USERNAME, BETFAIR_PASSWORD\n"
                "Obtén App Key delayed (gratis):\n"
                "  https://apps.betfair.com/visualisers/api-ng-account-operations/[/]"
            )
            return
        fb.update("[yellow]⏳ Login Betfair...[/]")
        try:
            client = BetfairExchangeClient()
            ok = await client.login()
            client.logout()
            if ok:
                fb.update("[green]✅ Betfair Exchange: login OK[/]")
            else:
                fb.update("[red]❌ Login falló — revisa credenciales en .env[/]")
        except Exception as exc:
            fb.update(f"[red]❌ {exc!s:.120}[/]")

    async def _test_us_books(self, fb: Static) -> None:
        """Intenta DK+FD+MGM solo si los flags APUESTAS_ENABLE_* están true."""
        import os as _os

        enabled = [
            b
            for b in ("DK", "FANDUEL", "BETMGM")
            if _os.environ.get(f"APUESTAS_ENABLE_{b}", "").lower() in ("1", "true", "yes")
        ]
        if not enabled:
            fb.update(
                "[yellow]⚠ US books deshabilitados[/]\n"
                "[dim]Para activar, setea en .env:\n"
                "  APUESTAS_ENABLE_DK=true\n"
                "  APUESTAS_ENABLE_FANDUEL=true\n"
                "  APUESTAS_ENABLE_BETMGM=true\n"
                "Requiere camoufox (ya instalado).[/]"
            )
            return
        fb.update(f"[yellow]⏳ Probando US books habilitados: {', '.join(enabled)}...[/]")
        try:
            from apuestas.ingest.us_books_scraper import fetch_all

            results = await fetch_all(["nba", "nfl"])
            lines = ["[green]✅ US books:[/]"]
            for book, odds in results.items():
                n = len(odds)
                color = "green" if n > 0 else "yellow"
                lines.append(f"  [{color}]● {book}: {n} odds[/]")
            fb.update("\n".join(lines))
        except Exception as exc:
            fb.update(f"[red]❌ {exc!s:.120}[/]")

    async def action_refresh(self) -> None:
        await self.refresh_data()


def _systemctl(action: str, *args: str) -> None:
    """Wrapper sync a systemctl --user."""
    import subprocess

    cmd = ["systemctl", "--user", action] + list(args)
    try:
        subprocess.run(cmd, check=False, capture_output=True, timeout=10)
    except Exception as exc:
        logger.warning("setup.systemctl_fail", cmd=" ".join(cmd), error=str(exc))


def _systemctl_state(unit: str) -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["systemctl", "--user", "is-active", unit],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ═══════════════════════════ Command palette ═════════════════════════════


class ApuestasCommands(Provider):
    """Provider custom para Textual Command Palette (Ctrl+P).

    Expone acciones del bot como comandos buscables: switch tabs, analyze,
    refresh, pause, ayuda, tutorial, clear cache, etc.
    """

    COMMANDS: list[tuple[str, str, str]] = [
        # (título mostrado, ayuda, acción-string del app)
        ("📊 Ir a Dashboard", "Overview: hit rate, alertas vivas/resueltas", "switch_tab('dash')"),
        ("🤖 Ir a Models", "Estado modelos ML + drift", "switch_tab('models')"),
        ("🎯 Ir a Calibración", "Gap predicho vs real", "switch_tab('calibration')"),
        ("🧠 Ir a Memoria", "cuba-memorys status + contenido", "switch_tab('memory')"),
        ("🌎 Ir a Regional", "Line shopping MX vs US", "switch_tab('regional')"),
        ("💬 Ir a LLM", "Consumo tokens + costo DeepSeek", "switch_tab('llm')"),
        ("📜 Ir a Logs", "Tail en vivo de structlog", "switch_tab('logs')"),
        ("🎛 Ir a Setup", "Panel con botones: servicios + integraciones", "switch_tab('setup')"),
        ("🧙 Wizard Telegram", "Configurar bot + chat_id automático", "setup_telegram"),
        ("🧙 Wizard Reddit", "Upgrade a OAuth (opcional)", "setup_reddit"),
        ("▶ Analizar eventos 48h", "Ingesta + devig + LLM + picks", "analyze"),
        ("⏸ Pausar/Reanudar bot", "Con confirmación Y/N", "toggle_pause"),
        ("🔄 Refrescar TODOS los tabs", "Ctrl+R", "refresh_all"),
        ("💡 Toggle sidebar ayuda", "Mostrar/ocultar guía lateral", "toggle_sidebar"),
        ("📚 Abrir tutorial paso a paso", "Guía interactiva", "show_tutorial"),
        ("📖 Menú de ayuda completo", "? o F1", "help"),
    ]

    async def discover(self) -> Hits:
        """Comandos visibles sin filtro (al abrir el palette)."""
        for title, help_text, action in self.COMMANDS:
            yield DiscoveryHit(
                title,
                self._runner(action),
                help=help_text,
            )

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for title, help_text, action in self.COMMANDS:
            score = matcher.match(title)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(title),
                    self._runner(action),
                    help=help_text,
                )

    def _runner(self, action: str) -> Any:
        app = self.app

        async def _run() -> None:
            await app.run_action(action)

        return _run


# ═══════════════════════════ App principal ═══════════════════════════════


class ApuestasTUI(App[None]):
    COMMANDS = App.COMMANDS | {ApuestasCommands}
    CSS = """
    Screen { background: $surface; }
    SystemStatus { height: 3; margin: 0 1; }
    #cards_grid {
        grid-size: 4 1;
        grid-columns: 1fr 1fr 1fr 1fr;
        height: 9;
        padding: 0 1;
    }
    MetricCard { height: 8; margin: 0 1; }
    WelcomeCard { height: 14; margin: 1 2; }
    EmptyState { height: 12; margin: 1 2; min-height: 10; }
    HelpSidebar {
        width: 34;
        height: 100%;
        background: $panel;
        border-left: solid $primary;
    }
    HelpSidebar.-hidden { display: none; }
    .section {
        padding: 1 2 0 2;
        text-style: bold;
    }
    DataTable {
        margin: 0 2;
        height: auto;
        max-height: 14;
        border: round $primary-darken-1;
    }
    #bankroll_plot {
        height: 18;
        margin: 1 2;
        padding: 1;
        background: $panel;
        border: round $accent;
    }
    #mem_status { padding: 0 2 1 2; height: auto; }
    #help_box {
        align: center middle;
        width: 90;
        height: auto;
        max-height: 40;
        background: $surface;
    }
    TabbedContent { height: 1fr; }
    TabPane { padding: 0; }
    Tabs { background: $panel; }
    """

    BINDINGS = [
        Binding("q", "quit", "Salir"),
        Binding("question_mark", "help", "Ayuda"),
        Binding("f1", "help", "Ayuda"),
        Binding("h", "toggle_sidebar", "Toggle ayuda"),
        Binding("d", "switch_tab('dash')", "Dashboard"),
        Binding("p", "switch_tab('pm')", "Post-mortems"),
        # Binding b retirado: tab "bankroll" ya no existe tras pivote detector puro.
        Binding("m", "switch_tab('models')", "Models"),
        Binding("c", "switch_tab('calibration')", "Calibración"),
        Binding("e", "switch_tab('memory')", "Memoria"),
        Binding("g", "switch_tab('regional')", "Regional"),
        Binding("l", "switch_tab('llm')", "LLM"),
        Binding("L", "switch_tab('logs')", "Logs"),
        Binding("s", "switch_tab('setup')", "Setup"),
        Binding("S", "switch_tab('setup')", "Setup"),
        Binding("a", "analyze", "Analizar"),
        Binding("A", "analyze", "Analizar"),
        Binding("P", "toggle_pause", "Pausa"),
        Binding("t", "show_tutorial", "Tutorial"),
        Binding("ctrl+r", "refresh_all", "Refresh all"),
    ]

    TITLE = "🎯 Apuestas Bot"
    SUB_TITLE = "TUI interactiva · presiona ? para ayuda"

    sidebar_visible: reactive[bool] = reactive(True)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield SystemStatus(id="sys_status")
        with Horizontal(id="main_row"):
            with Container(id="tabs_wrap"):
                with TabbedContent(initial="dash", id="main_tabs"):
                    with TabPane("📊 Dashboard", id="dash"):
                        yield DashboardScreen()
                    # Post-mortems y Bankroll retirados en pivote detector puro
                    # (2026-04-23). Sprint 2 añadirá AnalysisScreen basada en
                    # pick_analysis + SHAP en su lugar.
                    with TabPane("🤖 Models", id="models"):
                        yield DriftScreen()
                    with TabPane("🎯 Calibración", id="calibration"):
                        yield CalibrationScreen()
                    with TabPane("🧠 Memoria", id="memory"):
                        yield MemoryScreen()
                    with TabPane("🌎 Regional", id="regional"):
                        yield RegionalScreen()
                    with TabPane("💬 LLM", id="llm"):
                        yield LLMScreen()
                    with TabPane("📜 Logs", id="logs"):
                        yield LogsScreen()
                    with TabPane("🎛 Setup", id="setup"):
                        yield SetupScreen()
            yield HelpSidebar(id="help_sidebar")
        yield Footer()

    async def on_mount(self) -> None:
        try:
            from apuestas.mcp import memory as mcp_memory

            await mcp_memory.jornada_start()
        except Exception as exc:
            logger.debug("tui.jornada_start_fail", error=str(exc))

        await self._refresh_system_status()
        self.set_interval(30.0, self._refresh_system_status)
        self.notify(
            "💡 Presiona [ ? ] para ver todos los atajos y conceptos clave",
            severity="information",
            timeout=6,
        )

    async def on_unmount(self) -> None:
        try:
            from apuestas.mcp import memory as mcp_memory
            from apuestas.mcp.client import MCPClient

            await mcp_memory.jornada_end()
            await MCPClient.get().stop()
        except Exception as exc:
            logger.debug("tui.jornada_end_fail", error=str(exc))

    async def _refresh_system_status(self) -> None:
        try:
            data = await fetch_system_status()
        except Exception:
            return
        sb = self.query_one("#sys_status", SystemStatus)
        sb.bd_ok = data["bd_ok"]
        sb.llm_ok = data["llm_ok"]
        sb.mem_ok = data["mem_ok"]
        sb.api_credits = data["api_credits"]
        sb.last_refresh = datetime.now(tz=UTC).strftime("%H:%M:%S UTC")

    def action_switch_tab(self, tab_id: str) -> None:
        tabs = self.query_one(TabbedContent)
        tabs.active = tab_id
        # Actualizar sidebar
        sidebar = self.query_one(HelpSidebar)
        sidebar.current_tab = tab_id

    async def action_help(self) -> None:
        await self.push_screen(HelpScreen())

    async def action_show_tutorial(self) -> None:
        await self.push_screen(TutorialScreen())

    async def action_setup_telegram(self) -> None:
        await self.push_screen(TelegramSetupWizard())

    async def action_setup_reddit(self) -> None:
        await self.push_screen(RedditSetupWizard())

    def action_toggle_sidebar(self) -> None:
        sb = self.query_one(HelpSidebar)
        self.sidebar_visible = not self.sidebar_visible
        if self.sidebar_visible:
            sb.remove_class("-hidden")
        else:
            sb.add_class("-hidden")

    async def action_analyze(self) -> None:
        for screen in self.query(DashboardScreen):
            await screen.action_analyze()
            self.query_one(TabbedContent).active = "dash"
            return

    def action_toggle_pause(self) -> None:
        """Binding P → lanza worker (push_screen_wait requiere worker en Textual >=0.72)."""
        self._toggle_pause_worker()

    @work(exclusive=True)
    async def _toggle_pause_worker(self) -> None:
        try:
            from apuestas.bot.control import is_bot_paused, pause_bot, resume_bot

            paused, _ = await is_bot_paused()
            if paused:
                confirmed = await self.push_screen_wait(
                    ConfirmScreen(
                        title="Reanudar el bot",
                        message=("El bot volverá a emitir picks con EV ≥ 3%.\n¿Confirmas?"),
                    )
                )
                if confirmed:
                    await resume_bot()
                    self.notify("▶ Bot reanudado", severity="information")
            else:
                confirmed = await self.push_screen_wait(
                    ConfirmScreen(
                        title="Pausar el bot",
                        message=(
                            "Mientras está pausado NO emitirá picks nuevos.\n"
                            "Las bets pendientes siguen abiertas.\n"
                            "¿Confirmas la pausa?"
                        ),
                        danger=True,
                    )
                )
                if confirmed:
                    await pause_bot(reason="manual_tui", triggered_by="tui")
                    self.notify(
                        "⏸ Bot pausado · no emite picks hasta reanudar",
                        severity="warning",
                    )
        except Exception as exc:
            self.notify(f"⚠ {exc!s:.80}", severity="error")

    async def action_refresh_all(self) -> None:
        self.notify("🔄 Refrescando todos los tabs...", timeout=2)
        for screen_class in (
            DashboardScreen,
            PostMortemsScreen,
            BankrollScreen,
            DriftScreen,
            CalibrationScreen,
            MemoryScreen,
            RegionalScreen,
            LLMScreen,
            LogsScreen,
            SetupScreen,
        ):
            for screen in self.query(screen_class):
                if hasattr(screen, "refresh_data"):
                    try:
                        await screen.refresh_data()
                    except Exception as exc:
                        logger.debug(
                            "tui.refresh_all_fail",
                            screen=screen_class.__name__,
                            error=str(exc),
                        )
        await self._refresh_system_status()
        self.notify("✅ Refresh completo", severity="information", timeout=2)


def main() -> None:
    import os

    os.environ["APUESTAS_TUI_ACTIVE"] = "1"
    configure_logging()
    app = ApuestasTUI()
    app.run()


if __name__ == "__main__":
    main()
