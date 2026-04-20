# Backlog post-MVP

## Gap analysis vs sharps profesionales (deep research 2026-04-19)

Comparativa de qué datos/features usan los sharps de élite y qué tiene
el bot actualmente. Fuentes: Voulgaris (NBA, 70% WR peak), Tony Bloom
(Starlizard, 160+ cuants), Billy Walters, industry reports Datarade/LSports.

### Tier S — Usan pros, **tenemos equivalente funcional**
| Feature | Pros usan | Bot tiene |
|---------|-----------|-----------|
| De-vigging robusto | Shin / power | ✅ `betting/devig.py` 3 métodos |
| Closing Line Value | Trackean religiosamente vs Pinnacle | ✅ capture job + `bets.clv` |
| Kelly fractional | ¼-½ Kelly + caps | ✅ `betting/ev.py` con cap 5% |
| Ensemble models | GBDTs + neural + stacking | ✅ LGBM + XGBoost + CatBoost stacker |
| Calibración | Platt/isotonic + ECE tracking | ✅ `ml/calibrate.py` + MAPIE CI |
| SHAP explicability | Starlizard usa equivalente | ✅ `predictions.shap_top5` |
| Line shopping | Docenas de books (Walters) | ✅ Regional MX + US (17 books) |
| Post-mortems | Starlizard tiene review team | ✅ `post_mortems` con LLM narrative |
| Memoria histórica | Decisions DB propietaria | ✅ cuba-memorys feedback loop |
| Weather × performance | Todos los pros | ✅ `features/weather_perf.py` + buckets |

### ✅ Tier A — 9 features pro IMPLEMENTADAS (2026-04-19)
| Feature | Estado | Módulo / Tabla |
|---------|--------|----------------|
| **Play-by-play granular NBA** (Voulgaris edge #1) | ✅ | `ingest/nba_pbp.py` + `play_by_play` |
| **Referee bias profiles** (~25% del edge Voulgaris) | ✅ | `features/referee_bias.py` + `referee_bias_profile` |
| **Coaching clutch tendencies** (T≤3min) | ✅ | `features/coaching_clutch.py` + `coaching_tendencies` |
| **Steam move detector** (Walters #1) | ✅ | `betting/steam_detector.py` + `steam_moves` |
| **F5/Q1/H1 asymmetric markets exploit** | ✅ | `betting/half_period_markets.py` |
| **Player tracking proxies** (reemplaza Second Spectrum) | ✅ | `features/tracking_proxies.py` + `player_tracking_proxies` |
| **Injury feed estructurado cross-validated** | ✅ | `ingest/injury_feed.py` + `injury_feed` |
| **Bluesky sentiment** (reemplaza Twitter/X $100/mes) | ✅ | `ingest/bluesky_sentiment.py` + `bluesky_posts` |
| **Polymarket futures** (benchmark fair value) | ✅ | `ingest/polymarket.py` + `polymarket_markets` |

**Migración**: 0007 aplicada (9 tablas nuevas).
**Tests**: 22 nuevos en `tests/unit/test_tier_a_features.py` (227 total PASS).
**Integración deep_analysis**: inyecta referee bias + coaching clutch + steam moves activos al prompt del LLM.
**Integración PickDetailScreen**: nueva sección `🎯 TIER A` muestra todos los features Tier A visible al usuario con Enter sobre un pick.

### Inaccesible sin inversión >$500/mes (justificado no hacer)
| Gap | Por qué no | Workaround nuestro |
|---|---|---|
| **Second Spectrum NBA tracking** | Cerrado a equipos/sportsbooks | ✅ `tracking_proxies` derivadas de PBP (~60-70% de la señal) |
| **Synergy Sports film analysis** | $$ B2B | No hay workaround de calidad equivalente |
| **Sportradar / Genius Sports feeds oficiales** | $500+/mes overkill personal | ✅ football-data.org + The Odds API + ESPN hidden API |
| **Twitter/X Pro** | $100/mes | ✅ Bluesky beat writers + RSS + Reddit RSS |
| **Player tracking data** | Second Spectrum NBA / Opta fútbol (privado, $$) | Solo stats agregadas | Inaccesible <$500/mes |
| **Sportradar / Genius feeds** | Sportsbooks + Starlizard | Usamos gratis | $500+/mes (overkill para personal) |

### Tier B — Nice-to-have (roadmap fase 2-3)
- [ ] **Reranker BGE-v2-m3** sobre pgvector hybrid search
- [ ] **smolagents + Pydantic-AI** orquestador agente pre-match
- [ ] **Query expansion** con Qwen antes de retrieval
- [ ] **ragas** evaluation contra set de goldens
- [ ] **TabPFN v2.5** integración para props con <500 samples
- [ ] **Chronos-bolt-small** feature extractor sobre line movement
- [ ] **Venn-Abers predictors** como complemento MAPIE
- [ ] **StatsBomb 360 + SkillCorner open** para fútbol tracking
- [ ] **NFL Next Gen Stats scraping** — tracking gratis
- [ ] **Bluesky sentiment** via atproto (reemplaza Twitter/X)
- [ ] **Betfair Delayed feed** gratis como benchmark de-vigging
- [ ] **Live flow** (§17.10) post CLV+ validado 3 meses
- [ ] **Bayesian update in-play** en `live/ml_live.py`
- [ ] **Polymarket futures** (NBA MVP, Ballon d'Or, etc.)

### Veredicto honesto del gap analysis

El bot cubre **~70% de las capacidades cuantitativas** de un sharp amateur-pro
por ~$1 USD/mes (DeepSeek) vs los $millones de infraestructura de Starlizard.

**Para cerrar el gap con pros individuales como Voulgaris (NBA) bastaría:**
1. Añadir NBA Play-by-Play ingest (2-3 días, gratis) — alto ROI
2. Referee profiles scraping (1 semana, gratis) — alto ROI
3. Coaching tendencies features (1 semana, datos derivables)
4. First-half / second-half totals como mercado adicional (3 días)
5. Steam move detector básico (1 semana) — medio ROI

**Inaccesible para uso personal sin inversión >$500/mes:**
- Player tracking (Second Spectrum, Synergy)
- Sportradar/Genius official feeds
- Insider injury network a nivel Starlizard

Para 3 meses de paper trading con el bot actual ya se puede validar edge real
contra Pinnacle-Shin-devigged. Si CLV promedio > 0 consistente → invertir en
Tier A. Si CLV ≈ 0 → el gap es data granular (no modelos), considerar
suscripciones específicas o abandonar.

---



Seguimiento de features **explícitamente fuera del MVP 12 semanas** pero
planeadas en el plan maestro `~/.claude/plans/analiza-a-detalle-analiza-radiant-cherny.md`.

## Pendientes post-auditoría TUI (2026-04-19)

Gaps honestos detectados que **NO son bloqueantes** pero podrían mejorar UX:

### UX mejoras (nice-to-have)
- [ ] **Command palette real** con búsqueda fuzzy (Ctrl+P) — Textual trae uno built-in minimal, falta customizar con comandos del bot (analyze, settle, retrain, ...)
- [ ] **Tab "Regional" dedicado** comparativa MX vs US line shopping
- [ ] **Tab "Logs" con tail en vivo** de structlog/prefect (read-only)
- [ ] **Tab "LLM calls"** visualización de `llm_calls` tabla (tokens + costo + latencia)
- [ ] **Tutorial interactivo first-run** — actualmente solo hay `WelcomeCard` estático
- [ ] **Hedge suggestions** dentro de `PickDetailScreen` si el line movement cambia

### Integraciones pendientes
- [ ] **Reddit OAuth** — las keys del usuario aún no se obtuvieron (`https://www.reddit.com/prefs/apps`)
- [ ] **Telegram bot** — código existe (`src/apuestas/bot/telegram.py`) pero sin token configurado → no probado en prod
- [ ] **settle_bets con trigger PG AFTER UPDATE** automático (actualmente manual con `apuestas settle`)
- [ ] **Auto-populate `odds_api_credits_remaining`** ya implementado en `http_base._cache_api_credits`, falta correr un `apuestas analyze` real para llenar Valkey
- [ ] **cuba-memorys signatures v0.6** — `cuba_decreto` requiere payload distinto al asumido; los helpers funcionan pero algunos retornan None por mismatch

### Conocidos (cosméticos, no bloquean)
- [ ] `tui.jornada_end_fail` con "cancel scope in a different task" — mitigado con `asyncio.shield` pero aún logea debug
- [ ] SVG screenshots de Textual se ven pequeños por auto-scale, render real en terminal es full-size

No se implementan en la fase inicial para mantener foco; cada item tiene la
sección del plan donde se describe a detalle.

## Fase 2 (semanas 13-20)

### LLM + RAG avanzado
- [ ] **Reranker BGE-v2-m3** sobre pgvector hybrid search (§19.3)
- [ ] **smolagents + Pydantic-AI** orquestador agente pre-match (§15.6)
- [ ] **Query expansion** con Qwen antes de retrieval (§19.3)
- [ ] **ragas** evaluation contra set de goldens (§19.3)

### ML avanzado
- [ ] **TabPFN v2.5** integración completa para props con <500 samples/jugador (§15.5)
- [ ] **Chronos-bolt-small** feature extractor sobre line movement (§15.6)
- [ ] **Venn-Abers predictors** como complemento MAPIE (§15.6)

### Datos
- [ ] **StatsBomb 360 + SkillCorner open** para fútbol tracking (§15.6)
- [ ] **NFL Next Gen Stats scraping** — tracking gratis (§15.6)
- [ ] **Bluesky sentiment** via atproto (reemplaza Twitter/X) (§15.6)
- [ ] **Betfair Delayed feed** gratis como benchmark de-vigging (§15.4)

## Fase 3 (semanas 21+, post paper-trading)

### UI + acceso
- [ ] **Mobile-friendly PWA** en `/dashboard` con htmx + responsive (§19.15)
- [ ] **Inline keyboards Telegram** (✅ tomé / ❌ no / 💰 cambió) (§19.19)
- [ ] **Charts PNG inline** en Telegram (`/chart/bankroll.png`) (§19.19)

### Live betting (post CLV+ validado 3 meses)
- [ ] **Live flow** (§17.10) con WebSocket The Odds API paid
- [ ] **Bayesian update in-play** en `live/ml_live.py`
- [ ] **Triggers cash-out/hedge automáticos** (sugerencias)

### Mercados adicionales
- [ ] **SGP (same game parlay) builder** con correlation matrix (§23.9)
- [ ] **Polymarket futures** (NBA MVP, Ballon d'Or, etc.)
- [ ] **Arbitraje cross-región MX+US** si precios implícitos <1 (§22.7)

### Operaciones
- [ ] **DuckDB OLAP** sobre parquet históricos para backtests acelerados (§2)
- [ ] **UPS integration** con `apcupsd` hooks (§19.13)
- [ ] **pg_stat_statements** panel + `make db-vacuum` (§19.18)
- [ ] **Tailscale mesh** para dashboards móviles remote (§11)

## Requiere decisión explícita del usuario

- [ ] **Go/no-go a dinero real**: tras 3 meses paper trading, >1000 picks,
      CLV promedio positivo consistente vs Pinnacle-Shin devigged, drawdown
      simulado <25% (§ Referencias clave).
- [ ] **Expansión Perú/Colombia** (1xBet LATAM, Betcris) si se quiere ampliar
      jurisdicción (§22.7).
- [ ] **Live API Betfair £499 one-off** si Delayed feed demuestra valor (§15.4).

## Rechazado explícitamente (NO implementar)

- ❌ Feast/Featureform/Tecton — overkill laptop single-tenant (§15.7)
- ❌ AutoGluon/H2O/FLAML — opacidad (§15.7)
- ❌ RL/bandits para bet sizing — sin evidencia que supere ¼ Kelly (§15.7)
- ❌ Migrar pgvector → Qdrant/LanceDB — pgvector HNSW basta <3M vectores (§15.7)
- ❌ ZenML/Kubeflow/KServe/Seldon — overkill (§15.7)
- ❌ LLM como meta-learner en stacking — alucinaciones catastróficas (§15.7)
- ❌ LangGraph — pesado, smolagents es mejor DX local (§15.7)
- ❌ Port a Rust — Python con Polars+LightGBM ya es óptimo en este hardware
- ❌ Pinnacle/Bet365 directo — no operan con SEGOB MX (blueprint §8)
- ❌ Twitter/X API paga $100/mes — Bluesky cubre (§15.6)
- ❌ Auto-start 24/7 — decisión on-demand del usuario (§11)
