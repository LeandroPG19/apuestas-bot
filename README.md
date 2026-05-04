# apuestas — Bot de Apuestas Deportivas 100% Local

Bot multi-deporte (NBA · MLB · NFL · Fútbol · Tenis · NHL · Boxeo) que detecta valor (+EV)
sobre odds *fair* Pinnacle/Polymarket de-vigged y emite alertas por Telegram.

> ⚠ **Disclaimer**
> Herramienta personal de análisis. **No es asesoría financiera ni promueve apostar**.
> El bot **detecta y alerta** — la ejecución manual de la apuesta corre por cuenta del usuario,
> en casas con licencia de su jurisdicción. Cumplir con regulación local antes de usar.

---

## Tabla de contenidos

1. [Qué es](#qué-es)
2. [Tamaño y alcance](#tamaño-y-alcance)
3. [Características técnicas](#características-técnicas)
4. [Arquitectura](#arquitectura)
5. [Stack](#stack)
6. [Modelos ML](#modelos-ml)
7. [Fuentes de odds](#fuentes-de-odds)
8. [Setup paso a paso](#setup-paso-a-paso)
9. [Uso diario](#uso-diario)
10. [Bot Telegram](#bot-telegram)
11. [Modo on-demand](#modo-on-demand--sin-timers-en-background)
12. [Troubleshooting](#troubleshooting)
13. [Documentación](#documentación)
14. [Autor y licencia](#autor-y-licencia)

---

## Qué es

`apuestas` es un sistema completo de detección de valor en apuestas deportivas que corre
**100% local** en una laptop. Está diseñado bajo la filosofía "detector puro":

- **NO** gestiona stake, banca, Kelly, PnL ni reportes fiscales — el usuario decide.
- **SÍ** emite alertas con EV positivo basadas en odds *fair* Pinnacle/Polymarket de-vigged.
- **SÍ** calibra cada modelo con métricas Brier/ECE/BSS y rechaza promociones que fallen el
  KPI gate.
- **SÍ** trackea Closing Line Value (CLV) post-match para validar la edge real del detector.

El proyecto nace de la observación de que las casas de apuestas en jurisdicciones reguladas
(MX/US) ofrecen odds suficientemente desviadas del consenso *sharp* (Pinnacle, Polymarket,
Kalshi) como para detectar oportunidades sistemáticas sin depender de ejecución algorítmica
ni de líneas exclusivas.

---

## Tamaño y alcance

| Métrica | Valor |
|---|---|
| Líneas de código Python | ~85k LoC en 271 módulos |
| Migraciones Alembic | 32 (schema completo + identity resolution + retention/compression) |
| Tests | 67 archivos (unit + integration + E2E) |
| Make targets | 84 (install, retrain, backtest, ingest-*, etc.) |
| Configs YAML | 25 archivos (umbrales, scrapers selectors, scheduling) |
| Dependencias pineadas | 108 (`==` exactas, audit trimestral) |
| Ingesters externos | 58 (odds, weather, lineups, social, scraping) |
| Módulos ML | 44 (train, calibrate, stacker, conformal, drift, etc.) |

---

## Características técnicas

### Detector

- **De-vigging**: 3 métodos (Multiplicativo, Power, Shin) con fallback automático
- **EV adaptativo por deporte/fase**: NBA regular 4%, NBA playoff 8%, MLB 5%, soccer 3%
  (ver `config/ev_thresholds.yaml`)
- **Conformal prediction** (MAPIE): rechaza picks con bandas de incertidumbre > 0.10-0.12
- **Sample size guard**: rechaza si modelo entrenó con < 50 matches
- **Slippage guard**: cancela pick si la cuota actual cae > 5% bajo la emitida
- **CLV anti-stale**: cancela si Pinnacle se mueve > 2% en contra en 30 min
- **Correlation filter**: elimina picks h2h+spread del mismo lado (Koopman & Lit 2015)
- **Steam detector**: identifica movimientos sharp coordinados cross-book

### Confidence labeling

Cada pick recibe tier (S/A/B/C) calculado con 5 componentes:

1. EV vs threshold del deporte
2. Modelo vs Pinnacle de-vigged delta
3. Sharp consensus (Pinnacle + Polymarket + Kalshi)
4. Conformal interval width
5. Soft tags (late_line, weather_extreme, key_player_out, anticipated_clv)

### Calibración y monitoring

- **Métricas primarias**: log-loss · Brier · BSS · ECE · hit_rate − implied_rate (NUNCA accuracy)
- **Umbrales NBA gate**: Brier ≤ 0.22, BSS ≥ +0.03, ECE ≤ 0.05
- **Page-Hinkley drift monitor** + River ADWIN adaptativo → triggers retrain event-driven
- **Walk-forward backtest** + isotonic post-hoc + Venn-Abers (Vovk 2025)
- **SHAP top-5** persistido en cada pick para explicabilidad

---

## Arquitectura

```
┌──────────────────────────────────────────────────────────────────────┐
│ INGESTA (58 fuentes)                                                 │
│  Pinnacle · Polymarket · Kalshi · OddsJam · Kambi · Caliente ·       │
│  Football-data · TheSportsDB · OpenWeatherMap · Reddit · Twitter ·   │
│  StatsBomb · ClubElo · NBA PBP · MLB Statcast · Sackmann tennis      │
└──────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────┐
│ FEATURE STORE (Postgres + TimescaleDB + pgvector)                    │
│  features/{nba,mlb,nfl,soccer,tennis,nhl}.py + sports_advanced.py    │
│  Elo ratings · xT · clutch · Stuff+ · weather × performance · CLV    │
└──────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────┐
│ ML PIPELINE (44 módulos)                                             │
│  LightGBM + XGBoost + CatBoost + Dixon-Coles + MAPIE conformal       │
│  Stacker LogReg/LGBM monotonic · Venn-Abers · TabPFN opt-in          │
└──────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────┐
│ DETECTOR + DEEP ANALYSIS                                             │
│  flows/deep_analysis.py · betting/ev.py · betting/devig.py           │
│  EV gate + conformal gate + correlation filter + steam detector      │
└──────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────┐
│ ALERTING                                                             │
│  bot/telegram.py (long-polling) · API/metrics · Grafana dashboards   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Stack

| Capa | Tecnología |
|---|---|
| Lenguaje | Python **3.14.4** free-threaded (PEP 779) |
| Package manager | `uv` (lock pineado) |
| Web framework | FastAPI 0.128 + **granian 2.5** (ASGI) |
| ORM | SQLAlchemy 2.0 async + Alembic |
| Validation | Pydantic 2.12 + msgspec (hot paths) |
| Database | PostgreSQL 16 + TimescaleDB 2.20 + pgvector 0.8 HNSW |
| Cache + broker | Valkey 8.0 |
| Object storage | MinIO (S3-compatible) |
| ML | LightGBM 4.5 + XGBoost 2.1 + CatBoost 1.2 + MAPIE conformal |
| LLM | llama.cpp + Qwen 2.5 7B Q4_K_M (GPU) **ó** DeepSeek API |
| Embeddings | TEI + BGE-M3 INT8 |
| Tracking | MLflow 3.11 + Optuna |
| Orchestration | Prefect 3.4 + TaskIQ 0.12 |
| Data | Polars 1.34 + DuckDB 1.3 |
| Scraping | camoufox 0.5 (anti-Cloudflare) |
| Observability | Prometheus + Grafana + SignOz (OpenTelemetry) |

---

## Modelos ML

| Deporte | Modelo principal | Mercados |
|---|---|---|
| **NBA regular** | LGBM ensemble + isotonic + market stacker | h2h · totals · spreads |
| **NBA playoffs** | Modelo dedicado con guard (skip si overfitting) | h2h · totals |
| **MLB** | LGBM + Poisson GLM offense/defense + park factors | moneyline · totals · runline |
| **NFL** | LGBM regular + KPI gate (degradate a shadow si log_loss > 0.68) | h2h · ATS · totals |
| **Fútbol** | Dixon-Coles cross-liga + LGBM por-liga (16 ligas) | 1X2 · BTTS · AH · totals |
| **Tenis** | Markov chain Sackmann (36k matches ATP+WTA) | match winner · sets |
| **NHL** | Skeleton (off por default — train con NST/MoneyPuck pendiente) | h2h · totals · puckline |

**16 ligas soccer** con thresholds de empate calibrados: Premier, Championship, La Liga,
La Liga 2, Bundesliga, Bundesliga 2, Serie A, Serie B, Ligue 1, Ligue 2, Eredivisie, Liga
MX, MLS, Brasileirão, Liga Argentina, Champions League.

### Estado de deportes (2026-04-27)

| Deporte | Emit | Retrain | Notas |
|---|---|---|---|
| MLB | ✅ | ✅ | 46% volumen actual |
| NBA | ✅ | ✅ | 13% volumen, playoff guard activo |
| Fútbol | ✅ | ✅ | 34% volumen, 7 ligas promovidas |
| NHL | ❌ | ❌ | Reactivar con features NST/MoneyPuck |
| Tenis | ❌ | ❌ | Reactivar cuando Odds API tennis_atp/wta estable |
| NFL | ❌ | ❌ | Auto-reactivar 2026-09-01 (inicio season) |
| Boxeo / MMA | ❌ | ❌ | Volumen histórico insuficiente |

---

## Fuentes de odds

### Sin auth (siempre activas)

- **Pinnacle** guest API — benchmark sharp universal
- **Polymarket** games — futures + market making
- **Kalshi** — event contracts US
- **Kambi multi-operador** — Unibet/ComeOn/Betsson CDN
- **OddsJam** backend — 78+ books US gratis

### Con auth gratis

- **football-data.org** — Big-5 + Champions + Libertadores (10 req/min)
- **TheSportsDB** — Liga MX + MLS + ligas exóticas (key=`3`)
- **OpenWeatherMap** — 1k req/día
- **The Odds API** — 500 créditos/mes gratis

### Scraping (camoufox)

- **Caliente.mx**, **Codere.mx**, **Strendus** — books MX SEGOB
- **DraftKings**, **FanDuel**, **BetMGM** — books US (requiere VPN US)

---

## Setup paso a paso

### Requisitos

| Recurso | Mínimo |
|---|---|
| OS | Linux (Ubuntu 22.04+ / Debian 12). macOS sin probar. |
| RAM | 16 GB |
| Disco | 30 GB libres |
| GPU | Opcional — RTX ≥6 GB VRAM para LLM local. Sin GPU usa DeepSeek API. |
| Docker | 24+ con `docker compose` plugin |
| Python | 3.14.4 (free-threaded recomendado) |
| `uv` | https://docs.astral.sh/uv/ |

### 1. Clonar y crear `.env`

```bash
git clone <ESTE_REPO>.git apuestas
cd apuestas
cp .env.example .env
```

### 2. Llenar variables **mínimas** en `.env`

```ini
# Postgres — pon una password fuerte
POSTGRES_PASSWORD=<random-32-chars>
DATABASE_URL=postgresql+asyncpg://apuestas:<random-32-chars>@postgres:5432/apuestas

# Valkey
VALKEY_PASSWORD=<random-32-chars>
VALKEY_URL=redis://:<random-32-chars>@valkey:6379/0
TASKIQ_BROKER_URL=redis://:<random-32-chars>@valkey:6379/1

# MinIO
MINIO_ROOT_PASSWORD=<random-32-chars>
AWS_SECRET_ACCESS_KEY=<random-32-chars>   # mismo valor que MINIO_ROOT_PASSWORD
```

Genera cada password con `openssl rand -hex 32`.

### 3. Configurar APIs externas

#### 🟢 Mínimo viable (todas gratis, sin tarjeta)

| API | Cómo obtener | Variable |
|---|---|---|
| **football-data.org** | https://www.football-data.org/client/register | `FOOTBALL_DATA_ORG_KEY` |
| **TheSportsDB** | Sin registro, usar `3` | `THESPORTSDB_KEY=3` |
| **OpenWeatherMap** | https://openweathermap.org/api (1k/día gratis) | `OPENWEATHERMAP_KEY` |
| **The Odds API** | https://the-odds-api.com/ (500 créditos/mes gratis) | `THE_ODDS_API_KEY` |

> Pinnacle, Polymarket, Kalshi y OddsJam funcionan **sin auth**.

#### 🔵 LLM backend — elige uno

**Opción A — DeepSeek API** (recomendado si no tienes GPU):

```ini
LLM_BACKEND=deepseek
DEEPSEEK_API_KEY=sk-...   # https://platform.deepseek.com (~$0.27/M tokens input)
```

**Opción B — LLM local** (Qwen 2.5 7B Q4 en GPU):

```ini
LLM_BACKEND=llama_local
```

Después de `make cold-start` ejecuta `make download-models` para bajar el GGUF (~4.7 GB).

#### 🟡 Opcionales

| Variable | Uso |
|---|---|
| `API_FOOTBALL_KEY` | $19/mes — Liga MX odds + lineups realtime |
| `REDDIT_CLIENT_ID/SECRET` | Sentiment + injury rumors |
| `BETFAIR_APP_KEY` | Exchange odds (delayed gratis) |
| `VISUAL_CROSSING_KEY` | Backfill weather histórico 5 años |
| `SENTRY_DSN` | Error tracking |

### 4. Configurar Telegram

```bash
# 4.1 Crea un bot:
#     - Telegram → @BotFather → /newbot
#     - Nombre: cualquiera ("Mi Apuestas Bot")
#     - Username: termina en "bot" (ej. mi_apuestas_bot)
#     - BotFather te da un token: 123456789:ABCdef...

# 4.2 Wizard automático:
apuestas bot-setup
#   - Pide el token, captura tu chat_id, escribe a .env, envía mensaje de prueba
```

### 5. Levantar stack y arrancar

```bash
make install            # wizard primera vez (NVIDIA toolkit, modelos, .env)
make cold-start         # build + migrate + seed histórico (~20 min)
apuestas                # arranca Docker + bot + 1 ciclo catchup+analyze
```

---

## Uso diario

```bash
apuestas                # arranca todo + 1 ciclo de análisis
apuestas analyze        # ciclo extra durante la sesión
apuestas stop           # apaga bot + procesos. Docker queda UP.
apuestas stop --full    # también apaga Docker (volúmenes intactos)
```

Patrón típico: 1h en la mañana + 1h en la noche, todo on-demand.

### Comandos útiles

```bash
apuestas status          # health check (servicios + VRAM + picks)
apuestas picks           # picks de las últimas 24h
apuestas catchup         # solo ingesta de odds (sin emit)
apuestas region          # detecta VPN MX/US
apuestas backup          # pg_dump + MinIO snapshot
apuestas tui             # TUI Textual (dashboard live)
apuestas help            # lista completa
```

### Make targets relevantes

```bash
make install              # primera vez
make cold-start           # build + migrate + seed
make up / make down       # stack docker
make analyze              # ciclo análisis 360° próximos 48h
make audit-deps           # audit dependencias
make backup               # pg_dump + MinIO
make retrain SPORT=nba    # retrain modelo
make backtest-dc LEAGUE=4 SEASONS=2023,2024  # walk-forward DC
make ablation-elo SPORT=mlb YEARS=2023,2024  # ablation features
```

---

## Bot Telegram

Comandos disponibles:

| Comando | Función |
|---|---|
| `/start` | Bienvenida + guía rápida |
| `/help` | Conceptos EV/CLV/de-vigging con ejemplos |
| `/today` | Picks activos |
| `/clv` | Closing Line Value 7d/30d |
| `/region` | Re-detecta VPN MX/US |
| `/menu` | Submenú avanzado inline |
| `/analizar <equipo1> vs <equipo2>` | Análisis on-demand de un partido |
| `/explain <pick_id>` | SHAP top-5 features que llevaron al pick |
| `/pausar` / `/resumir` | Control emisión |

UX v2: teclado de 3 botones (Picks · Mi cuenta · Más).

---

## Modo on-demand — sin timers en background

NO hay `*.timer` activos. El bot **sólo trabaja cuando lo arrancas**. Razones:

- **Ahorrar quota** de The Odds API: ~28k → ~600 unidades/día (**−98%**)
- **Evitar consumo** eléctrico/recursos cuando no hay sesión activa

Para ciclos extra durante la sesión: `apuestas analyze` o `/analyze` desde Telegram.

---

## Troubleshooting

<details>
<summary><b>El bot no recibe mensajes</b></summary>

```bash
apuestas status                 # ¿bot corriendo?
tail -f logs/telegram.log       # ¿errores en polling?
# Verifica TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID en .env
```
</details>

<details>
<summary><b>The Odds API quota agotada</b></summary>

Sube `THE_ODDS_API_BUDGET_MONTHLY` o desactiva `THE_ODDS_API_KEY` (Pinnacle/Polymarket
funcionan sin él).
</details>

<details>
<summary><b>Postgres no levanta — port 5432 en uso</b></summary>

El compose mapea `POSTGRES_HOST_PORT=5433` por default. Si tu host también lo usa, cambia
a `5434` en `.env`.
</details>

<details>
<summary><b>VRAM insuficiente</b></summary>

Cambia a `LLM_BACKEND=deepseek` en `.env`. No necesitas GPU.
</details>

<details>
<summary><b>Drift detectado, modelo degradado a shadow</b></summary>

```bash
make retrain SPORT=nba           # retrain con data fresca
# si pasa KPI gate, se promueve automáticamente; si no, queda shadow
```
</details>

<details>
<summary><b>Ningún pick emitido en 24h</b></summary>

Probablemente: thresholds correctos pero sin valor real. Revisa:
```bash
apuestas tui                     # dashboard con picks rechazados + razón
tail -f logs/analyze.log
```
Casos comunes: `ev_below_threshold`, `conformal_width_too_high`, `correlation_with_existing_pick`.
</details>

---

## Documentación

- [CLAUDE.md](CLAUDE.md) — instrucciones de proyecto + sprints completados (1-14)
- [docs/arquitectura.md](docs/arquitectura.md) — diagramas C4 + flujos
- [docs/runbook.md](docs/runbook.md) — operaciones diarias
- [docs/onboarding.md](docs/onboarding.md) — guía nuevo dev
- [docs/anti-patterns-checklist.md](docs/anti-patterns-checklist.md) — qué NO hacer
- [docs/runbook_dr.md](docs/runbook_dr.md) — disaster recovery
- [docs/odds_sources.md](docs/odds_sources.md) — detalle por fuente
- [docs/security-review.md](docs/security-review.md) — security posture
- [BACKLOG.md](BACKLOG.md) — gap analysis vs sharps profesionales
- [.github/SECURITY.md](.github/SECURITY.md) — política de reporte de vulnerabilidades

---

## Autor y licencia

**Autor**: Leandro Pérez G.

**Licencia**: MIT.

Uso personal. Revisa la regulación de tu jurisdicción antes de usar este software para
apostar dinero real. El autor no se hace responsable del uso que terceros le den a este
código.

---

## Contribuciones

Issues y PRs bienvenidos. Antes de abrir un PR:

```bash
make audit-deps           # verificar lock al día
uv run pytest -x          # tests pasan
uv run ruff check         # lint clean
uv run mypy src/          # types OK
pre-commit run --all-files  # detect-secrets + gitleaks + ruff format
```
