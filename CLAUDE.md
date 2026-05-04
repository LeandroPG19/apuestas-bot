# Bot de Apuestas Deportivas — Configuración Claude Code

Proyecto: bot multi-deporte (NBA, MLB, NFL, Fútbol, Tenis, NHL, Boxeo), 100% local,
on-demand, Python 3.14.4 free-threaded.

**⚠ Pivote 2026-04-23 — "detector puro"**: el bot ya **no gestiona bankroll,
stake, Kelly, PnL, CLV ni fiscal**. Solo emite alertas de valor (EV positivo
sobre odds fair Pinnacle/Polymarket/Kalshi de-vigged). El usuario decide stake
y ejecuta manualmente. Plan maestro en
`~/.claude/plans/analiza-a-detalle-todos-immutable-reef.md`.

**Sprints completados (Sprint 1-8 + 5b):**
- **Sprint 1** demolición bankroll (4 migraciones, 15 archivos borrados, tests adaptados)
- **Sprint 2** alert_store + classify_confidence 5-componentes
- **Sprint 3** live_scores resolver 4-capas + mark_alert_results + TTL
- **Sprint 4** OddsJam + weather + regional + calibration UI + steam_watcher + odds_spike tags
- **Sprint 5** walk_forward backtest + Brier/BSS/ECE + isotonic + migración 0023 snapshots inmutables + devig Power default 2-way
- **Sprint 5b** NFL KPI gate (si log_loss>0.68 degradar a shadow)
- **Sprint 6** Polymarket + Kalshi + market_consensus (3-source sharp)
- **Sprint 7** Page-Hinkley drift monitor + event-driven retrain + SLO multi-burn-rate YAML
- **Sprint 7b** kpi_gate.passes_kpi_gate() para promote_production gate
- **Sprint 8** SHAP top-5 + /explain Telegram command

**Mejoras post-análisis 7 picks 23 abr (2026-04-24 late)**:
- `config/ev_thresholds.yaml` + `ev_threshold_for(sport, stage)`: NBA 0.04, NBA_playoff 0.08, MLB 0.05, soccer 0.03 — filtra picks marginales tipo #112 TEX-PIT (EV 3.41% MLB → lost)
- Draw guard soccer 3-way (`soccer_max_draw_prob=0.25`): evita picks tipo #105 Go Ahead 0-0 AZ
- Conformal width filter (`conformal_max_width=0.15`): rechaza picks con incertidumbre alta tipo #104 NBA playoff
- NBA playoff guard (`block_playoff_sports={"nba"}`): skip hasta tener train_nba_playoffs dedicado
- `late_line` soft_tag (kickoff < 90 min) afecta tier en classify_confidence
- `EventOdds.stage` poblado desde `matches.stage`

**Bugs arreglados post-auditoría (2026-04-24)**:
- 5 `assert` en prod convertidos a raise/log defensivo (`_match_resolver.py x2`, `mcp/client.py`, `news_pipeline.py`, `deep_analysis.py`) — evita crashes con `python -O`
- SQL f-strings auditados (whitelist de columnas, sin user input inyectado)
- **`_classify_alert` spreads/away bug (live_scores.py:224)**: `inv = away - home - line` invertía el signo en picks `spreads/away` con `line` positiva, tratando la línea como si el away fuese favorito. Fix: `inv = away - home + line`. Detectado con NYM 3-2 MIN spreads/away +1.5 marcado `lost` en vez de `won`. Regresión añadida en `test_mark_alert_results.py`.
- **Hit rate real auditado**: 58 picks emitidos 22-23 abr, 24 resueltos → 5 won / 19 lost = **20.8%** (no 32.1% como en conteo parcial previo). ROI flat $1 ≈ −55%. Bot sobre-emite en MLB spreads y soccer h2h draw markets; thresholds adaptativos (`ev_thresholds.yaml`) activos en siguiente run deberían comprimir a picks de EV > 5% en MLB.

**Deuda técnica final cerrada (2026-04-24)**:
- `river` ADWIN dep opcional + adaptador `_RiverAdapter` (fallback Page-Hinkley en Python 3.14)
- `config/prometheus/recording_rules.yml` + `alerting_rules.yml` (burn-rate SLO)
- `src/apuestas/obs/metrics.py` con 10 métricas wireadas en emit_alerts + drift
- `/metrics` endpoint FastAPI en `api/main.py`
- `config/grafana/bot_health_dashboard.json` con 8 panels (Brier/BSS/ECE/hit_rate/alerts/drift/watchdog/live_scores)
- `.github/workflows/api_contract.yml` schemathesis CI sobre /openapi.json
- `config/vulture_whitelist.py` + 0 dead code confirmado en src/
- `src/apuestas/flows/retrain_on_drift.py` event-driven + `scripts/trigger_nfl_retrain.sh`
- `hishel` dep opcional con fallback a httpx sin cache
- **0 errores de import** en walk-package sweep de `apuestas.*`

**Sprint 10 — accuracy improvements COMPLETO (2026-04-24)**:
- **Fase 1**: `DetectorConfig.conformal_max_width_by_sport` — mlb 0.10, soccer 0.08, resto 0.12, default global 0.12
- **Fase 1**: `src/apuestas/betting/correlation_filter.py` elimina picks h2h+spread del mismo side (Koopman & Lit 2015) → wireado en `deep_analysis.py:780+` antes de emit. 9 tests
- **Fase 2**: `src/apuestas/features/elo_builder.py` + `features/common.py::add_elo_features` — Elo bidireccional (Hvattum & Arntzen 2010) + rest_days + b2b por deporte. 10 tests. Wireado en `train_nba.py:167`, `train_mlb.py:95`, `train_nfl.py:178`
- **Fase 2**: `src/apuestas/ml/stacker.py::MarketAwareStacker` LGBM shallow con monotonic_constraints (OOF=+1, consensus_delta=-1, sharp_agreement=+1) + fallback LogReg. 5 tests. Opt-in en `train_base.py::_fit_l1` via `APUESTAS_USE_MARKET_STACKER=true`
- **Fase 3**: `src/apuestas/ml/dixon_coles.py::DixonColesModel` — Poisson bivariado + ρ correction low-scores + decay temporal ξ (Dixon & Coles 1997). Predicts 1X2/totals/BTTS. 9 tests
- **Fase 3**: `src/apuestas/ml/mlb_poisson.py::MLBPoissonModel` — GLM Poisson offense/defense + park factors MLB (Coors +15%, Petco −7%) + HFA log-scale. Predicts moneyline/totals/runline. 10 tests
- **72/72 tests nuevos pasan**, ruff clean, 0 errores de import

**Sprint 11 — deep research driven upgrades (2026-04-24)**:
- **Fase A** Venn-Abers wrapper completo (`ml/calibrate.py::_VennAbersWrapper`) ya no cae a sigmoid si librería disponible; Vovk 2025 ICML finite-sample guarantees. Focal loss opt-in en `MarketAwareStacker` (Mukhoti 2020 NeurIPS) via `focal_loss=True` + α/γ configurables
- **Fase B** `ml/tabpfn_stacker.py::TabPFNStacker` — foundation model tabular (arXiv 2501.02945 Jan 2025), opt-in en `train_base` via `APUESTAS_USE_TABPFN_STACKER=true`
- **Fase C** `betting/closing_line_predictor.py` — Ridge sobre log-odds + line_movement 4h/1h + sharp consensus + public_pct; `anticipated_clv()` devuelve CLV esperado (Buchdahl 2023)
- **Fase D** `betting/book_power_ratings.py` — SQL 90d rolling edge_bps por (bookmaker, league) + `rank_books_for()` para priorizar casas soft
- **Fase E** `features/soccer_xt.py` — xT aproximado (Karun Singh 2018 grid) + rolling windows [5,10]. VAEP proxy via stats de match (sin event-level data requerido)
- **Fase F** `features/nba_clutch.py` — clutch stats desde PBP (período ≥4, <5min, margen ≤5) + lineup 5-man net rating (Valencia-Cardot 2021)
- **Fase G** `features/mlb_pitching_plus.py` — Stuff+/Location+/Pitching+ aproximados desde Statcast (spin, velo, whiff, release consistency); Sarris 2022 Driveline
- **Fase H** `ml/ft_transformer.py::FTTransformerClassifier` — PyTorch FT-Transformer (Gorishniy 2023) solo para datasets n>2000, d>50; fallback 50/50 si torch no disponible
- **Fase I** `betting/live_kalman.py::LiveKalmanFilter` — state-space 1D para live betting (Ötting 2024); goal/minor/score_delta observers con variance calibrada por deporte
- **Fase J** `betting/execution_timing.py` + `betting/information_edge.py` — ventanas óptimas UTC por deporte + weather (Nathan 2008 MLB) + RLM sharp divergence detection
- **129/129 tests pasan** (unit + integration Sprint 10+11), ruff clean
- **Ablations ejecutadas 2026-04-24**: NBA/MLB/NFL → Elo REJECT en los 3 (Brier empeora marginal, ECE mejora). Decisión: NO promover Elo. Feature set actual ya rico
- **Wires Sprint 11 integrados al pipeline**:
  - `features/soccer.py::build_soccer_feature_frame` → xT rolling vía `APUESTAS_ENABLE_XT`
  - `features/nba.py::build_nba_feature_frame` → clutch rolling vía `APUESTAS_ENABLE_NBA_CLUTCH`
  - `features/mlb.py::build_mlb_feature_frame` → Stuff+/Pitching+ vía `APUESTAS_ENABLE_MLB_STUFF_PLUS`
  - `betting/ev.py::line_shopping` → book_power_ratings vía `APUESTAS_USE_BOOK_POWER` (requiere arg `league`)
  - `flows/deep_analysis.py::_apply_sprint11_soft_tags` → execution_timing + information_edge vía `APUESTAS_SPRINT11_SOFT_TAGS`
- **Features completamiento post-research** (`features/sports_advanced.py`): xA soccer, on/off court NBA, star_out_adjustment, umpire_k_adjustment MLB, line_injury_ev_adjustment NFL, pdo_regression_signal NHL, contrarian_signal public/sharp (12 tests)
- **Make targets nuevos**: `make retrain-sprint11` (stack completo con env flags), `make ablation-elo SPORT=X YEARS=Y`, `make backtest-dc LEAGUE=N SEASONS=S`, `make sprint11-status` (check flags activos)
- **Verificación deps**: `venn-abers==1.5.0` OK, `tabpfn==7.1.1` OK, `torch==2.10.0+cu128` OK en Python 3.14 free-threaded

**Sprint 11 — OPERACIONAL final (2026-04-24)**:
- **book_power_ratings LIVE con data real**: 82 profiles computados desde 1.1M rows `odds_history`. Top soccer: betfair_ex_eu (+908 bps), leovegas_se (+797), fanatics (+757), williamhill_us (+741), fanduel (+715). `line_shopping` con `league=X` ya re-ordena quotes por edge histórico
- **Cache 2-tier**: Valkey (cuando disponible) + archivo local `artifacts/book_power/latest.json` (fallback). Lazy-load automático en `get_cached_edge()`
- **Daily refresh job**: `flows/enrich_features.py::enrich_features_flow` + make target `make enrich-features`. Orquesta book_power + placeholders NBA PBP / MLB Statcast / Soccer shots
- **Focal loss bug fix**: `_FocalLGBMAdapter` en `stacker.py` expone predict_proba 2D cuando LGBM usa custom obj (focal). Previene crash en CalibratedClassifierCV downstream
- **Stack completo activable via `make retrain-sprint11`**: market_stacker + focal + xT/clutch/Stuff+ + book_power + Sprint 11 soft tags todos ON
- **Fixes columnas DB**: `odds_history.ts` (no `recorded_at`) en `book_power_ratings` + `closing_line_predictor`
- **Data gaps honestos**: play_by_play 0 rows (clutch NBA noop hasta ingest), team_games soccer sin possession/shots (xT noop), pitcher_games sin Statcast (Stuff+ noop). book_power es el único feature Sprint 11 con signal real hoy

**Sprint 12 — Backfill masivo data histórica (2026-04-24)**:
- **Migración 0026** crea 11 tablas nuevas: `team_elo_daily`, `odds_history_archive`, `statsbomb_events`, `injury_reports_normalized`, `weather_stadium_archive`, `power_rankings_external`, `nfl_epa_plays`, `nba_lineup_5man_efficiency`, `nba_hustle_stats`, `fangraphs_team_stats_daily`, `pitcher_game_stats` (particionado BRIN + GIN sobre jsonb)
- **6 ingesters bulk operacionales + Makefile targets**: `ingest-football-data` (52k odds soccer 18 ligas 2018-2026), `ingest-clubelo` (~500k Elo ratings), `ingest-statsbomb` (2.1M+ eventos event-level 75 competitions), `ingest-nflfastr` (389k NFL plays EPA/CPOE), `ingest-sackmann-tennis` (36k ATP+WTA matches), `ingest-fangraphs` (wRC+/FIP/xFIP/WAR team-season)
- **Backfills rate-limited background**: NBA PBP 2023-24+2024-25 (~16h) vía `ingest_nba_pbp_range`, MLB Statcast 2022-2024 (~30h) vía `ingest_mlb_statcast_range`
- **Data verificada al 2026-04-23 (ayer)**: football-data cubre hasta ayer, clubelo hasta 2026-04-12, todas las otras fuentes HTTP 200 OK. Sin fallbacks configurados — fuentes seguras.
- **Eliminado del plan por endpoints muertos**: 538 projects.fivethirtyeight.com (cerrado post ABC shutdown), Massey Ratings (403 sin header browser), MoneyPuck CSV (devuelve 3 rows, formato cambiado).
- Target `make ingest-all-bulk`: ejecuta Fase 1 completa secuencial. Target `make ingest-data-status`: dashboard de conteos.
- **Sprint 12 Fase 5 wire** — `features/historical_data_features.py` consume las 6 tablas nuevas y expone helpers async: `fetch_clubelo_for_match`, `fetch_nfl_epa_rolling`, `fetch_fangraphs_team`, `fetch_pitcher_stuff_plus`, `fetch_closing_odds_implied_prob`. Wireado en `deep_analysis.py::_enrich_with_historical_features` invocado tras consensus/regional/weather. Flag `ENABLE_HIST_FEATURES=true` (default).
- **Fix bug MLB Statcast**: `game_date` era string, no cast a DATE → 0 inserts. Añadido helper `_to_date()`. Tras fix: 5,871 pitcher-game rows en DB y creciendo.
- **Estado ingestas al cierre sesión**: statsbomb 3.1M eventos (completo), nfl_epa 389k (completo), clubelo 91k (~20% progreso, seguirá hasta ~500k), football-data 52k (completo), sackmann 36k (completo), pitcher_statcast 5.9k (arrancando), nba_pbp 0 (ScoreboardV3 de fechas futuras sin data — re-correr con range 2023-10-01/2024-06-30 manualmente).

**Sprint 12 cont. — bugs resueltos + retrains (2026-04-24)**:
- **NBA PBP 8 bugs**: ScoreboardV3 dfs[1]+dfs[2] no dfs[0]; gameCode `YYYYMMDD/AWYHOM` 6 chars sin `@`; home/away mapping invertido; asyncpg requiere date objects no strings; PBP v3 usa `get_data_frames()` (get_normalized_dict vacío); actionType v3 string no int; score casting filtrar `""/NaN`. Con fix: 56k eventos ingestados en 40 min (de 2024-10-01 a 2024-11-10, continúa).
- **MLB Statcast bug**: `game_date` string vs DATE col → 0 inserts. Fix: `_to_date()` helper. Tras fix: 14.6k pitcher-games insertados y creciendo.
- **`scripts/train_tennis_sackmann.py`** nuevo trainer tennis usando 36k matches Sackmann. **Métricas finales**: log_loss=0.6609, Brier=0.2342, **ECE=0.0287** ✓ (pasa KPI gate ECE). Primer modelo tennis con data histórica real.
- **`src/apuestas/features/statsbomb_features.py`** — agregador event-level StatsBomb: xg_total, pass_completion_pct, progressive_passes, pressures, ball_recoveries. Validado con match real (4M+ eventos procesables).
- **NFL retrain Sprint 12**: log_loss **0.6677** (vs baseline 0.85 random). Modelo NFL finalmente fuera de random.
- **8/8 bugs pendientes resueltos** en esta sesión. 9 procesos background activos trabajando. NBA PBP ETA ~40 min, MLB Statcast ETA ~2h.

**Sprint 12 — CLV tracking full stack (2026-04-24)**:
- **Migración 0027**: tabla `pick_closing_lines` (snapshots Pinnacle closing odds pre-kickoff) + cols `pick_alerts.clv_pct, closing_pinn_odds, closing_captured_at`
- **`flows/capture_closing_lines.py`**: Prefect flow que captura snapshots 30-60 min pre-kickoff, calcula CLV post-match, expone `clv_rolling_stats(days)`. Fórmula Buchdahl 2023: `CLV = odds_at_pick/odds_closing - 1`
- **Soft tags anticipated_clv** en `deep_analysis._apply_sprint11_soft_tags`: usa `ClosingLinePredictor` para predecir el cierre; si anticipated_clv < −2% → `anticipated_clv_negative` tag (baja tier); si > +3% → `anticipated_clv_positive` (boost confidence)
- **Metrics Prometheus nuevas**: `apuestas_clv_rolling_pct{sport,window_days}`, `apuestas_clv_positive_ratio`, `apuestas_closing_lines_captured_total`
- **Make targets**: `make capture-closing-lines`, `make clv-stats`, `make seed-venue-coords`, `make ingest-weather`
- **Weather archive operacional**: 95 venues con coords (30 MLB + 32 NFL + 30 NBA + alt names), Open-Meteo ERA5 ingestando (161k+ rows en 6 venues ya). Habilita `information_edge.compute_weather_adjustment_mlb/nfl`
- **6/6 tests CLV pasan**: compute_clv_pct validado con casos Buchdahl (positive/negative/zero/longshot)

**Sprint 11 — data engineering ingesters (2026-04-24)**:
- `ingest/nba_pbp.py::ingest_nba_pbp_for_date/range` — PlayByPlayV3 + ScoreboardV3 fuzzy match a matches DB por fecha+teams. Rate limit 1.2s. Popula `play_by_play` para feature `nba_clutch`
- `ingest/mlb_statcast.py::ingest_mlb_statcast_for_date/range` — pybaseball pitch-level agregado a pitcher×game_pk en tabla nueva `pitcher_game_stats` (creada auto) con spin_rate_avg/velo_avg/whiff_pct/release_consistency. Feeds `mlb_pitching_plus::estimate_stuff_plus`
- `ingest/soccer_shots.py::ingest_soccer_shots_for_match_fbref` — scrape FBref Team Stats table por match_fbref_id. Auto-crea/altera `team_games` con possession_pct/shots_total/shots_on_target. Opt-in via `APUESTAS_ENABLE_SOCCER_SHOTS=true` (riesgo Cloudflare)
- `flows/enrich_features.py` — ahora invoca ingesters reales (último día) en vez de noop placeholders
- Make targets nuevos: `make ingest-nba-pbp DATE=...`, `make ingest-mlb-statcast DATE=...`, `make enrich-features` (daily 03:00 UTC cron-ready)
- **Ablations Elo NBA/MLB/NFL → REJECT los 3**: feature set actual ya captura strength rolling. Elo solo mejora ECE marginalmente

**Sprint 10 Fase 3 wire + validation tooling (2026-04-24)**:
- `MLBPoissonSklearnWrapper` sklearn-compatible en `src/apuestas/ml/mlb_poisson.py` (fit/predict_proba/classes_) — 8 tests
- `train_mlb.py::_add_poisson_prediction` añade `poisson_p_home` como feature al ensemble LGBM/XGB/CatBoost via flag `APUESTAS_MLB_POISSON_ENSEMBLE=true`
- `add_elo_features` respeta flag `APUESTAS_ELO_FEATURES_DISABLED=true` para ablation — 4 tests
- `scripts/backtest_dc_vs_lgbm.py --league X --seasons 2023,2024,2025` walk-forward DC vs baseline prior; reporte en `artifacts/backtest_reports/`
- `scripts/retrain_elo_ablation.py --sport nba --years 2023,2024` retrain comparativo con/sin Elo + KPI gate automático; reporte en `artifacts/elo_ablation/`
- **84/84 tests pasan**, ruff clean, smoke test de imports + scripts OK

**Gaps A1-A18 + wire faltantes cerrados (2026-04-24)**:
- Migraciones 0024 (polymarket/kalshi/consensus cols) + 0025 (retention/BRIN snapshots)
- `market_consensus_delta` real en `classify_confidence` via `_enrich_with_consensus`
- SHAP top-5 wire persistente en `pick_alerts.shap_top5` (`_persist_shap_top5`)
- Isotonic calibration integrada en `train_base.py` post-hoc chain
- `kpi_gate` universal en `register_model_in_db` (degrade production→shadow si falla)
- Bayesian Beta-Binomial prior en `ml/bayesian_prior.py` para deportes con pocas muestras
- `config/prefect_schedules.py` con cron de 8 flows
- `monitors/watchdog.py` + `scripts/sanity_check.py` (Postgres/Valkey/Models/Telegram/orphans)
- `apuestas/cache.py` Valkey get/set/delete + `ingest/rate_limiter.py` token-bucket distribuido
- `config/sport_seasons.yaml` + `betting/season.py::is_sport_active` wired en detector
- `_LEGAL_DISCLAIMER` en `cmd_start`
- Honeypots adversariales en `tests/unit/test_honeypots.py`
- Batching Telegram 250ms si >5 picks en emit

**Métricas primarias operativas** (usar estas, NO accuracy):
- `log_loss` · `brier` · `brier_skill_score` · `ece` · `hit_rate − implied_rate`
- Umbrales NBA: Brier≤0.22, BSS≥+0.03, ECE≤0.05, HR−impl≥+2pp
- Comando `apuestas backtest --sport X --since YYYY-MM-DD` genera reporte md

## Stack

- **Python 3.14.4** con free-threading (PEP 779), `PYTHON_GIL=0` por default
- **PostgreSQL 16** + TimescaleDB 2.20 + pgvector 0.8 HNSW
- **Valkey 8.0** (cache + TaskIQ broker)
- **FastAPI 0.128 + granian 2.5** (ASGI)
- **SQLAlchemy 2.0 async** + Alembic
- **Polars 1.32 + DuckDB 1.3** (features + OLAP)
- **LightGBM 4.6 + XGBoost 3.2 + CatBoost 1.2.9** + stacking LogReg + MAPIE conformal
- **llama.cpp + Qwen 2.5 7B Q4** (GPU RTX 4050)
- **TEI + BGE-M3 INT8** embeddings
- **MLflow 2.22 + Optuna 4.5**
- **TaskIQ 0.11 + Prefect 3.4**
- **msgspec** (hot paths) + **Pydantic 2.12** (boundaries)
- **camoufox 0.5** (scraping Caliente.mx anti-Cloudflare)

## Skills habilitadas (auto-activación)

### Backend core (usar siempre)
- `tessl-guide` — activador maestro Tessl
- `tessl__fastapi-pro` — patrones async FastAPI + SQLAlchemy 2.0 + Pydantic V2
- `tessl__fastapi-error-handling` — exception handlers, async errors, graceful shutdown
- `tessl__fastapi-security-basics` — CORS, rate limiting, security headers, trusted hosts
- `tessl__sqlalchemy-best-practices` — engine/session/relationships, raw SQL safety
- `tessl__postgresql-python-best-practices` — asyncpg/psycopg3 pooling
- `tessl__postgresql` — schema PG-specific, índices, constraints
- `tessl__security` — OWASP, input validation en boundaries

### Testing + Docker
- `tessl__pytest-api-testing` — httpx AsyncClient, conftest fixtures, parametrize
- `tessl__docker-expert` — multi-stage builds, hardening, Compose orchestration

### Scraping
- `tessl__playwright-testing` — patrones browser automation (aplicable a camoufox)

### Infra + memoria
- `mcp-guide` — referencia de MCP servers
- `cuba-memorys-guide` — memoria persistente 19 tools

### Slash commands útiles
- `commit`, `pr`, `migrate`, `research`, `audit`, `lint`, `docker`, `status`, `plan`, `tasks`, `specify`

## MCP servers en el workflow

Todos auto-activados según contexto:

| MCP | Cuándo |
|---|---|
| **Context7** | ANTES de usar cualquier library (fastapi, sqlalchemy, polars, lightgbm, etc.) — mandatory |
| **postgres** | Data/schema queries, debugging — mandatory |
| **cuba-memorys** | Inicio de sesión `cuba_jornada(start)` + `cuba_faro` — mandatory |
| **cuba-search** | Research de libraries/papers no estándar |
| **semgrep** | SAST tras cada feature/fix importante |
| **playwright** | Testing E2E si expone UI |

## Reglas del proyecto

1. **Python 3.14.4**: el lock asume `cp314t` (free-threaded). Fallback automático a `cp314` en Docker build si alguna wheel falla.
2. **Versiones `==` pineadas**: validadas 2026-04-19. Re-audit trimestral con `make audit-deps`.
3. **Anti-leakage temporal**: todas las features rolling cierran en t-1; `TimeSeriesSplit` con gap=7d.
4. **Calibración primaria**: log-loss + Brier + ECE. NUNCA accuracy como métrica primaria.
5. **¼ Kelly + cap 5%**: correlation-aware para múltiples picks; stop-loss 30%.
6. **CLV tracking obligatorio**: closing line vs Pinnacle de-vigged con Shin.
7. **Regional MX/US**: auto-detect VPN con `region/auto_detect.py` (5 fuentes fallback).
8. **Semi-automatizado**: el bot detecta+alerta; el usuario ejecuta manualmente por TOS.
9. **On-demand**: `apuestas go` / `apuestas stop` (no 24/7); systemd timers disabled por default.
10. **Solo LAN**: Telegram polling, dashboards bind a LAN. Sin Cloudflare Tunnel.

## Estado 2026-04-21 (post-sesión validación)

- **Fuentes odds**: Pinnacle guest + Kambi Unibet CDN + OddsJam backend (78+ books gratis sin auth).
- **Modelos production**: `nba_moneyline v20260421_1912` (log_loss 0.6358), `nfl_ats v20260421_1301` (log_loss 0.8515).
- **Pipeline validado**: primer pick real #19 (Magic @ novig 4.55, EV +1.47%) persistido + enviado Telegram.
- **Thresholds temporales relajados** (fase validación): `EV_THRESHOLD=0.01`, `MIN_ODDS=1.10`, `MAX_ODDS=15.0`, `APUESTAS_MIRROR_MIN_COMPLETENESS=0.15`. Subir a producción (0.03/1.50/4.00/0.7) cuando CLV+ rate > 52% tras 30+ picks.
- **UX Telegram v2**: teclado 3 botones (Picks/Mi cuenta/Más), /menu submenu inline, pick format enriquecido con hora+role+alternativas+probabilidad+edge.

## Backlog documentado (fuera de scope MVP)

- Soccer DC optimizer multi-season fallback a LGBM ensemble.
- Tennis/NHL trainers dedicados (Markov chain + rolling).
- MLB histórico via Retrosheet (pybaseball rate-limited).
- Liga MX fbref 403 bypass (live Pinnacle cubre).
- NFL 2023-2026 data source alternativo (flancast90 termina 2022).
- Feature extractor real `build_match_features()` para poblar `team_stats_rolling_*` y divergir `p_model` de Pinnacle.

## Comandos Make principales

```bash
make install        # wizard primera vez (nvidia toolkit, modelos, .env)
make cold-start     # build + migrate + seed (~20 min primera vez)
make up             # levanta stack
make analyze        # ejecuta análisis 360° de eventos próximos 48h
make status         # servicios + VRAM + picks activos
make down           # apaga limpio
make audit-deps     # audit trimestral de dependencias
make backup         # pg_dump + MinIO snapshot
make retrain SPORT=nba          # retrain modelo de un deporte
make live-scores                # ingesta scores partidos finalizados
make settle-bets                # liquida bets pendientes + post_mortem
make fiscal-export MONTH=YYYY-MM  # CSV SAT mensual (§17.11)
make tui                        # TUI Textual (dashboard + bankroll + drift)
```

## CLI `apuestas` (wrapper modo on-demand)

```bash
apuestas              # TUI + arranca stack + pregunta al cerrar
apuestas go           # arranca stack 24/7 background sin TUI
apuestas stop         # apaga graceful (bot + timers + procs huérfanos)
apuestas stop --full  # + apaga Docker containers
apuestas status       # health check rápido
apuestas picks        # picks real 24h
apuestas analyze      # forzar deep_analysis ahora (~3 min)
apuestas catchup      # solo ingesta odds Pinnacle+Kambi+OddsJam
apuestas region       # detecta VPN y setea flags US books
apuestas tui-debug    # TUI con errores visibles (debug)
```

## Bot Telegram `@<TU_BOT>` (configura el tuyo en `.env`)

Comandos principales (minimalista, UX v2):
- `/start` — bienvenida + guía rápida
- `/help` — conceptos EV/Kelly/CLV/line shopping con ejemplos
- `/menu` — submenú avanzado inline
- `/today` — picks activos
- `/bankroll` — curva + ROI
- `/clv` — Closing Line Value 7d/30d
- `/region` — re-detecta VPN on-demand
- `/confirm_bet <id> <odds>` / `/mark_not_taken <id>` — tras pick
- `/pausar` / `/resumir` — control emisión

## Anti-patterns (blueprint §9 + §17-21)

Revisar `docs/anti-patterns-checklist.md` antes de:
- Cada deploy de modelo nuevo
- Cada cambio de threshold EV/Kelly
- Cada vez que se sospeche de data leakage

## Plan maestro

Todo el diseño en `~/.claude/plans/analiza-a-detalle-analiza-radiant-cherny.md`
(25 secciones principales, 2,500+ líneas).
