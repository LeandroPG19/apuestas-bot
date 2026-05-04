# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- Tennis model con Sackmann + ATP rankings live (currently shadow)
- NHL model con NST/MoneyPuck features
- StatsBomb event-level integration completa para soccer xT
- TabPFN stacker en producción tras validación walk-forward
- Live betting Kalman filter (state-space) para in-play markets

---

## [0.1.0] — 2026-05-04

Initial public release.

### Added

#### Detector core
- De-vigging con 3 métodos: Multiplicativo, Power, Shin (con fallback automático)
- EV adaptativo por deporte/fase via `config/ev_thresholds.yaml`
  (NBA regular 4%, NBA playoff 8%, MLB 5%, soccer 3%)
- Conformal prediction (MAPIE) con bandas de incertidumbre por deporte
- Sample size guard (rechaza modelos con < 50 matches de entrenamiento)
- Slippage guard (cancela picks si la cuota actual cae > 5% bajo la emitida)
- CLV anti-stale (cancela si Pinnacle se mueve > 2% en contra en 30 min)
- Correlation filter (Koopman & Lit 2015) elimina picks h2h+spread mismo lado
- Steam detector cross-book

#### Modelos ML
- LightGBM + XGBoost + CatBoost + MAPIE conformal por deporte
- Dixon-Coles cross-liga + por-liga para soccer (16 ligas)
- MLB Poisson GLM offense/defense + park factors
- Markov chain Sackmann para tennis (36k matches ATP+WTA)
- Stacker LogReg/LGBM monotonic con focal loss opt-in
- Venn-Abers calibration (Vovk 2025)
- TabPFN stacker (opt-in, foundation model tabular)
- KPI gate universal con degradate a shadow si log_loss > 0.68

#### Confidence labeling 5-componentes
- EV vs threshold del deporte
- Modelo vs Pinnacle de-vigged delta
- Sharp consensus (Pinnacle + Polymarket + Kalshi)
- Conformal interval width
- Soft tags (late_line, weather_extreme, key_player_out, anticipated_clv)

#### Calibración y monitoring
- Métricas primarias log-loss · Brier · BSS · ECE · hit_rate − implied_rate
- Page-Hinkley drift monitor + River ADWIN adaptativo
- Walk-forward backtest + isotonic post-hoc
- SHAP top-5 persistido por pick
- Prometheus + Grafana dashboards (8 panels)

#### Ingesta (58 fuentes)
- Pinnacle guest API · Polymarket · Kalshi · OddsJam · Kambi multi-operador
- football-data.org · TheSportsDB · The Odds API · OpenWeatherMap
- Caliente.mx + Codere + Strendus (camoufox anti-Cloudflare)
- StatsBomb open events · ClubElo · NBA PBP · MLB Statcast · Sackmann tennis
- Reddit · Twitter insiders · Sofascore · NFL play-by-play (nflfastr)

#### Infrastructure
- Python 3.14.4 free-threaded (PEP 779)
- PostgreSQL 16 + TimescaleDB 2.20 + pgvector 0.8 HNSW (32 migraciones)
- Valkey 8.0 (cache + TaskIQ broker)
- MinIO S3-compatible
- FastAPI 0.128 + granian 2.5
- Prefect 3.4 + TaskIQ 0.12
- MLflow 3.11 + Optuna 4.5
- Docker Compose orchestration

#### Bot Telegram
- Long-polling con UX v2 (teclado 3 botones + submenu inline)
- Comandos: `/today`, `/clv`, `/region`, `/menu`, `/analizar`, `/explain`, `/pausar`, `/resumir`
- SHAP explicability per pick
- Rate limiting per chat_id

#### CLI `apuestas`
- Modo on-demand puro (sin timers en background)
- `apuestas`, `apuestas analyze`, `apuestas stop`, `apuestas status`, `apuestas tui`

### Security
- Secret scanning + push protection enabled
- Pre-commit hooks: ruff, mypy, detect-secrets, gitleaks, no-commit-to-main
- 21-CFR Part 11 audit trail style en migraciones
- Branch protection en `main`: no force-push, no deletions

### Documentation
- README de 478+ líneas con onboarding completo
- CONTRIBUTING.md, CODE_OF_CONDUCT.md, CITATION.cff
- docs/ con runbook, arquitectura, anti-patterns, security review
- BACKLOG.md con gap analysis vs sharps profesionales

[Unreleased]: https://github.com/LeandroPG19/apuestas-bot/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/LeandroPG19/apuestas-bot/releases/tag/v0.1.0
