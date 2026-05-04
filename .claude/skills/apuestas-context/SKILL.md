---
name: apuestas-context
description: Carga el contexto completo del bot de apuestas multi-deporte (NBA/MLB/NFL/Fútbol/Tenis/NHL/Boxeo). Se activa cuando trabajes en cualquier archivo del repo `apuestas`, escribas código Python para features de sports betting, modelos calibrados, de-vigging, Kelly correlation-aware, o uses los MCP cuba-memorys/cuba-search para memoria/research.
---

# Apuestas Bot — Contexto del proyecto

## Stack pineado (2026-04-19)

- Python 3.14.4 free-threaded (`PYTHON_GIL=0`)
- FastAPI 0.128 + granian 2.5 + msgspec + Pydantic 2.12
- SQLAlchemy 2.0 async + Alembic
- PostgreSQL 16 + TimescaleDB 2.20 + pgvector 0.8 HNSW
- Polars 1.32 (LazyFrame primary) + DuckDB 1.3 (OLAP backtests)
- LightGBM 4.6 + XGBoost 3.2 + CatBoost 1.2.9 + LogReg stacker
- MAPIE 1.0 (conformal) + SHAP 0.48 + NannyML 0.13 (drift)
- TaskIQ 0.11 + Prefect 3.4
- llama.cpp (Qwen 2.5 7B Q4) + TEI (BGE-M3 INT8) GPU RTX 4050
- camoufox 0.5 scraper anti-Cloudflare

## Skills hermanas — usa siempre ANTES de escribir código

- **Context7 MCP** — docs canónicas de cada library ANTES de implementar (mandatory)
- **postgres MCP** — verificar schema/queries antes de tocar DB
- **tessl__fastapi-pro** — patrones FastAPI + SQLAlchemy 2.0
- **tessl__sqlalchemy-best-practices** — session management, relationships
- **tessl__postgresql-python-best-practices** — asyncpg pooling
- **tessl__postgresql** — schema design, migrations
- **tessl__pytest-api-testing** — httpx AsyncClient + fixtures
- **tessl__docker-expert** — multi-stage builds
- **tessl__security** — input validation en boundaries
- **tessl__fastapi-security-basics** — CORS, rate limiting, headers
- **tessl__fastapi-error-handling** — exception handlers
- **cuba-memorys-guide** — MCP memoria persistente
- **mcp-guide** — referencia todos los MCPs

## Layout del repo

```
src/apuestas/
├── api/              FastAPI + granian
├── betting/          devig, ev, detector, clv, portfolio, hedge, odds_spike, regional
├── bot/              telegram (polling)
├── features/         common, nba, soccer, mlb, nfl, tennis, nhl, props_*, weather_perf
├── flows/            Prefect 3 flows (ingest_odds, pre_match, deep_analysis, retrain)
├── ingest/           http_base + por deporte + news_rss + reddit + weather + scrapers
├── llm/              client (llama.cpp), embed (TEI), grammars, prompts, rag, router
├── mcp/              clients para cuba-memorys + cuba-search
├── ml/               calibrate, train_base, train_{nba,mlb,nfl,soccer,props}, registry,
│                     shap_explain, backtest, drift, discrepancy, props_distributions
├── models/           SQLAlchemy ORM (catalog, matches, analysis)
├── monitors/         calibration_audit, drift_monitor, bad_run
├── obs/              logging (structlog), metrics (OTel + Prometheus)
├── risk/             kelly (correlation-aware), montecarlo (risk of ruin)
├── schemas/          llm.py (msgspec), props.py (catálogo props por deporte)
├── tasks/            TaskIQ (broker, ingest, ml, scrape)
└── validators/       schemas.py (Pandera) + mirror_check.py (simetría home/away)
```

## Principios no negociables

1. **Anti-leakage temporal absoluto**: rolling en t-1 siempre, `TimeSeriesSplit(gap≥7d)`.
2. **Calibración primaria**: log-loss, Brier, ECE < 0.03. Accuracy solo como secundaria.
3. **¼ Kelly + cap 5%** con `correlation_aware_kelly` para multi-pick. Full Kelly prohibido.
4. **Threshold EV ≥ 3%**. Conformal filter: `p_lower > implied_prob + 0.01`.
5. **CLV como brújula real**: Buchdahl 2023, ~65–100 picks revelan skill vs ~9,600 para ROI.
6. **De-vigging Shin** como default (Power fallback, Multiplicative solo ablation).
7. **Regional MX/US** evaluado en cada pick con `net_profit_adjustment` por tolerancia a sharps.
8. **Semi-automatizado**: detectar+alertar, ejecutar manualmente por TOS Caliente/Strendus.
9. **On-demand**: `make up` arranca, `make down` apaga. No auto-start.
10. **Solo LAN**: Telegram polling, dashboards a 192.168.x.x. No Cloudflare Tunnel.

## Multi-deporte — mercados + props cubiertos

| Deporte | Mercados principales | Props |
|---|---|---|
| NBA | ML, spread, total, quarters | points, rebs, asts, 3PM, steals, blocks, PRA, P+R, P+A, R+A, double-double |
| MLB | ML, runline, total, F5, NRFI/YRFI | HR (yes/no), TB, hits, RBI, runs; pitcher K's, outs, ER |
| NFL | ML, spread, total, 1H/1Q | QB pass yds/TDs/completions, RB rush yds/carries, WR rec yds/recs, TD scorer |
| Fútbol | 1X2, AH, O/U 2.5, BTTS, DC | anytime_goalscorer, shots, SOT, cards, assists |
| Tenis | match_winner, set_betting, total_games, handicap | aces, double_faults, break_points_won |
| NHL | ML, puckline, total, period | player SOG, points, anytime_goal, goalie_saves, hits, blocks |
| Boxeo/MMA | ML, method, rounds | KO probability, go_distance, over_rounds |

## Anti-patterns checklist

Revisar `docs/anti-patterns-checklist.md` antes de cada deploy de modelo.
Si algo salta sí → fix antes de promote a production.

## Plan maestro

`~/.claude/plans/analiza-a-detalle-analiza-radiant-cherny.md` — 25 secciones:
1-14 fundamentos · 15 potenciadores 2026 · 16 análisis 360° · 17-20 detalles críticos ·
21 post-mortem automático · 22 regional MX/US · 23 player props · 24 weather-performance ·
25 tenis + NHL.
