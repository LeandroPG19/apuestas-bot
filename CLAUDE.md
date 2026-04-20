# Bot de Apuestas Deportivas — Configuración Claude Code

Proyecto: bot multi-deporte (NBA, MLB, NFL, Fútbol, Tenis, NHL, Boxeo), 100% local,
on-demand, Python 3.14.4 free-threaded, stack ya implementado en 9 fases.

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
7. **Regional MX/US**: cada pick evalúa ambas jurisdicciones y recomienda.
8. **Semi-automatizado**: el bot detecta+alerta; el usuario ejecuta manualmente por TOS.
9. **On-demand**: usuario inicia/apaga con `make up` / `make down` (no 24/7).
10. **Solo LAN**: Telegram polling, dashboards bind a LAN. Sin Cloudflare Tunnel.

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

## Anti-patterns (blueprint §9 + §17-21)

Revisar `docs/anti-patterns-checklist.md` antes de:
- Cada deploy de modelo nuevo
- Cada cambio de threshold EV/Kelly
- Cada vez que se sospeche de data leakage

## Plan maestro

Todo el diseño en `~/.claude/plans/analiza-a-detalle-analiza-radiant-cherny.md`
(25 secciones principales, 2,500+ líneas).
