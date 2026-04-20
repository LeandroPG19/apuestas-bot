# apuestas — Bot de Apuestas Deportivas 100% Local

Bot multi-deporte (NBA, MLB, NFL, Fútbol, Boxeo) que corre 100% en la laptop del usuario.
Modo **on-demand**: arrancas con `make up`, analizas con `make analyze`, apagas con `make down`.

## Stack

- **Python 3.13** + `uv` + `asyncio`
- **PostgreSQL 16 + TimescaleDB 2.20 + pgvector 0.8 HNSW** (dockerizado)
- **Valkey 8** (cache + broker TaskIQ)
- **llama.cpp + Qwen 2.5 7B Q4_K_M** (GPU RTX 4050)
- **TEI + BGE-M3 INT8** (embeddings GPU)
- **LightGBM/XGBoost/CatBoost + MAPIE conformal + MLflow**
- **FastAPI + granian**, **Prefect 3**, **TaskIQ**
- **SigNoz + Grafana** observabilidad
- **MCP**: `cuba-memorys` (memoria persistente) + `cuba-search` (research web)

## Quick start

```bash
make install            # wizard primera vez (nvidia toolkit, modelos, .env)
make cold-start         # build imágenes + migrate + seed histórico
make up                 # levanta stack
make analyze            # ejecuta análisis completo de eventos próximos 48 h
make status             # estado servicios + VRAM + picks activos
make down               # apaga limpio
```

## Documentación

- [docs/arquitectura.md](docs/arquitectura.md)
- [docs/runbook.md](docs/runbook.md)
- [docs/onboarding.md](docs/onboarding.md)
- [docs/anti-patterns-checklist.md](docs/anti-patterns-checklist.md)
- [docs/runbook_dr.md](docs/runbook_dr.md)

## Disclaimer

Herramienta personal de análisis. Modo semi-automatizado: el bot detecta y alerta, el usuario
ejecuta manualmente las apuestas en casas con permiso SEGOB (Caliente/Strendus/Codere).
Cumplir con LFPIORPI y reportes SAT según aplique.
