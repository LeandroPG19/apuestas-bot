# Runbook — operación cotidiana

## Arranque / apagado

| Acción | Comando |
|---|---|
| Encender stack | `make up` |
| Análisis on-demand | `make analyze` |
| Estado | `make status` |
| Logs streaming | `make logs` |
| Apagar limpio | `make down` |
| Reiniciar un servicio | `docker compose restart api` |

## Problemas comunes

### llm container no arranca — `CUDA_ERROR_OUT_OF_MEMORY`
- Verifica que NO haya otro proceso usando GPU: `nvidia-smi`
- Cierra navegador y apps GUI pesadas
- Si persiste, baja `LLAMA_ARG_CTX_SIZE` de 8192 a 4096 en `docker-compose.gpu.yml`

### Postgres healthcheck failing tras cold-start
- Espera 60 s: TimescaleDB `init-db` tarda
- `docker compose logs postgres | tail -50` → busca errores de extensión
- Si extensiones faltan, borra volumen: `docker compose down -v && make cold-start`

### VRAM saturada (>5.9 GB)
- Qwen (4.7 GB) + TEI (0.6 GB) + driver = ~5.3 GB deja ~700 MB
- Si pasas ese margen, bajar a quantización Q4_0 (más liviano): editar `LLAMA_ARG_MODEL`
- O usar BGE-M3 en quantización aún más agresiva (INT4)

### Telegram bot offline
- Verifica `TELEGRAM_BOT_TOKEN` en `.env`
- `docker compose logs telegram` → busca autenticación rechazada
- Reconfigura token con @BotFather y reinicia: `docker compose restart telegram`

### Rate limit APIs externas
- The Odds API free = 500 créditos/mes: mira `/metrics` → `api_rate_limit_remaining`
- API-Football Pro = 7500/día: distribuye ingesta con exponential backoff
- Si agotas, cae a RSS/Reddit como fallback automático (circuit breaker)

## Comandos útiles ad-hoc

```bash
# Ver queries lentas últimas 24 h
docker compose exec postgres psql -U apuestas -c \
  "SELECT mean_exec_time, calls, query FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10"

# Tamaño tablas
docker compose exec postgres psql -U apuestas -c \
  "SELECT relname, pg_size_pretty(pg_total_relation_size(relid)) FROM pg_catalog.pg_statio_user_tables ORDER BY pg_total_relation_size(relid) DESC LIMIT 20"

# Comprimir chunks TimescaleDB manualmente
docker compose exec postgres psql -U apuestas -c \
  "SELECT compress_chunk(c) FROM show_chunks('odds_history', older_than => INTERVAL '3 days') c"

# Estado Prefect flows
docker compose exec api prefect deployment ls

# Ver último backup
ls -lh backups/ | tail -3
```

## Mantenimiento semanal recomendado

1. `make backup` (cada domingo 03:00 idealmente vía cron personal)
2. `make audit-python` — pip-audit
3. Revisar dashboard "Calibración" y post-mortems flagged
4. `make retrain SPORT=nba` si CLV 7d negativo
5. Limpiar backups antiguos: `find backups/ -name "pg_*.dump" -mtime +14 -delete`

## Emergencia

### Disco lleno
```bash
# Top consumidores
du -sh /var/lib/docker/* | sort -h | tail
# Vaciar logs antiguos
docker compose logs --no-color 2>&1 | head -0  # truncates
sudo journalctl --vacuum-time=7d
# Purgar imágenes sin usar
docker system prune -af
```

### Corrupción DB → rollback
```bash
make down
docker compose up -d postgres valkey
make restore FILE=backups/pg_YYYY-MM-DD.dump
make up
```

Ver [runbook_dr.md](runbook_dr.md) para DR completo.
