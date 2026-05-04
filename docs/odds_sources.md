# Fuentes de odds configuradas

El bot combina 4+ fuentes para tener cobertura completa sin depender de APIs pagas.

## Stack actual (todas gratis)

| Fuente | Status default | Costo | Cobertura | Integración |
|---|---|---|---|---|
| **Pinnacle guest API** | ✅ Activa siempre | $0 | NBA, MLB, NFL, NHL, EPL, LaLiga, Bundesliga, Serie A, UCL, Liga MX, Tenis ATP/WTA | `ingest/pinnacle_scraper.py` · token guest público · incluida en catchup_flow |
| **Caliente.mx (scraping)** | ✅ Activa siempre | $0 | Liga MX + principales MX | `ingest/caliente.py` · camoufox anti-Cloudflare |
| **API-Football** | ⚠️ Requiere key | free/paid | Fútbol solo · 100 req/día free | `ingest/api_football.py` |
| **The Odds API** | ⚠️ Requiere key | $30/mes | Multi-book consenso | `ingest/odds_api.py` · circuit breaker 24h si quota |
| **Polymarket** | ✅ Activa siempre | $0 | Prediction markets NBA/NFL/MLB/UCL | `ingest/polymarket.py` |
| **Betfair Exchange** | ⚠️ Opcional (recomendado) | $0 + £10 depósito activación cuenta | EPL, UCL, tenis, NBA/MLB/NFL parcial | `ingest/betfair_exchange.py` · fail-soft sin creds |
| **DraftKings (scrape)** | ⚠️ Opt-in flag | $0 | NBA, NFL, MLB, NHL, EPL US | `ingest/us_books_scraper.py` · camoufox |
| **FanDuel (scrape)** | ⚠️ Opt-in flag | $0 | NBA, NFL, MLB, NHL | `ingest/us_books_scraper.py` · camoufox |
| **BetMGM (scrape)** | ⚠️ Opt-in flag | $0 | NBA, NFL, MLB, NHL | `ingest/us_books_scraper.py` · camoufox |
| **football-data.org** | ⚠️ Requiere key free | $0 | Fixtures EPL/LaLiga/UCL (sin odds, solo eventos) | `ingest/free_sources.py` |
| **ESPN / nba_api / MLB / NHL Stats** | ✅ Activas | $0 | Scoreboards, box scores | `ingest/*.py` |

## Cómo activar cada opcional

### 1. Betfair Exchange (recomendado #1)

Es la fuente sharp peer-to-peer con los precios más cercanos a probabilidad real. **Delayed API key es gratis** (lag 1-180s, OK para pre-match).

Pasos:
1. Crear cuenta en https://www.betfair.com/ (requiere depósito £10 único para activar).
2. Visitar https://apps.betfair.com/visualisers/api-ng-account-operations/ → login → "Get developer app keys" → elegir **Delayed** (gratis, sin cap mensual).
3. Añadir al `.env`:
   ```
   BETFAIR_APP_KEY=xxxxxxxxxxxxx
   BETFAIR_USERNAME=tu-email@example.com
   BETFAIR_PASSWORD=tu-password
   BETFAIR_CERT_PATH=  # opcional si generaste client cert para login non-interactive
   ```
4. Validar: TUI → tab Setup → `🐴 Test Betfair Exchange`.

### 2. DraftKings / FanDuel / BetMGM

Los 3 US books tienen endpoints JSON públicos pero con Akamai Bot Manager. Se usa `camoufox` (ya instalado para Caliente).

Pasos:
1. Setear flags en `.env` (default `false`):
   ```
   APUESTAS_ENABLE_DK=true
   APUESTAS_ENABLE_FANDUEL=true
   APUESTAS_ENABLE_BETMGM=true
   APUESTAS_US_BOOKS_STATE=nj   # o pa, co, in, mi, ny, va, oh, il, ks, md, tn, ia
   ```
2. Validar: TUI → tab Setup → `🇺🇸 Test US books`.

**Nota TOS**: estos scrapers violan los Terms of Service de los 3 sitios. Es tu responsabilidad aceptar el riesgo. Uso recomendado: benchmarking interno, no compartir datos scrapeados.

### 3. The Odds API (si quieres pagar)

Si renuevas la key:
```
THE_ODDS_API_KEY=your-paid-key
```
El bot usa circuit breaker 24h automáticamente si se detecta `Quota exhausted` para no spammear retries.

## Orden de preferencia (fair-value de-vigging §7)

Cuando `betting/clv.py` busca closing line, usa esta prioridad:
```python
CASE WHEN bookmaker = 'pinnacle' THEN 0
     WHEN bookmaker = 'circa'    THEN 1
     WHEN bookmaker = 'betfair'  THEN 2
     WHEN bookmaker = 'bookmaker' THEN 3
     ELSE 10 END
```
Pinnacle es fair sharp, Betfair es fair exchange. Para `multiplicative/power/shin` de-vigging (§7), usa Pinnacle si disponible, Betfair como fallback.

## Ejecución automática

`catchup_flow` (parte de `deep_analysis`) corre en paralelo:
- `catchup_pinnacle_guest` · siempre
- `catchup_betfair_exchange` · fail-soft si no hay creds
- `catchup_us_books` · solo books habilitados en `.env`
- `catchup_soccer_fixtures` · API-Football o football-data.org
- `catchup_soccer_odds` · API-Football (si key válida)
- `catchup_odds_api` · The Odds API (circuit-breaker si quota)
- `catchup_nba_scoreboard` · nba_api scoreboard
- `catchup_news` · RSS + Bluesky + Reddit

Cuando ejecutas `/analyze` desde TUI o Telegram, todo esto corre ~2-3 min y persiste en `odds_history` + `matches`.

## Verificación live

```bash
# En TUI
apuestas
# → tab Setup → 🎯 Test Pinnacle guest
# → tab Dashboard → A (disparar análisis)

# Vía CLI / scripts
PYTHONPATH=src python -m apuestas.flows.catchup
```
