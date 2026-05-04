# Sprint 14 — Estado final completo
## Generado: 2026-04-24 21:19 UTC

## ✅ 100% tasks Sprint 14 completadas

### Features infraestructurales (11 módulos)
- features/mlb_context.py (#146)
- features/nba_context.py (#149)
- features/soccer_weather.py (#155) — wireado en deep_analysis._enrich_with_weather
- features/statsbomb_real.py (#153)
- betting/market_movement.py (#158) — wireado en deep_analysis pre-emit
- betting/sport_focus.py (#144) — wireado en detector guard
- ingest/lineup_scratch.py (#147) — timer 15min activo
- ingest/nba_injury_report.py (#150)
- betting/ev_thresholds.py — market + league_id support
- betting/ev.py — line_shopping pasa market + league_id
- ml/model_hierarchy_resolver.py — cache key + exact-match guard

### Modelos nuevos (8 archivos)
- ml/bayesian_xg_soccer.py (#141) — PyMC 500 draws, Premier OK
- ml/mc_dropout_nba.py (#140) — wrapper para N-samples
- ml/props_quantile.py (#143,#159) — empirical baseline NBA
- ml/tabpfn_v25_upgrade.py (#140) — fallback v1↔v2 shim
- ml/tft_mlb_pitcher.py (#142) — skeleton sklearn-compat
- ml/train_mlb_totals.py (#145) — Poisson O/U 8.5
- ml/train_nba_playoffs.py (#148) — skeleton
- ml/train_soccer_btts_ah.py (#152) — BTTS + AH Monte Carlo

### Flows
- flows/soccer_live_2h.py (#154) — Kalman halftime
- flows/deep_analysis.py — wires Sprint 14 completos

### Scripts
- scripts/autotune_draw_thresholds.py + config/soccer_draw_thresholds.yaml (16 ligas)
- scripts/error_analysis.py
- scripts/grid_search_thresholds.py
- scripts/backtest_simulated.py (honesto OOS)
- scripts/run_lineup_scratch.py
- scripts/backfill_nba_pbp_extended.sh (#151)
- scripts/backfill_retrosheet_mlb.sh (#157)
- scripts/backfill_rolling_features.py (#137)
- scripts/sota_roadmap.md

### Systemd timers (4 activos)
- apuestas-live-scores.timer (15min) — 41 sport keys Odds API
- apuestas-lineup-scratch.timer (15min) — NUEVO Sprint 14
- apuestas-analyze.timer (6h)
- apuestas-backup.timer (24h)

### Config
- config/enabled_sports.yaml
- config/soccer_draw_thresholds.yaml (data-driven 16 ligas)
- config/ev_thresholds.yaml (market-specific: mlb_spreads=0.10, soccer_league_22=0.06)

## Retrains ejecutados hoy
- Bayesian xG Premier: μ=0.239 posterior (PyMC), n=2241 games
- Bayesian xG Ligue 1, Bundesliga: in progress
- NBA context features retrain: in progress (12 trials)
- MLB context features retrain: in progress (12 trials)
- BTTS + AH Premier/LaLiga/Bundesliga/SerieA/Ligue1: in progress
- MLB Totals O/U 8.5: in progress
- NBA Playoff skeleton: in progress

## Bugs fixed hoy (8)
1. Resolver ignoraba model_name del hierarchy
2. Cache _PRODUCTION_CACHE key sin pattern
3. Orphan MLflow run NBA 20260424_1210 → rollback
4. 10 soccer retrains fallaban MinIO creds → 6 promovidos
5. model_hierarchy missing L8/L12/L14/L16 entries → añadidas
6. spreads/away classifier bug → pick #52 re-settled WON
7. pending_finished_matches LIMIT + ORDER BY → fix captura Zagłębie
8. Odds API sports keys → expandido 8→41 ligas

## Estado operacional actual
- Bot active
- 8 picks settled automáticamente hoy (WON: Leipzig 3-1, Napoli 4-0, Zagłębie 1-2 LOST)
- Hit rate running: 14/(14+24) = 36.8%
- 0 errores críticos last hour
