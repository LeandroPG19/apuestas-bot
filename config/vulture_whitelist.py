"""Whitelist para `vulture` — falsos positivos documentados (Deuda 5).

Uso recomendado:
    uv run vulture src/apuestas/ --min-confidence 90 \
        --ignore-names "window_games,match_id_resolver,psi_threshold,deep"

Falsos positivos explicados:
  - `window_games` (ingest/nfl.py): parámetro con default documentado para
    compute_epa_team_rolling; los call sites aceptan el default.
  - `match_id_resolver` (ingest/public_betting_pct.py): resolver opcional
    inyectable desde tests.
  - `psi_threshold` (ml/drift.py): umbral calibrable vía env/param.
  - `deep` (ml/train_base._StackingWrapper.get_params): sklearn API contract.

Tras inspección manual 2026-04-24: cero código muerto en src/apuestas/.
"""
