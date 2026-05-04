"""B3: Bootstrap Dixon-Coles priors desde FBref + penaltyblog.

Estrategia:
1. `soccerdata.FBref` scrapea schedule + goles finales 2 últimas temporadas.
2. `penaltyblog.DixonColesGoalModel` fit por liga (no pool inter-liga, cada una
   tiene escala distinta).
3. Extrae `attack_X`, `defence_X` por equipo.
4. Resuelve team_id interno via `team_resolver` (fuzzy con RapidFuzz).
5. Upsert `team_strength_bayesian` con variance=0.10 (informed prior) y
   n_matches=20 sintéticos (para que 2-3 settlements no descalibren).

Penaltyblog param convention:
- `attack_X`: log-linear, >0 → más ofensivo que promedio liga.
- `defence_X`: log-linear, <0 → MEJOR defensa (concede menos goles).
- `home_advantage`: constante por liga.

Para alinear con schema bayesian_dc.py (attack_rating, defense_rating positivos
con PRIOR_ATTACK=1.0, PRIOR_DEFENSE=1.0 multiplicativo), convertimos:
- attack_rating = exp(attack_X + home_advantage/2)  # ~ baseline 1.0
- defense_rating = exp(-defence_X + home_advantage/2)  # invertir signo: menos golpe = mejor
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from apuestas.db import session_scope
from apuestas.ingest.team_resolver import resolve_team_id
from apuestas.obs.logging import configure_logging, get_logger

logger = get_logger(__name__)

# FBref league code → sport_code interno (convención BD: sin prefijo 'soccer_').
# Fallback secundario: 'soccer' genérico si la liga específica no existe en teams.
LEAGUE_MAP: dict[str, tuple[str, ...]] = {
    "ENG-Premier League": ("epl", "soccer"),
    "ESP-La Liga": ("laliga", "soccer"),
    "GER-Bundesliga": ("bundesliga", "soccer"),
    "ITA-Serie A": ("seriea", "soccer"),
    "FRA-Ligue 1": ("ligue1", "soccer"),
    "MEX-Liga MX": ("liga_mx", "soccer"),
}


async def upsert_bayesian_prior(
    *,
    team_id: int,
    attack: float,
    defense: float,
    variance: float,
    n_matches: int,
) -> None:
    async with session_scope() as session:
        await session.execute(
            text(
                """
                INSERT INTO team_strength_bayesian
                    (team_id, attack_rating, defense_rating, variance, n_matches, updated_at)
                VALUES (:tid, :a, :d, :v, :n, NOW())
                ON CONFLICT (team_id) DO UPDATE
                  SET attack_rating = EXCLUDED.attack_rating,
                      defense_rating = EXCLUDED.defense_rating,
                      variance = EXCLUDED.variance,
                      n_matches = EXCLUDED.n_matches,
                      updated_at = NOW()
                """
            ),
            {"tid": team_id, "a": attack, "d": defense, "v": variance, "n": n_matches},
        )


def _fit_dc_for_league(schedule: Any) -> dict[str, Any] | None:
    """Fit penaltyblog goal model. Intenta Dixon-Coles (con tau correction)
    y cae a Poisson simple si el optimizer SLSQP falla (caso común en EPL
    por correlaciones en la constraint sum_attack = n_teams).

    Retorna dict con attack/defence por team.
    """
    from penaltyblog.models import DixonColesGoalModel, PoissonGoalsModel

    df = schedule[
        schedule["home_team"].notna()
        & schedule["away_team"].notna()
        & schedule["home_score"].notna()
        & schedule["away_score"].notna()
    ].copy()

    if len(df) < 50:
        logger.warning("bootstrap_dc.insufficient_data", matches=len(df))
        return None

    goals_h = df["home_score"].astype(int).tolist()
    goals_a = df["away_score"].astype(int).tolist()
    teams_h = df["home_team"].tolist()
    teams_a = df["away_team"].tolist()

    # Attempt 1: full Dixon-Coles (tau correction for low-score draws).
    model_kind = "dixon_coles"
    try:
        model = DixonColesGoalModel(goals_h, goals_a, teams_h, teams_a)
        model.fit()
        params = model.get_params()
    except Exception as exc:
        logger.info(
            "bootstrap_dc.fallback_to_poisson",
            matches=len(df),
            dc_error=str(exc)[:120],
        )
        # Attempt 2: simpler Poisson goal model (no tau). Mismos params
        # attack_X/defence_X/home_advantage, sin rho.
        try:
            model = PoissonGoalsModel(goals_h, goals_a, teams_h, teams_a)
            model.fit()
            params = model.get_params()
            model_kind = "poisson"
        except Exception as exc2:
            logger.warning("bootstrap_dc.both_models_failed", error=str(exc2)[:200])
            return None
    logger.info("bootstrap_dc.model_fit", kind=model_kind, matches=len(df))

    teams = list(set(df["home_team"]).union(df["away_team"]))
    # Penaltyblog Poisson/DC convention: attack_X, defence_X centrados tal que
    # sum(attack_X) = n_teams (constraint). Team promedio de la liga → ~1.0.
    # Para alinear con bayesian_dc.PRIOR_ATTACK=1.0 / PRIOR_DEFENSE=1.0,
    # normalizamos dividiendo por la media liga (debería ser ~1 pero robustez):
    attacks_raw = [float(params[f"attack_{t}"]) for t in teams if f"attack_{t}" in params]
    defences_raw = []
    for t in teams:
        v = params.get(f"defence_{t}") or params.get(f"defense_{t}")
        if v is not None:
            defences_raw.append(float(v))
    if not attacks_raw or not defences_raw:
        return None
    mean_att = float(np.mean(attacks_raw)) or 1.0
    mean_def = float(np.mean(defences_raw)) or 1.0

    out: dict[str, dict[str, float]] = {}
    for team in teams:
        att = params.get(f"attack_{team}")
        defv = params.get(f"defence_{team}") or params.get(f"defense_{team}")
        if att is None or defv is None:
            continue
        # Ratio normalizado: team_attack / league_mean_attack. Valores típicos
        # [0.5, 2.0] con mean=1.0. Cap razonable para outliers extremos.
        attack_rating = float(att) / mean_att
        defense_rating = float(defv) / mean_def
        attack_rating = max(0.4, min(attack_rating, 2.5))
        defense_rating = max(0.4, min(defense_rating, 2.5))
        out[team] = {"attack": attack_rating, "defense": defense_rating}
    return out


async def bootstrap_league(fb_league: str, sport_codes: tuple[str, ...]) -> dict[str, int]:
    """Bootstrap DC for one FBref league. sport_codes es tupla ordenada de
    códigos internos a probar (ej ('epl', 'soccer')). Se usa el primero que
    encuentre matches en la BD con ese sport_code.

    Returns {fitted, resolved, inserted}.
    """
    # Choose sport_code that has teams in DB
    async with session_scope() as session:
        sport_code = sport_codes[0]
        for sc in sport_codes:
            r = await session.execute(
                text("SELECT 1 FROM teams WHERE sport_code = :sp LIMIT 1"),
                {"sp": sc},
            )
            if r.first():
                sport_code = sc
                break
    import soccerdata as sd

    logger.info("bootstrap_dc.league_start", league=fb_league, sport=sport_code)

    try:
        fb = sd.FBref(leagues=[fb_league], seasons=["2023-2024", "2024-2025"])
        schedule = fb.read_schedule()
        schedule = schedule.reset_index()
    except Exception as exc:
        logger.warning("bootstrap_dc.fbref_fail", league=fb_league, error=str(exc)[:200])
        return {"fitted": 0, "resolved": 0, "inserted": 0}

    # Normalize column names - soccerdata returns 'score' as "0–3" (em-dash),
    # "2-1" (hyphen) o NaN si el match no se jugó. Regex robusto:
    if "score" in schedule.columns and "home_score" not in schedule.columns:
        import re

        _RE = re.compile(r"(\d+)\s*[–\-—]\s*(\d+)")

        def _parse(s: Any) -> tuple[int | None, int | None]:
            if not isinstance(s, str):
                return None, None
            m = _RE.search(s)
            if not m:
                return None, None
            return int(m.group(1)), int(m.group(2))

        parsed = schedule["score"].apply(_parse)
        schedule["home_score"] = [p[0] for p in parsed]
        schedule["away_score"] = [p[1] for p in parsed]

    team_params = _fit_dc_for_league(schedule)
    if team_params is None:
        return {"fitted": 0, "resolved": 0, "inserted": 0}

    resolved = 0
    inserted = 0
    for team_name, params in team_params.items():
        try:
            team_id = await resolve_team_id(
                source="fbref",
                external_id=team_name,
                external_name=team_name,
                sport_code=sport_code,
                auto_link_threshold=88.0,  # más conservador en bootstrap
            )
        except Exception as exc:
            logger.debug("bootstrap_dc.resolve_fail", team=team_name, error=str(exc)[:100])
            continue

        if team_id is None:
            continue
        resolved += 1

        try:
            await upsert_bayesian_prior(
                team_id=team_id,
                attack=params["attack"],
                defense=params["defense"],
                variance=0.10,
                n_matches=20,
            )
            inserted += 1
        except Exception as exc:
            logger.warning("bootstrap_dc.upsert_fail", team_id=team_id, error=str(exc)[:100])

    logger.info(
        "bootstrap_dc.league_done",
        league=fb_league,
        sport=sport_code,
        fitted=len(team_params),
        resolved=resolved,
        inserted=inserted,
    )
    return {"fitted": len(team_params), "resolved": resolved, "inserted": inserted}


async def main() -> None:
    configure_logging()
    total = {"fitted": 0, "resolved": 0, "inserted": 0}
    for fb_league, sport_codes in LEAGUE_MAP.items():
        try:
            c = await bootstrap_league(fb_league, sport_codes)
            for k, v in c.items():
                total[k] += v
        except Exception as exc:
            logger.exception("bootstrap_dc.league_fail", league=fb_league, error=str(exc))
    logger.info("bootstrap_dc.all_done", **total)


if __name__ == "__main__":
    asyncio.run(main())
