"""Backtest HONESTO sin trampas — valida el bot out-of-sample.

Reglas anti-trampa:
  1. Modelos production actuales (entrenados hasta 2024-25). Test window >=
     2025-08-01 garantiza out-of-sample (sin leakage).
  2. `build_match_features` extrae features t-1 (ya filtra data anterior al kickoff).
  3. `estimator.predict_proba(X)` — p_model REAL, no aproximación.
  4. Odds de SOFT books (betfair_ex_eu, draftkings, etc), no Pinnacle. Pinnacle
     sirve solo para p_fair_reference (devig).
  5. Label = outcome real del match. Sin stake, sin Kelly.
  6. Thresholds adaptativos del YAML + guards DetectorConfig aplicados.

Uso:
  python scripts/backtest_simulated.py --sport soccer --league 4 --since 2025-08-01 --until 2026-04-22
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.pop("PYTHON_GIL", None)
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minio-admin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "change-me-minio-password")

from sqlalchemy import text

from apuestas.betting.ev_thresholds import ev_threshold_for
from apuestas.betting.ev_thresholds import reset_cache as reset_ev_cache
from apuestas.db import session_scope
from apuestas.ml.model_hierarchy_resolver import resolve_and_load_model


def devig_mult(odds: list[float]) -> list[float]:
    inv = [1 / o for o in odds]
    s = sum(inv)
    return [v / s for v in inv]


SOFT_BOOKS = (
    "betfair_ex_eu",
    "betfair_ex_uk",
    "draftkings",
    "fanduel",
    "betmgm",
    "caesars",
    "gtbets",
    "coolbet",
    "leovegas_se",
    "williamhill_us",
    "unibet",
    "betway",
    "bet365",
    "betfair",
    "1xbet",
    "onexbet",
    "matchbook",
    "betfred",
    "ladbrokes",
    "pointsbetus",
)


async def fetch_ood_matches(
    sport: str, league_id: int | None, since: str, until: str, limit: int = 3000
) -> list[dict]:
    """Matches OOS con Pinnacle + best soft-book odds pre-kickoff."""
    async with session_scope() as s:
        q = """
            WITH pinn_snap AS (
              SELECT DISTINCT ON (oh.match_id, oh.outcome)
                oh.match_id, oh.outcome, oh.odds
              FROM odds_history oh
              JOIN matches m ON m.id = oh.match_id
              WHERE oh.bookmaker='pinnacle' AND oh.market='h2h'
                AND m.sport_code=:sport
                AND m.start_time BETWEEN :since AND :until
                AND m.home_score IS NOT NULL
                AND oh.ts <= m.start_time
                AND oh.ts >= m.start_time - INTERVAL '6 hours'
              ORDER BY oh.match_id, oh.outcome, oh.ts DESC
            ),
            soft_best AS (
              SELECT oh.match_id, oh.outcome, MAX(oh.odds) best_odds
              FROM odds_history oh
              JOIN matches m ON m.id=oh.match_id
              WHERE oh.bookmaker = ANY(:books)
                AND oh.market='h2h'
                AND m.sport_code=:sport
                AND m.start_time BETWEEN :since AND :until
                AND oh.ts <= m.start_time
                AND oh.ts >= m.start_time - INTERVAL '6 hours'
              GROUP BY oh.match_id, oh.outcome
            )
            SELECT m.id match_id, m.league_id, m.stage,
                   m.home_score hs, m.away_score as_,
                   MAX(CASE WHEN ps.outcome='home' THEN ps.odds END) pinn_home,
                   MAX(CASE WHEN ps.outcome='draw' THEN ps.odds END) pinn_draw,
                   MAX(CASE WHEN ps.outcome='away' THEN ps.odds END) pinn_away,
                   MAX(CASE WHEN sb.outcome='home' THEN sb.best_odds END) soft_home,
                   MAX(CASE WHEN sb.outcome='draw' THEN sb.best_odds END) soft_draw,
                   MAX(CASE WHEN sb.outcome='away' THEN sb.best_odds END) soft_away
            FROM matches m
            JOIN pinn_snap ps ON ps.match_id=m.id
            LEFT JOIN soft_best sb ON sb.match_id=m.id
            WHERE m.sport_code=:sport
              AND m.start_time BETWEEN :since AND :until
        """
        since_dt = (
            datetime.fromisoformat(since).replace(tzinfo=UTC) if isinstance(since, str) else since
        )
        until_dt = (
            datetime.fromisoformat(until).replace(tzinfo=UTC) if isinstance(until, str) else until
        )
        params = {
            "sport": sport,
            "since": since_dt,
            "until": until_dt,
            "lim": limit,
            "books": list(SOFT_BOOKS),
        }
        if league_id is not None:
            q += " AND m.league_id=:lg"
            params["lg"] = league_id
        q += """
            GROUP BY m.id
            ORDER BY m.start_time
            LIMIT :lim
        """
        rows = (await s.execute(text(q), params)).fetchall()
    return [dict(r._mapping) for r in rows]


async def compute_p_model(
    estimator, match_row: dict, sport: str, outcomes: list[str]
) -> dict[str, float] | None:
    """Llama build_match_features + estimator.predict_proba. Sin trampas."""
    try:
        from sqlalchemy import text as _text

        from apuestas.features.feature_store import build_match_features

        async with session_scope() as s:
            tr = (
                await s.execute(
                    _text(
                        "SELECT home_team_id, away_team_id, start_time FROM matches WHERE id=:mid"
                    ),
                    {"mid": match_row["match_id"]},
                )
            ).first()
        if tr is None:
            return None
        expected = getattr(estimator, "feature_names_in_", None)
        feature_names = list(expected) if expected is not None else []
        if not feature_names:
            return None
        features = await build_match_features(
            sport_code=sport,
            home_team_id=int(tr.home_team_id),
            away_team_id=int(tr.away_team_id),
            match_start=tr.start_time,
            feature_names=feature_names,
            min_coverage=0.30,
        )
    except Exception:
        return None
    if features is None:
        return None

    # features = np.ndarray shape (n_features,)
    try:
        import numpy as _np

        X = _np.asarray(features).reshape(1, -1)
        proba = estimator.predict_proba(X)[0]
        classes = getattr(estimator, "classes_", None)
        if classes is None:
            # 3-way: [home, draw, away]; 2-way: [home, away]
            if len(proba) == 3:
                return {"home": float(proba[0]), "draw": float(proba[1]), "away": float(proba[2])}
            return {"home": float(proba[0]), "away": float(proba[1])}
        out: dict[str, float] = {}
        for cls, pr in zip(classes, proba, strict=False):
            key = str(cls).lower()
            if key not in ("home", "away", "draw"):
                if key in ("0", "h"):
                    key = "home"
                elif key in ("1", "d"):
                    key = "draw"
                elif key in ("2", "a"):
                    key = "away"
            out[key] = float(pr)
        return out
    except Exception:
        return None


def simulate_pick(
    *,
    sport: str,
    league_id: int | None,
    stage: str | None,
    outcome: str,
    p_model: float,
    market_odds: float,
    p_draw_pinn: float | None,
    use_new: bool,
) -> bool:
    ev = market_odds * p_model - 1.0
    thr = (
        ev_threshold_for(sport=sport, market="h2h", stage=stage, league_id=league_id)
        if use_new
        else 0.03
    )
    if ev < thr:
        return False
    if use_new:
        if sport == "nba" and stage == "playoff":
            return False
        if p_draw_pinn is not None and outcome in ("home", "away"):
            draw_thr = 0.30 if league_id in (22, 253) else 0.25
            if p_draw_pinn >= draw_thr:
                return False
    return True


def classify(outcome: str, hs: int, as_: int) -> str:
    if hs > as_:
        w = "home"
    elif as_ > hs:
        w = "away"
    else:
        w = "draw"
    return "won" if outcome == w else "lost"


async def run(sport: str, league_id: int | None, since: str, until: str, use_new: bool):
    reset_ev_cache()
    rows = await fetch_ood_matches(sport, league_id, since, until)
    print(f"  [{('NEW' if use_new else 'LEGACY')}] {len(rows)} OOS candidate matches")

    async with session_scope() as s:
        resolved = await resolve_and_load_model(
            s, sport_code=sport, market="h2h", league_id=league_id
        )
    if resolved is None:
        print("  [!] No model available")
        return None
    info, estimator = resolved
    print(f"  Model: {info.model_name} v{info.model_version} (priority={info.priority})")

    picks = []
    n_no_features = 0
    for r in rows:
        if r["pinn_home"] is None or r["pinn_away"] is None:
            continue
        n3 = r["pinn_draw"] is not None
        pinn_odds = (
            [r["pinn_home"], r["pinn_draw"], r["pinn_away"]]
            if n3
            else [r["pinn_home"], r["pinn_away"]]
        )
        soft_odds = (
            [r["soft_home"], r["soft_draw"], r["soft_away"]]
            if n3
            else [r["soft_home"], r["soft_away"]]
        )
        labels = ["home", "draw", "away"] if n3 else ["home", "away"]
        pinn_fair = devig_mult([float(o) for o in pinn_odds])
        p_draw_pinn = pinn_fair[1] if n3 else None

        p_model_map = await compute_p_model(estimator, r["match_id"], sport, labels)
        if p_model_map is None:
            n_no_features += 1
            continue

        for i, outcome in enumerate(labels):
            soft_price = soft_odds[i]
            if soft_price is None:
                continue  # no soft book offered esta línea
            odds_market = float(soft_price)
            p_model = p_model_map.get(outcome)
            if p_model is None or p_model <= 0.0:
                continue
            emit = simulate_pick(
                sport=sport,
                league_id=r["league_id"],
                stage=r.get("stage"),
                outcome=outcome,
                p_model=p_model,
                market_odds=odds_market,
                p_draw_pinn=p_draw_pinn,
                use_new=use_new,
            )
            if not emit:
                continue
            result = classify(outcome, int(r["hs"]), int(r["as_"]))
            picks.append(
                {
                    "match_id": r["match_id"],
                    "outcome": outcome,
                    "odds": odds_market,
                    "p_model": p_model,
                    "p_pinn": pinn_fair[i],
                    "result": result,
                    "league_id": r["league_id"],
                }
            )
    if n_no_features:
        print(f"  [warn] {n_no_features} matches skipped (no features available)")

    if not picks:
        return {
            "n_picks": 0,
            "won": 0,
            "lost": 0,
            "hit_rate": 0.0,
            "avg_odds": 0.0,
            "implied": 0.0,
            "hr_minus_implied": 0.0,
            "roi": 0.0,
            "profit": 0.0,
            "brier": 0.0,
        }

    won = sum(1 for p in picks if p["result"] == "won")
    lost = len(picks) - won
    profit = sum((p["odds"] - 1) if p["result"] == "won" else -1.0 for p in picks)
    avg_odds = sum(p["odds"] for p in picks) / len(picks)
    implied = 1 / avg_odds if avg_odds else 0.0
    hit_rate = won / len(picks)
    brier = sum((p["p_model"] - (1.0 if p["result"] == "won" else 0.0)) ** 2 for p in picks) / len(
        picks
    )
    return {
        "n_picks": len(picks),
        "won": won,
        "lost": lost,
        "hit_rate": hit_rate,
        "avg_odds": avg_odds,
        "implied": implied,
        "hr_minus_implied": hit_rate - implied,
        "roi": profit / len(picks),
        "profit": profit,
        "brier": brier,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", required=True)
    ap.add_argument("--league", type=int, default=None)
    ap.add_argument("--since", required=True)
    ap.add_argument("--until", required=True)
    args = ap.parse_args()

    print("=" * 78)
    print(f"BACKTEST HONESTO — {args.sport} lg={args.league} OOS {args.since} → {args.until}")
    print("Modelos production (train ≤ 2024-25), test OOS, soft-book odds, sin leakage")
    print("=" * 78)

    legacy = await run(args.sport, args.league, args.since, args.until, use_new=False)
    new = await run(args.sport, args.league, args.since, args.until, use_new=True)

    for label, r in [
        ("LEGACY (EV≥0.03 flat, sin guards)", legacy),
        ("NEW (thresholds adaptativos + guards)", new),
    ]:
        if r is None:
            continue
        print(f"\n--- {label} ---")
        print(f"  n_picks={r['n_picks']}  won={r['won']}  lost={r['lost']}")
        if r["n_picks"]:
            print(
                f"  hit_rate={r['hit_rate']:.3f}  avg_odds={r['avg_odds']:.3f}  implied={r['implied']:.3f}"
            )
            print(f"  HR−implied={r['hr_minus_implied']:+.3f}  (Buchdahl: >+0.02 = skill real)")
            print(f"  ROI flat $1/pick={r['roi']:+.4f}  profit_total=${r['profit']:+.2f}")
            print(f"  Brier={r['brier']:.4f}")

    if legacy and new and legacy["n_picks"]:
        print(
            f"\n>>> n_picks: {legacy['n_picks']} → {new['n_picks']} ({(new['n_picks'] - legacy['n_picks']) / legacy['n_picks'] * 100:+.1f}%)"
        )
        print(f">>> ROI delta: {new['roi'] - legacy['roi']:+.4f}")
        print(f">>> HR delta: {new['hit_rate'] - legacy['hit_rate']:+.3f}")


if __name__ == "__main__":
    asyncio.run(main())
