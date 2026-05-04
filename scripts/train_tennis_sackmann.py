"""Trainer tennis desde `tennis_matches_sackmann` — Sprint 12.

Entrena modelo binario P(winner=player_A) usando 36k+ matches ATP+WTA
con features serve/return rolling + rank diff + surface one-hot.

Uso:
    uv run python scripts/train_tennis_sackmann.py --tours atp,wta --since 2018
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apuestas.ml.train_base import TrainConfig, train_ensemble
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def load_matches(tours: list[str], since: int) -> pl.DataFrame:
    from sqlalchemy import text

    from apuestas.db import session_scope

    async with session_scope() as s:
        rows = (
            await s.execute(
                text(
                    """
                    SELECT tour, tourney_id, tourney_date, surface, tourney_level,
                           winner_id, winner_rank, winner_age,
                           loser_id, loser_rank, loser_age,
                           w_ace, w_df, w_svpt, w_1stin, w_1stwon, w_2ndwon, w_bpsaved, w_bpfaced,
                           l_ace, l_df, l_svpt, l_1stin, l_1stwon, l_2ndwon, l_bpsaved, l_bpfaced
                    FROM tennis_matches_sackmann
                    WHERE tour = ANY(:tours)
                      AND EXTRACT(YEAR FROM tourney_date) >= :since
                      AND tourney_date IS NOT NULL
                    ORDER BY tourney_date ASC
                    """
                ),
                {"tours": tours, "since": since},
            )
        ).fetchall()
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame([dict(r._mapping) for r in rows])


def build_frame(matches: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Expande 1 match → 2 rows simétricos (winner/loser) con y=1/0."""
    if matches.height == 0:
        return pl.DataFrame(), []

    def _side(df: pl.DataFrame, is_winner: bool) -> pl.DataFrame:
        pfx = "w" if is_winner else "l"
        opfx = "l" if is_winner else "w"
        return df.select(
            pl.col("tourney_date").alias("match_date"),
            pl.col("surface"),
            pl.col(f"{pfx}_ace").alias("ace"),
            pl.col(f"{pfx}_df").alias("df"),
            pl.col(f"{pfx}_svpt").alias("svpt"),
            pl.col(f"{pfx}_1stin").alias("first_in"),
            pl.col(f"{pfx}_1stwon").alias("first_won"),
            pl.col(f"{pfx}_2ndwon").alias("second_won"),
            pl.col(f"{pfx}_bpsaved").alias("bp_saved"),
            pl.col(f"{pfx}_bpfaced").alias("bp_faced"),
            pl.col(f"{'winner' if is_winner else 'loser'}_id").alias("player_id"),
            pl.col(f"{'loser' if is_winner else 'winner'}_id").alias("opponent_id"),
            pl.col(f"{'winner' if is_winner else 'loser'}_rank").alias("rank"),
            pl.col(f"{'loser' if is_winner else 'winner'}_rank").alias("opp_rank"),
            pl.col(f"{'winner' if is_winner else 'loser'}_age").alias("age"),
            pl.col(f"{'loser' if is_winner else 'winner'}_age").alias("opp_age"),
            pl.lit(1 if is_winner else 0).alias("y"),
        )

    a = _side(matches, True)
    b = _side(matches, False)
    df = pl.concat([a, b]).sort(["player_id", "match_date"])

    df = df.with_columns(
        [
            (pl.col("first_in").cast(pl.Float64) / pl.col("svpt").cast(pl.Float64).clip(1)).alias(
                "first_in_pct"
            ),
            (
                pl.col("first_won").cast(pl.Float64) / pl.col("first_in").cast(pl.Float64).clip(1)
            ).alias("first_won_pct"),
            (
                pl.col("bp_saved").cast(pl.Float64) / pl.col("bp_faced").cast(pl.Float64).clip(1)
            ).alias("bp_save_pct"),
            (pl.col("ace").cast(pl.Float64) / pl.col("svpt").cast(pl.Float64).clip(1)).alias(
                "ace_pct"
            ),
            (pl.col("df").cast(pl.Float64) / pl.col("svpt").cast(pl.Float64).clip(1)).alias(
                "df_pct"
            ),
            (pl.col("rank") - pl.col("opp_rank")).alias("rank_diff"),
            (pl.col("age") - pl.col("opp_age")).alias("age_diff"),
        ]
    )

    for col in ("ace_pct", "df_pct", "first_in_pct", "first_won_pct", "bp_save_pct"):
        df = df.with_columns(
            pl.col(col)
            .shift(1)
            .rolling_mean(window_size=10, min_samples=3)
            .over("player_id")
            .alias(f"{col}_roll_10")
        )

    df = df.drop_nulls(subset=["ace_pct_roll_10", "first_won_pct_roll_10"])

    df = df.with_columns(
        [
            (pl.col("surface") == "Hard").cast(pl.Int8).alias("surface_hard"),
            (pl.col("surface") == "Clay").cast(pl.Int8).alias("surface_clay"),
            (pl.col("surface") == "Grass").cast(pl.Int8).alias("surface_grass"),
        ]
    )

    feat_cols = [
        "rank_diff",
        "age_diff",
        "ace_pct_roll_10",
        "df_pct_roll_10",
        "first_in_pct_roll_10",
        "first_won_pct_roll_10",
        "bp_save_pct_roll_10",
        "surface_hard",
        "surface_clay",
        "surface_grass",
    ]
    return df, feat_cols


async def main_async(tours: list[str], since: int, n_trials: int) -> int:
    import mlflow

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "file:///tmp/mlflow_tennis_sackmann")
    )
    mlflow.set_experiment("tennis_sackmann")

    matches = await load_matches(tours, since)
    if matches.height == 0:
        print("Sin matches; ejecuta make ingest-sackmann-tennis primero")
        return 1

    logger.info("tennis_sackmann.loaded", matches=matches.height, tours=tours)

    frame, feat_cols = build_frame(matches)
    if frame.height == 0:
        print("Build frame vacío — rolling insuficiente")
        return 1

    logger.info("tennis_sackmann.frame", rows=frame.height, n_features=len(feat_cols))

    frame = frame.sort("match_date")
    n = frame.height
    n_train = int(n * 0.75)
    n_cal = int(n * 0.15)
    train_df = frame.slice(0, n_train)
    cal_df = frame.slice(n_train, n_cal)
    holdout_df = frame.slice(n_train + n_cal, n - n_train - n_cal)

    def _xy(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = df.select(feat_cols).fill_nan(0.0).fill_null(0.0).to_numpy()
        y = df["y"].to_numpy().astype(np.int8)
        return X, y

    X_tr, y_tr = _xy(train_df)
    X_ca, y_ca = _xy(cal_df)
    X_ho, y_ho = _xy(holdout_df)

    with mlflow.start_run(
        run_name=f"tennis_sackmann_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M')}"
    ):
        mlflow.log_params(
            {
                "tours": ",".join(tours),
                "since": since,
                "n_features": len(feat_cols),
                "n_train": len(y_tr),
                "n_holdout": len(y_ho),
            }
        )
        result = train_ensemble(
            X_tr,
            y_tr,
            X_ca,
            y_ca,
            X_ho,
            y_ho,
            feature_names=feat_cols,
            cfg=TrainConfig(n_trials=n_trials, random_state=42),
        )
        for k, v in result.metrics.items():
            try:
                mlflow.log_metric(k, float(v))
            except (TypeError, ValueError):
                pass

    print(
        f"✓ Tennis Sackmann trained: "
        f"log_loss={result.holdout_log_loss:.4f} "
        f"brier={result.holdout_brier:.4f} "
        f"ece={result.holdout_ece:.4f}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tours", type=str, default="atp,wta")
    parser.add_argument("--since", type=int, default=2018)
    parser.add_argument("--n-trials", type=int, default=10)
    args = parser.parse_args()
    return asyncio.run(
        main_async(
            [t.strip() for t in args.tours.split(",")],
            args.since,
            args.n_trials,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
