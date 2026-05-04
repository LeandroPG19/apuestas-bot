"""Fase 4.8 + 4.12 + 4.14: slippage + SHAP + latency columns.

Revision ID: 0013
Revises: 0012
Create Date: 2026-04-21

Añade a `bets`:
  - slippage_bps: diferencia bps entre odds_displayed vs odds_obtained (Fase 4.14)
  - detection_to_placement_seconds: latencia detector→bet (Fase 4.8)

Añade a `predictions`:
  - shap_top_features: top-5 features con SHAP values por pick (Fase 4.12)
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "bets",
        sa.Column(
            "slippage_bps",
            sa.Integer,
            nullable=True,
            comment="Basis points diff odds_displayed vs odds_obtained",
        ),
    )
    op.add_column(
        "bets",
        sa.Column(
            "detection_to_placement_seconds",
            sa.Integer,
            nullable=True,
            comment="Latencia desde detector emite pick hasta usuario coloca",
        ),
    )
    op.add_column(
        "bets",
        sa.Column(
            "odds_displayed",
            sa.Numeric(8, 3),
            nullable=True,
            comment="Odds mostradas en el bot al emitir pick",
        ),
    )
    op.add_column(
        "predictions",
        sa.Column(
            "shap_top_features",
            sa.dialects.postgresql.JSONB,
            nullable=True,
            comment="[{feature, value, contribution}, ...] top 5 SHAP",
        ),
    )
    op.create_index(
        "idx_bets_slippage",
        "bets",
        ["slippage_bps"],
        postgresql_where=sa.text("slippage_bps IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("idx_bets_slippage", table_name="bets")
    op.drop_column("predictions", "shap_top_features")
    op.drop_column("bets", "odds_displayed")
    op.drop_column("bets", "detection_to_placement_seconds")
    op.drop_column("bets", "slippage_bps")
