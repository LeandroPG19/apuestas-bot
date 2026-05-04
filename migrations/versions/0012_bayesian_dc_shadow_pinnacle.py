"""Fase 3.5 + 2.3: team_strength_bayesian + shadow_pinnacle_predictions.

Revision ID: 0012
Revises: 0011
Create Date: 2026-04-21
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ─── team_strength_bayesian ──────────────────────────────────────────
    op.create_table(
        "team_strength_bayesian",
        sa.Column(
            "team_id",
            sa.BigInteger,
            sa.ForeignKey("teams.id"),
            primary_key=True,
        ),
        sa.Column("attack_rating", sa.Numeric(6, 4), nullable=False, server_default="1.0"),
        sa.Column("defense_rating", sa.Numeric(6, 4), nullable=False, server_default="1.0"),
        sa.Column("variance", sa.Numeric(6, 4), nullable=False, server_default="0.25"),
        sa.Column("n_matches", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "idx_team_strength_n_matches",
        "team_strength_bayesian",
        ["n_matches"],
    )

    # ─── shadow_pinnacle_predictions (auditoría Fase 2.3) ────────────────
    op.create_table(
        "shadow_pinnacle_predictions",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("match_id", sa.BigInteger, sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("outcome", sa.Text, nullable=False),
        sa.Column("p_model_primary", sa.Numeric(6, 5), nullable=False),
        sa.Column("p_shadow_pinnacle", sa.Numeric(6, 5), nullable=False),
        sa.Column("divergence", sa.Numeric(6, 5), nullable=False),
        sa.Column(
            "tier",
            sa.Text,
            nullable=False,
            comment="aligned | normal | divergent",
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "tier IN ('aligned', 'normal', 'divergent')",
            name="ck_shadow_tier",
        ),
    )
    op.create_index(
        "idx_shadow_match",
        "shadow_pinnacle_predictions",
        ["match_id", "created_at"],
    )
    op.create_index(
        "idx_shadow_tier",
        "shadow_pinnacle_predictions",
        ["tier", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_shadow_tier", table_name="shadow_pinnacle_predictions")
    op.drop_index("idx_shadow_match", table_name="shadow_pinnacle_predictions")
    op.drop_table("shadow_pinnacle_predictions")
    op.drop_index(
        "idx_team_strength_n_matches",
        table_name="team_strength_bayesian",
    )
    op.drop_table("team_strength_bayesian")
