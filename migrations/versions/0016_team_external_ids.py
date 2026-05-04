"""B1: Team entity resolution — cross-source ID mapping + review queue.

Revision ID: 0016
Revises: 0015
Create Date: 2026-04-22

Resuelve el problema de team duplicados entre fuentes:
- Sofascore crea teams con IDs propios (16363 "Vanagai") mientras el id=15
  "Detroit Pistons" tiene toda la historia rolling.
- El feature_store retorna None para matches con IDs de Sofascore.

Diseño (recomendado por splink/dedupe community):
- `team_external_id`: 1-to-many, cualquier source → canonical team_id.
- `team_match_review`: cola para matches fuzzy 75-95% que requieren revisión
  manual antes de auto-link.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0016"
down_revision: str | None = "0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "team_external_id",
        sa.Column(
            "team_id", sa.Integer, sa.ForeignKey("teams.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column(
            "source",
            sa.Text,
            nullable=False,
            comment="sofascore|nba_api|nhl_api|odds_api|pinnacle|clubelo|fbref|football_data_org",
        ),
        sa.Column("external_id", sa.Text, nullable=False),
        sa.Column("confidence", sa.Numeric(3, 2), nullable=True, comment="0.00-1.00"),
        sa.Column("verified", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
        sa.PrimaryKeyConstraint("source", "external_id", name="pk_team_external_id"),
    )
    op.create_index("ix_team_external_id_team", "team_external_id", ["team_id"])
    op.create_index(
        "ix_team_external_id_source_verified",
        "team_external_id",
        ["source", "verified"],
        postgresql_where=sa.text("verified = true"),
    )

    op.create_table(
        "team_match_review",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("source", sa.Text, nullable=False),
        sa.Column("external_id", sa.Text, nullable=False),
        sa.Column("external_name", sa.Text, nullable=False),
        sa.Column("sport_code", sa.Text, nullable=True),
        sa.Column(
            "candidate_team_id",
            sa.Integer,
            sa.ForeignKey("teams.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("candidate_name", sa.Text, nullable=True),
        sa.Column("score", sa.Numeric(3, 2), nullable=True, comment="fuzzy match score 0-1"),
        sa.Column(
            "status",
            sa.Text,
            nullable=False,
            server_default=sa.text("'pending'"),
            comment="pending|approved|rejected|new_team",
        ),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("resolved_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.create_index("ix_team_match_review_status", "team_match_review", ["status"])
    op.create_unique_constraint(
        "uq_team_match_review_source_ext",
        "team_match_review",
        ["source", "external_id"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_team_match_review_source_ext", "team_match_review", type_="unique")
    op.drop_index("ix_team_match_review_status", table_name="team_match_review")
    op.drop_table("team_match_review")
    op.drop_index("ix_team_external_id_source_verified", table_name="team_external_id")
    op.drop_index("ix_team_external_id_team", table_name="team_external_id")
    op.drop_table("team_external_id")
