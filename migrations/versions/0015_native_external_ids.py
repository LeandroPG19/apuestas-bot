"""P1: External IDs nativos para NBA/NHL + The Odds API.

Revision ID: 0015
Revises: 0014
Create Date: 2026-04-22

Añade a `matches`:
  - external_id_nba TEXT: game_id oficial nba_api (ej: '0022300001')
  - external_id_nhl TEXT: gameId oficial NHL API (ej: '2024020001')
  - external_id_odds_api TEXT: event.id de The Odds API

Permite `live_scores_flow` llamar APIs nativas directamente cuando el
ID nativo existe. Si es NULL → fallback a fuzzy team-name matching.

Índices parciales para no inflar índice global (solo rows con ID nativo).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0015"
down_revision: str | None = "0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "matches",
        sa.Column(
            "external_id_nba",
            sa.Text,
            nullable=True,
            comment="NBA game_id oficial (nba_api), ej: '0022300001'",
        ),
    )
    op.add_column(
        "matches",
        sa.Column(
            "external_id_nhl",
            sa.Text,
            nullable=True,
            comment="NHL gameId oficial (api-web.nhle.com), ej: '2024020001'",
        ),
    )
    op.add_column(
        "matches",
        sa.Column(
            "external_id_odds_api",
            sa.Text,
            nullable=True,
            comment="event.id de The Odds API, guardado al ingestar odds",
        ),
    )
    op.create_index(
        "ix_matches_external_id_nba",
        "matches",
        ["external_id_nba"],
        postgresql_where=sa.text("external_id_nba IS NOT NULL"),
    )
    op.create_index(
        "ix_matches_external_id_nhl",
        "matches",
        ["external_id_nhl"],
        postgresql_where=sa.text("external_id_nhl IS NOT NULL"),
    )
    op.create_index(
        "ix_matches_external_id_odds_api",
        "matches",
        ["external_id_odds_api"],
        postgresql_where=sa.text("external_id_odds_api IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_matches_external_id_odds_api", table_name="matches")
    op.drop_index("ix_matches_external_id_nhl", table_name="matches")
    op.drop_index("ix_matches_external_id_nba", table_name="matches")
    op.drop_column("matches", "external_id_odds_api")
    op.drop_column("matches", "external_id_nhl")
    op.drop_column("matches", "external_id_nba")
