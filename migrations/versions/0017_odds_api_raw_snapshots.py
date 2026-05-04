"""Raw snapshots stage tables para The Odds API paid tier.

Revision ID: 0017
Revises: 0016
Create Date: 2026-04-22

Añade tablas stage JSONB para:
- `odds_api_event_snapshots`: raw response de /events/{id}/odds (player props + alternates).
- `odds_api_historical_snapshots`: raw response de /historical/sports/{sport}/odds.

Arquitectura:
- Ingest flow hace bulk fetch y persiste en JSONB (cero resolución player_id).
- Downstream ETL async resuelve player_id + persiste normalized en `player_prop_lines`.
- Desacopla fetch (time-sensitive, cuesta créditos) de parsing (lento, bloqueos DB).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0017"
down_revision: str | None = "0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "odds_api_event_snapshots",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("sport_key", sa.Text, nullable=False, comment="basketball_nba, etc."),
        sa.Column("event_id", sa.Text, nullable=False, comment="The Odds API event id"),
        sa.Column(
            "internal_match_id", sa.BigInteger, nullable=True, comment="FK matches.id si resolvible"
        ),
        sa.Column(
            "markets",
            sa.Text,
            nullable=False,
            comment="csv: player_points,...",
        ),
        sa.Column("payload", sa.dialects.postgresql.JSONB, nullable=False),
        sa.Column("captured_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("processed", sa.Boolean, nullable=False, server_default=sa.text("false")),
    )
    op.create_index(
        "ix_odds_event_snap_event", "odds_api_event_snapshots", ["event_id", "captured_at"]
    )
    op.create_index(
        "ix_odds_event_snap_unprocessed",
        "odds_api_event_snapshots",
        ["captured_at"],
        postgresql_where=sa.text("processed = false"),
    )

    op.create_table(
        "odds_api_historical_snapshots",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("sport_key", sa.Text, nullable=False),
        sa.Column(
            "snapshot_ts",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            comment="timestamp del snapshot en la API",
        ),
        sa.Column("markets", sa.Text, nullable=False),
        sa.Column("regions", sa.Text, nullable=False),
        sa.Column("payload", sa.dialects.postgresql.JSONB, nullable=False),
        sa.Column("captured_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
        sa.UniqueConstraint(
            "sport_key",
            "snapshot_ts",
            "markets",
            "regions",
            name="uq_historical_snap",
        ),
    )
    op.create_index(
        "ix_historical_snap_sport_ts",
        "odds_api_historical_snapshots",
        ["sport_key", "snapshot_ts"],
    )


def downgrade() -> None:
    op.drop_index("ix_historical_snap_sport_ts", table_name="odds_api_historical_snapshots")
    op.drop_table("odds_api_historical_snapshots")
    op.drop_index("ix_odds_event_snap_unprocessed", table_name="odds_api_event_snapshots")
    op.drop_index("ix_odds_event_snap_event", table_name="odds_api_event_snapshots")
    op.drop_table("odds_api_event_snapshots")
