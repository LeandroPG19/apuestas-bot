"""Tennis + NHL schema (§25).

Revision ID: 0005
Revises: 0004
Create Date: 2026-04-19

Añade:
- sports codes: tennis, nhl (si no existen ya)
- tennis_surface_ratings (Elo por superficie)
- nhl_goalie_stats (save%, GSAx, games, rolling)
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Añadir sports codes (idempotente)
    op.execute(
        """
        INSERT INTO sports (code, name, has_draws) VALUES
            ('tennis', 'Tennis ATP/WTA', false),
            ('nhl', 'NHL Hockey', false)
        ON CONFLICT (code) DO NOTHING
        """
    )

    # ─── Tennis surface ratings ──────────────────────────────────────────
    op.create_table(
        "tennis_surface_ratings",
        sa.Column("player_id", sa.BigInteger, sa.ForeignKey("players.id"), primary_key=True),
        sa.Column("surface", sa.Text, primary_key=True),
        sa.Column("elo", sa.Numeric(6, 1), nullable=False, server_default="1500"),
        sa.Column("games_played", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "last_updated",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "surface IN ('hard','clay','grass','indoor_hard')",
            name="ck_tennis_surface",
        ),
    )
    op.create_index(
        "idx_tennis_elo_by_surface",
        "tennis_surface_ratings",
        ["surface", "elo"],
        postgresql_using="btree",
    )

    # ─── NHL goalie stats ────────────────────────────────────────────────
    op.create_table(
        "nhl_goalie_stats",
        sa.Column("player_id", sa.BigInteger, sa.ForeignKey("players.id"), primary_key=True),
        sa.Column("games_played", sa.Integer, nullable=False, server_default="0"),
        sa.Column("saves_total", sa.Integer, nullable=False, server_default="0"),
        sa.Column("shots_against_total", sa.Integer, nullable=False, server_default="0"),
        sa.Column("save_pct", sa.Numeric(5, 4)),
        sa.Column("gsax_total", sa.Numeric(8, 4), server_default="0"),
        sa.Column("last_10_save_pct", sa.Numeric(5, 4)),
        sa.Column(
            "last_updated",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # ─── Tennis match details (sets, aces) ───────────────────────────────
    op.create_table(
        "tennis_match_details",
        sa.Column("match_id", sa.BigInteger, sa.ForeignKey("matches.id"), primary_key=True),
        sa.Column("surface", sa.Text, nullable=False),
        sa.Column("tournament_level", sa.Text),  # GS/M1000/M500/M250
        sa.Column("best_of", sa.Integer, server_default="3"),
        sa.Column("winner_sets", sa.Integer),
        sa.Column("loser_sets", sa.Integer),
        sa.Column("winner_aces", sa.Integer),
        sa.Column("loser_aces", sa.Integer),
        sa.Column("winner_double_faults", sa.Integer),
        sa.Column("loser_double_faults", sa.Integer),
        sa.Column("winner_first_serve_pct", sa.Numeric(5, 4)),
        sa.Column("loser_first_serve_pct", sa.Numeric(5, 4)),
        sa.Column("winner_break_points_saved", sa.Integer),
        sa.Column("loser_break_points_saved", sa.Integer),
        sa.Column("duration_minutes", sa.Integer),
        sa.CheckConstraint("best_of IN (3, 5)", name="ck_tennis_best_of"),
        sa.CheckConstraint(
            "surface IN ('hard','clay','grass','indoor_hard')",
            name="ck_tennis_match_surface",
        ),
    )


def downgrade() -> None:
    op.drop_table("tennis_match_details")
    op.drop_table("nhl_goalie_stats")
    op.drop_index("idx_tennis_elo_by_surface", table_name="tennis_surface_ratings")
    op.drop_table("tennis_surface_ratings")
    op.execute("DELETE FROM sports WHERE code IN ('tennis', 'nhl')")
