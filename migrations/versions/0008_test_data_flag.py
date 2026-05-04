"""test_data flag — aísla demo/backtest data de métricas production.

Añade `test_data BOOLEAN DEFAULT false` a `bets` y `predictions`. Queries de
dashboard/CLV/bankroll/post_mortems filtran por `test_data = false` para no
contaminar métricas con data sintética (demos, sandbox, backtests manuales).

Marca retroactivamente bets/predictions anteriores a 2026-04-20 16:00 UTC
como test_data (corresponde al cutover post-cleanup de la sesión de demo).

Revision ID: 0008
Revises: 0007
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "bets",
        sa.Column(
            "test_data",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.add_column(
        "predictions",
        sa.Column(
            "test_data",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    # Índice parcial: solo filas "reales" (99% de lecturas) para speed en filtros.
    op.create_index(
        "idx_bets_not_test",
        "bets",
        ["id"],
        postgresql_where=sa.text("test_data = false"),
    )
    op.create_index(
        "idx_predictions_not_test",
        "predictions",
        ["id"],
        postgresql_where=sa.text("test_data = false"),
    )
    # Marcar data demo histórica (cutover 2026-04-20 16:00 UTC)
    op.execute(
        sa.text(
            """
            UPDATE bets
            SET test_data = true
            WHERE placed_at < '2026-04-20 16:00:00+00';
            """
        )
    )
    op.execute(
        sa.text(
            """
            UPDATE predictions
            SET test_data = true
            WHERE created_at < '2026-04-20 16:00:00+00';
            """
        )
    )
    # Columna auxiliar para trazabilidad de envío Telegram (Gap #11).
    op.add_column(
        "bets",
        sa.Column(
            "notification_sent_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("bets", "notification_sent_at")
    op.drop_index("idx_predictions_not_test", table_name="predictions")
    op.drop_index("idx_bets_not_test", table_name="bets")
    op.drop_column("predictions", "test_data")
    op.drop_column("bets", "test_data")
