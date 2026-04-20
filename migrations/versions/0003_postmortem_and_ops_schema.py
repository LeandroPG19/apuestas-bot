"""Post-mortem + infra ops: decision_log, data_lineage, feature_versions, llm_calls, embeddings_cache, fiscal_events, post_mortems, calibration_rolling, audit triggers.

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-19

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ─── Audit trail (append-only) ───────────────────────────────────────
    op.execute(
        """
        CREATE TABLE audit.audit_log (
            id BIGSERIAL PRIMARY KEY,
            tabla TEXT NOT NULL,
            accion TEXT NOT NULL,
            row_pk JSONB,
            usuario TEXT,
            correlation_id TEXT,
            datos_anteriores JSONB,
            datos_nuevos JSONB,
            ip INET,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX idx_audit_tabla_created ON audit.audit_log (tabla, created_at DESC)")

    # Bloquear updates/deletes en audit log
    op.execute(
        """
        CREATE OR REPLACE FUNCTION audit.block_mutation()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION 'audit_log is append-only; % prohibited', TG_OP;
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        """
        CREATE TRIGGER audit_log_no_update BEFORE UPDATE ON audit.audit_log
        FOR EACH ROW EXECUTE FUNCTION audit.block_mutation()
        """
    )
    op.execute(
        """
        CREATE TRIGGER audit_log_no_delete BEFORE DELETE ON audit.audit_log
        FOR EACH ROW EXECUTE FUNCTION audit.block_mutation()
        """
    )

    # ─── Decision log ────────────────────────────────────────────────────
    op.create_table(
        "decision_log",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("event_id", sa.BigInteger, sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("market", sa.Text, nullable=False),
        sa.Column("outcome", sa.Text, nullable=False),
        sa.Column("line", sa.Numeric(6, 2)),
        sa.Column("p_model", sa.Numeric(6, 5)),
        sa.Column("p_lower", sa.Numeric(6, 5)),
        sa.Column("p_upper", sa.Numeric(6, 5)),
        sa.Column("fair_odds", sa.Numeric(8, 3)),
        sa.Column("best_offer", sa.Numeric(8, 3)),
        sa.Column("best_bookmaker", sa.Text),
        sa.Column("edge", sa.Numeric(6, 4)),
        sa.Column("decision", sa.Text, nullable=False),  # bet|skip
        sa.Column(
            "skip_reason", sa.Text
        ),  # low_edge|conformal_width|repetition_flag|data_limited|user_blacklist
        sa.Column("bet_id", sa.BigInteger, sa.ForeignKey("bets.id")),
        sa.Column("correlation_id", sa.Text),
        sa.Column(
            "created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("idx_decision_event", "decision_log", ["event_id"])
    op.create_index(
        "idx_decision_skip",
        "decision_log",
        ["skip_reason"],
        postgresql_where=sa.text("skip_reason IS NOT NULL"),
    )

    # ─── Data lineage ────────────────────────────────────────────────────
    op.create_table(
        "data_lineage",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("source", sa.Text, nullable=False),
        sa.Column("dataset_name", sa.Text, nullable=False),
        sa.Column(
            "ingested_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("version_hash", sa.Text),
        sa.Column("row_count", sa.BigInteger),
        sa.Column("flow_run_id", sa.Text),
        sa.Column("schema_version", sa.Text),
        sa.Column("metadata", sa.JSON),
    )
    op.create_index("idx_lineage_source_ingested", "data_lineage", ["source", "ingested_at"])

    # ─── Feature versions ────────────────────────────────────────────────
    op.create_table(
        "feature_versions",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("feature_set_name", sa.Text, nullable=False),
        sa.Column("version", sa.Text, nullable=False),
        sa.Column("sport_code", sa.Text, sa.ForeignKey("sports.code")),
        sa.Column(
            "introduced_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("deprecated_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("schema_json", sa.JSON, nullable=False),
        sa.Column("mlflow_models", sa.dialects.postgresql.ARRAY(sa.Text)),
        sa.UniqueConstraint("feature_set_name", "version", name="uq_feature_set_version"),
    )

    # ─── LLM calls tracking ──────────────────────────────────────────────
    op.create_table(
        "llm_calls",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("ts", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("task_kind", sa.Text, nullable=False),
        sa.Column("model", sa.Text, nullable=False),
        sa.Column("prompt_version", sa.Text),
        sa.Column("tokens_in", sa.Integer),
        sa.Column("tokens_out", sa.Integer),
        sa.Column("latency_ms", sa.Integer),
        sa.Column("cost_usd", sa.Numeric(10, 6), server_default="0"),  # local = 0
        sa.Column("success", sa.Boolean),
        sa.Column("retry_count", sa.Integer, server_default="0"),
        sa.Column("correlation_id", sa.Text),
        sa.Column("prediction_id", sa.BigInteger, sa.ForeignKey("predictions.id")),
    )
    op.create_index("idx_llm_calls_ts", "llm_calls", ["ts"])
    op.create_index("idx_llm_calls_task", "llm_calls", ["task_kind", "ts"])

    # ─── Embeddings cache ────────────────────────────────────────────────
    op.create_table(
        "embeddings_cache",
        sa.Column("content_hash", sa.Text, primary_key=True),  # sha256
        sa.Column("model", sa.Text, nullable=False),
        sa.Column(
            "created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "last_used_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("hits", sa.BigInteger, server_default="0"),
    )
    # Columna embedding añadida por separado (Vector necesita import)
    op.execute("ALTER TABLE embeddings_cache ADD COLUMN embedding vector(1024)")

    # ─── Fiscal events (SAT) ─────────────────────────────────────────────
    op.create_table(
        "fiscal_events",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("event_date", sa.Date, nullable=False),
        sa.Column("event_type", sa.Text, nullable=False),  # deposit|withdraw|winnings|loss
        sa.Column("bookmaker", sa.Text),
        sa.Column("amount_mxn", sa.Numeric(14, 2), nullable=False),
        sa.Column("iva_retained_mxn", sa.Numeric(14, 2), server_default="0"),
        sa.Column("isr_retained_mxn", sa.Numeric(14, 2), server_default="0"),
        sa.Column("uma_threshold_crossed", sa.Boolean, server_default=sa.false()),
        sa.Column("uif_report_required", sa.Boolean, server_default=sa.false()),
        sa.Column("bet_id", sa.BigInteger, sa.ForeignKey("bets.id")),
        sa.Column("notes", sa.Text),
    )
    op.create_index("idx_fiscal_date", "fiscal_events", ["event_date"])

    # ─── Post-mortems ────────────────────────────────────────────────────
    op.create_table(
        "post_mortems",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("bet_id", sa.BigInteger, sa.ForeignKey("bets.id"), nullable=False, unique=True),
        sa.Column("event_id", sa.BigInteger, sa.ForeignKey("matches.id"), nullable=False),
        # Snapshot
        sa.Column("prediction_snapshot", sa.JSON, nullable=False),
        sa.Column("features_snapshot", sa.JSON, nullable=False),
        sa.Column("shap_top5", sa.JSON, nullable=False),
        sa.Column("llm_analysis_snapshot", sa.JSON, nullable=False),
        sa.Column("ev_predicted", sa.Numeric(6, 4), nullable=False),
        sa.Column("kelly_predicted", sa.Numeric(6, 4), nullable=False),
        # Realidad
        sa.Column("outcome", sa.Text, nullable=False),
        sa.Column("actual_final_score", sa.JSON),
        sa.Column("actual_lineups", sa.JSON),
        sa.Column("actual_key_events", sa.JSON),
        sa.Column("pnl_units", sa.Numeric(10, 3), nullable=False),
        sa.Column("clv", sa.Numeric(6, 4)),
        # Discrepancia
        sa.Column("prediction_error", sa.Numeric(6, 4)),
        sa.Column("calibration_miss", sa.Numeric(6, 4)),
        sa.Column("ev_realized", sa.Numeric(6, 4)),
        sa.Column("ev_realized_vs_predicted", sa.Numeric(6, 4)),
        sa.Column("llm_alignment_score", sa.Numeric(4, 3)),
        sa.Column("shap_attribution_check", sa.Numeric(4, 3)),
        sa.Column("line_movement_assessment_correct", sa.Boolean),
        sa.Column("discrepancy_score", sa.Numeric(5, 3)),
        # Narrativa
        sa.Column("narrative", sa.JSON, nullable=False),
        # Metadata
        sa.Column(
            "post_mortem_generated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("review_status", sa.Text, nullable=False, server_default="auto"),
        sa.Column("human_notes", sa.Text),
        sa.Column("pattern_tags", sa.dialects.postgresql.ARRAY(sa.Text)),
        sa.CheckConstraint(
            "review_status IN ('auto','reviewed','flagged')", name="ck_pm_review_status"
        ),
    )
    op.create_index("idx_pm_discrepancy", "post_mortems", ["discrepancy_score"])
    op.create_index("idx_pm_outcome", "post_mortems", ["outcome", "discrepancy_score"])
    op.create_index("idx_pm_event", "post_mortems", ["event_id"])
    op.create_index("idx_pm_generated", "post_mortems", ["post_mortem_generated_at"])
    op.create_index("idx_pm_tags", "post_mortems", ["pattern_tags"], postgresql_using="gin")

    # ─── Calibration rolling ─────────────────────────────────────────────
    op.create_table(
        "calibration_rolling",
        sa.Column("sport_code", sa.Text, sa.ForeignKey("sports.code"), primary_key=True),
        sa.Column("market", sa.Text, primary_key=True),
        sa.Column("confidence_bucket", sa.Text, primary_key=True),  # ej. "p=[0.55,0.60)"
        sa.Column("window_days", sa.Integer, primary_key=True),  # 7, 30, 90
        sa.Column("n_predictions", sa.Integer, nullable=False),
        sa.Column("mean_predicted", sa.Numeric(6, 4)),
        sa.Column("mean_actual", sa.Numeric(6, 4)),
        sa.Column("calibration_gap", sa.Numeric(6, 4)),
        sa.Column("brier_realized", sa.Numeric(6, 4)),
        sa.Column("ece_realized", sa.Numeric(6, 4)),
        sa.Column(
            "last_computed",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # ─── Model registry extension (shadow/champion) ──────────────────────
    op.create_table(
        "model_registry_meta",
        sa.Column("mlflow_run_id", sa.Text, primary_key=True),
        sa.Column("model_name", sa.Text, nullable=False),
        sa.Column("model_version", sa.Text, nullable=False),
        sa.Column("sport_code", sa.Text, sa.ForeignKey("sports.code")),
        sa.Column("stage", sa.Text, nullable=False),  # shadow|production|archived
        sa.Column("promoted_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("promoted_by", sa.Text),
        sa.Column("retired_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("performance_30d", sa.JSON),
        sa.Column("drift_status", sa.Text),
        sa.Column("notes", sa.Text),
        sa.CheckConstraint("stage IN ('shadow','production','archived')", name="ck_stage"),
    )
    op.create_index("idx_model_reg_stage", "model_registry_meta", ["model_name", "stage"])

    # ─── Market catalog ──────────────────────────────────────────────────
    op.create_table(
        "market_catalog",
        sa.Column("sport_code", sa.Text, sa.ForeignKey("sports.code"), primary_key=True),
        sa.Column("market", sa.Text, primary_key=True),
        sa.Column("display_name", sa.Text, nullable=False),
        sa.Column("category", sa.Text, nullable=False),  # main|nicho|props|futures
        sa.Column("outcomes_schema", sa.JSON, nullable=False),
        sa.Column("active", sa.Boolean, nullable=False, server_default=sa.true()),
    )

    # ─── Pattern blacklist (anti-repetición) ─────────────────────────────
    op.create_table(
        "pattern_blacklist",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("tag", sa.Text, nullable=False, unique=True),
        sa.Column("sport_code", sa.Text, sa.ForeignKey("sports.code")),
        sa.Column("reason", sa.Text),
        sa.Column("active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column(
            "added_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("expires_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("confidence_penalty", sa.Numeric(4, 3), server_default="0.1"),
    )

    # ─── Monte Carlo risk snapshots ──────────────────────────────────────
    op.create_table(
        "risk_snapshots",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("ts", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("n_simulations", sa.Integer, nullable=False),
        sa.Column("n_bets_per_path", sa.Integer, nullable=False),
        sa.Column("prob_dd_25pct", sa.Numeric(5, 4)),
        sa.Column("prob_dd_40pct", sa.Numeric(5, 4)),
        sa.Column("prob_double", sa.Numeric(5, 4)),
        sa.Column("expected_bankroll_6m", sa.Numeric(14, 4)),
        sa.Column("p10_bankroll_6m", sa.Numeric(14, 4)),
        sa.Column("p90_bankroll_6m", sa.Numeric(14, 4)),
        sa.Column("params", sa.JSON),
    )


def downgrade() -> None:
    op.drop_table("risk_snapshots")
    op.drop_table("pattern_blacklist")
    op.drop_table("market_catalog")
    op.drop_table("model_registry_meta")
    op.drop_table("calibration_rolling")
    op.drop_table("post_mortems")
    op.drop_table("fiscal_events")
    op.drop_table("embeddings_cache")
    op.drop_table("llm_calls")
    op.drop_table("feature_versions")
    op.drop_table("data_lineage")
    op.drop_table("decision_log")
    op.execute("DROP TRIGGER IF EXISTS audit_log_no_delete ON audit.audit_log")
    op.execute("DROP TRIGGER IF EXISTS audit_log_no_update ON audit.audit_log")
    op.execute("DROP FUNCTION IF EXISTS audit.block_mutation()")
    op.execute("DROP TABLE IF EXISTS audit.audit_log CASCADE")
