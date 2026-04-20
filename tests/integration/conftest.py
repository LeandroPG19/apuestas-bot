"""Fixtures para tests integration con testcontainers Postgres real.

Requiere Docker corriendo. Cada test recibe un engine limpio con schema
aplicado vía alembic upgrade head.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Generator
from typing import Any

import pytest
import pytest_asyncio


@pytest.fixture(scope="session")
def pg_container() -> Generator[dict[str, Any]]:
    """Levanta Postgres 16 con testcontainers, expone URL asyncpg."""
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:
        pytest.skip("testcontainers no disponible", allow_module_level=False)

    with PostgresContainer("postgres:16-alpine") as pg:
        raw_url = pg.get_connection_url()  # postgresql+psycopg2://...
        async_url = raw_url.replace("postgresql+psycopg2", "postgresql+asyncpg")
        sync_url = raw_url.replace("postgresql+psycopg2", "postgresql+psycopg")
        yield {"async_url": async_url, "sync_url": sync_url, "container": pg}


@pytest_asyncio.fixture
async def integration_engine(pg_container: dict[str, Any]) -> AsyncGenerator[Any]:
    """Engine asyncpg conectado al Postgres del contenedor.

    NOTE: No corre migraciones reales (usan TimescaleDB que no está en alpine).
    Crea tablas mínimas necesarias para los tests integration directamente.
    """
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    engine = create_async_engine(pg_container["async_url"], echo=False)
    await _create_minimal_schema(engine)
    os.environ["DATABASE_URL"] = pg_container["async_url"]
    sessionmaker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    yield {"engine": engine, "sessionmaker": sessionmaker}
    await engine.dispose()


async def _create_minimal_schema(engine: Any) -> None:
    """Schema reducido sin TimescaleDB/pgvector, suficiente para integration tests."""
    from sqlalchemy import text

    ddl = [
        """CREATE TABLE IF NOT EXISTS sports (
            code TEXT PRIMARY KEY, name TEXT NOT NULL, has_draws BOOLEAN DEFAULT false
        )""",
        """CREATE TABLE IF NOT EXISTS teams (
            id BIGSERIAL PRIMARY KEY, sport_code TEXT REFERENCES sports(code),
            external_id TEXT, name TEXT NOT NULL, country TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS matches (
            id BIGSERIAL PRIMARY KEY,
            sport_code TEXT REFERENCES sports(code),
            league_id BIGINT,
            external_id TEXT,
            home_team_id BIGINT REFERENCES teams(id),
            away_team_id BIGINT REFERENCES teams(id),
            start_time TIMESTAMPTZ NOT NULL,
            status TEXT NOT NULL DEFAULT 'scheduled',
            home_score INTEGER, away_score INTEGER,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )""",
        """CREATE TABLE IF NOT EXISTS predictions (
            id BIGSERIAL PRIMARY KEY,
            match_id BIGINT REFERENCES matches(id),
            model_name TEXT, model_version TEXT,
            market TEXT NOT NULL, outcome TEXT NOT NULL,
            line NUMERIC(6,2),
            probability NUMERIC(6,5),
            p_lower NUMERIC(6,5), p_upper NUMERIC(6,5),
            ev NUMERIC(6,4), kelly_fraction NUMERIC(6,4),
            features_snapshot JSONB, shap_top5 JSONB, llm_analysis JSONB,
            decision TEXT DEFAULT 'skip',
            created_at TIMESTAMPTZ DEFAULT NOW()
        )""",
        """CREATE TABLE IF NOT EXISTS bets (
            id BIGSERIAL PRIMARY KEY,
            match_id BIGINT REFERENCES matches(id),
            prediction_id BIGINT REFERENCES predictions(id),
            bookmaker TEXT NOT NULL,
            market TEXT NOT NULL, outcome TEXT NOT NULL,
            line NUMERIC(6,2),
            stake_units NUMERIC(10,3) NOT NULL,
            odds_placed NUMERIC(8,3) NOT NULL,
            placed_at TIMESTAMPTZ DEFAULT NOW(),
            status TEXT DEFAULT 'pending',
            is_paper BOOLEAN DEFAULT true,
            pnl_units NUMERIC(10,3),
            closing_line NUMERIC(8,3), clv NUMERIC(6,4),
            settled_at TIMESTAMPTZ
        )""",
        """CREATE TABLE IF NOT EXISTS bankroll_history (
            ts TIMESTAMPTZ NOT NULL,
            is_paper BOOLEAN DEFAULT true,
            bankroll_units NUMERIC(14,4) NOT NULL,
            delta_units NUMERIC(14,4),
            bet_id BIGINT, event TEXT, notes TEXT
        )""",
        """INSERT INTO sports(code,name,has_draws) VALUES
            ('nba','NBA',false),('soccer','Soccer',true),('mlb','MLB',false),
            ('nfl','NFL',false),('nhl','NHL',false),('tennis','Tennis',false)
            ON CONFLICT DO NOTHING""",
    ]
    async with engine.begin() as conn:
        for stmt in ddl:
            await conn.execute(text(stmt))
