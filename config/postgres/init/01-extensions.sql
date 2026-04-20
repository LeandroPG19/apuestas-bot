-- Extensiones obligatorias para el proyecto apuestas.
-- Se ejecuta UNA sola vez al inicializar el volumen postgres_data.

CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- Parámetros base para auditoría y correlación
ALTER DATABASE apuestas SET timezone TO 'UTC';

-- Schemas
CREATE SCHEMA IF NOT EXISTS apuestas;
CREATE SCHEMA IF NOT EXISTS audit;

GRANT ALL ON SCHEMA apuestas TO apuestas;
GRANT ALL ON SCHEMA audit TO apuestas;
