# syntax=docker/dockerfile:1.7
# Multi-stage build (builder → runtime) para minimizar superficie final.
#
# Python 3.14.4 free-threaded (PEP 703). Las imágenes oficiales python:3.14.4-*
# exponen dos intérpretes:
#   - `python3.14`  → con GIL (legacy).
#   - `python3.14t` → free-threaded (sin GIL).
# Usamos `PYTHON_GIL=0` para activar free-threading en runtime.
# Si alguna C-extension no tiene wheel cp314t, el build cae automáticamente
# al intérprete con GIL (escape hatch en uv sync || retry).

ARG PYTHON_VERSION=3.14.4
ARG UV_VERSION=0.5.18

# ═══════════════════════ Stage 1: builder ═══════════════════════════════
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_SYSTEM_PYTHON=0 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHON_GIL=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libpq-dev \
        libomp-dev \
        ca-certificates \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

ARG UV_VERSION
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && uv --version

WORKDIR /app

COPY pyproject.toml uv.lock* README.md ./

# Intenta free-threaded (cp314t). Si alguna wheel no existe, fallback a GIL.
RUN uv venv --python 3.14.4 /app/.venv \
    && uv sync --frozen \
        --extra ml --extra sports --extra llm --extra scraping \
        --extra bot --extra monitoring --extra dev \
    || (echo "⚠ cp314t sync falló — reintentando con GIL cp314..." \
        && rm -rf /app/.venv \
        && uv venv --python 3.14.4 /app/.venv \
        && uv sync --frozen \
            --extra ml --extra sports --extra llm --extra scraping \
            --extra bot --extra monitoring --extra dev)


# ═══════════════════════ Stage 2: runtime ═══════════════════════════════
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHON_GIL=0

# Runtime deps (sin build-essential para reducir superficie)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libpq5 \
        libomp-dev \
        curl \
        ca-certificates \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 1000 apuestas \
    && useradd -u 1000 -g 1000 -m -s /bin/bash apuestas

WORKDIR /app

COPY --from=builder --chown=apuestas:apuestas /app/.venv /app/.venv

COPY --chown=apuestas:apuestas src /app/src
COPY --chown=apuestas:apuestas migrations /app/migrations
COPY --chown=apuestas:apuestas alembic.ini /app/alembic.ini
COPY --chown=apuestas:apuestas config /app/config
COPY --chown=apuestas:apuestas prompts /app/prompts
COPY --chown=apuestas:apuestas scripts /app/scripts
COPY --chown=apuestas:apuestas pyproject.toml README.md /app/

USER apuestas

HEALTHCHECK --interval=15s --timeout=5s --retries=8 --start-period=30s \
    CMD curl -fsS http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["granian", "--interface", "asgi", "--host", "0.0.0.0", "--port", "8000", "apuestas.api.main:app"]
