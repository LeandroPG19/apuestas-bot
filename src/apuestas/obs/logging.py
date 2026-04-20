"""Logging estructurado con structlog + OpenTelemetry correlation."""

import logging
import os
import sys
from pathlib import Path

import structlog

from apuestas.config import Environment, get_settings


def configure_logging() -> None:
    """Configura structlog + stdlib logging coherentes.

    Si APUESTAS_TUI_ACTIVE=1, redirige logs a archivo para no pisar la TUI.
    El ring buffer sigue alimentando el tab Logs dentro de la TUI.
    """
    settings = get_settings()
    is_local = settings.apuestas_env == Environment.LOCAL
    tui_active = os.environ.get("APUESTAS_TUI_ACTIVE", "").strip() == "1"

    from apuestas.obs.log_buffer import log_capture_processor

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.dict_tracebacks,
        log_capture_processor,  # type: ignore[list-item]
    ]

    if tui_active:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    elif is_local:
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    if tui_active:
        log_dir = Path.home() / ".local" / "share" / "apuestas" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "app.log"
        handler: logging.Handler = logging.FileHandler(log_file, encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(settings.apuestas_log_level.value)

    for noisy in ("uvicorn.access", "granian.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
