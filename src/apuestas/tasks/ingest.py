"""Placeholder tasks módulo ingest — se poblará en Fase 3-4."""

from apuestas.obs.logging import get_logger
from apuestas.tasks.broker import broker

logger = get_logger(__name__)


@broker.task(task_name="ingest.ping")
async def ping() -> str:
    logger.info("tasks.ingest.ping")
    return "pong"
