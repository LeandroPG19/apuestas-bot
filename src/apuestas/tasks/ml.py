"""Placeholder tasks módulo ML — se poblará en Fase 5-6."""

from apuestas.obs.logging import get_logger
from apuestas.tasks.broker import broker

logger = get_logger(__name__)


@broker.task(task_name="ml.ping")
async def ping() -> str:
    logger.info("tasks.ml.ping")
    return "pong"
