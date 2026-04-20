"""Placeholder tasks módulo scraping — camoufox Caliente.mx (Fase 3-4)."""

from apuestas.obs.logging import get_logger
from apuestas.tasks.broker import broker

logger = get_logger(__name__)


@broker.task(task_name="scrape.ping")
async def ping() -> str:
    logger.info("tasks.scrape.ping")
    return "pong"
