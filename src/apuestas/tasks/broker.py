"""TaskIQ broker singleton — Valkey/Redis backend."""

from taskiq_redis import RedisAsyncResultBackend, RedisStreamBroker

from apuestas.config import get_settings

_settings = get_settings()
_broker_url = (
    str(_settings.valkey.taskiq_broker_url)
    if _settings.valkey.taskiq_broker_url
    else str(_settings.valkey.url)
)

broker = RedisStreamBroker(url=_broker_url).with_result_backend(
    RedisAsyncResultBackend(redis_url=_broker_url, result_ex_time=3600)
)
