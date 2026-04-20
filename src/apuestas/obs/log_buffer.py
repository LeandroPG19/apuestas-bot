"""Ring buffer en memoria para los últimos N eventos de structlog.

El tab Logs de la TUI lee de este buffer para mostrar tail en vivo sin tener
que parsear stdout o escribir a disco. El handler se añade al root logger
durante `configure_logging()`.
"""

from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Any

# Deque global compartida entre procesos del mismo interpreter.
_BUFFER: deque[dict[str, Any]] = deque(maxlen=500)
_LOCK = Lock()


def log_capture_processor(
    _logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Processor de structlog: copia cada evento al ring buffer antes de renderizar."""
    with _LOCK:
        entry = {
            "ts": event_dict.get("timestamp") or event_dict.get("_record_created"),
            "level": event_dict.get("level") or method_name.upper(),
            "event": event_dict.get("event", ""),
            "logger": event_dict.get("logger_name") or event_dict.get("logger") or "",
            "extra": {
                k: v
                for k, v in event_dict.items()
                if k
                not in ("timestamp", "level", "event", "logger_name", "logger", "_record_created")
            },
        }
        _BUFFER.append(entry)
    return event_dict


def recent_logs(limit: int = 200, level_min: str = "DEBUG") -> list[dict[str, Any]]:
    """Retorna los últimos `limit` eventos, filtrados por nivel mínimo."""
    priority = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
    min_p = priority.get(level_min.upper(), 10)
    with _LOCK:
        snapshot = list(_BUFFER)
    out = [e for e in snapshot if priority.get(str(e.get("level", "")).upper(), 0) >= min_p]
    return out[-limit:]


def clear_buffer() -> int:
    with _LOCK:
        n = len(_BUFFER)
        _BUFFER.clear()
    return n
