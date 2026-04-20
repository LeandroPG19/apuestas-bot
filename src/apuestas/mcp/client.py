"""Cliente MCP base para conectar con cuba-memorys y cuba-search vía stdio.

Implementación sobre el SDK oficial `mcp` (>=1.15). Los servidores MCP se
lanzan como subprocess y se habla JSON-RPC por stdio. Gestionamos la
conexión como async context manager persistente durante una sesión del bot.

Si `APUESTAS_USE_MCP=false` o los stdio commands no están configurados,
los helpers en memory.py / research.py degradan graciosamente a no-op.
"""

from __future__ import annotations

import asyncio
import shlex
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from apuestas.config import get_settings
from apuestas.obs.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = get_logger(__name__)


@dataclass(slots=True)
class MCPConnection:
    name: str
    session: Any  # mcp.ClientSession
    enabled: bool


class MCPClient:
    """Singleton perezoso de conexiones MCP (cuba-memorys + cuba-search)."""

    _instance: MCPClient | None = None

    def __init__(self) -> None:
        self._connections: dict[str, MCPConnection] = {}
        self._exit_stack: AsyncExitStack | None = None
        self._lock = asyncio.Lock()
        settings = get_settings()
        self._use_mcp = settings.apuestas_enable_mcp and settings.mcp.apuestas_use_mcp
        self._memory_cmd = settings.mcp.cuba_memorys_stdio_cmd
        self._search_cmd = settings.mcp.cuba_search_stdio_cmd

    @classmethod
    def get(cls) -> MCPClient:
        if cls._instance is None:
            cls._instance = MCPClient()
        return cls._instance

    async def start(self) -> None:
        if not self._use_mcp:
            logger.info("mcp.disabled_via_flag")
            return

        if self._exit_stack is not None:
            return  # ya started

        async with self._lock:
            if self._exit_stack is not None:
                return
            self._exit_stack = AsyncExitStack()
            await self._exit_stack.__aenter__()

            for name, cmd in (("memorys", self._memory_cmd), ("search", self._search_cmd)):
                if not cmd:
                    logger.info("mcp.skip_unconfigured", name=name)
                    continue
                try:
                    conn = await self._connect(name, cmd)
                    self._connections[name] = conn
                except Exception as exc:
                    logger.warning("mcp.connect_failed", name=name, error=str(exc))

    async def _connect(self, name: str, cmd: str) -> MCPConnection:
        """Lanza subprocess MCP server y abre sesión JSON-RPC stdio."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            msg = "mcp SDK requerido: pip install mcp"
            raise RuntimeError(msg) from exc

        assert self._exit_stack is not None

        parts = shlex.split(cmd)
        params = StdioServerParameters(command=parts[0], args=parts[1:] if len(parts) > 1 else [])
        read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        logger.info("mcp.connected", name=name)
        return MCPConnection(name=name, session=session, enabled=True)

    async def stop(self) -> None:
        if self._exit_stack is None:
            return
        # Cerrar siempre en el mismo task donde se entró. Si el caller está en
        # otro task (ej. Textual teardown), shielding evita propagar el
        # "Attempted to exit cancel scope in a different task" como excepción
        # ruidosa hacia arriba — es cosmético, la conexión stdio ya murió.
        try:
            await asyncio.shield(self._exit_stack.__aexit__(None, None, None))
        except (RuntimeError, asyncio.CancelledError) as exc:
            logger.debug("mcp.stop_warning", error=str(exc))
        finally:
            self._exit_stack = None
            self._connections.clear()
            logger.info("mcp.stopped")

    def is_connected(self, name: str) -> bool:
        return name in self._connections and self._connections[name].enabled

    async def call(
        self, server: str, tool: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Invoca una tool del server MCP. Auto-arranca la conexión si no existe.
        Retorna None si MCP está deshabilitado o la conexión falló.
        """
        if not self._use_mcp:
            return None
        # Auto-start lazy (primera llamada arranca subprocess)
        if self._exit_stack is None:
            try:
                await self.start()
            except Exception as exc:
                logger.debug("mcp.autostart_failed", error=str(exc))
                return None
        if server not in self._connections:
            return None
        conn = self._connections[server]
        try:
            result = await conn.session.call_tool(tool, arguments or {})
            return _extract_content(result)
        except Exception as exc:
            logger.warning("mcp.call_failed", server=server, tool=tool, error=str(exc))
            return None


def _extract_content(result: Any) -> dict[str, Any]:
    """Normaliza CallToolResult a dict."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    # mcp devuelve CallToolResult con .content list
    out: dict[str, Any] = {}
    for item in getattr(result, "content", []):
        if hasattr(item, "text"):
            out.setdefault("text_chunks", []).append(item.text)
        if hasattr(item, "data"):
            out.setdefault("data_chunks", []).append(item.data)
    if hasattr(result, "isError"):
        out["is_error"] = result.isError
    return out


@asynccontextmanager
async def mcp_session() -> AsyncIterator[MCPClient]:
    """Context manager para uso one-shot."""
    client = MCPClient.get()
    await client.start()
    try:
        yield client
    finally:
        pass  # mantener conexión; stop global en shutdown de FastAPI
