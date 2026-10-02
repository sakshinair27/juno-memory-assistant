"""MCP client bridge: expose the tasks server's tools to Claude.

Tools are discovered over MCP (list_tools) and converted to Anthropic tool
definitions; tool calls from Claude are forwarded with call_tool. The graph is
synchronous, so each call opens a short-lived MCP session on its own event loop.
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Callable, Coroutine, TypeVar

from mcp import Client

from .config import settings
from .tracing import traced

log = logging.getLogger(__name__)
T = TypeVar("T")


def _target() -> Any:
    if settings.mcp_server_url:
        return settings.mcp_server_url
    from mcp_server.tasks_server import mcp  # in-process MCPServer instance

    return mcp


def _attr(obj: Any, *names: str) -> Any:
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


async def _list() -> list[dict]:
    async with Client(_target()) as c:
        res = await c.list_tools()
    tools = []
    for t in res.tools:
        tools.append({
            "name": t.name,
            "description": t.description or "",
            "input_schema": _attr(t, "input_schema", "inputSchema") or {"type": "object", "properties": {}},
        })
    return tools


async def _call(name: str, args: dict) -> tuple[str, bool]:
    async with Client(_target()) as c:
        res = await c.call_tool(name, args)
    parts = [getattr(b, "text", "") for b in (res.content or [])]
    return "\n".join(p for p in parts if p) or "(no output)", bool(_attr(res, "is_error", "isError"))


def _run(make_coro: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run a coroutine from sync code, even if the calling thread already has a
    running event loop (e.g. FastAPI's lifespan) — then use a worker thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(make_coro())
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(make_coro())).result()


class TaskTools:
    def __init__(self) -> None:
        self._defs: list[dict] | None = None

    def definitions(self) -> list[dict]:
        if self._defs is not None:
            return self._defs
        try:
            self._defs = _run(_list)  # only cache a successful discovery
            return self._defs
        except Exception as e:  # the assistant still works without its action tool
            log.warning("MCP tool discovery failed, continuing without tools: %s", e)
            return []

    @traced("mcp_call_tool", as_type="tool")
    def call(self, name: str, args: dict) -> tuple[str, bool]:
        try:
            return _run(lambda: _call(name, args))
        except Exception as e:
            log.exception("MCP tool call failed")
            return f"Tool error: {e}", True


@lru_cache(maxsize=1)
def get_task_tools() -> TaskTools:
    return TaskTools()
