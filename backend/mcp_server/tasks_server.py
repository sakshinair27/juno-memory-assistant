"""A tiny MCP server: a tasks / reminders store backed by the same Postgres.

Run standalone:
    python -m mcp_server.tasks_server            # stdio (for any MCP client)
    python -m mcp_server.tasks_server --http     # streamable HTTP on :8765/mcp

The assistant backend connects to it in-process by default (still speaking
MCP), or over HTTP when MCP_SERVER_URL is set.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.mcpserver import MCPServer  # noqa: E402
from psycopg import Connection  # noqa: E402
from psycopg.rows import dict_row  # noqa: E402

from app.config import settings  # noqa: E402

mcp = MCPServer(
    "juno-tasks",
    instructions="A personal to-do / reminder list. Use add_task when the user says 'remind me to...' or asks to note a to-do.",
)


def _conn() -> Connection:
    schema = os.getenv("TASKS_SCHEMA", settings.db_schema)
    return Connection.connect(settings.database_url, autocommit=True, row_factory=dict_row,
                              options=f"-c search_path={schema},public")


def _fmt(t: dict) -> str:
    due = f" (due {t['due']})" if t.get("due") else ""
    return f"#{t['id']} [{'x' if t['done'] else ' '}] {t['title']}{due}"


@mcp.tool()
def add_task(title: str, due: str | None = None) -> str:
    """Add a task or reminder. `title` is what to do; `due` is a free-text time like 'tomorrow 9am' or '2026-10-05'."""
    with _conn() as c:
        dup = c.execute(
            "SELECT * FROM tasks WHERE NOT done AND lower(title) = lower(%s) AND due IS NOT DISTINCT FROM %s",
            (title.strip(), due)).fetchone()
        if dup:  # idempotent: asking twice for the same reminder doesn't create two
            return f"Already on the list, not added again: {_fmt(dup)}"
        t = c.execute("INSERT INTO tasks (title, due) VALUES (%s, %s) RETURNING *", (title.strip(), due)).fetchone()
    return f"Added task {_fmt(t)}"


@mcp.tool()
def list_tasks(include_done: bool = False) -> str:
    """List the user's tasks. Open tasks only unless include_done is true."""
    with _conn() as c:
        q = "SELECT * FROM tasks" + ("" if include_done else " WHERE NOT done") + " ORDER BY done, id"
        rows = c.execute(q).fetchall()
    return "\n".join(_fmt(t) for t in rows) or "No tasks."


@mcp.tool()
def complete_task(task_id: int) -> str:
    """Mark a task as done by its numeric id."""
    with _conn() as c:
        t = c.execute("UPDATE tasks SET done = true, completed_at = now() WHERE id = %s RETURNING *", (task_id,)).fetchone()
    return f"Completed {_fmt(t)}" if t else f"No task with id {task_id}."


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if args.http:
        mcp.run("streamable-http", host="127.0.0.1", port=args.port)
    else:
        mcp.run("stdio")
