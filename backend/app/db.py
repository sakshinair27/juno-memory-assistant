"""Postgres + pgvector connection pool and schema bootstrap."""
from __future__ import annotations

import re
from functools import lru_cache

from pgvector.psycopg import register_vector
from psycopg import Connection, sql
from psycopg_pool import ConnectionPool

from .config import settings

_SCHEMA_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def _schema_ddl(dim: int) -> list[str]:
    return [
        """
        CREATE TABLE IF NOT EXISTS memories (
            id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            content          TEXT NOT NULL,
            category         TEXT NOT NULL DEFAULT 'other',
            pinned           BOOLEAN NOT NULL DEFAULT false,
            confidence       REAL NOT NULL DEFAULT 1.0,
            embedding        vector(%d) NOT NULL,
            source_session   TEXT,
            created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_accessed_at TIMESTAMPTZ,
            access_count     INTEGER NOT NULL DEFAULT 0
        )
        """ % dim,
        "CREATE INDEX IF NOT EXISTS memories_embedding_hnsw ON memories USING hnsw (embedding vector_cosine_ops)",
        """
        CREATE TABLE IF NOT EXISTS memory_events (
            id          BIGSERIAL PRIMARY KEY,
            memory_id   UUID,
            op          TEXT NOT NULL,
            old_content TEXT,
            new_content TEXT,
            reason      TEXT,
            session_id  TEXT,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
        "CREATE INDEX IF NOT EXISTS memory_events_memory_id ON memory_events (memory_id)",
        """
        CREATE TABLE IF NOT EXISTS quarantined_facts (
            id             BIGSERIAL PRIMARY KEY,
            content        TEXT NOT NULL,
            category       TEXT NOT NULL,
            reason         TEXT,
            source_message TEXT,
            session_id     TEXT,
            created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id           BIGSERIAL PRIMARY KEY,
            title        TEXT NOT NULL,
            due          TEXT,
            done         BOOLEAN NOT NULL DEFAULT false,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            completed_at TIMESTAMPTZ
        )
        """,
    ]


def _configure(conn: Connection) -> None:
    register_vector(conn)


def init_db(database_url: str, schema: str, dim: int) -> None:
    """Create the extension, the schema and the tables if they don't exist."""
    if not _SCHEMA_RE.match(schema):
        raise ValueError(f"invalid schema name: {schema!r}")
    # Fail fast (e.g. Docker not running) instead of hanging on connect.
    with Connection.connect(database_url, autocommit=True, connect_timeout=5) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
        conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
        for stmt in _schema_ddl(dim):
            conn.execute(stmt)


def make_pool(database_url: str | None = None, schema: str | None = None, dim: int | None = None) -> ConnectionPool:
    database_url = database_url or settings.database_url
    schema = schema or settings.db_schema
    init_db(database_url, schema, dim or settings.embed_dim)
    pool = ConnectionPool(
        database_url,
        min_size=1,
        max_size=10,
        kwargs={"options": f"-c search_path={schema},public", "autocommit": True, "connect_timeout": 5},
        configure=_configure,
        open=True,
    )
    pool.wait(timeout=10)
    return pool


@lru_cache(maxsize=1)
def get_pool() -> ConnectionPool:
    return make_pool()
