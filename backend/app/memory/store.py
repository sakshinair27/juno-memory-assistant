"""pgvector-backed fact store.

We store *facts*, not transcripts: one row per durable statement about the
user, with its embedding. Every mutation is also appended to memory_events so
the UI (and evals) can show "lives in Seattle -> lives in Austin" history.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from uuid import UUID

import numpy as np
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


@dataclass
class Memory:
    id: str
    content: str
    category: str
    pinned: bool
    confidence: float
    created_at: datetime
    updated_at: datetime
    source_session: str | None = None
    access_count: int = 0
    similarity: float | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        d["updated_at"] = self.updated_at.isoformat()
        return d


_COLS = "id::text, content, category, pinned, confidence, created_at, updated_at, source_session, access_count"


def _row(r: dict) -> Memory:
    return Memory(**r)


class MemoryStore:
    def __init__(self, pool: ConnectionPool):
        self.pool = pool

    # ---------- reads ----------
    def search(self, embedding: np.ndarray, k: int, min_sim: float = 0.0,
               pinned: bool | None = None) -> list[Memory]:
        where = "WHERE 1 - (embedding <=> %(e)s) >= %(min)s"
        if pinned is not None:
            where += " AND pinned = %(pinned)s"
        q = f"""
            SELECT {_COLS}, 1 - (embedding <=> %(e)s) AS similarity
            FROM memories {where}
            ORDER BY embedding <=> %(e)s
            LIMIT %(k)s
        """
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(q, {"e": embedding, "min": min_sim, "k": k, "pinned": pinned})
            return [_row(r) for r in cur.fetchall()]

    def pinned(self, limit: int) -> list[Memory]:
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(f"SELECT {_COLS} FROM memories WHERE pinned ORDER BY updated_at DESC LIMIT %s", (limit,))
            return [_row(r) for r in cur.fetchall()]

    def get(self, memory_id: str) -> Memory | None:
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(f"SELECT {_COLS} FROM memories WHERE id = %s", (UUID(memory_id),))
            r = cur.fetchone()
            return _row(r) if r else None

    def list_all(self) -> list[Memory]:
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(f"SELECT {_COLS} FROM memories ORDER BY pinned DESC, updated_at DESC")
            return [_row(r) for r in cur.fetchall()]

    def events(self, limit: int = 50) -> list[dict]:
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT id, memory_id::text, op, old_content, new_content, reason, session_id, created_at "
                "FROM memory_events ORDER BY id DESC LIMIT %s", (limit,))
            rows = cur.fetchall()
        for r in rows:
            r["created_at"] = r["created_at"].isoformat()
        return rows

    def touch(self, ids: list[str]) -> None:
        if not ids:
            return
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE memories SET last_accessed_at = now(), access_count = access_count + 1 WHERE id = ANY(%s)",
                ([UUID(i) for i in ids],))

    # ---------- writes ----------
    def add(self, content: str, category: str, embedding: np.ndarray, *, pinned: bool = False,
            confidence: float = 1.0, session_id: str | None = None, reason: str | None = None) -> Memory:
        with self.pool.connection() as conn, conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"INSERT INTO memories (content, category, pinned, confidence, embedding, source_session) "
                f"VALUES (%s, %s, %s, %s, %s, %s) RETURNING {_COLS}",
                (content, category, pinned, confidence, embedding, session_id))
            m = _row(cur.fetchone())
            cur.execute(
                "INSERT INTO memory_events (memory_id, op, new_content, reason, session_id) VALUES (%s, 'ADD', %s, %s, %s)",
                (UUID(m.id), content, reason, session_id))
        return m

    def update(self, memory_id: str, content: str, embedding: np.ndarray, *, category: str | None = None,
               pinned: bool | None = None, session_id: str | None = None, reason: str | None = None) -> Memory | None:
        with self.pool.connection() as conn, conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT content FROM memories WHERE id = %s FOR UPDATE", (UUID(memory_id),))
            old = cur.fetchone()
            if old is None:
                return None
            cur.execute(
                f"UPDATE memories SET content = %s, embedding = %s, category = COALESCE(%s, category), "
                f"pinned = COALESCE(%s, pinned), source_session = COALESCE(%s, source_session), updated_at = now() "
                f"WHERE id = %s RETURNING {_COLS}",
                (content, embedding, category, pinned, session_id, UUID(memory_id)))
            m = _row(cur.fetchone())
            cur.execute(
                "INSERT INTO memory_events (memory_id, op, old_content, new_content, reason, session_id) "
                "VALUES (%s, 'UPDATE', %s, %s, %s, %s)",
                (UUID(memory_id), old["content"], content, reason, session_id))
        return m

    def delete(self, memory_id: str, *, session_id: str | None = None, reason: str | None = None) -> bool:
        with self.pool.connection() as conn, conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
            cur.execute("DELETE FROM memories WHERE id = %s RETURNING content", (UUID(memory_id),))
            old = cur.fetchone()
            if old is None:
                return False
            cur.execute(
                "INSERT INTO memory_events (memory_id, op, old_content, reason, session_id) VALUES (%s, 'DELETE', %s, %s, %s)",
                (UUID(memory_id), old["content"], reason, session_id))
        return True

    # ---------- quarantine (memory-poisoning review queue) ----------
    def quarantine(self, content: str, category: str, reason: str, *, source_message: str | None = None,
                   session_id: str | None = None) -> int:
        with self.pool.connection() as conn:
            return conn.execute(
                "INSERT INTO quarantined_facts (content, category, reason, source_message, session_id) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (content, category, reason, source_message, session_id)).fetchone()[0]

    def quarantined(self, limit: int = 50) -> list[dict]:
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT id, content, category, reason, source_message, created_at "
                        "FROM quarantined_facts ORDER BY id DESC LIMIT %s", (limit,))
            rows = cur.fetchall()
        for r in rows:
            r["created_at"] = r["created_at"].isoformat()
        return rows

    def clear(self) -> None:
        with self.pool.connection() as conn:
            conn.execute("TRUNCATE memories, memory_events, quarantined_facts")
