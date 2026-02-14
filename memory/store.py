"""
Memory Store abstractions and deterministic implementations.

Design goals:
- Clear API (save/get/search)
- Structured JSON values stored outside the LLM
- Deterministic retrieval for planner/context enrichment
"""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any


class MemoryStore(ABC):
    """Memory abstraction used by planner/orchestrator components."""

    @abstractmethod
    def save(self, key: str, value: Any, namespace: str = "global") -> None:
        """Persist a value by key in a namespace."""

    @abstractmethod
    def get(self, key: str, namespace: str = "global") -> Any | None:
        """Retrieve a value by key from a namespace."""

    @abstractmethod
    def search(self, query: str, namespace: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """Search memory entries by query text."""


class SQLiteMemoryStore(MemoryStore):
    """SQLite-backed deterministic memory store."""

    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            isolation_level="DEFERRED",
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_entries (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                search_text TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(namespace, key)
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memory_search
            ON memory_entries(search_text)
            """
        )
        self._conn.commit()

    def save(self, key: str, value: Any, namespace: str = "global") -> None:
        key_norm = key.strip()
        namespace_norm = (namespace or "global").strip()
        if not key_norm:
            raise ValueError("Memory key cannot be empty.")

        value_json = json.dumps(value, ensure_ascii=False, sort_keys=True)
        search_text = f"{key_norm} {value_json}".lower()
        now = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            """
            INSERT INTO memory_entries(namespace, key, value_json, search_text, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(namespace, key) DO UPDATE SET
                value_json = excluded.value_json,
                search_text = excluded.search_text,
                updated_at = excluded.updated_at
            """,
            (namespace_norm, key_norm, value_json, search_text, now),
        )
        self._conn.commit()

    def get(self, key: str, namespace: str = "global") -> Any | None:
        row = self._conn.execute(
            """
            SELECT value_json
            FROM memory_entries
            WHERE namespace = ? AND key = ?
            LIMIT 1
            """,
            ((namespace or "global").strip(), key.strip()),
        ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row["value_json"])
        except Exception:
            return row["value_json"]

    def search(self, query: str, namespace: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        query_text = (query or "").strip().lower()
        if not query_text:
            return []
        like = f"%{query_text}%"

        if namespace:
            rows = self._conn.execute(
                """
                SELECT namespace, key, value_json, updated_at
                FROM memory_entries
                WHERE namespace = ?
                  AND search_text LIKE ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (namespace.strip(), like, max(1, limit)),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT namespace, key, value_json, updated_at
                FROM memory_entries
                WHERE search_text LIKE ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (like, max(1, limit)),
            ).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            try:
                parsed_value = json.loads(row["value_json"])
            except Exception:
                parsed_value = row["value_json"]
            results.append(
                {
                    "namespace": row["namespace"],
                    "key": row["key"],
                    "value": parsed_value,
                    "updated_at": row["updated_at"],
                }
            )
        return results

    def close(self) -> None:
        self._conn.close()
