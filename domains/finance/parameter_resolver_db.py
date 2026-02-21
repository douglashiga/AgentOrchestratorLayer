"""
Parameter Resolver Database — SQLite store for deterministic parameter mappings.

Responsibility:
- Store parameter_name + input_value → resolved_value mappings
- Support bulk seeding from alias dicts
- Provide fast lookup for deterministic resolution
- Store LLM-learned mappings for auto-learning

Follows patterns from memory/store.py (WAL, DEFERRED, row_factory).
"""

import logging
import sqlite3
import unicodedata
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "domains/finance/parameter_resolver.db"


def _normalize_key(value: str) -> str:
    """Normalize input key: strip, lowercase, unicode NFKC."""
    text = (value or "").strip().lower()
    return unicodedata.normalize("NFKC", text)


class ParameterResolverDB:
    """SQLite-backed deterministic parameter mapping store for the finance domain."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
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
        logger.info("ParameterResolverDB initialized: %s", db_path)

    def _init_db(self) -> None:
        """Create tables and indexes."""
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS parameter_mappings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parameter_name TEXT NOT NULL,
                input_value TEXT NOT NULL,
                resolved_value TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'manual',
                confidence REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(parameter_name, input_value)
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_param_lookup
            ON parameter_mappings(parameter_name, input_value)
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_param_name
            ON parameter_mappings(parameter_name)
            """
        )
        self._conn.commit()

    def lookup(self, parameter_name: str, input_value: str) -> dict[str, Any] | None:
        """
        Look up a deterministic mapping.

        Returns {"resolved_value": str, "source": str, "confidence": float} or None.
        Input value is normalized before lookup.
        """
        param = _normalize_key(parameter_name)
        value = _normalize_key(input_value)
        if not param or not value:
            return None

        row = self._conn.execute(
            "SELECT resolved_value, source, confidence FROM parameter_mappings "
            "WHERE parameter_name = ? AND input_value = ?",
            (param, value),
        ).fetchone()

        if row:
            return {
                "resolved_value": row["resolved_value"],
                "source": row["source"],
                "confidence": row["confidence"],
            }
        return None

    def insert_mapping(
        self,
        parameter_name: str,
        input_value: str,
        resolved_value: str,
        source: str = "manual",
        confidence: float = 1.0,
    ) -> None:
        """Insert or update a parameter mapping (UPSERT)."""
        param = _normalize_key(parameter_name)
        value = _normalize_key(input_value)
        if not param or not value:
            return

        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO parameter_mappings
                (parameter_name, input_value, resolved_value, source, confidence, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(parameter_name, input_value) DO UPDATE SET
                resolved_value = excluded.resolved_value,
                source = excluded.source,
                confidence = excluded.confidence,
                updated_at = excluded.updated_at
            """,
            (param, value, resolved_value, source, confidence, now, now),
        )
        self._conn.commit()

    def bulk_seed(
        self,
        parameter_name: str,
        mappings: dict[str, str],
        source: str = "seed",
    ) -> int:
        """
        Seed many mappings at once for a parameter.
        Uses UPSERT — idempotent, safe to call on every startup.
        Returns count of inserted/updated rows.
        """
        param = _normalize_key(parameter_name)
        if not param or not mappings:
            return 0

        now = datetime.now(timezone.utc).isoformat()
        count = 0
        for raw_input, resolved in mappings.items():
            key = _normalize_key(raw_input)
            if not key:
                continue
            self._conn.execute(
                """
                INSERT INTO parameter_mappings
                    (parameter_name, input_value, resolved_value, source, confidence, created_at, updated_at)
                VALUES (?, ?, ?, ?, 1.0, ?, ?)
                ON CONFLICT(parameter_name, input_value) DO UPDATE SET
                    resolved_value = excluded.resolved_value,
                    source = excluded.source,
                    confidence = excluded.confidence,
                    updated_at = excluded.updated_at
                """,
                (param, key, resolved, source, now, now),
            )
            count += 1

        self._conn.commit()
        logger.info("Seeded %d mappings for parameter '%s'", count, param)
        return count

    def list_mappings(
        self,
        parameter_name: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List mappings, optionally filtered by parameter_name."""
        if parameter_name:
            param = _normalize_key(parameter_name)
            rows = self._conn.execute(
                "SELECT parameter_name, input_value, resolved_value, source, confidence, created_at, updated_at "
                "FROM parameter_mappings WHERE parameter_name = ? ORDER BY input_value LIMIT ?",
                (param, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT parameter_name, input_value, resolved_value, source, confidence, created_at, updated_at "
                "FROM parameter_mappings ORDER BY parameter_name, input_value LIMIT ?",
                (limit,),
            ).fetchall()

        return [dict(r) for r in rows]

    def delete_mapping(self, parameter_name: str, input_value: str) -> bool:
        """Delete a specific mapping. Returns True if deleted."""
        param = _normalize_key(parameter_name)
        value = _normalize_key(input_value)
        cursor = self._conn.execute(
            "DELETE FROM parameter_mappings WHERE parameter_name = ? AND input_value = ?",
            (param, value),
        )
        self._conn.commit()
        return bool(cursor.rowcount)

    def get_stats(self) -> dict[str, Any]:
        """Return counts per parameter_name and source."""
        rows = self._conn.execute(
            "SELECT parameter_name, source, COUNT(*) as cnt "
            "FROM parameter_mappings GROUP BY parameter_name, source "
            "ORDER BY parameter_name, source"
        ).fetchall()

        stats: dict[str, dict[str, int]] = {}
        total = 0
        for row in rows:
            param = row["parameter_name"]
            src = row["source"]
            cnt = row["cnt"]
            if param not in stats:
                stats[param] = {}
            stats[param][src] = cnt
            total += cnt

        return {"total": total, "by_parameter": stats}

    def close(self) -> None:
        """Close database connection."""
        self._conn.close()
