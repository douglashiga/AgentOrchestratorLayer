"""
Conversation State Manager — SQLite-backed.

Responsibility:
- Store text history per session_id
- Retrieve history
- Persist new interactions

Performance:
- Persistent SQLite connection (no reconnect per query)
- WAL mode for concurrent reads

Prohibitions:
- Never alters business rules
- Never alters domain context
- Never executes calculations
"""

import json
import sqlite3
from datetime import datetime, timezone


class ConversationManager:
    """SQLite-backed conversation state manager with persistent connection."""

    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = db_path
        # Persistent connection — reused for all operations
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            isolation_level="DEFERRED",
        )
        self._conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrent read performance
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_id
            ON conversations(session_id)
        """)
        self._conn.commit()

    def save(self, session_id: str, role: str, content: str, metadata: dict | None = None) -> None:
        """Persist a conversation turn."""
        self._conn.execute(
            """INSERT INTO conversations (session_id, role, content, metadata, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                session_id,
                role,
                content,
                json.dumps(metadata or {}),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def get_history(self, session_id: str, limit: int = 20) -> list[dict]:
        """Retrieve conversation history for a session."""
        rows = self._conn.execute(
            """SELECT role, content, created_at
               FROM conversations
               WHERE session_id = ?
               ORDER BY id DESC
               LIMIT ?""",
            (session_id, limit),
        ).fetchall()

        # Return in chronological order
        return [
            {"role": row["role"], "content": row["content"], "created_at": row["created_at"]}
            for row in reversed(rows)
        ]

    def clear_session(self, session_id: str) -> None:
        """Clear all history for a session."""
        self._conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
        self._conn.commit()

    def close(self) -> None:
        """Close the persistent connection."""
        self._conn.close()
