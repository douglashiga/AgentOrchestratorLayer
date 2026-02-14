"""Persistent task/event store for declarative workflow runtime."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from shared.workflow_contracts import TaskInstance, WorkflowEvent


class TaskStateStore:
    """SQLite-backed store for workflow task snapshots and events."""

    def __init__(self, db_path: str = "agent.db"):
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
            CREATE TABLE IF NOT EXISTS workflow_tasks (
                task_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                idempotency_key TEXT,
                task_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._ensure_column("workflow_tasks", "idempotency_key", "TEXT")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_events (
                event_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_workflow_events_task_id
            ON workflow_events(task_id)
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_workflow_tasks_idempotency_key
            ON workflow_tasks(idempotency_key)
            """
        )
        self._conn.commit()

    def _ensure_column(self, table_name: str, column_name: str, declaration: str) -> None:
        rows = self._conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        columns = {str(row["name"]) for row in rows}
        if column_name in columns:
            return
        self._conn.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {declaration}"
        )

    def save_task(self, task: TaskInstance) -> None:
        payload = task.model_dump(mode="json")
        idempotency_key = ""
        if isinstance(task.metadata, dict):
            idempotency_key = str(task.metadata.get("idempotency_key", "")).strip()
        self._conn.execute(
            """
            INSERT INTO workflow_tasks (task_id, status, idempotency_key, task_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                status=excluded.status,
                idempotency_key=excluded.idempotency_key,
                task_json=excluded.task_json,
                updated_at=excluded.updated_at
            """,
            (
                task.task_id,
                task.status,
                idempotency_key,
                json.dumps(payload, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def get_task(self, task_id: str) -> TaskInstance | None:
        row = self._conn.execute(
            "SELECT task_json FROM workflow_tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            return None

        payload = json.loads(row["task_json"])
        return TaskInstance(**payload)

    def save_event(self, event: WorkflowEvent) -> None:
        payload = event.model_dump(mode="json")
        self._conn.execute(
            """
            INSERT OR REPLACE INTO workflow_events (event_id, task_id, event_type, event_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.task_id,
                event.event_type,
                json.dumps(payload, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def list_events(self, task_id: str) -> list[WorkflowEvent]:
        rows = self._conn.execute(
            """
            SELECT event_json
            FROM workflow_events
            WHERE task_id = ?
            ORDER BY created_at ASC
            """,
            (task_id,),
        ).fetchall()
        return [WorkflowEvent(**json.loads(row["event_json"])) for row in rows]

    def find_latest_task_by_idempotency_key(self, idempotency_key: str) -> TaskInstance | None:
        key = str(idempotency_key or "").strip()
        if not key:
            return None
        row = self._conn.execute(
            """
            SELECT task_json
            FROM workflow_tasks
            WHERE idempotency_key = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (key,),
        ).fetchone()
        if row is None:
            return None
        return TaskInstance(**json.loads(row["task_json"]))

    def close(self) -> None:
        self._conn.close()
