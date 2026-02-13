"""
Registry Database — SQLite storage for Dynamic Domains.

Responsibility:
- Store Domain configurations (URL, auth, changes)
- Store Capability schemas
- Persist Registry state across restarts
"""

import sqlite3
import json
import logging
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

DB_PATH = "registry.db"

class RegistryDB:
    """SQLite-backed Registry for Domains and Capabilities."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS domains (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    type TEXT NOT NULL,  -- 'local', 'remote_http'
                    config TEXT NOT NULL, -- JSON
                    enabled BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS capabilities (
                    id TEXT PRIMARY KEY,
                    domain_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    input_schema TEXT, -- JSON
                    metadata TEXT DEFAULT '{}', -- JSON for UI hints, templates, etc.
                    FOREIGN KEY(domain_id) REFERENCES domains(id) ON DELETE CASCADE,
                    UNIQUE(domain_id, name)
                );
            """)

    def register_domain(self, name: str, domain_type: str, config: dict[str, Any]) -> str:
        """Register or update a domain."""
        domain_id = str(uuid.uuid4())
        config_json = json.dumps(config)
        
        with self._get_conn() as conn:
            # Check if exists
            existing = conn.execute("SELECT id FROM domains WHERE name = ?", (name,)).fetchone()
            if existing:
                domain_id = existing["id"]
                conn.execute(
                    "UPDATE domains SET type = ?, config = ?, enabled = 1 WHERE id = ?",
                    (domain_type, config_json, domain_id)
                )
                logger.info("Updated domain: %s (%s)", name, domain_type)
            else:
                conn.execute(
                    "INSERT INTO domains (id, name, type, config) VALUES (?, ?, ?, ?)",
                    (domain_id, name, domain_type, config_json)
                )
                logger.info("Registered new domain: %s (%s)", name, domain_type)
        
        return domain_id

    def register_capability(self, domain_name: str, capability: str, description: str = "", schema: dict | None = None, metadata: dict | None = None) -> None:
        """Register a capability for a domain."""
        with self._get_conn() as conn:
            domain = conn.execute("SELECT id FROM domains WHERE name = ?", (domain_name,)).fetchone()
            if not domain:
                raise ValueError(f"Domain not found: {domain_name}")
            
            domain_id = domain["id"]
            schema_json = json.dumps(schema) if schema else "{}"
            metadata_json = json.dumps(metadata) if metadata else "{}"
            
            # Upsert capability
            existing = conn.execute(
                "SELECT id FROM capabilities WHERE domain_id = ? AND name = ?", 
                (domain_id, capability)
            ).fetchone()

            if existing:
                 conn.execute(
                    "UPDATE capabilities SET description = ?, input_schema = ?, metadata = ? WHERE id = ?",
                    (description, schema_json, metadata_json, existing["id"])
                )
            else:
                cap_id = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO capabilities (id, domain_id, name, description, input_schema, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                    (cap_id, domain_id, capability, description, schema_json, metadata_json)
                )
            
            logger.info("Registered capability: %s -> %s", domain_name, capability)

    def get_domain(self, name: str) -> Optional[dict[str, Any]]:
        """Get domain config by name."""
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM domains WHERE name = ?", (name,)).fetchone()
            if row:
                return dict(row)
        return None

    def list_domains(self) -> list[dict[str, Any]]:
        """List all enabled domains."""
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM domains WHERE enabled = 1").fetchall()
            return [dict(r) for r in rows]

    def list_capabilities(self, domain_name: str | None = None) -> list[dict[str, Any]]:
        """List capabilities, optionally filtered by domain."""
        with self._get_conn() as conn:
            query = """
                SELECT c.name as capability, c.description, d.name as domain, d.type, d.config, c.input_schema, c.metadata
                FROM capabilities c
                JOIN domains d ON c.domain_id = d.id
                WHERE d.enabled = 1
            """
            params = []
            if domain_name:
                query += " AND d.name = ?"
                params.append(domain_name)
            
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
