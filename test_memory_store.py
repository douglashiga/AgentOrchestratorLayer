from __future__ import annotations

from pathlib import Path

from memory.store import SQLiteMemoryStore


def test_memory_store_save_get_search(tmp_path: Path):
    db_path = tmp_path / "memory_test.db"
    store = SQLiteMemoryStore(db_path=str(db_path))

    try:
        store.save("preferred_market", "SE", namespace="global")
        store.save("risk_mode", "moderate", namespace="session:s1")
        store.save("wheel_active", True, namespace="session:s1")

        assert store.get("preferred_market", namespace="global") == "SE"
        assert store.get("risk_mode", namespace="session:s1") == "moderate"
        assert store.get("wheel_active", namespace="session:s1") is True

        results = store.search("moderate", namespace="session:s1", limit=10)
        assert results
        assert any(row["key"] == "risk_mode" for row in results)
    finally:
        store.close()
