from __future__ import annotations

import json

from registry.db import RegistryDB
from registry.domain_registry import HandlerRegistry
from registry.loader import RegistryLoader


def test_sync_capabilities_uses_manifest_and_reconciles_stale_rows(monkeypatch, tmp_path) -> None:
    db = RegistryDB(db_path=str(tmp_path / "registry.db"))
    registry = HandlerRegistry()
    loader = RegistryLoader(db=db, runtime_registry=registry)

    db.register_domain("finance", "remote_http", {"url": "http://finance.example", "timeout": 5})
    db.register_capability("finance", "stale_capability", description="old", schema={}, metadata={})

    class _FakeHttpHandler:
        def __init__(self, base_url: str, auth_token: str | None = None, timeout: float = 60.0):
            self.base_url = base_url

        def fetch_manifest(self):
            return {
                "domain": "finance",
                "capabilities": [
                    {
                        "name": "get_stock_price",
                        "description": "Get stock price",
                        "schema": {"type": "object", "properties": {"symbol": {"type": "string"}}},
                        "metadata": {"source": "manifest"},
                    }
                ],
            }

        def fetch_openapi(self):
            raise AssertionError("OpenAPI fallback must not be used when manifest succeeds")

    monkeypatch.setattr("registry.loader.HttpDomainHandler", _FakeHttpHandler)

    ok = loader.sync_capabilities("finance")
    assert ok is True

    caps = db.list_capabilities("finance")
    names = sorted(str(item.get("capability", "")) for item in caps)
    assert names == ["get_stock_price"]


def test_sync_capabilities_falls_back_to_openapi_when_manifest_fails(monkeypatch, tmp_path) -> None:
    db = RegistryDB(db_path=str(tmp_path / "registry.db"))
    registry = HandlerRegistry()
    loader = RegistryLoader(db=db, runtime_registry=registry)

    db.register_domain("finance", "remote_http", {"url": "http://finance.example", "timeout": 5})

    class _FakeHttpHandler:
        def __init__(self, base_url: str, auth_token: str | None = None, timeout: float = 60.0):
            self.base_url = base_url

        def fetch_manifest(self):
            raise RuntimeError("manifest unavailable")

        def fetch_openapi(self):
            return {
                "openapi": "3.1.0",
                "paths": {
                    "/health": {
                        "get": {"operationId": "health_check", "summary": "Health"},
                    },
                    "/quotes/price": {
                        "post": {
                            "operationId": "get_stock_price",
                            "summary": "Get stock price",
                            "requestBody": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {"symbol": {"type": "string"}},
                                            "required": ["symbol"],
                                        }
                                    }
                                }
                            },
                        }
                    },
                },
            }

    monkeypatch.setattr("registry.loader.HttpDomainHandler", _FakeHttpHandler)

    ok = loader.sync_capabilities("finance")
    assert ok is True

    caps = db.list_capabilities("finance")
    by_name = {str(item.get("capability", "")): item for item in caps}
    assert "get_stock_price" in by_name

    metadata = json.loads(str(by_name["get_stock_price"].get("metadata", "{}")))
    assert metadata.get("source") == "openapi"
