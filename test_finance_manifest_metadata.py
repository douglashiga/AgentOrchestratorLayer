from __future__ import annotations

from domains.finance import server as finance_server


class _FakeMCPAdapter:
    def list_tools(self):
        return [
            {
                "name": "get_stock_price",
                "description": "raw stock price tool",
                "schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "source": {"type": "string", "default": "close"},
                    },
                    "required": ["symbol"],
                },
            },
            {
                "name": "compare_fundamentals",
                "description": "raw compare tool",
                "schema": {
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["symbols"],
                },
            },
        ]


def _capability_map(manifest: dict) -> dict[str, dict]:
    caps = manifest.get("capabilities")
    assert isinstance(caps, list)
    by_name: dict[str, dict] = {}
    for item in caps:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            if name:
                by_name[name] = item
    return by_name


def test_finance_manifest_enriches_stock_price_metadata(monkeypatch) -> None:
    monkeypatch.setattr(finance_server, "mcp_adapter", _FakeMCPAdapter())
    manifest = finance_server.get_manifest()
    by_name = _capability_map(manifest)

    stock_price = by_name["get_stock_price"]
    metadata = stock_price.get("metadata")
    assert isinstance(metadata, dict)
    assert stock_price.get("description", "").startswith("Get current stock price")
    assert isinstance(metadata.get("intent_hints"), dict)
    assert "qual o valor" in metadata.get("intent_hints", {}).get("keywords", [])
    assert metadata.get("parameter_specs", {}).get("symbol", {}).get("required") is True
    assert metadata.get("parameter_specs", {}).get("source", {}).get("default") == "close"
    assert metadata.get("flow", {}).get("pre", [{}])[0].get("type") == "resolve_symbol"

    domain_hints = manifest.get("domain_intent_hints")
    assert isinstance(domain_hints, dict)
    assert "fundamentos" in domain_hints.get("keywords", [])


def test_finance_manifest_enriches_compare_fundamentals_metadata(monkeypatch) -> None:
    monkeypatch.setattr(finance_server, "mcp_adapter", _FakeMCPAdapter())
    manifest = finance_server.get_manifest()
    by_name = _capability_map(manifest)

    compare = by_name["compare_fundamentals"]
    metadata = compare.get("metadata")
    assert isinstance(metadata, dict)
    assert isinstance(metadata.get("intent_hints"), dict)
    assert "comparar fundamentos" in metadata.get("intent_hints", {}).get("keywords", [])
    assert metadata.get("parameter_specs", {}).get("symbols", {}).get("required") is True
    assert metadata.get("flow", {}).get("pre", [{}])[0].get("type") == "resolve_symbol_list"
