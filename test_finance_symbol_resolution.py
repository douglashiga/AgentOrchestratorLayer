from __future__ import annotations

from domains.finance.handler import FinanceDomainHandler
from shared.models import Decision


class DummyGateway:
    def __init__(self, result):
        self._result = result

    def execute(self, skill_name, parameters):
        return self._result


class DummyGatewayByAction:
    def __init__(self, by_action):
        self.by_action = by_action

    def execute(self, skill_name, parameters):
        action = parameters.get("_action")
        return self.by_action.get(action, {"success": False, "error": "not-found"})


def test_symbol_resolution_returns_clarification_for_ambiguous_match():
    gateway = DummyGateway(
        {
            "success": True,
            "data": {
                "results": [
                    {"symbol": "VALE3.SA", "name": "Vale ON"},
                    {"symbol": "VALE", "name": "Vale ADR"},
                ]
            },
        }
    )
    handler = FinanceDomainHandler(skill_gateway=gateway, registry=None)
    out = handler._resolve_symbol_value("vale2", step={"search_capability": "search_symbol"})
    assert getattr(out, "status", "") == "clarification"
    assert "VALE3.SA" in out.explanation


def test_symbol_resolution_uses_alias_for_plain_company_name():
    gateway = DummyGateway({"success": False, "error": "should_not_be_called"})
    handler = FinanceDomainHandler(skill_gateway=gateway, registry=None)
    out = handler._resolve_symbol_value("vale", step={"search_capability": "search_symbol"})
    assert out == "VALE3.SA"


def test_symbol_resolution_picks_single_match():
    gateway = DummyGateway(
        {
            "success": True,
            "data": {
                "results": [
                    {"symbol": "PETR4.SA", "name": "Petrobras PN"},
                ]
            },
        }
    )
    handler = FinanceDomainHandler(skill_gateway=gateway, registry=None)
    out = handler._resolve_symbol_value("petrobras", step={"search_capability": "search_symbol"})
    assert out == "PETR4.SA"


def test_symbol_resolution_returns_clarification_when_lookup_fails():
    gateway = DummyGateway(
        {
            "success": False,
            "error": "network",
        }
    )
    handler = FinanceDomainHandler(skill_gateway=gateway, registry=None)
    out = handler._resolve_symbol_value("PETRO4", step={"search_capability": "search_symbol"})
    assert getattr(out, "status", "") == "clarification"


def test_infer_symbol_from_query_text_b3_base():
    handler = FinanceDomainHandler(skill_gateway=DummyGateway({"success": True, "data": {}}), registry=None)
    inferred = handler._infer_symbol_from_query_text("qual o valor da petro4 hoje?")
    assert inferred == "PETR4.SA"


def test_infer_symbol_from_query_text_plain_company_token():
    handler = FinanceDomainHandler(skill_gateway=DummyGateway({"success": True, "data": {}}), registry=None)
    inferred = handler._infer_symbol_from_query_text("me diga o valor da petro e manda no telegram")
    assert inferred == "PETRO"


def test_pre_flow_resolves_inferred_plain_token_via_search():
    gateway = DummyGatewayByAction(
        {
            "yahoo_search": {
                "success": True,
                "data": {
                    "quotes": [
                        {"symbol": "PETR4.SA", "shortname": "Petrobras PN"},
                    ]
                },
            }
        }
    )
    handler = FinanceDomainHandler(skill_gateway=gateway, registry=None)
    metadata = {
        "schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
        "flow": {"pre": [{"type": "resolve_symbol", "param": "symbol", "required": True}]},
    }
    out = handler._apply_pre_flow(
        capability="get_stock_price",
        params={},
        metadata=metadata,
        original_query="me diga o valor da petro e envia no telegram",
    )
    assert isinstance(out, dict)
    assert out.get("symbol") == "PETR4.SA"


def test_get_flow_steps_marks_required_from_schema_even_for_explicit_flow():
    handler = FinanceDomainHandler(skill_gateway=DummyGateway({"success": True, "data": {}}), registry=None)
    metadata = {
        "schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
        "flow": {"pre": [{"type": "resolve_symbol", "param": "symbol"}]},
    }
    steps = handler._get_flow_steps("get_stock_price", {}, metadata)
    assert steps and steps[0]["required"] is True


def test_stock_price_cache_fallback_returns_success():
    gateway = DummyGatewayByAction(
        {
            "get_historical_data_cached": {
                "success": True,
                "data": {
                    "success": True,
                    "data": [
                        {"date": "2026-02-10", "close": 10.0},
                        {"date": "2026-02-11", "close": 11.5},
                    ],
                },
            }
        }
    )
    handler = FinanceDomainHandler(skill_gateway=gateway, registry=None)

    class Ctx:
        currency = "BRL"

    out = handler._fallback_stock_price_from_cache(
        params={"symbol": "PETR4.SA"},
        domain_context=Ctx(),
        original_error="ib_not_connected",
    )
    assert out is not None
    assert out.status == "success"
    assert out.result.get("price") == 11.5
    assert out.metadata.get("source") == "cache_fallback"


def test_validate_required_params_from_schema_returns_clarification():
    handler = FinanceDomainHandler(skill_gateway=DummyGateway({"success": True, "data": {}}), registry=None)
    metadata = {
        "schema": {
            "type": "object",
            "required": ["symbol", "period"],
            "properties": {
                "symbol": {"type": "string"},
                "period": {"type": "string"},
            },
        }
    }
    out = handler._validate_required_params_from_schema(
        capability="get_historical_data",
        params={"symbol": "PETR4.SA"},
        metadata=metadata,
    )
    assert out is not None
    assert out.status == "clarification"
    assert "period" in out.explanation


def test_operational_error_is_mapped_to_clarification():
    handler = FinanceDomainHandler(skill_gateway=DummyGateway({"success": True, "data": {}}), registry=None)
    decision = Decision(
        action="get_stock_price",
        success=False,
        error="ib_not_connected | Not connected to IB Gateway",
        explanation="Failed",
    )
    out = handler._map_operational_error_to_clarification("get_stock_price", decision)
    assert out is not None
    assert out.status == "clarification"
