from __future__ import annotations

from domains.finance.handler import FinanceDomainHandler


class DummyGateway:
    def __init__(self, result):
        self._result = result

    def execute(self, skill_name, parameters):
        return self._result


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
    out = handler._resolve_symbol_value("vale", step={"search_capability": "search_symbol"})
    assert getattr(out, "status", "") == "clarification"
    assert "VALE3.SA" in out.explanation


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
