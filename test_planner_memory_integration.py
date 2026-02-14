from __future__ import annotations

from typing import Any

from planner.service import PlannerService
from shared.models import IntentOutput


class DummyMemoryStore:
    def __init__(self):
        self._data: dict[tuple[str, str], Any] = {}

    def save(self, key: str, value: Any, namespace: str = "global") -> None:
        self._data[(namespace, key)] = value

    def get(self, key: str, namespace: str = "global") -> Any | None:
        return self._data.get((namespace, key))

    def search(self, query: str, namespace: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        query_l = query.lower()
        rows: list[dict[str, Any]] = []
        for (ns, key), value in self._data.items():
            if namespace and ns != namespace:
                continue
            blob = f"{key} {value}".lower()
            if query_l in blob:
                rows.append({"namespace": ns, "key": key, "value": value})
        return rows[:limit]

    def close(self) -> None:
        return


def test_planner_injects_market_from_memory_when_missing():
    memory = DummyMemoryStore()
    memory.save("preferred_market", "sweden", namespace="session:s1")

    catalog = [
        {
            "domain": "finance",
            "capability": "get_top_gainers",
            "description": "",
            "schema": {
                "type": "object",
                "properties": {
                    "market": {"type": "string"},
                    "period": {"type": "string"},
                },
            },
            "metadata": {},
        }
    ]

    planner = PlannerService(capability_catalog=catalog, model_selector=None, memory_store=memory)
    intent = IntentOutput(
        domain="finance",
        capability="get_top_gainers",
        confidence=1.0,
        parameters={},
        original_query="quero as maiores altas",
    )

    plan = planner.generate_plan(intent, session_id="s1")
    assert plan.steps[0].params.get("market") == "SE"


def test_planner_does_not_override_explicit_param():
    memory = DummyMemoryStore()
    memory.save("preferred_market", "SE", namespace="session:s1")

    catalog = [
        {
            "domain": "finance",
            "capability": "get_top_gainers",
            "description": "",
            "schema": {
                "type": "object",
                "properties": {
                    "market": {"type": "string"},
                },
            },
            "metadata": {},
        }
    ]

    planner = PlannerService(capability_catalog=catalog, model_selector=None, memory_store=memory)
    intent = IntentOutput(
        domain="finance",
        capability="get_top_gainers",
        confidence=1.0,
        parameters={"market": "BR"},
        original_query="quero as maiores altas no brasil",
    )

    plan = planner.generate_plan(intent, session_id="s1")
    assert plan.steps[0].params.get("market") == "BR"
