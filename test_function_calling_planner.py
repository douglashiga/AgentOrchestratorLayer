import os

from planner.function_calling_planner import FunctionCallingPlanner
from shared.models import ExecutionIntent, ExecutionPlan, ExecutionStep


class DummySelector:
    def __init__(self, response):
        self.response = response

    def generate(self, messages, policy, session_id=None):
        return self.response


def _base_plan() -> ExecutionPlan:
    return ExecutionPlan(
        execution_mode="sequential",
        combine_mode="last",
        steps=[
            ExecutionStep(
                id=1,
                domain="finance",
                capability="get_stock_price",
                params={"symbol": "AAPL"},
                depends_on=[],
                required=True,
                output_key="primary",
            )
        ],
    )


def test_function_calling_planner_adds_valid_step():
    os.environ["PLANNER_FUNCTION_CALLING_ENABLED"] = "true"
    os.environ["PLANNER_FUNCTION_CHOICE_MODE"] = "auto"

    selector = DummySelector(
        {
            "decision": "add_step",
            "combine_mode": "report",
            "step": {
                "domain": "communication",
                "capability": "send_telegram_message",
                "params": {
                    "chat_id": "${ENV:TELEGRAM_DEFAULT_CHAT_ID}",
                    "message": "${1.explanation}",
                    "ignored_field": "x",
                },
                "depends_on": [1],
                "required": False,
                "output_key": "notification",
            },
        }
    )
    catalog = [
        {
            "domain": "finance",
            "capability": "get_stock_price",
            "description": "",
            "schema": {"type": "object", "properties": {"symbol": {"type": "string"}}},
            "metadata": {},
        },
        {
            "domain": "communication",
            "capability": "send_telegram_message",
            "description": "",
            "schema": {
                "type": "object",
                "properties": {
                    "chat_id": {"type": "string"},
                    "message": {"type": "string"},
                },
            },
            "metadata": {},
        },
    ]

    planner = FunctionCallingPlanner(model_selector=selector, capability_catalog=catalog)
    intent = ExecutionIntent(
        domain="finance",
        capability="get_stock_price",
        confidence=0.99,
        parameters={"symbol": "AAPL", "notify": True},
        original_query="pega o preco e envia no telegram",
    )

    plan = planner.expand_plan(intent, _base_plan())
    assert len(plan.steps) == 2
    assert plan.execution_mode == "dag"
    assert plan.steps[1].capability == "send_telegram_message"
    assert "ignored_field" not in plan.steps[1].params


def test_function_calling_planner_rejects_unknown_capability():
    os.environ["PLANNER_FUNCTION_CALLING_ENABLED"] = "true"
    os.environ["PLANNER_FUNCTION_CHOICE_MODE"] = "auto"

    selector = DummySelector(
        {
            "decision": "add_step",
            "combine_mode": "report",
            "step": {
                "domain": "communication",
                "capability": "non_existing_cap",
                "params": {},
                "depends_on": [1],
            },
        }
    )
    catalog = [
        {
            "domain": "finance",
            "capability": "get_stock_price",
            "description": "",
            "schema": {"type": "object", "properties": {"symbol": {"type": "string"}}},
            "metadata": {},
        }
    ]

    planner = FunctionCallingPlanner(model_selector=selector, capability_catalog=catalog)
    intent = ExecutionIntent(
        domain="finance",
        capability="get_stock_price",
        confidence=0.99,
        parameters={"symbol": "AAPL"},
        original_query="consulta preco",
    )

    plan = planner.expand_plan(intent, _base_plan())
    assert len(plan.steps) == 1


def test_function_calling_planner_rejects_notifier_without_explicit_notify():
    os.environ["PLANNER_FUNCTION_CALLING_ENABLED"] = "true"
    os.environ["PLANNER_FUNCTION_CHOICE_MODE"] = "auto"

    selector = DummySelector(
        {
            "decision": "add_step",
            "combine_mode": "report",
            "step": {
                "domain": "communication",
                "capability": "send_telegram_message",
                "params": {
                    "chat_id": "${ENV:TELEGRAM_DEFAULT_CHAT_ID}",
                    "message": "${1.explanation}",
                },
                "depends_on": [1],
            },
        }
    )
    catalog = [
        {
            "domain": "finance",
            "capability": "get_stock_price",
            "description": "",
            "schema": {"type": "object", "properties": {"symbol": {"type": "string"}}},
            "metadata": {},
        },
        {
            "domain": "communication",
            "capability": "send_telegram_message",
            "description": "",
            "schema": {
                "type": "object",
                "properties": {
                    "chat_id": {"type": "string"},
                    "message": {"type": "string"},
                },
            },
            "metadata": {"composition": {"role": "notifier"}},
        },
    ]

    planner = FunctionCallingPlanner(model_selector=selector, capability_catalog=catalog)
    intent = ExecutionIntent(
        domain="finance",
        capability="get_stock_price",
        confidence=0.99,
        parameters={"symbol": "AAPL"},
        original_query="qual preco da aapl",
    )

    plan = planner.expand_plan(intent, _base_plan())
    assert len(plan.steps) == 1
