from planner.task_decomposer import TaskDecomposer
from shared.models import IntentOutput


def test_finance_notify_plan_is_multi_domain():
    decomposer = TaskDecomposer(
        known_capabilities=[
            "get_stock_price",
            "send_telegram_message",
        ]
    )
    intent = IntentOutput(
        domain="finance",
        capability="get_stock_price",
        confidence=1.0,
        parameters={"symbol": "AAPL"},
        original_query="pegue o preco da aapl e envie no telegram",
    )

    plan = decomposer.decompose(intent)
    assert plan.execution_mode == "dag"
    assert plan.combine_mode == "report"
    assert len(plan.steps) == 2
    assert plan.steps[0].domain == "finance"
    assert plan.steps[1].domain == "communication"
    assert plan.steps[1].depends_on == [1]


def test_default_plan_is_single_step():
    decomposer = TaskDecomposer(known_capabilities=["get_stock_price"])
    intent = IntentOutput(
        domain="finance",
        capability="get_stock_price",
        confidence=1.0,
        parameters={"symbol": "AAPL"},
        original_query="qual o preco da aapl",
    )

    plan = decomposer.decompose(intent)
    assert plan.execution_mode == "sequential"
    assert len(plan.steps) == 1
