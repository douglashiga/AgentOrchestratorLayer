from planner.task_decomposer import TaskDecomposer
from shared.models import ExecutionIntent


def test_finance_notify_plan_is_multi_domain():
    decomposer = TaskDecomposer(
        capability_catalog=[
            {
                "domain": "finance",
                "capability": "get_stock_price",
                "metadata": {
                    "composition": {
                        "followup_roles": ["notifier"],
                        "enabled_if": {"path": "parameters.notify", "equals": True},
                    }
                },
            },
            {
                "domain": "communication",
                "capability": "send_telegram_message",
                "metadata": {
                    "composition": {
                        "role": "notifier",
                        "priority": 100,
                        "param_map": {
                            "chat_id": {
                                "from_parameters": ["chat_id"],
                                "default": "${ENV:TELEGRAM_DEFAULT_CHAT_ID}",
                            },
                            "message": {
                                "from_parameters": ["message"],
                                "default": "${1.explanation}",
                            },
                        },
                    }
                },
            },
        ]
    )
    intent = ExecutionIntent(
        domain="finance",
        capability="get_stock_price",
        confidence=1.0,
        parameters={"symbol": "AAPL", "notify": True},
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
    decomposer = TaskDecomposer(
        capability_catalog=[
            {
                "domain": "finance",
                "capability": "get_stock_price",
                "metadata": {
                    "composition": {
                        "followup_roles": ["notifier"],
                        "enabled_if": {"path": "parameters.notify", "equals": True},
                    }
                },
            }
        ]
    )
    intent = ExecutionIntent(
        domain="finance",
        capability="get_stock_price",
        confidence=1.0,
        parameters={"symbol": "AAPL"},
        original_query="qual o preco da aapl",
    )

    plan = decomposer.decompose(intent)
    assert plan.execution_mode == "sequential"
    assert len(plan.steps) == 1


def test_method_contract_takes_precedence_over_legacy_composition() -> None:
    decomposer = TaskDecomposer(
        capability_catalog=[
            {
                "domain": "finance",
                "capability": "get_stock_price",
                "metadata": {
                    "workflow": {
                        "execution_mode": "dag",
                        "nodes": [
                            {"id": "resolve", "kind": "resolve"},
                            {"id": "fetch", "kind": "call"},
                        ],
                        "edges": [
                            {"from_node": "resolve", "to_node": "fetch"},
                        ],
                    },
                    "composition": {
                        "followup_roles": ["notifier"],
                        "enabled_if": {"path": "parameters.notify", "equals": True},
                    },
                },
            },
            {
                "domain": "communication",
                "capability": "send_telegram_message",
                "metadata": {
                    "composition": {
                        "role": "notifier",
                        "priority": 100,
                        "param_map": {"message": {"default": "${1.explanation}"}},
                    }
                },
            },
        ]
    )
    intent = ExecutionIntent(
        domain="finance",
        capability="get_stock_price",
        confidence=1.0,
        parameters={"symbol": "AAPL", "notify": True},
        original_query="pegue preço e mande",
    )

    plan = decomposer.decompose(intent)
    assert plan.execution_mode == "sequential"
    assert len(plan.steps) == 1
    assert plan.steps[0].capability == "get_stock_price"


def test_multi_symbol_get_price_generates_parallel_dag() -> None:
    decomposer = TaskDecomposer(
        capability_catalog=[
            {
                "domain": "finance",
                "capability": "get_stock_price",
                "metadata": {
                    "decomposition": {
                        "array_params": [
                            {"param_name": "symbols_text", "single_param_name": "symbol_text", "max_concurrency": 4}
                        ]
                    }
                },
            }
        ]
    )
    intent = ExecutionIntent(
        domain="finance",
        capability="get_stock_price",
        confidence=1.0,
        parameters={"symbols_text": ["PETR4", "VALE3"]},
        original_query="qual o valor da petro e da vale",
    )

    plan = decomposer.decompose(intent)
    assert plan.execution_mode == "dag"
    assert plan.combine_mode == "report"
    assert len(plan.steps) == 2
    assert plan.steps[0].params.get("symbol_text") == "PETR4"
    assert plan.steps[1].params.get("symbol_text") == "VALE3"
    assert all(step.capability == "get_stock_price" for step in plan.steps)


def test_multi_symbol_notify_adds_notifier_followup_when_catalog_supports_it() -> None:
    decomposer = TaskDecomposer(
        capability_catalog=[
            {
                "domain": "finance",
                "capability": "get_stock_price",
                "metadata": {
                    "decomposition": {
                        "array_params": [
                            {"param_name": "symbols_text", "single_param_name": "symbol_text", "max_concurrency": 4}
                        ]
                    },
                    "composition": {
                        "followup_roles": ["notifier"],
                        "enabled_if": {"path": "parameters.notify", "equals": True},
                        "followup_required": False,
                        "followup_output_key": "notification",
                    },
                },
            },
            {
                "domain": "communication",
                "capability": "send_telegram_message",
                "metadata": {
                    "composition": {
                        "role": "notifier",
                        "priority": 100,
                        "param_map": {
                            "chat_id": {
                                "from_parameters": ["chat_id"],
                                "default": "${ENV:TELEGRAM_DEFAULT_CHAT_ID}",
                            },
                            "message": {
                                "from_parameters": ["message"],
                                "default": "${1.explanation}",
                            },
                        },
                    }
                },
            },
        ]
    )
    intent = ExecutionIntent(
        domain="finance",
        capability="get_stock_price",
        confidence=1.0,
        parameters={"symbols_text": ["PETR4", "VALE3"], "notify": True},
        original_query="qual o valor da petro e vale e manda no telegram",
    )

    plan = decomposer.decompose(intent)
    assert plan.execution_mode == "dag"
    assert plan.combine_mode == "report"
    assert len(plan.steps) == 3
    assert plan.steps[0].capability == "get_stock_price"
    assert plan.steps[1].capability == "get_stock_price"
    assert plan.steps[2].capability == "send_telegram_message"
    assert plan.steps[2].depends_on == [1, 2]


def test_decomposer_prefers_execution_steps_hint_from_intent_parameters() -> None:
    decomposer = TaskDecomposer(capability_catalog=[])
    intent = ExecutionIntent(
        domain="finance",
        capability="get_stock_price",
        confidence=1.0,
        parameters={
            "symbol": "PETR4.SA",
            "_execution_steps": [
                {
                    "domain": "finance",
                    "capability": "get_stock_price",
                    "params": {"symbol": "PETR4.SA"},
                    "required": True,
                },
                {
                    "domain": "communication",
                    "capability": "send_telegram_message",
                    "params": {"message": "${1.explanation}"},
                    "depends_on": [1],
                    "required": False,
                },
            ],
        },
        original_query="qual o valor da petr4 e envie no telegram",
    )

    plan = decomposer.decompose(intent)
    assert plan.execution_mode == "dag"
    assert len(plan.steps) == 2
    assert plan.steps[0].capability == "get_stock_price"
    assert plan.steps[1].capability == "send_telegram_message"
    assert plan.steps[1].depends_on == [1]
