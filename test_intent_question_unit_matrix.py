from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from intent.adapter import IntentAdapter
from planner.task_decomposer import TaskDecomposer


class ScenarioModelSelector:
    """Deterministic model stub keyed by question text."""

    def __init__(self, expected_by_question: dict[str, dict[str, Any]]):
        self.expected_by_question = expected_by_question

    def generate(self, messages, policy, session_id=None):
        question = str(messages[-1].get("content", "")).strip()
        expected = self.expected_by_question.get(question, {})
        domain = str(expected.get("intent_domain", "general")).strip() or "general"
        action = str(expected.get("intent_capability", "chat")).strip() or "chat"
        params: dict[str, Any] = {}
        if domain == "general" and action == "chat":
            params = {"message": question}
        return {
            "domain": domain,
            "action": action,
            "confidence": 0.91,
            "parameters": params,
        }


def _capability_catalog() -> list[dict[str, Any]]:
    return [
        {
            "domain": "general",
            "capability": "chat",
            "description": "General conversation.",
            "schema": {"type": "object", "properties": {"message": {"type": "string"}}},
            "metadata": {},
        },
        {
            "domain": "finance",
            "capability": "get_stock_price",
            "description": "Get stock price by symbol/ticker.",
            "schema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "symbols": {"type": "array", "items": {"type": "string"}},
                    "currency": {"type": "string"},
                    "exchange": {"type": "string"},
                    "notify": {"type": "boolean"},
                },
            },
            "metadata": {
                "parameter_specs": {
                    "currency": {
                        "infer_from_symbol_suffix": {
                            ".SA": "BRL",
                            ".ST": "SEK",
                        },
                        "default": "USD",
                    },
                    "exchange": {
                        "infer_from_symbol_suffix": {
                            ".SA": "BOVESPA",
                            ".ST": "SFB",
                        },
                        "default": "SMART",
                    },
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
            "domain": "finance",
            "capability": "compare_fundamentals",
            "description": "Compare fundamentals for multiple symbols.",
            "schema": {
                "type": "object",
                "properties": {"symbols": {"type": "array", "items": {"type": "string"}}},
                "required": ["symbols"],
            },
            "metadata": {},
        },
        {
            "domain": "communication",
            "capability": "send_telegram_message",
            "description": "Send a Telegram message after primary action when user asks notify/send.",
            "schema": {
                "type": "object",
                "properties": {
                    "chat_id": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["message"],
            },
            "metadata": {
                "composition": {
                    "role": "notifier",
                    "priority": 100,
                    "param_map": {
                        "chat_id": {
                            "from_parameters": ["chat_id", "group_id"],
                            "default": "${ENV:TELEGRAM_DEFAULT_CHAT_ID}",
                        },
                        "message": {
                            "from_parameters": ["message", "notify_message"],
                            "default": "${1.explanation}",
                        },
                    },
                }
            },
        },
    ]


def _load_cases() -> list[dict[str, Any]]:
    path = Path(__file__).resolve().parent / "scripts" / "questions.intent.unit.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("questions.intent.unit.json must be a JSON array")
    out: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        out.append(item)
    if not out:
        raise ValueError("questions.intent.unit.json must contain at least one question item")
    return out


def _as_symbols(params: dict[str, Any]) -> list[str]:
    symbols = params.get("symbols")
    if isinstance(symbols, list):
        return [str(v).strip() for v in symbols if str(v).strip()]
    symbol = params.get("symbol")
    if isinstance(symbol, str) and symbol.strip():
        return [symbol.strip()]
    return []


def _assert_expected(case: dict[str, Any], intent: Any, plan: Any) -> None:
    expected = case.get("expected", {})
    if not isinstance(expected, dict):
        expected = {}

    params = dict(intent.parameters or {})
    step_caps = [str(step.capability) for step in plan.steps]

    exp_domain = expected.get("intent_domain")
    if isinstance(exp_domain, str) and exp_domain.strip():
        assert intent.domain == exp_domain

    exp_capability = expected.get("intent_capability")
    if isinstance(exp_capability, str) and exp_capability.strip():
        assert intent.capability == exp_capability

    exp_params_contains = expected.get("intent_parameters_contains")
    if isinstance(exp_params_contains, dict):
        for key, value in exp_params_contains.items():
            assert params.get(key) == value

    exp_has_keys = expected.get("intent_parameters_has_keys")
    if isinstance(exp_has_keys, list):
        for key in [str(v).strip() for v in exp_has_keys if str(v).strip()]:
            assert key in params

    exp_not_has_keys = expected.get("intent_parameters_not_has_keys")
    if isinstance(exp_not_has_keys, list):
        for key in [str(v).strip() for v in exp_not_has_keys if str(v).strip()]:
            assert key not in params

    exp_symbols = expected.get("intent_symbols_exact")
    if isinstance(exp_symbols, list):
        assert _as_symbols(params) == [str(v) for v in exp_symbols]

    exp_min_steps = expected.get("plan_min_steps")
    if isinstance(exp_min_steps, int):
        assert len(plan.steps) >= exp_min_steps

    exp_plan_caps = expected.get("plan_contains_step_capabilities")
    if isinstance(exp_plan_caps, list):
        for cap in [str(v).strip() for v in exp_plan_caps if str(v).strip()]:
            assert cap in step_caps


@pytest.mark.parametrize(
    "case",
    _load_cases(),
    ids=lambda row: str(row.get("question", ""))[:80],
)
def test_intent_matrix_with_prompt_expectations(case: dict[str, Any]) -> None:
    question = str(case.get("question", "")).strip()
    expected = case.get("expected", {})
    prompt_contains = case.get("prompt_contains", [])
    if not isinstance(prompt_contains, list):
        prompt_contains = []

    model = ScenarioModelSelector(expected_by_question={question: expected if isinstance(expected, dict) else {}})
    catalog = _capability_catalog()
    adapter = IntentAdapter(model_selector=model, capability_catalog=catalog)
    decomposer = TaskDecomposer(capability_catalog=catalog)

    messages = adapter._build_messages(question, history=[])
    assert str(messages[-1].get("content", "")).strip() == question
    system_prompt = str(messages[0].get("content", ""))
    for snippet in [str(v).strip() for v in prompt_contains if str(v).strip()]:
        assert snippet in system_prompt

    intent = adapter.extract(question, history=[], session_id="unit-matrix")
    plan = decomposer.decompose(intent)
    _assert_expected(case=case, intent=intent, plan=plan)
