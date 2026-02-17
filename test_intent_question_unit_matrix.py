from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from intent.adapter import IntentAdapter
from planner.goal_resolver import GoalResolver
from planner.task_decomposer import TaskDecomposer


class ScenarioModelSelector:
    """Deterministic model stub keyed by question text.

    Returns goal-based output matching the new IntentOutput schema.
    """

    def __init__(self, expected_by_question: dict[str, dict[str, Any]]):
        self.expected_by_question = expected_by_question

    def generate(self, messages, policy, session_id=None):
        question = str(messages[-1].get("content", "")).strip()
        expected = self.expected_by_question.get(question, {})
        primary_domain = str(expected.get("intent_primary_domain", "general")).strip() or "general"
        goal = str(expected.get("intent_goal", "CHAT")).strip() or "CHAT"
        entities: dict[str, Any] = {}
        return {
            "primary_domain": primary_domain,
            "goal": goal,
            "confidence": 0.91,
            "entities": entities,
        }


def _goal_catalog() -> list[dict[str, Any]]:
    """Goal catalog matching the finance + general + communication domains."""
    return [
        {
            "domain": "general",
            "goal": "CHAT",
            "description": "General conversation.",
            "hints": {"keywords": ["oi", "ola", "hello", "ajuda", "help", "deploy", "kubernetes"]},
        },
        {
            "domain": "finance",
            "goal": "GET_QUOTE",
            "description": "Get current stock price/quote for one or more symbols.",
            "hints": {"keywords": ["valor", "preco", "cotacao", "quanto esta"]},
            "entities_schema": {
                "symbol_text": {"type": "string", "required": True, "description": "Symbol or company name as user said"},
                "symbols_text": {"type": "array", "description": "Multiple symbols"},
            },
        },
        {
            "domain": "finance",
            "goal": "COMPARE_STOCKS",
            "description": "Compare fundamentals for multiple companies.",
            "hints": {"keywords": ["compare", "comparar", "fundamentos"]},
            "entities_schema": {
                "symbols_text": {"type": "array", "required": True, "description": "Companies to compare"},
            },
        },
    ]


def _goal_catalog_dict() -> dict[str, dict[str, Any]]:
    """Build the dict keyed by 'domain:GOAL_NAME' for GoalResolver."""
    catalog = {
        "general:CHAT": {
            "capabilities": ["chat"],
        },
        "finance:GET_QUOTE": {
            "capabilities": ["get_stock_price"],
            "entities_schema": {
                "symbol_text": {"type": "string", "required": True},
                "symbols_text": {"type": "array"},
            },
        },
        "finance:COMPARE_STOCKS": {
            "capabilities": ["compare_fundamentals"],
            "entities_schema": {
                "symbols_text": {"type": "array", "required": True},
            },
        },
    }
    return catalog


def _capability_catalog() -> list[dict[str, Any]]:
    """Capability catalog for TaskDecomposer (still needed for decomposition metadata)."""
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
                    "notify": {"type": "boolean"},
                },
            },
            "metadata": {
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


def _assert_expected(case: dict[str, Any], intent: Any, resolved: Any, plan: Any) -> None:
    expected = case.get("expected", {})
    if not isinstance(expected, dict):
        expected = {}

    step_caps = [str(step.capability) for step in plan.steps]

    # IntentOutput fields (goal-based)
    exp_primary_domain = expected.get("intent_primary_domain")
    if isinstance(exp_primary_domain, str) and exp_primary_domain.strip():
        assert intent.primary_domain == exp_primary_domain

    exp_goal = expected.get("intent_goal")
    if isinstance(exp_goal, str) and exp_goal.strip():
        assert intent.goal == exp_goal

    # ExecutionIntent fields (post-resolver)
    exp_capability = expected.get("resolved_capability")
    if isinstance(exp_capability, str) and exp_capability.strip():
        assert resolved.capability == exp_capability

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
    goal_cat = _goal_catalog()
    goal_cat_dict = _goal_catalog_dict()
    cap_catalog = _capability_catalog()

    adapter = IntentAdapter(model_selector=model, goal_catalog=goal_cat)
    resolver = GoalResolver(goal_catalog_dict=goal_cat_dict)
    decomposer = TaskDecomposer(capability_catalog=cap_catalog)

    messages = adapter._build_messages(question, history=[])
    assert str(messages[-1].get("content", "")).strip() == question

    system_prompt = str(messages[0].get("content", ""))
    for snippet in [str(v).strip() for v in prompt_contains if str(v).strip()]:
        assert snippet in system_prompt, (
            f"Expected '{snippet}' in system prompt.\n"
            f"Prompt excerpt:\n{system_prompt[:500]}"
        )

    # Full pipeline: adapter → resolver → decomposer
    intent = adapter.extract(question, history=[], session_id="unit-matrix")
    resolved = resolver.resolve(intent)
    plan = decomposer.decompose(resolved)
    _assert_expected(case=case, intent=intent, resolved=resolved, plan=plan)
