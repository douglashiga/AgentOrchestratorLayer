from __future__ import annotations

from intent.adapter import IntentAdapter


class DummyModelSelector:
    def __init__(self, payload):
        self.payload = payload

    def generate(self, messages, policy, session_id=None):
        return self.payload


class FailingModelSelector:
    def generate(self, messages, policy, session_id=None):
        raise RuntimeError("model unavailable")


def _goal_catalog() -> list[dict]:
    return [
        {
            "domain": "general",
            "goal": "CHAT",
            "description": "General conversation and help.",
            "hints": {"keywords": ["oi", "ola", "hello", "ajuda"]},
        },
        {
            "domain": "finance",
            "goal": "GET_QUOTE",
            "description": "Get current stock price/quote.",
            "hints": {"keywords": ["preco", "cotacao", "valor", "quanto esta"]},
            "entities_schema": {
                "symbol_text": {"type": "string", "required": True, "description": "Symbol or company name as user said it"},
            },
        },
    ]


def test_intent_adapter_passthrough_finance_payload() -> None:
    adapter = IntentAdapter(
        model_selector=DummyModelSelector(
            {
                "primary_domain": "finance",
                "goal": "GET_QUOTE",
                "confidence": 0.91,
                "entities": {"symbol_text": "Petrobras"},
            }
        ),
        goal_catalog=_goal_catalog(),
    )

    intent = adapter.extract("qual o preco da petro?")
    assert intent.primary_domain == "finance"
    assert intent.goal == "GET_QUOTE"
    assert intent.confidence == 0.91
    assert intent.entities.get("symbol_text") == "Petrobras"


def test_intent_adapter_does_not_infer_or_mutate_parameters() -> None:
    adapter = IntentAdapter(
        model_selector=DummyModelSelector(
            {
                "primary_domain": "finance",
                "goal": "GET_QUOTE",
                "confidence": 0.85,
                "entities": {},
            }
        ),
        goal_catalog=_goal_catalog(),
    )

    intent = adapter.extract("qual o preco da petro e manda no telegram")
    assert intent.primary_domain == "finance"
    assert intent.goal == "GET_QUOTE"
    # No ticker inferred — only what was in model output
    assert "symbol" not in intent.entities
    assert intent.entities == {}


def test_intent_adapter_fallback_prefers_general_chat_without_catalog_evidence() -> None:
    adapter = IntentAdapter(
        model_selector=FailingModelSelector(),
        goal_catalog=_goal_catalog(),
    )

    intent = adapter.extract("oi")
    assert intent.primary_domain == "general"
    assert intent.goal in ("CHAT", "chat", "GENERAL")


def test_intent_adapter_fallback_uses_catalog_hints_when_model_fails() -> None:
    adapter = IntentAdapter(
        model_selector=FailingModelSelector(),
        goal_catalog=_goal_catalog(),
    )

    intent = adapter.extract("qual o valor da petro4?")
    assert intent.primary_domain == "finance"
    assert intent.goal == "GET_QUOTE"
    assert intent.confidence > 0.0


def test_intent_adapter_prompt_includes_goal_hints() -> None:
    adapter = IntentAdapter(
        model_selector=DummyModelSelector(
            {
                "primary_domain": "general",
                "goal": "CHAT",
                "confidence": 0.9,
                "entities": {},
            }
        ),
        goal_catalog=[
            {
                "domain": "finance",
                "goal": "TOP_MOVERS",
                "description": "Find biggest gainers/losers.",
                "hints": {
                    "keywords": ["maiores altas", "bovespa"],
                    "examples": ["quais as maiores altas do bovespa?"],
                },
            },
            {"domain": "general", "goal": "CHAT", "description": "General chat."},
        ],
    )

    prompt = adapter._build_system_prompt()
    assert "maiores altas" in prompt
    assert "bovespa" in prompt


def test_intent_adapter_returns_goal_based_output_not_capability() -> None:
    """Intent output has goal, not capability. GoalResolver handles capability mapping."""
    adapter = IntentAdapter(
        model_selector=DummyModelSelector(
            {
                "primary_domain": "finance",
                "goal": "GET_QUOTE",
                "confidence": 0.88,
                "entities": {"symbol_text": "PETR4"},
            }
        ),
        goal_catalog=_goal_catalog(),
    )

    intent = adapter.extract("qual o valor da petr4")
    # New model: primary_domain + goal, NOT domain + capability
    assert hasattr(intent, "primary_domain")
    assert hasattr(intent, "goal")
    assert hasattr(intent, "entities")
    assert not hasattr(intent, "capability")
    assert intent.goal == "GET_QUOTE"
    assert intent.entities.get("symbol_text") == "PETR4"
