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


def test_intent_adapter_passthrough_finance_payload() -> None:
    adapter = IntentAdapter(
        model_selector=DummyModelSelector(
            {
                "domain": "finance",
                "action": "get_stock_price",
                "confidence": 0.91,
                "parameters": {"symbol": "PETR4.SA"},
            }
        ),
        capability_catalog=[
            {"domain": "finance", "capability": "get_stock_price"},
            {"domain": "general", "capability": "chat"},
        ],
    )

    intent = adapter.extract("qual o preco da petro?")
    assert intent.domain == "finance"
    assert intent.capability == "get_stock_price"
    assert intent.confidence == 0.91
    assert intent.parameters == {"symbol": "PETR4.SA"}


def test_intent_adapter_does_not_infer_or_mutate_parameters() -> None:
    adapter = IntentAdapter(
        model_selector=DummyModelSelector(
            {
                "domain": "finance",
                "action": "get_stock_price",
                "confidence": 0.85,
                "parameters": {},
            }
        ),
        capability_catalog=[
            {"domain": "finance", "capability": "get_stock_price"},
        ],
    )

    intent = adapter.extract("qual o preco da petro e manda no telegram")
    assert intent.domain == "finance"
    assert intent.capability == "get_stock_price"
    assert intent.parameters == {}


def test_intent_adapter_fallbacks_to_general_chat_on_model_failure() -> None:
    adapter = IntentAdapter(
        model_selector=FailingModelSelector(),
        capability_catalog=[
            {"domain": "finance", "capability": "get_stock_price"},
            {"domain": "general", "capability": "chat"},
        ],
    )

    intent = adapter.extract("qual o preco da petro?")
    assert intent.domain == "general"
    assert intent.capability == "chat"
    assert intent.parameters.get("message") == "qual o preco da petro?"


def test_intent_adapter_prompt_includes_catalog_intent_hints() -> None:
    adapter = IntentAdapter(
        model_selector=DummyModelSelector(
            {
                "domain": "general",
                "action": "chat",
                "confidence": 0.9,
                "parameters": {"message": "oi"},
            }
        ),
        capability_catalog=[
            {
                "domain": "finance",
                "capability": "get_top_gainers",
                "description": "List top gainers",
                "metadata": {
                    "intent_hints": {
                        "keywords": ["maiores altas", "bovespa"],
                        "examples": ["quais as maiores altas do bovespa?"],
                    }
                },
            },
            {"domain": "general", "capability": "chat"},
        ],
    )

    prompt = adapter._build_system_prompt()
    assert "intent keywords" in prompt
    assert "maiores altas" in prompt
