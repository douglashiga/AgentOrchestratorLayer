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
    assert intent.parameters.get("symbol") == "PETR4.SA"
    assert isinstance(intent.parameters.get("_execution_steps"), list)


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
    assert "symbol" not in intent.parameters
    assert isinstance(intent.parameters.get("_execution_steps"), list)


def test_intent_adapter_fallback_prefers_general_chat_without_catalog_evidence() -> None:
    adapter = IntentAdapter(
        model_selector=FailingModelSelector(),
        capability_catalog=[
            {"domain": "finance", "capability": "get_stock_price"},
            {"domain": "general", "capability": "chat"},
        ],
    )

    intent = adapter.extract("oi")
    assert intent.domain == "general"
    assert intent.capability == "chat"
    assert intent.parameters.get("message") == "oi"


def test_intent_adapter_fallback_uses_catalog_hints_when_model_fails() -> None:
    adapter = IntentAdapter(
        model_selector=FailingModelSelector(),
        capability_catalog=[
            {
                "domain": "finance",
                "capability": "get_stock_price",
                "description": "Consultar preco e cotacao de ativo por ticker.",
                "metadata": {
                    "intent_hints": {
                        "keywords": ["qual o valor", "preco", "cotacao"],
                    }
                },
            },
            {"domain": "general", "capability": "chat"},
        ],
    )

    intent = adapter.extract("qual o valor da petro4?")
    assert intent.domain == "finance"
    assert intent.capability == "get_stock_price"
    assert intent.confidence > 0.0


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


def test_intent_adapter_composes_execution_steps_in_single_pass(monkeypatch) -> None:
    monkeypatch.setenv("INTENT_MULTI_PASS_ENABLED", "false")
    adapter = IntentAdapter(
        model_selector=DummyModelSelector(
            {
                "domain": "finance",
                "action": "get_stock_price",
                "confidence": 0.88,
                "parameters": {"symbol": "PETR4.SA", "notify": True},
            }
        ),
        capability_catalog=[
            {
                "domain": "finance",
                "capability": "get_stock_price",
                "metadata": {
                    "parameter_specs": {"symbol": {"type": "string", "required": True}},
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
        ],
    )

    intent = adapter.extract("qual o valor da petr4 e me notifique")
    steps = intent.parameters.get("_execution_steps")
    assert isinstance(steps, list)
    assert len(steps) == 2
    assert steps[0].get("domain") == "finance"
    assert steps[0].get("capability") == "get_stock_price"
    assert steps[1].get("domain") == "communication"
    assert steps[1].get("capability") == "send_telegram_message"
