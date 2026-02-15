from __future__ import annotations

from intent.adapter import IntentAdapter


class DummyModelSelector:
    def __init__(self, payload):
        self.payload = payload

    def generate(self, messages, policy, session_id=None):
        return self.payload


def _catalog() -> list[dict]:
    return [
        {"domain": "general", "capability": "chat", "description": "General chat."},
        {
            "domain": "finance",
            "capability": "get_top_gainers",
            "description": "List top gainers for market ranking queries.",
            "metadata": {
                "intent_hints": {
                    "keywords": ["maiores altas", "bovespa", "top gainers"],
                    "examples": ["quais as maiores altas de hoje do bovespa?"],
                },
                "parameter_specs": {
                    "market": {
                        "type": "string",
                        "normalization": {"case": "upper"},
                        "aliases": {"BOVESPA": "BR", "IBOVESPA": "BR"},
                        "examples": ["BR", "US", "SE"],
                    }
                },
            },
        },
    ]


def test_catalog_intent_hints_are_rendered_in_system_prompt():
    adapter = IntentAdapter(
        model_selector=DummyModelSelector(
            {
                "domain": "general",
                "action": "chat",
                "confidence": 0.9,
                "parameters": {"message": "oi"},
            }
        ),
        capability_catalog=_catalog(),
    )

    prompt = adapter._build_system_prompt()
    assert "intent keywords" in prompt
    assert "maiores altas" in prompt
    assert "bovespa" in prompt


def test_model_output_parameters_are_preserved_without_normalization():
    adapter = IntentAdapter(
        model_selector=DummyModelSelector(
            {
                "domain": "finance",
                "action": "get_top_gainers",
                "confidence": 0.92,
                "parameters": {"market": "bovespa"},
            }
        ),
        capability_catalog=_catalog(),
    )

    intent = adapter.extract("quais as maiores altas de hoje do bovespa?")
    assert intent.domain == "finance"
    assert intent.capability == "get_top_gainers"
    assert intent.parameters.get("market") == "bovespa"
