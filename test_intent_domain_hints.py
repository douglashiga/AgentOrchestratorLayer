from __future__ import annotations

from intent.adapter import IntentAdapter


class SequenceModelSelector:
    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.calls = 0

    def generate(self, messages, policy, session_id=None):
        self.calls += 1
        if self._payloads:
            return self._payloads.pop(0)
        return {"domain": "general", "action": "chat", "confidence": 0.0, "parameters": {}}


def test_intent_multi_pass_uses_domain_and_capability_hints_with_obvious_params() -> None:
    selector = SequenceModelSelector(
        [
            {
                "relevant_domains": [{"domain": "finance", "confidence": 0.93}],
                "relevant_capabilities": [
                    {"domain": "finance", "action": "get_top_gainers", "confidence": 0.91}
                ],
                "obvious_parameters": {"market": "bovespa"},
            },
            {
                "domain": "finance",
                "action": "get_top_gainers",
                "confidence": 0.88,
                "parameters": {},
            },
        ]
    )
    adapter = IntentAdapter(
        model_selector=selector,
        capability_catalog=[
            {"domain": "general", "capability": "chat", "description": "General chat."},
            {
                "domain": "finance",
                "capability": "get_top_gainers",
                "description": "Top gainers in a market.",
                "metadata": {
                    "domain_description": "Market and stock data domain.",
                    "domain_intent_hints": {
                        "keywords": ["bolsa", "acoes", "cotacao"],
                    },
                    "intent_hints": {
                        "keywords": ["maiores altas", "top gainers"],
                    },
                    "parameter_specs": {
                        "market": {
                            "type": "string",
                            "normalization": {"case": "upper"},
                            "aliases": {"BOVESPA": "BR", "IBOVESPA": "BR"},
                            "required": True,
                        }
                    },
                },
            },
        ],
    )

    intent = adapter.extract("quais as maiores altas de hoje do bovespa?")
    assert selector.calls >= 2
    assert intent.domain == "finance"
    assert intent.capability == "get_top_gainers"
    assert intent.parameters.get("market") == "BR"


def test_system_prompt_renders_domain_intent_hints() -> None:
    selector = SequenceModelSelector(
        [{"domain": "general", "action": "chat", "confidence": 0.9, "parameters": {"message": "oi"}}]
    )
    adapter = IntentAdapter(
        model_selector=selector,
        capability_catalog=[
            {
                "domain": "finance",
                "capability": "get_stock_price",
                "description": "Get stock quote.",
                "metadata": {
                    "domain_description": "Market and quotes domain.",
                    "domain_intent_hints": {
                        "keywords": ["preco", "cotacao", "bolsa"],
                        "examples": ["qual o valor da petr4?"],
                    },
                    "intent_hints": {
                        "keywords": ["qual o preco", "cotacao"],
                    },
                },
            },
            {"domain": "general", "capability": "chat", "description": "General chat."},
        ],
    )

    prompt = adapter._build_system_prompt()
    assert "domain intent keywords" in prompt
    assert "preco" in prompt


def test_intent_multi_pass_composes_execution_steps_for_next_stage() -> None:
    selector = SequenceModelSelector(
        [
            {
                "relevant_domains": [{"domain": "finance", "confidence": 0.9}],
                "relevant_capabilities": [
                    {"domain": "finance", "action": "get_stock_price", "confidence": 0.9}
                ],
                "obvious_parameters": {"symbol": "PETR4.SA", "notify": True},
            },
            {
                "domain": "finance",
                "action": "get_stock_price",
                "confidence": 0.9,
                "parameters": {"symbol": "PETR4.SA", "notify": True},
            },
        ]
    )
    adapter = IntentAdapter(
        model_selector=selector,
        capability_catalog=[
            {
                "domain": "finance",
                "capability": "get_stock_price",
                "description": "Get stock quote.",
                "metadata": {
                    "intent_hints": {"keywords": ["preco", "cotacao"]},
                    "parameter_specs": {"symbol": {"type": "string", "required": True}},
                },
            },
            {
                "domain": "communication",
                "capability": "send_telegram_message",
                "description": "Send to Telegram.",
                "metadata": {
                    "composition": {
                        "role": "notifier",
                        "priority": 100,
                        "param_map": {
                            "message": {"default": "${1.explanation}"},
                        },
                    }
                },
            },
        ],
    )

    intent = adapter.extract("qual o valor da petr4 e envie no telegram")
    steps = intent.parameters.get("_execution_steps")
    assert isinstance(steps, list)
    assert len(steps) == 2
    assert steps[0].get("capability") == "get_stock_price"
    assert steps[1].get("capability") == "send_telegram_message"
