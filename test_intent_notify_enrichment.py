from __future__ import annotations

from intent.adapter import IntentAdapter


class DummyModelSelector:
    def __init__(self, payload):
        self.payload = payload

    def generate(self, messages, policy, session_id=None):
        return self.payload


def test_intent_enrichment_sets_notify_and_confidence_floor():
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
            {"domain": "communication", "capability": "send_telegram_message"},
        ],
    )

    intent = adapter.extract("me pegue o valor da petro e me envie no telegram")
    assert intent.parameters.get("notify") is True
    assert intent.confidence >= 0.95
    assert intent.domain == "finance"
    assert intent.capability == "get_stock_price"
