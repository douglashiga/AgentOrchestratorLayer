from __future__ import annotations

from intent.adapter import IntentAdapter


class DummyModelSelector:
    def __init__(self, payload):
        self.payload = payload

    def generate(self, messages, policy, session_id=None):
        return self.payload


def _goal_catalog() -> list[dict]:
    return [
        {"domain": "general", "goal": "CHAT", "description": "General chat."},
        {
            "domain": "finance",
            "goal": "TOP_MOVERS",
            "description": "List top gainers for market ranking queries.",
            "hints": {
                "keywords": ["maiores altas", "bovespa", "top gainers"],
                "examples": ["quais as maiores altas de hoje do bovespa?"],
            },
            "entities_schema": {
                "direction": {
                    "type": "enum",
                    "values": ["GAINERS", "LOSERS", "BOTH"],
                    "required": True,
                    "default": "BOTH",
                    "capability_map": {
                        "GAINERS": "get_top_gainers",
                        "LOSERS": "get_top_losers",
                        "BOTH": ["get_top_gainers", "get_top_losers"],
                    },
                },
                "market_text": {"type": "string", "description": "Market as user mentioned it"},
            },
        },
    ]


def test_goal_catalog_hints_are_rendered_in_system_prompt():
    adapter = IntentAdapter(
        model_selector=DummyModelSelector(
            {
                "primary_domain": "general",
                "goal": "CHAT",
                "confidence": 0.9,
                "entities": {},
            }
        ),
        goal_catalog=_goal_catalog(),
    )

    prompt = adapter._build_system_prompt()
    # Goal hints should appear in prompt
    assert "maiores altas" in prompt
    assert "bovespa" in prompt
    # Goal name should appear
    assert "TOP_MOVERS" in prompt


def test_model_output_entities_are_preserved() -> None:
    adapter = IntentAdapter(
        model_selector=DummyModelSelector(
            {
                "primary_domain": "finance",
                "goal": "TOP_MOVERS",
                "confidence": 0.92,
                "entities": {"direction": "GAINERS", "market_text": "bovespa"},
            }
        ),
        goal_catalog=_goal_catalog(),
    )

    intent = adapter.extract("quais as maiores altas de hoje do bovespa?")
    assert intent.primary_domain == "finance"
    assert intent.goal == "TOP_MOVERS"
    # Entities passed through from model output unchanged — no alias normalization in intent layer
    assert intent.entities.get("direction") == "GAINERS"
    assert intent.entities.get("market_text") == "bovespa"


def test_entities_schema_rendered_in_prompt() -> None:
    adapter = IntentAdapter(
        model_selector=DummyModelSelector(
            {"primary_domain": "general", "goal": "CHAT", "confidence": 0.9, "entities": {}}
        ),
        goal_catalog=_goal_catalog(),
    )

    prompt = adapter._build_system_prompt()
    # entities_schema should be rendered so LLM knows what to extract
    assert "direction" in prompt
    assert "GAINERS" in prompt
    assert "market_text" in prompt
