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
        return {"primary_domain": "general", "goal": "CHAT", "confidence": 0.0, "entities": {}}


def _goal_catalog() -> list[dict]:
    return [
        {"domain": "general", "goal": "CHAT", "description": "General chat."},
        {
            "domain": "finance",
            "goal": "GET_QUOTE",
            "description": "Get current stock price/quote.",
            "domain_description": "Market and stock data domain.",
            "domain_hints": {
                "keywords": ["bolsa", "acoes", "cotacao"],
            },
            "hints": {
                "keywords": ["preco", "cotacao", "quanto esta"],
                "examples": ["qual o preco da petro?"],
            },
            "entities_schema": {
                "symbol_text": {"type": "string", "required": True},
            },
        },
        {
            "domain": "finance",
            "goal": "TOP_MOVERS",
            "description": "Find biggest gainers or losers.",
            "domain_description": "Market and stock data domain.",
            "domain_hints": {
                "keywords": ["bolsa", "acoes", "cotacao"],
            },
            "hints": {
                "keywords": ["maiores altas", "maiores baixas", "top gainers"],
                "examples": ["quais as maiores altas do bovespa?"],
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
                "market_text": {"type": "string"},
            },
        },
    ]


def test_intent_extracts_goal_with_enum_entity() -> None:
    selector = SequenceModelSelector(
        [
            {
                "primary_domain": "finance",
                "goal": "TOP_MOVERS",
                "confidence": 0.88,
                "entities": {"direction": "GAINERS", "market_text": "bovespa"},
            }
        ]
    )
    adapter = IntentAdapter(model_selector=selector, goal_catalog=_goal_catalog())

    intent = adapter.extract("quais as maiores altas de hoje do bovespa?")
    assert intent.primary_domain == "finance"
    assert intent.goal == "TOP_MOVERS"
    assert intent.entities.get("direction") == "GAINERS"
    assert intent.entities.get("market_text") == "bovespa"


def test_system_prompt_renders_domain_and_goal_hints() -> None:
    selector = SequenceModelSelector(
        [{"primary_domain": "general", "goal": "CHAT", "confidence": 0.9, "entities": {}}]
    )
    adapter = IntentAdapter(
        model_selector=selector,
        goal_catalog=_goal_catalog(),
    )

    prompt = adapter._build_system_prompt()
    # Domain description should appear
    assert "Market and stock data domain" in prompt
    # Goal-level keywords should appear
    assert "maiores altas" in prompt


def test_intent_extracts_entities_for_notification_flow() -> None:
    selector = SequenceModelSelector(
        [
            {
                "primary_domain": "finance",
                "goal": "GET_QUOTE",
                "confidence": 0.9,
                "entities": {"symbol_text": "PETR4", "notify": True},
            }
        ]
    )
    adapter = IntentAdapter(
        model_selector=selector,
        goal_catalog=_goal_catalog(),
    )

    intent = adapter.extract("qual o valor da petr4 e envie no telegram")
    assert intent.primary_domain == "finance"
    assert intent.goal == "GET_QUOTE"
    assert intent.entities.get("symbol_text") == "PETR4"
    assert intent.entities.get("notify") is True


def test_fallback_scores_goal_hints_for_domain_routing() -> None:
    """When model fails, fallback uses goal hint scoring to choose domain."""
    from intent.adapter import IntentAdapter as Adapter

    class AlwaysFailSelector:
        def generate(self, messages, policy, session_id=None):
            raise RuntimeError("model down")

    catalog = [
        {
            "domain": "finance",
            "goal": "TOP_MOVERS",
            "description": "Biggest gainers and losers.",
            "hints": {"keywords": ["maiores ganhos", "maiores altas", "top gainers"]},
        },
        {
            "domain": "general",
            "goal": "CHAT",
            "description": "General conversation.",
            "hints": {"keywords": ["oi", "ola", "ajuda"]},
        },
    ]

    adapter = Adapter(model_selector=AlwaysFailSelector(), goal_catalog=catalog)
    intent = adapter.extract("quais os maiores ganhos do bovespa hoje?")
    # Should route to finance domain based on keyword matching
    assert intent.primary_domain == "finance"
    assert intent.goal == "TOP_MOVERS"
