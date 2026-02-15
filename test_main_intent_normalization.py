from __future__ import annotations

from main import ORCHESTRATOR_CONFIDENCE_THRESHOLD, _normalize_intent_parameters
from registry.domain_registry import HandlerRegistry
from shared.models import IntentOutput


def _registry_with_stock_price() -> HandlerRegistry:
    registry = HandlerRegistry()
    registry.register_capability(
        "get_stock_price",
        handler=object(),
        metadata={
            "domain": "finance",
            "description": "Get stock price by symbol.",
            "intent_hints": {
                "keywords": ["qual o valor", "valor", "qual o preco", "cotacao", "stock price"],
                "examples": ["qual o valor da petr4?"],
            },
            "parameter_specs": {
                "symbol": {
                    "type": "string",
                    "required": True,
                },
                "currency": {
                    "type": "string",
                    "infer_from_symbol_suffix": {".SA": "BRL", ".ST": "SEK"},
                    "default": "USD",
                },
                "exchange": {
                    "type": "string",
                    "infer_from_symbol_suffix": {".SA": "BOVESPA", ".ST": "SFB"},
                    "default": "SMART",
                },
            },
            "flow": {
                "pre": [
                    {
                        "type": "resolve_symbol",
                        "param": "symbol",
                        "search_capability": "search_symbol",
                    }
                ]
            },
        },
    )
    registry.register_capability(
        "send_telegram_message",
        handler=object(),
        metadata={
            "domain": "communication",
            "description": "Send notification messages to Telegram.",
            "channel": "telegram",
            "intent_hints": {
                "keywords": ["envie no telegram", "manda no telegram", "notifique no telegram"],
            },
            "composition": {"role": "notifier"},
            "parameter_specs": {
                "message": {"type": "string", "required": True},
            },
            "schema": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        },
    )
    return registry


def test_normalize_routes_general_chat_price_query_to_finance() -> None:
    registry = _registry_with_stock_price()
    raw = IntentOutput(
        domain="general",
        capability="chat",
        confidence=0.4,
        parameters={"message": "qual o valor da petro4?"},
        original_query="qual o valor da petro4?",
    )

    normalized = _normalize_intent_parameters(raw, registry, entry_request=None)
    assert normalized.domain == "finance"
    assert normalized.capability == "get_stock_price"
    assert normalized.parameters.get("symbol") is None
    assert normalized.parameters.get("currency") == "USD"
    assert normalized.parameters.get("exchange") == "SMART"
    assert normalized.confidence >= ORCHESTRATOR_CONFIDENCE_THRESHOLD


def test_normalize_keeps_general_chat_for_non_finance_message() -> None:
    registry = _registry_with_stock_price()
    raw = IntentOutput(
        domain="general",
        capability="chat",
        confidence=0.8,
        parameters={"message": "oi tudo bem?"},
        original_query="oi tudo bem?",
    )

    normalized = _normalize_intent_parameters(raw, registry, entry_request=None)
    assert normalized.domain == "general"
    assert normalized.capability == "chat"


def test_normalize_carries_notify_flag_when_routing_to_finance() -> None:
    registry = _registry_with_stock_price()
    raw = IntentOutput(
        domain="general",
        capability="chat",
        confidence=0.5,
        parameters={"message": "me diga o valor da petro4 e me envie no telegram"},
        original_query="me diga o valor da petro4 e me envie no telegram",
    )

    normalized = _normalize_intent_parameters(raw, registry, entry_request=None)
    assert normalized.domain == "finance"
    assert normalized.capability == "get_stock_price"
    assert normalized.parameters.get("symbol") is None
    assert normalized.parameters.get("currency") == "USD"
    assert normalized.parameters.get("exchange") == "SMART"
    assert normalized.parameters.get("notify") is True


def test_normalize_does_not_reroute_without_method_intent_hints() -> None:
    registry = HandlerRegistry()
    registry.register_capability(
        "get_company_info",
        handler=object(),
        metadata={
            "domain": "finance",
            "description": "Get company profile and sector info.",
            "parameter_specs": {"symbol": {"type": "string", "required": True}},
        },
    )
    raw = IntentOutput(
        domain="general",
        capability="chat",
        confidence=0.5,
        parameters={"message": "qual o valor da petro4?"},
        original_query="qual o valor da petro4?",
    )

    normalized = _normalize_intent_parameters(raw, registry, entry_request=None)
    assert normalized.domain == "general"
    assert normalized.capability == "chat"


def test_normalize_routes_plain_company_token_to_price_capability() -> None:
    registry = _registry_with_stock_price()
    raw = IntentOutput(
        domain="general",
        capability="chat",
        confidence=0.4,
        parameters={"message": "qual o valor da petro?"},
        original_query="qual o valor da petro?",
    )

    normalized = _normalize_intent_parameters(raw, registry, entry_request=None)
    assert normalized.domain == "finance"
    assert normalized.capability == "get_stock_price"
    assert normalized.parameters.get("symbol") is None
    assert normalized.parameters.get("currency") == "USD"
    assert normalized.parameters.get("exchange") == "SMART"


def test_normalize_canonicalizes_domain_from_registered_capability_metadata() -> None:
    registry = _registry_with_stock_price()
    raw = IntentOutput(
        domain="general",
        capability="get_stock_price",
        confidence=0.9,
        parameters={"symbol": "PETR4.SA"},
        original_query="qual o valor da petr4?",
    )

    normalized = _normalize_intent_parameters(raw, registry, entry_request=None)
    assert normalized.domain == "finance"
    assert normalized.capability == "get_stock_price"


def test_normalize_parameter_alias_matching_is_accent_insensitive() -> None:
    registry = HandlerRegistry()
    registry.register_capability(
        "get_top_gainers",
        handler=object(),
        metadata={
            "domain": "finance",
            "description": "Top gainers by market",
            "parameter_specs": {
                "market": {
                    "type": "string",
                    "aliases": {"SUÉCIA": "SE", "BOVESPA": "BR"},
                    "normalization": {"case": "upper"},
                }
            },
        },
    )
    raw = IntentOutput(
        domain="finance",
        capability="get_top_gainers",
        confidence=0.8,
        parameters={"market": "suecia"},
        original_query="maiores altas da suecia",
    )

    normalized = _normalize_intent_parameters(raw, registry, entry_request=None)
    assert normalized.parameters.get("market") == "SE"


def test_normalize_boosts_confidence_for_registered_capability_with_required_params() -> None:
    registry = _registry_with_stock_price()
    raw = IntentOutput(
        domain="finance",
        capability="get_stock_price",
        confidence=0.15,
        parameters={"symbol": "PETR4.SA"},
        original_query="qual o valor da petr4?",
    )

    normalized = _normalize_intent_parameters(raw, registry, entry_request=None)
    assert normalized.confidence >= ORCHESTRATOR_CONFIDENCE_THRESHOLD
    assert normalized.parameters.get("currency") == "BRL"
    assert normalized.parameters.get("exchange") == "BOVESPA"


def test_normalize_does_not_boost_confidence_when_required_params_are_missing() -> None:
    registry = _registry_with_stock_price()
    raw = IntentOutput(
        domain="finance",
        capability="get_stock_price",
        confidence=0.2,
        parameters={},
        original_query="oi tudo bem?",
    )

    normalized = _normalize_intent_parameters(raw, registry, entry_request=None)
    assert normalized.confidence == 0.2


def test_normalize_boosts_confidence_when_flow_can_resolve_required_params() -> None:
    registry = _registry_with_stock_price()
    raw = IntentOutput(
        domain="finance",
        capability="get_stock_price",
        confidence=0.46,
        parameters={},
        original_query="qual o valor da petr4.sa?",
    )

    normalized = _normalize_intent_parameters(raw, registry, entry_request=None)
    assert normalized.parameters.get("symbol") is None
    assert normalized.parameters.get("currency") == "USD"
    assert normalized.parameters.get("exchange") == "SMART"
    assert normalized.confidence >= ORCHESTRATOR_CONFIDENCE_THRESHOLD
