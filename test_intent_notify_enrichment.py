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


def test_intent_enrichment_extracts_multiple_symbols_for_price_queries():
    adapter = IntentAdapter(
        model_selector=DummyModelSelector(
            {
                "domain": "finance",
                "action": "get_stock_price",
                "confidence": 0.88,
                "parameters": {},
            }
        ),
        capability_catalog=[
            {"domain": "finance", "capability": "get_stock_price"},
        ],
    )

    intent = adapter.extract("qual o valor da petro e da vale?")
    assert intent.domain == "finance"
    assert intent.capability == "get_stock_price"
    assert intent.parameters.get("symbols") == ["PETR4.SA", "VALE3.SA"]
    assert intent.confidence >= 0.95


def test_intent_enrichment_does_not_capture_notify_verbs_as_symbols():
    adapter = IntentAdapter(
        model_selector=DummyModelSelector(
            {
                "domain": "finance",
                "action": "get_stock_price",
                "confidence": 0.90,
                "parameters": {"symbols": ["PETR4.SA", "MANDA"]},
            }
        ),
        capability_catalog=[
            {
                "domain": "finance",
                "capability": "get_stock_price",
                "metadata": {
                    "parameter_specs": {
                        "currency": {
                            "infer_from_symbol_suffix": {".SA": "BRL", ".ST": "SEK"},
                            "default": "USD",
                        },
                        "exchange": {
                            "infer_from_symbol_suffix": {".SA": "BOVESPA", ".ST": "SFB"},
                            "default": "SMART",
                        },
                    }
                },
            },
            {"domain": "communication", "capability": "send_telegram_message"},
        ],
    )

    intent = adapter.extract("qual o valor da petro e manda no telegram")
    assert intent.parameters.get("notify") is True
    assert intent.parameters.get("symbol") == "PETR4.SA"
    assert intent.parameters.get("symbols") is None
    assert intent.parameters.get("currency") == "BRL"
    assert intent.parameters.get("exchange") == "BOVESPA"


def test_intent_enrichment_merges_model_single_symbol_with_query_multi_symbols():
    adapter = IntentAdapter(
        model_selector=DummyModelSelector(
            {
                "domain": "finance",
                "action": "get_stock_price",
                "confidence": 0.90,
                "parameters": {"symbol": "VALE3.SA"},
            }
        ),
        capability_catalog=[
            {"domain": "finance", "capability": "get_stock_price"},
        ],
    )

    intent = adapter.extract("qual o valor da vale3 e petro4?")
    assert intent.domain == "finance"
    assert intent.capability == "get_stock_price"
    assert intent.parameters.get("symbol") == "VALE3.SA"
    assert intent.parameters.get("symbols") == ["VALE3.SA", "PETR4.SA"]
    assert intent.confidence >= 0.95


def test_intent_enrichment_applies_generic_parameter_contract_defaults_and_normalization():
    adapter = IntentAdapter(
        model_selector=DummyModelSelector(
            {
                "domain": "ops",
                "action": "restart_job",
                "confidence": 0.91,
                "parameters": {"environment": "prod"},
            }
        ),
        capability_catalog=[
            {
                "domain": "ops",
                "capability": "restart_job",
                "metadata": {
                    "parameter_specs": {
                        "environment": {
                            "type": "string",
                            "enum": ["DEV", "PROD"],
                            "normalization": {"case": "upper"},
                        },
                        "region": {
                            "type": "string",
                            "default": "us-east-1",
                        },
                    }
                },
            },
        ],
    )

    intent = adapter.extract("reinicie o job em prod")
    assert intent.domain == "ops"
    assert intent.capability == "restart_job"
    assert intent.parameters.get("environment") == "PROD"
    assert intent.parameters.get("region") == "us-east-1"


def test_intent_enrichment_inferrs_currency_and_exchange_from_symbol_suffix():
    adapter = IntentAdapter(
        model_selector=DummyModelSelector(
            {
                "domain": "finance",
                "action": "get_stock_price",
                "confidence": 0.95,
                "parameters": {"symbol": "PETR4.SA"},
            }
        ),
        capability_catalog=[
            {
                "domain": "finance",
                "capability": "get_stock_price",
                "metadata": {
                    "parameter_specs": {
                        "currency": {
                            "infer_from_symbol_suffix": {".SA": "BRL", ".ST": "SEK"},
                            "default": "USD",
                        },
                        "exchange": {
                            "infer_from_symbol_suffix": {".SA": "BOVESPA", ".ST": "SFB"},
                            "default": "SMART",
                        },
                    }
                },
            },
        ],
    )

    intent = adapter.extract("qual o valor da petro")
    assert intent.domain == "finance"
    assert intent.capability == "get_stock_price"
    assert intent.parameters.get("symbol") == "PETR4.SA"
    assert intent.parameters.get("currency") == "BRL"
    assert intent.parameters.get("exchange") == "BOVESPA"
