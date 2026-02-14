from registry.domain_registry import HandlerRegistry


class _DummyHandler:
    pass


def test_register_capability_builds_method_spec_from_metadata_workflow() -> None:
    registry = HandlerRegistry()
    handler = _DummyHandler()

    registry.register_capability(
        "get_stock_price",
        handler,
        metadata={
            "domain": "finance",
            "description": "Get stock price",
            "schema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                },
                "required": ["symbol"],
            },
            "workflow": {
                "execution_mode": "sequential",
                "nodes": [
                    {"id": "fetch", "kind": "call", "handler": "market_data.get_quote"},
                ],
            },
            "policy": {
                "default_timeout_seconds": 15.0,
                "human_validation": {
                    "enabled": True,
                    "auto_continue_confidence": 0.85,
                },
            },
        },
    )

    spec = registry.resolve_method_spec("get_stock_price")
    assert spec is not None
    assert spec.domain == "finance"
    assert spec.method == "get_stock_price"
    assert spec.workflow.nodes[0].kind == "call"
    assert spec.policy.default_timeout_seconds == 15.0


def test_register_capability_builds_method_spec_from_explicit_payload() -> None:
    registry = HandlerRegistry()
    handler = _DummyHandler()

    registry.register_capability(
        "send_telegram_message",
        handler,
        metadata={
            "method_spec": {
                "domain": "communication",
                "method": "send_telegram_message",
                "workflow": {
                    "execution_mode": "sequential",
                    "nodes": [
                        {
                            "id": "deliver",
                            "kind": "call",
                            "params": {
                                "capability": "send_telegram_message",
                            },
                        }
                    ],
                },
            }
        },
    )

    spec = registry.resolve_method_spec("send_telegram_message")
    assert spec is not None
    assert spec.domain == "communication"
    assert spec.method == "send_telegram_message"
