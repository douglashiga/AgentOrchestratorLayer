from api.openai_server import _format_progress_event_human, _format_progress_event_json


def test_human_intent_progress_renders_confidence() -> None:
    text = _format_progress_event_human(
        {
            "type": "intent_normalized",
            "intent": {
                "domain": "finance",
                "capability": "get_stock_price",
                "confidence": 0.94,
                "parameters": {"symbol": "PETR4.SA"},
            },
        },
        "intent_normalized",
    )
    assert "conf=0.94" in text


def test_json_step_progress_includes_confidence_fields() -> None:
    text = _format_progress_event_json(
        {
            "type": "step_completed",
            "step_id": 1,
            "domain": "finance",
            "capability": "get_stock_price",
            "status": "success",
            "confidence": 1.0,
            "intent_confidence": 0.94,
            "result": {"price": 37.25},
        },
        "step_completed",
    )
    assert '"confidence":1.0' in text
    assert '"intent_confidence":0.94' in text
