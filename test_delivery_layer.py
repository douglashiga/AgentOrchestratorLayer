from __future__ import annotations

from shared.delivery_layer import build_delivery_payload
from shared.models import DomainOutput


def test_delivery_layer_renders_table_for_list_rows():
    out = DomainOutput(
        status="success",
        result={"items": [{"symbol": "PETR4.SA", "price": 36.88}, {"symbol": "VALE3.SA", "price": 61.01}]},
        explanation="",
    )
    payload = build_delivery_payload(out)
    assert payload.kind == "table"
    assert "| symbol | price |" in payload.content


def test_delivery_layer_renders_video_link():
    out = DomainOutput(
        status="success",
        result={"url": "https://youtu.be/dQw4w9WgXcQ"},
        explanation="",
    )
    payload = build_delivery_payload(out)
    assert payload.kind == "video"
    assert "Abrir vídeo" in payload.content


def test_delivery_layer_falls_back_to_text():
    out = DomainOutput(
        status="success",
        result={"symbol": "PETR4.SA", "price": 36.8899, "currency": "BRL", "date": "2026-02-13"},
        explanation="",
    )
    payload = build_delivery_payload(out)
    assert payload.kind == "text"
    assert "36.89" in payload.content
