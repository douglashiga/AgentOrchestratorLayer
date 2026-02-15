from __future__ import annotations

from shared.models import DomainOutput
from shared.response_formatter import format_domain_output


def test_format_domain_output_rounds_price_to_two_decimals():
    out = DomainOutput(
        status="success",
        result={"symbol": "PETR4.SA", "price": 36.88999938964844, "currency": "BRL", "date": "2026-02-13"},
        explanation="",
    )
    text = format_domain_output(out)
    assert text == "PETR4.SA: 36.89 BRL (2026-02-13)"


def test_format_domain_output_dag_with_notification():
    out = DomainOutput(
        status="success",
        result={
            "steps": {
                "primary": {
                    "status": "success",
                    "result": {"symbol": "PETR4.SA", "price": 36.889999, "currency": "BRL", "date": "2026-02-13"},
                },
                "notification": {"status": "success"},
            }
        },
        explanation="Executed 2 task(s).",
    )
    text = format_domain_output(out)
    assert "PETR4.SA: 36.89 BRL (2026-02-13)" in text
    assert "Mensagem enviada no Telegram." in text


def test_format_domain_output_dag_multi_steps_with_arbitrary_keys():
    out = DomainOutput(
        status="success",
        result={
            "steps": {
                "foo": {
                    "step_id": 10,
                    "status": "success",
                    "capability": "get_stock_price",
                    "result": {"symbol": "VALE3.SA", "price": 61.0199, "currency": "BRL", "date": "2026-02-13"},
                },
                "bar": {
                    "step_id": 20,
                    "status": "success",
                    "capability": "get_stock_price",
                    "result": {"symbol": "PETR4.SA", "price": 36.8899, "currency": "BRL", "date": "2026-02-13"},
                },
            }
        },
        explanation="Executed 2 task(s).",
    )
    text = format_domain_output(out)
    assert "VALE3.SA: 61.02 BRL (2026-02-13)" in text
    assert "PETR4.SA: 36.89 BRL (2026-02-13)" in text
    assert "Concluído com sucesso." not in text


def test_format_domain_output_failure_is_objective():
    out = DomainOutput(
        status="failure",
        explanation="",
        metadata={"error": "MCP error: timeout 10.1234"},
    )
    text = format_domain_output(out)
    assert text.startswith("Não consegui concluir:")
    assert "10.12" in text
