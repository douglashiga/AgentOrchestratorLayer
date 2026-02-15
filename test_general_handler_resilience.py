from __future__ import annotations

from domains.general.handler import GeneralDomainHandler
from shared.models import IntentOutput


class FailingModelSelector:
    def generate(self, messages, policy, session_id=None):
        raise RuntimeError("provider unavailable")


def test_general_handler_returns_fallback_text_on_model_error():
    handler = GeneralDomainHandler(model_selector=FailingModelSelector())
    intent = IntentOutput(
        domain="general",
        capability="chat",
        confidence=1.0,
        parameters={"message": "oi"},
        original_query="oi",
    )

    output = handler.execute(intent)
    if hasattr(output, "__await__"):
        import asyncio

        output = asyncio.run(output)

    assert output.status == "success"
    assert isinstance(output.result.get("response"), str)
    assert "instabilidade no chat geral" in output.explanation.lower()
