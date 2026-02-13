import asyncio
import logging
from unittest.mock import MagicMock, AsyncMock

from orchestrator.orchestrator import Orchestrator
from shared.models import IntentOutput, DomainOutput

logging.basicConfig(level=logging.INFO)

def test_async_orchestration():
    async def _run() -> None:
        registry = MagicMock()
        handler = MagicMock()
        handler.execute = AsyncMock(return_value=DomainOutput(
            status="success", result={"foo": "bar"}, explanation="Async works!", confidence=1.0
        ))
        registry.resolve_capability.return_value = handler

        model_selector = MagicMock()
        orchestrator = Orchestrator(domain_registry=registry, model_selector=model_selector)

        intent = IntentOutput(
            domain="finance",
            capability="test_async",
            confidence=1.0,
            parameters={}
        )

        output = await orchestrator.process(intent)

        assert output.status == "success"
        assert output.explanation == "Async works!"
        handler.execute.assert_awaited_once()

    asyncio.run(_run())

if __name__ == "__main__":
    test_async_orchestration()
