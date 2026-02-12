import asyncio
import logging
from unittest.mock import MagicMock, AsyncMock

from orchestrator.orchestrator import Orchestrator
from shared.models import IntentOutput, DomainOutput

logging.basicConfig(level=logging.INFO)

async def test_async_orchestratration():
    print("\n--- Testing Async Orchestrator ---")
    
    # Mock Registry and Handler
    registry = MagicMock()
    handler = MagicMock()
    # Handler.execute MUST be an async mock (coroutine)
    handler.execute = AsyncMock(return_value=DomainOutput(
        status="success", result={"foo": "bar"}, explanation="Async works!", confidence=1.0
    ))
    
    registry.resolve_capability.return_value = handler
    
    # Mock ModelSelector
    model_selector = MagicMock()
    
    orchestrator = Orchestrator(domain_registry=registry, model_selector=model_selector)
    
    intent = IntentOutput(
        domain="finance",
        capability="test_async",
        confidence=0.99,
        parameters={}
    )
    
    print("Calling await orchestrator.process()...")
    output = await orchestrator.process(intent)
    
    print(f"Status: {output.status}")
    print(f"Explanation: {output.explanation}")
    
    if output.status == "success" and output.explanation == "Async works!":
        print("✅ SUCCESS: Orchestrator is async and awaited handler.")
    else:
        print("❌ FAILURE: Async chain failed.")

if __name__ == "__main__":
    asyncio.run(test_async_orchestratration())
