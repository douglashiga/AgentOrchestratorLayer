import asyncio
import logging
from unittest.mock import MagicMock

from domains.finance.handler import FinanceDomainHandler
from shared.models import IntentOutput
from skills.gateway import SkillGateway

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_typed")

async def test_missing_param_clarification():
    print("\n--- Testing Missing Param (Market) in get_top_gainers ---")
    
    # Mock Gateway & Registry
    gateway = MagicMock(spec=SkillGateway)
    registry = MagicMock()
    registry.get_metadata.return_value = {}  # No overrides needed for this test checking Pydantic

    handler = FinanceDomainHandler(skill_gateway=gateway, registry=registry)

    # Intent missing 'market' (Required by TopGainersInput)
    intent = IntentOutput(
        domain="finance",
        capability="get_top_gainers",
        confidence=1.0,
        parameters={"limit": 5} # Missing 'market' and 'period'
    )

    output = await handler.execute(intent)

    print(f"Status: {output.status}")
    print(f"Explanation: {output.explanation}")
    
    if output.status == "clarification" and "validation_error" in output.metadata:
        print("✅ SUCCESS: Caught Pydantic ValidationError and asked for clarification.")
    else:
        print("❌ FAILURE: Did not trigger clarification correctly.")
        print(output)

async def test_valid_execution():
    print("\n--- Testing Valid Execution for get_top_gainers ---")
    
    gateway = MagicMock(spec=SkillGateway)
    gateway.execute.return_value = {"success": True, "data": [{"ticker": "AAPL", "price": 150}]}
    
    registry = MagicMock()
    registry.get_metadata.return_value = {}

    handler = FinanceDomainHandler(skill_gateway=gateway, registry=registry)

    intent = IntentOutput(
        domain="finance",
        capability="get_top_gainers",
        confidence=1.0,
        parameters={"market": "US", "period": "1d", "limit": 5}
    )

    output = await handler.execute(intent)

    print(f"Status: {output.status}")
    print(f"Result: {output.result}")
    
    if output.status == "success":
        print("✅ SUCCESS: Valid parameters executed correctly.")
    else:
        print("❌ FAILURE: Valid execution failed.")

if __name__ == "__main__":
    asyncio.run(test_missing_param_clarification())
    asyncio.run(test_valid_execution())
