import asyncio
import logging
from unittest.mock import MagicMock

from domains.finance.handler import FinanceDomainHandler
from shared.models import ExecutionIntent
from skills.gateway import SkillGateway

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_typed")

def test_missing_param_clarification():
    async def _run() -> None:
        gateway = MagicMock(spec=SkillGateway)
        registry = MagicMock()
        registry.get_metadata.return_value = {}

        handler = FinanceDomainHandler(skill_gateway=gateway, registry=registry)
        intent = ExecutionIntent(
            domain="finance",
            capability="get_top_gainers",
            confidence=1.0,
            parameters={"limit": 5}
        )

        output = await handler.execute(intent)
        assert output.status == "clarification"
        assert "validation_error" in output.metadata

    asyncio.run(_run())

def test_valid_execution():
    async def _run() -> None:
        gateway = MagicMock(spec=SkillGateway)
        gateway.execute.return_value = {"success": True, "data": [{"ticker": "AAPL", "price": 150}]}

        registry = MagicMock()
        registry.get_metadata.return_value = {}

        handler = FinanceDomainHandler(skill_gateway=gateway, registry=registry)
        intent = ExecutionIntent(
            domain="finance",
            capability="get_top_gainers",
            confidence=1.0,
            parameters={"market": "US", "period": "1d", "limit": 5}
        )

        output = await handler.execute(intent)
        assert output.status == "success"
        assert "items" in output.result
        assert output.result["count"] == 1

    asyncio.run(_run())

if __name__ == "__main__":
    test_missing_param_clarification()
    test_valid_execution()
