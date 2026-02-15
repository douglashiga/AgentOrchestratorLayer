import asyncio
from unittest.mock import AsyncMock, MagicMock

from orchestrator.orchestrator import Orchestrator
from shared.models import DomainOutput, IntentOutput


def test_orchestrator_test_mode_stops_before_domain_execution(monkeypatch):
    async def _run() -> None:
        monkeypatch.setenv("ORCHESTRATOR_TEST_MODE", "true")
        registry = MagicMock()
        handler = MagicMock()
        handler.execute = AsyncMock(
            return_value=DomainOutput(
                status="success",
                result={"ok": True},
                explanation="should not run",
                confidence=1.0,
            )
        )
        registry.resolve_capability.return_value = handler
        registry.get_metadata.return_value = {}
        model_selector = MagicMock()
        orchestrator = Orchestrator(domain_registry=registry, model_selector=model_selector)

        intent = IntentOutput(
            domain="finance",
            capability="get_stock_price",
            confidence=0.20,  # below threshold; test mode should bypass confidence gating
            parameters={"symbol": "VALE3.SA"},
            original_query="qual o valor da vale3",
        )

        out = await orchestrator.process(intent)
        assert out.status == "success"
        assert out.metadata.get("test_mode") is True
        assert out.metadata.get("test_mode_stage") == "domain_routed"
        assert out.result.get("route", {}).get("handler_class") is not None
        handler.execute.assert_not_called()

    asyncio.run(_run())
