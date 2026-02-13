import asyncio
from unittest.mock import AsyncMock, MagicMock

from execution.engine import ExecutionEngine
from shared.models import DomainOutput, ExecutionPlan, ExecutionStep, IntentOutput


def test_execution_engine_dag_combines_results():
    async def _run() -> None:
        orchestrator = MagicMock()

        async def process(intent: IntentOutput) -> DomainOutput:
            if intent.capability == "get_stock_price":
                return DomainOutput(
                    status="success",
                    result={"price": 123.45, "symbol": intent.parameters.get("symbol")},
                    explanation="AAPL is trading at 123.45 USD",
                    confidence=1.0,
                )
            return DomainOutput(
                status="success",
                result={"sent": True, "message": intent.parameters.get("message")},
                explanation="Telegram message sent",
                confidence=1.0,
            )

        orchestrator.process = AsyncMock(side_effect=process)
        engine = ExecutionEngine(orchestrator=orchestrator)

        plan = ExecutionPlan(
            execution_mode="dag",
            combine_mode="report",
            steps=[
                ExecutionStep(
                    id=1,
                    domain="finance",
                    capability="get_stock_price",
                    params={"symbol": "AAPL"},
                    depends_on=[],
                    output_key="analysis",
                ),
                ExecutionStep(
                    id=2,
                    domain="communication",
                    capability="send_telegram_message",
                    params={"message": "${1.explanation}", "chat_id": "123"},
                    depends_on=[1],
                    required=False,
                    output_key="notification",
                ),
            ],
        )

        original_intent = IntentOutput(
            domain="finance",
            capability="get_stock_price",
            confidence=1.0,
            parameters={"symbol": "AAPL"},
            original_query="pegue e envie no telegram",
        )

        output = await engine.execute_plan(plan, original_intent=original_intent)
        assert output.status == "success"
        assert "steps" in output.result
        assert "analysis" in output.result["steps"]
        assert "notification" in output.result["steps"]
        assert output.result["steps"]["notification"]["result"]["message"] == "AAPL is trading at 123.45 USD"

    asyncio.run(_run())
