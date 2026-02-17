import asyncio

from execution.engine import ExecutionEngine
from execution.task_state_store import TaskStateStore
from registry.domain_registry import HandlerRegistry
from shared.models import DomainOutput, ExecutionIntent, ExecutionPlan, ExecutionStep
from shared.workflow_contracts import ClarificationAnswer


class _CatalogOrchestrator:
    def __init__(self) -> None:
        self.domain_registry = HandlerRegistry()
        self.calls: list[ExecutionIntent] = []

    async def process(self, intent: ExecutionIntent) -> DomainOutput:
        self.calls.append(intent)
        if intent.capability == "market_get_quote":
            symbol = str(intent.parameters.get("symbol", "")).strip()
            return DomainOutput(
                status="success",
                result={"symbol": symbol, "price": 111.25, "currency": "USD"},
                explanation=f"{symbol} at 111.25 USD",
                confidence=1.0,
            )
        return DomainOutput(status="failure", result={}, explanation="unsupported", confidence=0.0)


def test_execute_plan_uses_catalog_method_spec_with_pause_resume(tmp_path) -> None:
    async def _run() -> None:
        orchestrator = _CatalogOrchestrator()

        # Capability metadata mirrors a remote catalog contract (workflow + policy)
        orchestrator.domain_registry.register_capability(
            "get_stock_price",
            handler=object(),
            metadata={
                "domain": "finance",
                "schema": {
                    "type": "object",
                    "properties": {"symbol": {"type": "string"}},
                    "required": ["symbol"],
                },
                "workflow": {
                    "execution_mode": "dag",
                    "nodes": [
                        {
                            "id": "ask_symbol",
                            "kind": "human_gate",
                            "params": {
                                "question": "Qual ticker?",
                                "answer_target": "symbol",
                                "options": ["VALE3.SA", "PETR4.SA"],
                            },
                        },
                        {
                            "id": "fetch",
                            "kind": "call",
                            "params": {
                                "domain": "finance",
                                "capability": "market_get_quote",
                                "params": {"symbol": "${symbol}"},
                            },
                            "output_key": "quote",
                        },
                        {
                            "id": "done",
                            "kind": "return",
                            "params": {"from": "quote", "explanation": "Cotação pronta."},
                        },
                    ],
                    "edges": [
                        {"from_node": "ask_symbol", "to_node": "fetch"},
                        {"from_node": "fetch", "to_node": "done"},
                    ],
                    "start_node": "ask_symbol",
                    "terminal_nodes": ["done"],
                },
                "policy": {
                    "default_retry": {"max_attempts": 1},
                    "human_validation": {"enabled": True},
                },
            },
        )

        store = TaskStateStore(db_path=str(tmp_path / "workflow_catalog.db"))
        engine = ExecutionEngine(orchestrator=orchestrator, task_store=store)

        plan = ExecutionPlan(
            execution_mode="sequential",
            combine_mode="last",
            steps=[
                ExecutionStep(
                    id=1,
                    domain="finance",
                    capability="get_stock_price",
                    params={},
                    depends_on=[],
                    required=True,
                    output_key="primary",
                )
            ],
        )
        original_intent = ExecutionIntent(
            domain="finance",
            capability="get_stock_price",
            confidence=1.0,
            parameters={},
            original_query="quero preço",
        )

        first = await engine.execute_plan(plan, original_intent)
        assert first.status == "clarification"
        task_id = str(first.metadata.get("task_id", "")).strip()
        question_id = str(first.metadata.get("question_id", "")).strip()
        assert task_id and question_id

        resumed = await engine.resume_task(
            ClarificationAnswer(
                task_id=task_id,
                question_id=question_id,
                selected_option="VALE3.SA",
            )
        )
        assert resumed.status == "success"
        assert resumed.result["symbol"] == "VALE3.SA"
        assert resumed.result["price"] == 111.25

        assert orchestrator.calls
        assert orchestrator.calls[-1].capability == "market_get_quote"
        assert orchestrator.calls[-1].parameters["symbol"] == "VALE3.SA"

        engine.close()

    asyncio.run(_run())
