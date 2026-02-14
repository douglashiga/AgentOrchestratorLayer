import asyncio

from execution.engine import ExecutionEngine
from execution.task_state_store import TaskStateStore
from registry.domain_registry import HandlerRegistry
from shared.models import DomainOutput, IntentOutput
from shared.workflow_contracts import ClarificationAnswer, MethodSpec


class _DummyOrchestrator:
    def __init__(self) -> None:
        self.domain_registry = HandlerRegistry()
        self.calls: list[IntentOutput] = []

    async def process(self, intent: IntentOutput) -> DomainOutput:
        self.calls.append(intent)
        symbol = str(intent.parameters.get("symbol", "")).strip()
        return DomainOutput(
            status="success",
            result={"symbol": symbol, "price": 99.5, "currency": "USD"},
            explanation=f"{symbol} at 99.50 USD",
            confidence=1.0,
        )


def test_workflow_runtime_pause_and_resume(tmp_path) -> None:
    async def _run() -> None:
        store = TaskStateStore(db_path=str(tmp_path / "workflow_tasks.db"))
        orchestrator = _DummyOrchestrator()
        engine = ExecutionEngine(orchestrator=orchestrator, task_store=store)

        method_spec = MethodSpec(
            domain="finance",
            method="get_stock_price",
            workflow={
                "execution_mode": "dag",
                "nodes": [
                    {
                        "id": "ask_symbol",
                        "kind": "human_gate",
                        "params": {
                            "question": "Qual ticker você quer consultar?",
                            "options": ["VALE3.SA", "PETR4.SA"],
                            "answer_target": "symbol",
                        },
                    },
                    {
                        "id": "fetch_price",
                        "kind": "call",
                        "params": {
                            "domain": "finance",
                            "capability": "get_stock_price",
                            "params": {
                                "symbol": "${symbol}",
                            },
                        },
                        "output_key": "quote",
                    },
                    {
                        "id": "done",
                        "kind": "return",
                        "params": {
                            "from": "quote",
                            "explanation": "Cotação obtida.",
                        },
                    },
                ],
                "edges": [
                    {"from_node": "ask_symbol", "to_node": "fetch_price"},
                    {"from_node": "fetch_price", "to_node": "done"},
                ],
                "start_node": "ask_symbol",
                "terminal_nodes": ["done"],
            },
        )

        intent = IntentOutput(
            domain="finance",
            capability="get_stock_price",
            confidence=1.0,
            parameters={},
            original_query="pegue preço",
        )

        first = await engine.execute_method(intent=intent, method_spec=method_spec, task_id="task-1")
        assert first.status == "clarification"
        assert first.result["task_id"] == "task-1"

        question_id = str(first.result.get("question_id", ""))
        assert question_id

        paused_task = store.get_task("task-1")
        assert paused_task is not None
        assert paused_task.status == "waiting_user"
        assert paused_task.waiting_question is not None

        answer = ClarificationAnswer(
            question_id=question_id,
            task_id="task-1",
            selected_option="VALE3.SA",
        )
        second = await engine.resume_task(answer)

        assert second.status == "success"
        assert second.result["symbol"] == "VALE3.SA"
        assert second.result["price"] == 99.5
        assert orchestrator.calls
        assert orchestrator.calls[0].parameters["symbol"] == "VALE3.SA"

        completed_task = store.get_task("task-1")
        assert completed_task is not None
        assert completed_task.status == "completed"

        events = store.list_events("task-1")
        event_types = [evt.event_type for evt in events]
        assert "task_created" in event_types
        assert "clarification_required" in event_types
        assert "task_resumed" in event_types
        assert event_types[-1] == "task_completed"

        engine.close()

    asyncio.run(_run())
