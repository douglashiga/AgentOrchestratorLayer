import asyncio
import time

from execution.engine import ExecutionEngine
from execution.task_state_store import TaskStateStore
from registry.domain_registry import HandlerRegistry
from shared.models import DomainOutput, ExecutionIntent
from shared.workflow_contracts import MethodSpec


class _AdvancedOrchestrator:
    def __init__(self) -> None:
        self.domain_registry = HandlerRegistry()
        self.calls: list[ExecutionIntent] = []
        self._transient_attempts = 0

    async def process(self, intent: ExecutionIntent) -> DomainOutput:
        self.calls.append(intent)
        cap = intent.capability
        if cap == "transient_quote":
            self._transient_attempts += 1
            if self._transient_attempts == 1:
                return DomainOutput(
                    status="failure",
                    result={},
                    explanation="transient_network_error",
                    confidence=0.0,
                    metadata={"error": "transient_network_error"},
                )
            symbol = str(intent.parameters.get("symbol", "")).strip()
            return DomainOutput(
                status="success",
                result={"symbol": symbol, "price": 88.1},
                explanation="ok",
                confidence=1.0,
            )

        if cap in {"quote_a", "quote_b"}:
            await asyncio.sleep(0.25)
            value = 10.0 if cap == "quote_a" else 20.0
            return DomainOutput(
                status="success",
                result={"value": value},
                explanation=cap,
                confidence=1.0,
            )

        return DomainOutput(status="failure", result={}, explanation="unsupported", confidence=0.0)


def test_retry_and_idempotency_are_applied(tmp_path) -> None:
    async def _run() -> None:
        orchestrator = _AdvancedOrchestrator()
        store = TaskStateStore(db_path=str(tmp_path / "retry_idem.db"))
        engine = ExecutionEngine(orchestrator=orchestrator, task_store=store)

        spec = MethodSpec(
            domain="finance",
            method="get_stock_price",
            policy={
                "idempotency_key_fields": ["symbol"],
                "default_retry": {
                    "max_attempts": 2,
                    "backoff_seconds": 0.0,
                    "retry_on": ["transient_network_error"],
                },
            },
            workflow={
                "execution_mode": "dag",
                "nodes": [
                    {
                        "id": "fetch",
                        "kind": "call",
                        "params": {
                            "domain": "finance",
                            "capability": "transient_quote",
                            "params": {"symbol": "${symbol}"},
                        },
                        "output_key": "quote",
                    },
                    {
                        "id": "done",
                        "kind": "return",
                        "params": {"from": "quote"},
                    },
                ],
                "edges": [{"from_node": "fetch", "to_node": "done"}],
                "start_node": "fetch",
                "terminal_nodes": ["done"],
            },
        )

        intent = ExecutionIntent(
            domain="finance",
            capability="get_stock_price",
            confidence=1.0,
            parameters={"symbol": "AAPL"},
            original_query="AAPL",
        )

        first = await engine.execute_method(intent=intent, method_spec=spec, session_id="s1", user_id="u1")
        assert first.status == "success"
        assert first.result["symbol"] == "AAPL"

        # 1 failure + 1 retry success
        transient_calls = [c for c in orchestrator.calls if c.capability == "transient_quote"]
        assert len(transient_calls) == 2

        second = await engine.execute_method(intent=intent, method_spec=spec, session_id="s1", user_id="u1")
        assert second.status == "success"
        assert second.metadata.get("cached") is True

        # Should not call remote capability again due idempotency cache.
        transient_calls_after = [c for c in orchestrator.calls if c.capability == "transient_quote"]
        assert len(transient_calls_after) == 2

        engine.close()

    asyncio.run(_run())


def test_workflow_dag_runs_parallel_batch(tmp_path) -> None:
    async def _run() -> None:
        orchestrator = _AdvancedOrchestrator()
        store = TaskStateStore(db_path=str(tmp_path / "parallel_dag.db"))
        engine = ExecutionEngine(orchestrator=orchestrator, task_store=store)

        spec = MethodSpec(
            domain="finance",
            method="parallel_quotes",
            workflow={
                "execution_mode": "dag",
                "max_concurrency": 4,
                "nodes": [
                    {
                        "id": "a",
                        "kind": "call",
                        "params": {"domain": "finance", "capability": "quote_a", "params": {}},
                        "output_key": "qa",
                    },
                    {
                        "id": "b",
                        "kind": "call",
                        "params": {"domain": "finance", "capability": "quote_b", "params": {}},
                        "output_key": "qb",
                    },
                    {
                        "id": "done",
                        "kind": "return",
                        "params": {
                            "result": {
                                "a": "${qa.value}",
                                "b": "${qb.value}",
                            }
                        },
                    },
                ],
                "edges": [
                    {"from_node": "a", "to_node": "done"},
                    {"from_node": "b", "to_node": "done"},
                ],
                "start_node": "a",
                "terminal_nodes": ["done"],
            },
        )

        intent = ExecutionIntent(
            domain="finance",
            capability="parallel_quotes",
            confidence=1.0,
            parameters={},
            original_query="parallel",
        )

        start = time.perf_counter()
        out = await engine.execute_method(intent=intent, method_spec=spec, task_id="parallel-1")
        elapsed = time.perf_counter() - start

        assert out.status == "success"
        assert out.result["a"] in {"10.0", 10.0}
        assert out.result["b"] in {"20.0", 20.0}
        assert elapsed < 0.45

        engine.close()

    asyncio.run(_run())
