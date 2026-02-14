from pydantic import ValidationError

from shared.workflow_contracts import MethodSpec, WorkflowEvent


def test_method_spec_supports_human_gate_workflow() -> None:
    spec = MethodSpec(
        domain="finance",
        method="get_price",
        input_schema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
            },
            "required": ["symbol"],
        },
        workflow={
            "execution_mode": "dag",
            "nodes": [
                {"id": "normalize", "kind": "transform", "handler": "normalize_symbol"},
                {"id": "resolve", "kind": "resolve", "handler": "resolve_ticker"},
                {
                    "id": "gate",
                    "kind": "decision",
                    "when": "resolve.confidence < 0.85 or resolve.candidates_count != 1",
                },
                {"id": "ask_user", "kind": "human_gate", "handler": "ask_clarification"},
                {"id": "fetch", "kind": "call", "handler": "market_data.get_quote"},
                {"id": "return", "kind": "return", "output_key": "quote"},
            ],
            "edges": [
                {"from_node": "normalize", "to_node": "resolve"},
                {"from_node": "resolve", "to_node": "gate"},
                {"from_node": "gate", "to_node": "ask_user", "when": "gate == true"},
                {"from_node": "gate", "to_node": "fetch", "when": "gate == false"},
                {"from_node": "ask_user", "to_node": "fetch"},
                {"from_node": "fetch", "to_node": "return"},
            ],
            "start_node": "normalize",
            "terminal_nodes": ["return"],
        },
    )

    assert spec.method == "get_price"
    assert spec.workflow.execution_mode == "dag"
    assert spec.workflow.nodes[3].kind == "human_gate"


def test_method_spec_rejects_duplicate_node_ids() -> None:
    try:
        MethodSpec(
            domain="finance",
            method="get_price",
            workflow={
                "execution_mode": "sequential",
                "nodes": [
                    {"id": "step", "kind": "transform"},
                    {"id": "step", "kind": "call", "handler": "market_data.get_quote"},
                ],
            },
        )
        assert False, "expected duplicate id validation error"
    except ValidationError as exc:
        assert "unique" in str(exc)


def test_method_spec_rejects_unknown_edge_target() -> None:
    try:
        MethodSpec(
            domain="finance",
            method="get_price",
            workflow={
                "execution_mode": "dag",
                "nodes": [
                    {"id": "normalize", "kind": "transform"},
                    {"id": "fetch", "kind": "call", "handler": "market_data.get_quote"},
                ],
                "edges": [
                    {"from_node": "normalize", "to_node": "missing_node"},
                ],
            },
        )
        assert False, "expected edge validation error"
    except ValidationError as exc:
        assert "missing_node" in str(exc)


def test_method_spec_rejects_cycles_for_dag() -> None:
    try:
        MethodSpec(
            domain="finance",
            method="get_price",
            workflow={
                "execution_mode": "dag",
                "nodes": [
                    {"id": "a", "kind": "transform"},
                    {"id": "b", "kind": "call", "handler": "market_data.get_quote"},
                ],
                "edges": [
                    {"from_node": "a", "to_node": "b"},
                    {"from_node": "b", "to_node": "a"},
                ],
            },
        )
        assert False, "expected cycle validation error"
    except ValidationError as exc:
        assert "must not contain cycles" in str(exc)


def test_workflow_event_contract() -> None:
    event = WorkflowEvent(
        event_id="evt-1",
        task_id="task-1",
        domain="finance",
        method="get_price",
        event_type="task_created",
    )

    assert event.event_type == "task_created"
    assert event.timestamp.tzinfo is not None
