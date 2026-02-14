"""Generic workflow contracts for dynamic method execution.

These models are intentionally domain-agnostic and can represent
any capability workflow (finance, communication, CRM, etc.) without
hardcoded logic in the engine.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


ConditionType = bool | str | dict[str, Any]

NodeKind = Literal[
    "transform",
    "validate",
    "resolve",
    "decision",
    "human_gate",
    "call",
    "aggregate",
    "return",
]

ExecutionMode = Literal["sequential", "parallel", "dag"]
CombineMode = Literal["last", "report", "merge", "aggregate"]
TaskStatus = Literal["pending", "running", "waiting_user", "paused", "completed", "failed", "cancelled"]
EventType = Literal[
    "task_created",
    "task_started",
    "node_started",
    "node_completed",
    "node_failed",
    "clarification_required",
    "task_paused",
    "task_resumed",
    "task_completed",
    "task_failed",
]


class RetryPolicy(BaseModel):
    """Retry behavior reusable across nodes and methods."""

    model_config = {"frozen": True}

    max_attempts: int = Field(default=1, ge=1)
    backoff_seconds: float = Field(default=0.0, ge=0.0)
    jitter_seconds: float = Field(default=0.0, ge=0.0)
    retry_on: list[str] = Field(default_factory=list)


class HumanValidationPolicy(BaseModel):
    """Generic policy to decide when human confirmation is required."""

    model_config = {"frozen": True}

    enabled: bool = Field(default=True)
    auto_continue_confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    require_unique_candidate: bool = Field(default=True)
    max_candidates_without_confirmation: int = Field(default=1, ge=1)
    trigger_on_missing_required: bool = Field(default=True)
    trigger_on_invalid_input: bool = Field(default=True)
    trigger_on_ambiguity: bool = Field(default=True)
    max_user_turns: int = Field(default=3, ge=1)


class MethodPolicy(BaseModel):
    """Execution policy for a method/capability contract."""

    model_config = {"frozen": True}

    default_timeout_seconds: float | None = Field(default=None, gt=0.0)
    default_retry: RetryPolicy = Field(default_factory=RetryPolicy)
    human_validation: HumanValidationPolicy = Field(default_factory=HumanValidationPolicy)
    idempotency_key_fields: list[str] = Field(default_factory=list)


class WorkflowNodeSpec(BaseModel):
    """Reusable workflow node contract."""

    model_config = {"frozen": True}

    id: str
    kind: NodeKind
    handler: str | None = Field(default=None, description="Registry key of the node handler")
    params: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    retry: RetryPolicy | None = Field(default=None)
    required: bool = Field(default=True)
    output_key: str | None = Field(default=None)
    when: ConditionType | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowEdgeSpec(BaseModel):
    """Directed edge in a workflow graph."""

    model_config = {"frozen": True}

    from_node: str
    to_node: str
    when: ConditionType | None = Field(default=None)


class WorkflowSpec(BaseModel):
    """Declarative workflow graph executed by the runtime engine."""

    model_config = {"frozen": True}

    execution_mode: ExecutionMode = Field(default="dag")
    combine_mode: CombineMode = Field(default="report")
    max_concurrency: int = Field(default=4, ge=1)
    nodes: list[WorkflowNodeSpec]
    edges: list[WorkflowEdgeSpec] = Field(default_factory=list)
    start_node: str | None = Field(default=None)
    terminal_nodes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_graph(self) -> "WorkflowSpec":
        if not self.nodes:
            raise ValueError("WorkflowSpec.nodes must not be empty")

        node_ids = [node.id.strip() for node in self.nodes]
        if any(not node_id for node_id in node_ids):
            raise ValueError("Workflow nodes must use non-empty ids")

        if len(set(node_ids)) != len(node_ids):
            raise ValueError("Workflow node ids must be unique")

        node_id_set = set(node_ids)

        if self.start_node and self.start_node not in node_id_set:
            raise ValueError("Workflow start_node must reference an existing node")

        if self.terminal_nodes:
            unknown_terminals = [node for node in self.terminal_nodes if node not in node_id_set]
            if unknown_terminals:
                raise ValueError(f"Workflow terminal_nodes reference unknown ids: {unknown_terminals}")

        for edge in self.edges:
            if edge.from_node not in node_id_set:
                raise ValueError(f"Edge from_node '{edge.from_node}' not found in nodes")
            if edge.to_node not in node_id_set:
                raise ValueError(f"Edge to_node '{edge.to_node}' not found in nodes")

        if self.execution_mode == "dag" and len(self.nodes) > 1 and not self.edges:
            raise ValueError("DAG workflow with multiple nodes must declare edges")

        if self.execution_mode == "dag" and self._has_cycle(node_ids=node_ids):
            raise ValueError("DAG workflow must not contain cycles")

        return self

    def _has_cycle(self, node_ids: list[str]) -> bool:
        if not self.edges:
            return False

        adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        for edge in self.edges:
            adjacency.setdefault(edge.from_node, []).append(edge.to_node)

        visiting: set[str] = set()
        visited: set[str] = set()

        def dfs(node_id: str) -> bool:
            if node_id in visiting:
                return True
            if node_id in visited:
                return False

            visiting.add(node_id)
            for neighbor in adjacency.get(node_id, []):
                if dfs(neighbor):
                    return True
            visiting.remove(node_id)
            visited.add(node_id)
            return False

        for node in node_ids:
            if dfs(node):
                return True
        return False


class MethodSpec(BaseModel):
    """Full method/capability contract registered in the engine catalog."""

    model_config = {"frozen": True}

    domain: str
    method: str
    version: str = Field(default="1.0.0")
    description: str = Field(default="")
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    workflow: WorkflowSpec
    policy: MethodPolicy = Field(default_factory=MethodPolicy)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClarificationQuestion(BaseModel):
    """Question emitted when the workflow requires user confirmation."""

    model_config = {"frozen": True}

    question_id: str
    task_id: str
    node_id: str
    prompt: str
    options: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)


class ClarificationAnswer(BaseModel):
    """User answer used to resume a paused workflow."""

    model_config = {"frozen": True}

    question_id: str
    task_id: str
    selected_option: str | None = Field(default=None)
    free_text: str | None = Field(default=None)
    confirmed: bool = Field(default=True)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskInstance(BaseModel):
    """Runtime task snapshot persisted by the orchestration engine."""

    model_config = {"frozen": True}

    task_id: str
    domain: str
    method: str
    status: TaskStatus = Field(default="pending")
    session_id: str | None = Field(default=None)
    user_id: str | None = Field(default=None)
    correlation_id: str | None = Field(default=None)
    input_payload: dict[str, Any] = Field(default_factory=dict)
    normalized_payload: dict[str, Any] = Field(default_factory=dict)
    resolved_payload: dict[str, Any] = Field(default_factory=dict)
    output_payload: dict[str, Any] = Field(default_factory=dict)
    current_node_id: str | None = Field(default=None)
    waiting_question: ClarificationQuestion | None = Field(default=None)
    last_error: str | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WorkflowEvent(BaseModel):
    """Canonical event envelope for observability and pause/resume flows."""

    model_config = {"frozen": True}

    event_id: str
    task_id: str
    domain: str
    method: str
    event_type: EventType
    status: TaskStatus | None = Field(default=None)
    node_id: str | None = Field(default=None)
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
