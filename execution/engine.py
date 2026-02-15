"""Execution Engine.

Supports two execution modes:
- Legacy `ExecutionPlan` + `ExecutionStep` execution.
- Declarative `MethodSpec` workflow runtime with pause/resume and events.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import random
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from execution.result_combiner import ResultCombiner
from execution.task_state_store import TaskStateStore
from orchestrator.orchestrator import Orchestrator
from shared.models import DomainOutput, ExecutionPlan, ExecutionStep, IntentOutput
from shared.safe_eval import safe_eval_bool
from shared.workflow_contracts import (
    ClarificationAnswer,
    ClarificationQuestion,
    MethodSpec,
    TaskInstance,
    WorkflowEvent,
    WorkflowNodeSpec,
)

logger = logging.getLogger(__name__)

WorkflowHandler = Callable[..., Any]
ProgressCallback = Callable[[dict[str, Any]], Any]


@dataclass
class NodeExecutionOutcome:
    status: str = "success"  # success|failure|clarification
    result: Any = field(default_factory=dict)
    explanation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ExecutionEngine:
    """Engine to execute plans using the Orchestrator and declarative workflows."""

    def __init__(self, orchestrator: Orchestrator, task_store: TaskStateStore | None = None):
        self.orchestrator = orchestrator
        self.result_combiner = ResultCombiner()
        self.task_store = task_store or TaskStateStore()
        self.workflow_handlers: dict[str, WorkflowHandler] = {}

    def register_workflow_handler(self, handler_name: str, handler: WorkflowHandler) -> None:
        """Register a custom runtime handler for workflow nodes."""
        key = str(handler_name).strip()
        if not key:
            raise ValueError("handler_name must not be empty")
        self.workflow_handlers[key] = handler

    def close(self) -> None:
        """Release engine resources."""
        if self.task_store and hasattr(self.task_store, "close"):
            try:
                self.task_store.close()
            except Exception as exc:
                logger.warning("Failed to close TaskStateStore: %s", exc)

    async def execute_plan(
        self,
        plan: ExecutionPlan,
        original_intent: IntentOutput,
        progress_callback: ProgressCallback | None = None,
    ) -> DomainOutput:
        """Execute the plan step-by-step."""
        logger.info("Execution Engine started: %d steps", len(plan.steps))
        steps_by_id = {step.id: step for step in plan.steps}
        pending: dict[int, ExecutionStep] = dict(steps_by_id)
        outputs: dict[int, DomainOutput] = {}

        if not pending:
            return DomainOutput(
                status="failure",
                result={},
                explanation="Plan executed but produced no output.",
                confidence=0.0,
            )

        while pending:
            ready_steps = [
                step for step in pending.values() if all(dep_id in outputs for dep_id in step.depends_on)
            ]

            if not ready_steps:
                unresolved = sorted(pending.keys())
                return DomainOutput(
                    status="failure",
                    result={},
                    explanation="Execution plan has unresolved/cyclic dependencies.",
                    confidence=0.0,
                    metadata={"error": "cyclic_dependencies", "pending_steps": unresolved},
                )

            batch = self._select_batch(plan, ready_steps)
            logger.info("Executing batch: %s", [s.id for s in batch])

            results = await asyncio.gather(
                *(
                    self._execute_step(
                        step,
                        outputs,
                        original_intent,
                        progress_callback=progress_callback,
                    )
                    for step in batch
                )
            )

            must_stop = False
            for step_id, output in results:
                step = steps_by_id[step_id]
                outputs[step_id] = output
                pending.pop(step_id, None)

                if output.status in ("failure", "clarification") and step.required:
                    logger.info(
                        "Required step %d stopped execution: status=%s",
                        step.id,
                        output.status,
                    )
                    must_stop = True

            if must_stop:
                break

        return self.result_combiner.combine(plan=plan, steps_by_id=steps_by_id, outputs=outputs)

    async def execute_method(
        self,
        *,
        intent: IntentOutput,
        method_spec: MethodSpec,
        task_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        correlation_id: str | None = None,
    ) -> DomainOutput:
        """Execute a declarative MethodSpec workflow and persist runtime state/events."""
        input_payload = dict(intent.parameters or {})
        idempotency_key = self._build_idempotency_key(
            method_spec=method_spec,
            payload=input_payload,
            session_id=session_id,
            user_id=user_id,
        )

        if idempotency_key:
            existing = self.task_store.find_latest_task_by_idempotency_key(idempotency_key)
            if existing is not None and (task_id is None or existing.task_id != task_id):
                replay = self._domain_output_from_existing_task(existing_task=existing, method_spec=method_spec)
                if replay is not None:
                    return replay

        metadata: dict[str, Any] = {"method_spec": method_spec.model_dump(mode="json")}
        if idempotency_key:
            metadata["idempotency_key"] = idempotency_key

        task = TaskInstance(
            task_id=task_id or f"task-{uuid.uuid4().hex[:12]}",
            domain=method_spec.domain or intent.domain,
            method=method_spec.method,
            status="pending",
            session_id=session_id,
            user_id=user_id,
            correlation_id=correlation_id,
            input_payload=input_payload,
            normalized_payload=input_payload,
            resolved_payload=input_payload,
            metadata=metadata,
        )
        self.task_store.save_task(task)
        self._emit_event(task=task, event_type="task_created", status="pending")
        return await self._run_method_workflow(task=task, method_spec=method_spec, runtime_state=None, is_resume=False)

    async def resume_task(self, answer: ClarificationAnswer) -> DomainOutput:
        """Resume a paused workflow task after user clarification."""
        task = self.task_store.get_task(answer.task_id)
        if task is None:
            return DomainOutput(
                status="failure",
                result={},
                explanation=f"Task '{answer.task_id}' not found.",
                confidence=0.0,
                metadata={"error": "task_not_found", "task_id": answer.task_id},
            )

        if task.status != "waiting_user" or task.waiting_question is None:
            return DomainOutput(
                status="failure",
                result={},
                explanation=f"Task '{answer.task_id}' is not waiting for clarification.",
                confidence=0.0,
                metadata={"error": "task_not_waiting_user", "task_status": task.status},
            )

        if answer.question_id != task.waiting_question.question_id:
            return DomainOutput(
                status="failure",
                result={},
                explanation="Clarification answer does not match pending question.",
                confidence=0.0,
                metadata={
                    "error": "question_mismatch",
                    "expected_question_id": task.waiting_question.question_id,
                    "received_question_id": answer.question_id,
                },
            )

        method_spec_payload = task.metadata.get("method_spec") if isinstance(task.metadata, dict) else None
        runtime_state = task.metadata.get("runtime_state") if isinstance(task.metadata, dict) else None
        if not isinstance(method_spec_payload, dict) or not isinstance(runtime_state, dict):
            return DomainOutput(
                status="failure",
                result={},
                explanation="Task cannot be resumed because runtime metadata is missing.",
                confidence=0.0,
                metadata={"error": "missing_runtime_state", "task_id": task.task_id},
            )

        try:
            method_spec = MethodSpec(**method_spec_payload)
        except Exception as exc:
            return DomainOutput(
                status="failure",
                result={},
                explanation=f"Task cannot be resumed due to invalid method contract: {exc}",
                confidence=0.0,
                metadata={"error": "invalid_method_spec", "task_id": task.task_id},
            )

        payload = runtime_state.get("payload")
        if not isinstance(payload, dict):
            payload = {}
            runtime_state["payload"] = payload

        answer_target = str(task.waiting_question.context.get("answer_target", "clarification_answer")).strip() or "clarification_answer"
        answer_value = answer.selected_option if answer.selected_option is not None else answer.free_text
        payload[answer_target] = answer_value
        payload["clarification_confirmed"] = bool(answer.confirmed)

        answers = runtime_state.get("answers")
        if not isinstance(answers, list):
            answers = []
        answers.append(answer.model_dump(mode="json"))
        runtime_state["answers"] = answers

        completed_nodes = runtime_state.get("completed_nodes")
        if not isinstance(completed_nodes, list):
            completed_nodes = []
            runtime_state["completed_nodes"] = completed_nodes

        node_results = runtime_state.get("node_results")
        if not isinstance(node_results, dict):
            node_results = {}
            runtime_state["node_results"] = node_results

        pending_nodes = runtime_state.get("pending_nodes")
        if not isinstance(pending_nodes, list):
            pending_nodes = []
            runtime_state["pending_nodes"] = pending_nodes

        paused_node_id = str(runtime_state.get("paused_node_id", "")).strip()
        if paused_node_id:
            if paused_node_id not in completed_nodes:
                completed_nodes.append(paused_node_id)
            node_results[paused_node_id] = {
                "status": "success",
                "explanation": "User clarification received.",
                "result": {"confirmed": bool(answer.confirmed), "value": answer_value},
                "metadata": {"answer_target": answer_target},
            }
            if paused_node_id in pending_nodes:
                pending_nodes.remove(paused_node_id)
            runtime_state["paused_node_id"] = ""

        metadata = dict(task.metadata)
        metadata["runtime_state"] = self._json_safe(runtime_state)
        updated_task = task.model_copy(
            update={
                "status": "running",
                "waiting_question": None,
                "normalized_payload": payload,
                "resolved_payload": payload,
                "metadata": metadata,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self.task_store.save_task(updated_task)
        self._emit_event(task=updated_task, event_type="task_resumed", status="running")

        return await self._run_method_workflow(
            task=updated_task,
            method_spec=method_spec,
            runtime_state=runtime_state,
            is_resume=True,
        )

    def _select_batch(self, plan: ExecutionPlan, ready_steps: list[ExecutionStep]) -> list[ExecutionStep]:
        """Select next batch based on execution strategy."""
        if plan.execution_mode in ("parallel", "dag"):
            return ready_steps[: max(1, plan.max_concurrency)]
        return [ready_steps[0]]

    async def _execute_step(
        self,
        step: ExecutionStep,
        previous_outputs: dict[int, DomainOutput],
        original_intent: IntentOutput,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[int, DomainOutput]:
        """Execute a single step with parameter interpolation."""
        logger.info("Executing Step %d: %s (%s)", step.id, step.capability, step.domain or original_intent.domain)
        try:
            resolved_params = self._resolve_placeholders(step.params, previous_outputs)
            await self._emit_progress(
                progress_callback=progress_callback,
                payload={
                    "type": "step_started",
                    "step_id": step.id,
                    "domain": step.domain or original_intent.domain,
                    "capability": step.capability,
                    "params": self._json_safe(resolved_params),
                    "depends_on": list(step.depends_on),
                },
            )
            step_intent = IntentOutput(
                domain=step.domain or original_intent.domain,
                capability=step.capability,
                confidence=original_intent.confidence,
                parameters=resolved_params,
                original_query=original_intent.original_query,
            )

            method_spec = self._resolve_method_spec(step.capability)
            if method_spec is not None:
                output = await self.execute_method(
                    intent=step_intent,
                    method_spec=method_spec,
                    task_id=f"step-{step.id}-{uuid.uuid4().hex[:8]}",
                )
                await self._emit_progress(
                    progress_callback=progress_callback,
                    payload={
                        "type": "step_completed",
                        "step_id": step.id,
                        "domain": step.domain or original_intent.domain,
                        "capability": step.capability,
                        "status": output.status,
                        "result": self._json_safe(output.result),
                        "explanation": str(output.explanation or ""),
                    },
                )
                return step.id, output

            if step.timeout_seconds and step.timeout_seconds > 0:
                output = await asyncio.wait_for(
                    self.orchestrator.process(step_intent),
                    timeout=step.timeout_seconds,
                )
            else:
                output = await self.orchestrator.process(step_intent)
            await self._emit_progress(
                progress_callback=progress_callback,
                payload={
                    "type": "step_completed",
                    "step_id": step.id,
                    "domain": step.domain or original_intent.domain,
                    "capability": step.capability,
                    "status": output.status,
                    "result": self._json_safe(output.result),
                    "explanation": str(output.explanation or ""),
                },
            )
            return step.id, output
        except Exception as e:
            logger.critical("Engine crash on Step %d: %s", step.id, e)
            await self._emit_progress(
                progress_callback=progress_callback,
                payload={
                    "type": "step_completed",
                    "step_id": step.id,
                    "domain": step.domain or original_intent.domain,
                    "capability": step.capability,
                    "status": "failure",
                    "result": {},
                    "explanation": f"Execution Engine crashed at step {step.id}: {e}",
                },
            )
            return step.id, DomainOutput(
                status="failure",
                result={},
                explanation=f"Execution Engine crashed at step {step.id}: {e}",
                confidence=0.0,
                metadata={"error": str(e)},
            )

    async def _emit_progress(
        self,
        *,
        progress_callback: ProgressCallback | None,
        payload: dict[str, Any],
    ) -> None:
        if progress_callback is None:
            return
        try:
            maybe_result = progress_callback(self._json_safe(payload))
            if inspect.isawaitable(maybe_result):
                await maybe_result
        except Exception as exc:
            logger.warning("Failed to emit execution progress callback: %s", exc)

    async def _run_method_workflow(
        self,
        *,
        task: TaskInstance,
        method_spec: MethodSpec,
        runtime_state: dict[str, Any] | None,
        is_resume: bool,
    ) -> DomainOutput:
        workflow = method_spec.workflow
        node_ids = [node.id for node in workflow.nodes]
        node_by_id = {node.id: node for node in workflow.nodes}
        node_order = {node_id: idx for idx, node_id in enumerate(node_ids)}
        incoming_by_target, _outgoing_by_source = self._workflow_topology(method_spec)

        if runtime_state is None:
            payload = dict(task.resolved_payload or task.normalized_payload or task.input_payload)
            runtime_state = {
                "payload": payload,
                "node_results": {},
                "completed_nodes": [],
                "failed_nodes": [],
                "skipped_nodes": [],
                "pending_nodes": list(node_ids),
                "paused_node_id": "",
                "answers": [],
            }
        else:
            payload = runtime_state.get("payload")
            if not isinstance(payload, dict):
                payload = {}
            runtime_state["payload"] = payload

            node_results = runtime_state.get("node_results")
            if not isinstance(node_results, dict):
                node_results = {}
            runtime_state["node_results"] = node_results

            completed_nodes = runtime_state.get("completed_nodes")
            if not isinstance(completed_nodes, list):
                completed_nodes = []
            runtime_state["completed_nodes"] = completed_nodes

            failed_nodes = runtime_state.get("failed_nodes")
            if not isinstance(failed_nodes, list):
                failed_nodes = []
            runtime_state["failed_nodes"] = failed_nodes

            skipped_nodes = runtime_state.get("skipped_nodes")
            if not isinstance(skipped_nodes, list):
                skipped_nodes = []
            runtime_state["skipped_nodes"] = skipped_nodes

            pending_nodes = runtime_state.get("pending_nodes")
            if not isinstance(pending_nodes, list):
                pending_nodes = []
            if not pending_nodes:
                legacy_queue = runtime_state.get("queue", [])
                if isinstance(legacy_queue, list) and legacy_queue:
                    pending_nodes = [str(item) for item in legacy_queue if str(item).strip()]
            if not pending_nodes:
                done = set(completed_nodes + failed_nodes + skipped_nodes)
                pending_nodes = [node_id for node_id in node_ids if node_id not in done]
            runtime_state["pending_nodes"] = pending_nodes

        payload = runtime_state["payload"]
        node_results = runtime_state["node_results"]
        completed_nodes = runtime_state["completed_nodes"]
        failed_nodes = runtime_state["failed_nodes"]
        skipped_nodes = runtime_state["skipped_nodes"]
        pending_nodes = runtime_state["pending_nodes"]

        task_running = task.model_copy(
            update={
                "status": "running",
                "normalized_payload": payload,
                "resolved_payload": payload,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self.task_store.save_task(task_running)
        if not is_resume:
            self._emit_event(task=task_running, event_type="task_started", status="running")

        while pending_nodes:
            ready_nodes = self._ready_workflow_nodes(
                method_spec=method_spec,
                pending_nodes=pending_nodes,
                completed_nodes=completed_nodes,
                failed_nodes=failed_nodes,
                skipped_nodes=skipped_nodes,
                incoming_by_target=incoming_by_target,
                payload=payload,
                node_results=node_results,
                task=task_running,
                node_order=node_order,
            )

            if not ready_nodes:
                skipped_now = self._skip_unreachable_nodes(
                    pending_nodes=pending_nodes,
                    completed_nodes=completed_nodes,
                    failed_nodes=failed_nodes,
                    skipped_nodes=skipped_nodes,
                    incoming_by_target=incoming_by_target,
                    payload=payload,
                    node_results=node_results,
                    task=task_running,
                )
                if skipped_now:
                    continue

                metadata = dict(task_running.metadata)
                metadata["runtime_state"] = self._json_safe(runtime_state)
                failed_task = task_running.model_copy(
                    update={
                        "status": "failed",
                        "last_error": "Workflow has unresolved dependencies.",
                        "metadata": metadata,
                        "updated_at": datetime.now(timezone.utc),
                    }
                )
                self.task_store.save_task(failed_task)
                self._emit_event(task=failed_task, event_type="task_failed", status="failed")
                return DomainOutput(
                    status="failure",
                    result={"task_id": failed_task.task_id},
                    explanation="Workflow has unresolved dependencies.",
                    confidence=0.0,
                    metadata={"task_id": failed_task.task_id, "method": failed_task.method},
                )

            batch = self._select_workflow_nodes_batch(
                method_spec=method_spec,
                ready_nodes=ready_nodes,
            )

            for node_id in batch:
                self._emit_event(task=task_running, event_type="node_started", status="running", node_id=node_id)

            outcomes = await asyncio.gather(
                *(
                    self._execute_workflow_node_with_retry(
                        task=task_running,
                        node=node_by_id[node_id],
                        method_spec=method_spec,
                        payload=payload,
                        node_results=node_results,
                    )
                    for node_id in batch
                )
            )

            pending_nodes[:] = [node_id for node_id in pending_nodes if node_id not in set(batch)]
            runtime_state["pending_nodes"] = pending_nodes

            ordered_batch = sorted(
                zip(batch, outcomes),
                key=lambda item: node_order.get(item[0], 10**9),
            )

            for node_id, outcome in ordered_batch:
                node = node_by_id[node_id]
                task_running = task_running.model_copy(
                    update={
                        "current_node_id": node_id,
                        "updated_at": datetime.now(timezone.utc),
                    }
                )

                node_results[node_id] = {
                    "status": outcome.status,
                    "explanation": outcome.explanation,
                    "result": self._json_safe(outcome.result),
                    "metadata": self._json_safe(outcome.metadata),
                }

                if outcome.status == "clarification":
                    question = ClarificationQuestion(
                        question_id=f"q-{uuid.uuid4().hex[:10]}",
                        task_id=task_running.task_id,
                        node_id=node_id,
                        prompt=outcome.explanation or f"Confirm data for method '{task_running.method}'.",
                        options=[str(opt) for opt in outcome.metadata.get("options", [])]
                        if isinstance(outcome.metadata.get("options"), list)
                        else [],
                        context={
                            "answer_target": str(outcome.metadata.get("answer_target", "clarification_answer")),
                            "node_kind": node.kind,
                        },
                    )

                    runtime_state["paused_node_id"] = node_id
                    metadata = dict(task_running.metadata)
                    metadata["runtime_state"] = self._json_safe(runtime_state)
                    metadata["method_spec"] = method_spec.model_dump(mode="json")

                    paused_task = task_running.model_copy(
                        update={
                            "status": "waiting_user",
                            "waiting_question": question,
                            "metadata": metadata,
                            "updated_at": datetime.now(timezone.utc),
                        }
                    )
                    self.task_store.save_task(paused_task)
                    self._emit_event(
                        task=paused_task,
                        event_type="clarification_required",
                        status="waiting_user",
                        node_id=node_id,
                        payload={
                            "question_id": question.question_id,
                            "prompt": question.prompt,
                            "options": question.options,
                        },
                    )
                    self._emit_event(task=paused_task, event_type="task_paused", status="waiting_user", node_id=node_id)

                    return DomainOutput(
                        status="clarification",
                        result={
                            "task_id": paused_task.task_id,
                            "question_id": question.question_id,
                            "options": question.options,
                            "node_id": node_id,
                        },
                        explanation=question.prompt,
                        confidence=1.0,
                        metadata={
                            "task_id": paused_task.task_id,
                            "question_id": question.question_id,
                            "node_id": node_id,
                            "status": "waiting_user",
                        },
                    )

                if outcome.status == "failure":
                    if node_id not in failed_nodes:
                        failed_nodes.append(node_id)
                    self._emit_event(
                        task=task_running,
                        event_type="node_failed",
                        status="failed" if node.required else "running",
                        node_id=node_id,
                        payload={"explanation": outcome.explanation, "metadata": outcome.metadata},
                    )
                    if node.required:
                        metadata = dict(task_running.metadata)
                        metadata["runtime_state"] = self._json_safe(runtime_state)
                        failed_task = task_running.model_copy(
                            update={
                                "status": "failed",
                                "last_error": outcome.explanation or f"Workflow failed at '{node_id}'.",
                                "metadata": metadata,
                                "updated_at": datetime.now(timezone.utc),
                            }
                        )
                        self.task_store.save_task(failed_task)
                        self._emit_event(task=failed_task, event_type="task_failed", status="failed", node_id=node_id)
                        return DomainOutput(
                            status="failure",
                            result={"task_id": failed_task.task_id, "node_id": node_id},
                            explanation=outcome.explanation or f"Workflow failed at node '{node_id}'.",
                            confidence=0.0,
                            metadata={
                                "task_id": failed_task.task_id,
                                "node_id": node_id,
                                "method": failed_task.method,
                            },
                        )
                    continue

                # success path
                if node_id not in completed_nodes:
                    completed_nodes.append(node_id)
                self._emit_event(task=task_running, event_type="node_completed", status="running", node_id=node_id)

                if node.output_key:
                    payload[node.output_key] = outcome.result
                elif isinstance(outcome.result, dict):
                    payload.update(outcome.result)

                if node.kind == "return":
                    result_payload = outcome.result if isinstance(outcome.result, dict) else {"value": outcome.result}
                    final_output = DomainOutput(
                        status="success",
                        result=self._json_safe(result_payload),
                        explanation=outcome.explanation or f"Executed method '{task_running.method}'.",
                        confidence=1.0,
                        metadata={
                            "task_id": task_running.task_id,
                            "method": task_running.method,
                            "nodes": self._json_safe(node_results),
                        },
                    )
                    completed_task = task_running.model_copy(
                        update={
                            "status": "completed",
                            "output_payload": final_output.result,
                            "current_node_id": node_id,
                            "metadata": {
                                **dict(task_running.metadata),
                                "runtime_state": self._json_safe(runtime_state),
                            },
                            "updated_at": datetime.now(timezone.utc),
                        }
                    )
                    self.task_store.save_task(completed_task)
                    self._emit_event(task=completed_task, event_type="task_completed", status="completed", node_id=node_id)
                    return final_output

            metadata = dict(task_running.metadata)
            metadata["runtime_state"] = self._json_safe(runtime_state)
            task_running = task_running.model_copy(
                update={
                    "metadata": metadata,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            self.task_store.save_task(task_running)

        fallback_output = DomainOutput(
            status="success",
            result={
                "task_id": task_running.task_id,
                "method": task_running.method,
                "payload": self._json_safe(payload),
                "nodes": self._json_safe(node_results),
            },
            explanation=f"Executed method '{task_running.method}'.",
            confidence=1.0,
            metadata={"task_id": task_running.task_id, "method": task_running.method},
        )
        completed_task = task_running.model_copy(
            update={
                "status": "completed",
                "output_payload": fallback_output.result,
                "metadata": {
                    **dict(task_running.metadata),
                    "runtime_state": self._json_safe(runtime_state),
                },
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self.task_store.save_task(completed_task)
        self._emit_event(task=completed_task, event_type="task_completed", status="completed")
        return fallback_output

    def _workflow_topology(self, method_spec: MethodSpec) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        incoming_by_target: dict[str, list[Any]] = {}
        outgoing_by_source: dict[str, list[Any]] = {}
        for edge in method_spec.workflow.edges:
            incoming_by_target.setdefault(edge.to_node, []).append(edge)
            outgoing_by_source.setdefault(edge.from_node, []).append(edge)
        return incoming_by_target, outgoing_by_source

    def _ready_workflow_nodes(
        self,
        *,
        method_spec: MethodSpec,
        pending_nodes: list[str],
        completed_nodes: list[str],
        failed_nodes: list[str],
        skipped_nodes: list[str],
        incoming_by_target: dict[str, list[Any]],
        payload: dict[str, Any],
        node_results: dict[str, Any],
        task: TaskInstance,
        node_order: dict[str, int],
    ) -> list[str]:
        workflow = method_spec.workflow
        if not pending_nodes:
            return []

        done_nodes = set(completed_nodes + failed_nodes + skipped_nodes)

        if not workflow.edges:
            if workflow.execution_mode == "parallel":
                return sorted(pending_nodes, key=lambda node_id: node_order.get(node_id, 10**9))
            # sequential default
            first = min(pending_nodes, key=lambda node_id: node_order.get(node_id, 10**9))
            return [first]

        ready: list[str] = []
        for node_id in pending_nodes:
            incoming = incoming_by_target.get(node_id, [])
            if not incoming:
                ready.append(node_id)
                continue

            predecessors = {edge.from_node for edge in incoming}
            if not predecessors.issubset(done_nodes):
                continue

            active = False
            for edge in incoming:
                if not self._evaluate_condition(edge.when, payload=payload, node_results=node_results, task=task):
                    continue
                if edge.from_node in done_nodes:
                    active = True
                    break
            if active:
                ready.append(node_id)

        return sorted(ready, key=lambda node_id: node_order.get(node_id, 10**9))

    def _skip_unreachable_nodes(
        self,
        *,
        pending_nodes: list[str],
        completed_nodes: list[str],
        failed_nodes: list[str],
        skipped_nodes: list[str],
        incoming_by_target: dict[str, list[Any]],
        payload: dict[str, Any],
        node_results: dict[str, Any],
        task: TaskInstance,
    ) -> list[str]:
        done_nodes = set(completed_nodes + failed_nodes + skipped_nodes)
        skipped_now: list[str] = []
        for node_id in list(pending_nodes):
            incoming = incoming_by_target.get(node_id, [])
            if not incoming:
                continue

            predecessors = {edge.from_node for edge in incoming}
            if not predecessors.issubset(done_nodes):
                continue

            has_active_path = any(
                self._evaluate_condition(edge.when, payload=payload, node_results=node_results, task=task)
                for edge in incoming
            )
            if has_active_path:
                continue

            pending_nodes.remove(node_id)
            if node_id not in skipped_nodes:
                skipped_nodes.append(node_id)
            skipped_now.append(node_id)

        return skipped_now

    def _select_workflow_nodes_batch(self, *, method_spec: MethodSpec, ready_nodes: list[str]) -> list[str]:
        if not ready_nodes:
            return []
        workflow = method_spec.workflow
        if workflow.execution_mode in ("parallel", "dag"):
            return ready_nodes[: max(1, workflow.max_concurrency)]
        return [ready_nodes[0]]

    async def _execute_workflow_node_with_retry(
        self,
        *,
        task: TaskInstance,
        node: WorkflowNodeSpec,
        method_spec: MethodSpec,
        payload: dict[str, Any],
        node_results: dict[str, Any],
    ) -> NodeExecutionOutcome:
        max_attempts, backoff_seconds, jitter_seconds, retry_on = self._retry_policy_for_node(node, method_spec)

        for attempt in range(1, max_attempts + 1):
            try:
                timeout_seconds = node.timeout_seconds or method_spec.policy.default_timeout_seconds
                if timeout_seconds and timeout_seconds > 0:
                    outcome = await asyncio.wait_for(
                        self._execute_workflow_node(
                            task=task,
                            node=node,
                            method_spec=method_spec,
                            payload=payload,
                            node_results=node_results,
                        ),
                        timeout=timeout_seconds,
                    )
                else:
                    outcome = await self._execute_workflow_node(
                        task=task,
                        node=node,
                        method_spec=method_spec,
                        payload=payload,
                        node_results=node_results,
                    )
            except TimeoutError:
                outcome = NodeExecutionOutcome(
                    status="failure",
                    result={},
                    explanation=f"Workflow node '{node.id}' timed out.",
                    metadata={"error": "timeout"},
                )
            except Exception as exc:
                outcome = NodeExecutionOutcome(
                    status="failure",
                    result={},
                    explanation=f"Workflow node '{node.id}' failed: {exc}",
                    metadata={"error": str(exc)},
                )

            if outcome.status in ("success", "clarification"):
                return outcome

            if attempt >= max_attempts:
                return outcome

            if not self._should_retry_outcome(outcome=outcome, retry_on=retry_on):
                return outcome

            sleep_seconds = 0.0
            if backoff_seconds > 0:
                sleep_seconds += backoff_seconds * (2 ** (attempt - 1))
            if jitter_seconds > 0:
                sleep_seconds += random.uniform(0.0, jitter_seconds)
            if sleep_seconds > 0:
                await asyncio.sleep(sleep_seconds)

        return NodeExecutionOutcome(status="failure", result={}, explanation=f"Workflow node '{node.id}' failed.")

    def _retry_policy_for_node(self, node: WorkflowNodeSpec, method_spec: MethodSpec) -> tuple[int, float, float, list[str]]:
        retry = node.retry if node.retry is not None else method_spec.policy.default_retry
        max_attempts = 1
        backoff_seconds = 0.0
        jitter_seconds = 0.0
        retry_on: list[str] = []
        if retry is not None:
            max_attempts = max(1, int(retry.max_attempts))
            backoff_seconds = max(0.0, float(retry.backoff_seconds))
            jitter_seconds = max(0.0, float(retry.jitter_seconds))
            retry_on = [str(item).strip().lower() for item in (retry.retry_on or []) if str(item).strip()]
        return max_attempts, backoff_seconds, jitter_seconds, retry_on

    def _should_retry_outcome(self, *, outcome: NodeExecutionOutcome, retry_on: list[str]) -> bool:
        if outcome.status != "failure":
            return False
        if not retry_on:
            return True

        metadata_error = ""
        if isinstance(outcome.metadata, dict):
            metadata_error = str(outcome.metadata.get("error", "")).strip().lower()
        haystack = " ".join(
            part
            for part in [str(outcome.explanation or "").lower(), metadata_error]
            if part
        )
        return any(marker in haystack for marker in retry_on)

    def _build_idempotency_key(
        self,
        *,
        method_spec: MethodSpec,
        payload: dict[str, Any],
        session_id: str | None,
        user_id: str | None,
    ) -> str:
        fields = [str(item).strip() for item in (method_spec.policy.idempotency_key_fields or []) if str(item).strip()]
        if not fields:
            return ""

        key_payload = {
            field_name: self._resolve_context_path(payload, field_name)
            for field_name in fields
        }
        basis = {
            "domain": method_spec.domain,
            "method": method_spec.method,
            "session_id": session_id or "",
            "user_id": user_id or "",
            "fields": key_payload,
        }
        digest = hashlib.sha256(json.dumps(basis, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        return digest

    def _domain_output_from_existing_task(
        self,
        *,
        existing_task: TaskInstance,
        method_spec: MethodSpec,
    ) -> DomainOutput | None:
        if existing_task.status == "completed":
            return DomainOutput(
                status="success",
                result=dict(existing_task.output_payload or {}),
                explanation=f"Returned cached result for '{method_spec.method}'.",
                confidence=1.0,
                metadata={
                    "task_id": existing_task.task_id,
                    "cached": True,
                    "idempotency_key": (existing_task.metadata or {}).get("idempotency_key", ""),
                },
            )

        if existing_task.status == "waiting_user" and existing_task.waiting_question is not None:
            question = existing_task.waiting_question
            return DomainOutput(
                status="clarification",
                result={
                    "task_id": existing_task.task_id,
                    "question_id": question.question_id,
                    "options": question.options,
                    "node_id": question.node_id,
                },
                explanation=question.prompt,
                confidence=1.0,
                metadata={
                    "task_id": existing_task.task_id,
                    "question_id": question.question_id,
                    "node_id": question.node_id,
                    "status": "waiting_user",
                    "cached": True,
                },
            )

        if existing_task.status in {"pending", "running", "paused"}:
            return DomainOutput(
                status="clarification",
                result={"task_id": existing_task.task_id},
                explanation=f"Task '{existing_task.task_id}' is already in progress.",
                confidence=1.0,
                metadata={"task_id": existing_task.task_id, "status": existing_task.status, "cached": True},
            )

        return None

    async def _execute_workflow_node(
        self,
        *,
        task: TaskInstance,
        node: WorkflowNodeSpec,
        method_spec: MethodSpec,
        payload: dict[str, Any],
        node_results: dict[str, Any],
    ) -> NodeExecutionOutcome:
        resolved_params = self._resolve_workflow_placeholders(
            node.params,
            payload=payload,
            node_results=node_results,
            task=task,
        )

        if node.handler:
            handler = self.workflow_handlers.get(node.handler)
            if handler is not None:
                maybe_result = handler(
                    payload=payload,
                    node=node,
                    method_spec=method_spec,
                    task=task,
                    node_results=node_results,
                    params=resolved_params,
                )
                if inspect.isawaitable(maybe_result):
                    maybe_result = await maybe_result
                return self._normalize_handler_outcome(maybe_result)

        if node.kind == "decision":
            expr = resolved_params.get("expr", node.when if node.when is not None else False)
            decision_value = self._evaluate_condition(expr, payload=payload, node_results=node_results, task=task)
            return NodeExecutionOutcome(
                status="success",
                result={"value": decision_value},
                explanation="",
            )

        if node.kind == "human_gate":
            prompt = str(resolved_params.get("question", "")).strip() or f"Please confirm data for '{task.method}'."
            options = resolved_params.get("options", [])
            answer_target = str(resolved_params.get("answer_target", "clarification_answer")).strip() or "clarification_answer"
            return NodeExecutionOutcome(
                status="clarification",
                result={},
                explanation=prompt,
                metadata={
                    "options": [str(opt) for opt in options] if isinstance(options, list) else [],
                    "answer_target": answer_target,
                },
            )

        if node.kind == "validate":
            required_fields = resolved_params.get("required_fields")
            if not isinstance(required_fields, list):
                required_fields = []
                schema_required = method_spec.input_schema.get("required") if isinstance(method_spec.input_schema, dict) else []
                if isinstance(schema_required, list):
                    required_fields = [str(item).strip() for item in schema_required if str(item).strip()]

            missing: list[str] = []
            for field_name in required_fields:
                value = self._resolve_context_path(payload, str(field_name))
                if value in (None, ""):
                    missing.append(str(field_name))

            if missing:
                return NodeExecutionOutcome(
                    status="failure",
                    result={"missing": missing},
                    explanation=f"Missing required fields: {', '.join(missing)}",
                )
            return NodeExecutionOutcome(status="success", result={"valid": True})

        if node.kind == "call":
            call_params = dict(resolved_params) if isinstance(resolved_params, dict) else {}
            capability = str(call_params.pop("capability", "")).strip() or method_spec.method
            domain = str(call_params.pop("domain", "")).strip() or method_spec.domain or task.domain

            params_block = call_params.pop("params", None)
            if isinstance(params_block, dict):
                action_params = params_block
            else:
                action_params = call_params
            if not action_params:
                action_params = dict(payload)

            intent = IntentOutput(
                domain=domain,
                capability=capability,
                confidence=1.0,
                parameters=action_params,
                original_query=f"workflow:{task.method}",
            )
            output = await self.orchestrator.process(intent)

            return NodeExecutionOutcome(
                status=output.status,
                result=output.result,
                explanation=output.explanation,
                metadata={
                    "domain_output": output.model_dump(mode="json"),
                    **(output.metadata or {}),
                },
            )

        if node.kind == "aggregate":
            sources = resolved_params.get("sources")
            if isinstance(sources, list) and sources:
                aggregate = {str(path): self._resolve_context_path(payload, str(path)) for path in sources}
                return NodeExecutionOutcome(status="success", result=aggregate)
            return NodeExecutionOutcome(status="success", result=dict(payload))

        if node.kind == "return":
            if "result" in resolved_params:
                result_payload = resolved_params.get("result")
            else:
                result_path = str(resolved_params.get("from", "")).strip()
                if result_path:
                    result_payload = self._resolve_context_path(payload, result_path)
                else:
                    result_payload = dict(payload)
            explanation = str(resolved_params.get("explanation", "")).strip()
            return NodeExecutionOutcome(status="success", result=result_payload, explanation=explanation)

        # transform / resolve default behavior
        assign_block = resolved_params.get("assign") if isinstance(resolved_params, dict) else None
        if isinstance(assign_block, dict):
            return NodeExecutionOutcome(status="success", result=assign_block)
        return NodeExecutionOutcome(status="success", result={})

    def _normalize_handler_outcome(self, raw: Any) -> NodeExecutionOutcome:
        if isinstance(raw, NodeExecutionOutcome):
            return raw
        if isinstance(raw, DomainOutput):
            return NodeExecutionOutcome(
                status=raw.status,
                result=raw.result,
                explanation=raw.explanation,
                metadata={"domain_output": raw.model_dump(mode="json"), **(raw.metadata or {})},
            )
        if isinstance(raw, dict):
            return NodeExecutionOutcome(status="success", result=raw)
        if raw is None:
            return NodeExecutionOutcome(status="success", result={})
        return NodeExecutionOutcome(status="success", result={"value": raw})

    def _emit_event(
        self,
        *,
        task: TaskInstance,
        event_type: str,
        status: str | None = None,
        node_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        try:
            event = WorkflowEvent(
                event_id=f"evt-{uuid.uuid4().hex[:14]}",
                task_id=task.task_id,
                domain=task.domain,
                method=task.method,
                event_type=event_type,  # type: ignore[arg-type]
                status=status,
                node_id=node_id,
                payload=payload or {},
            )
            self.task_store.save_event(event)
        except Exception as exc:
            logger.warning("Failed to persist workflow event '%s': %s", event_type, exc)

    def _evaluate_condition(
        self,
        condition: Any,
        *,
        payload: dict[str, Any],
        node_results: dict[str, Any],
        task: TaskInstance,
    ) -> bool:
        if condition is None:
            return True
        if isinstance(condition, bool):
            return condition

        if isinstance(condition, dict):
            if "all" in condition and isinstance(condition["all"], list):
                return all(
                    self._evaluate_condition(item, payload=payload, node_results=node_results, task=task)
                    for item in condition["all"]
                )
            if "any" in condition and isinstance(condition["any"], list):
                return any(
                    self._evaluate_condition(item, payload=payload, node_results=node_results, task=task)
                    for item in condition["any"]
                )
            if "not" in condition:
                return not self._evaluate_condition(
                    condition["not"],
                    payload=payload,
                    node_results=node_results,
                    task=task,
                )

            path = condition.get("path")
            if not isinstance(path, str) or not path.strip():
                return False
            value = self._resolve_condition_path(path, payload=payload, node_results=node_results, task=task)

            if condition.get("exists") is True:
                return value is not None
            if condition.get("truthy") is True:
                return bool(value)
            if "equals" in condition:
                return value == condition.get("equals")
            if "in" in condition and isinstance(condition["in"], list):
                return value in condition["in"]
            return False

        if isinstance(condition, str):
            expr = condition.strip()
            if not expr:
                return True

            local_ctx: dict[str, Any] = {
                "payload": payload,
                "nodes": node_results,
                "task": {
                    "task_id": task.task_id,
                    "domain": task.domain,
                    "method": task.method,
                    "status": task.status,
                },
            }
            for key, value in payload.items():
                if isinstance(key, str) and key.isidentifier():
                    local_ctx[key] = value
            for key, value in node_results.items():
                if isinstance(key, str) and key.isidentifier():
                    if isinstance(value, dict) and "result" in value:
                        local_ctx[key] = value["result"]
                    else:
                        local_ctx[key] = value

            return safe_eval_bool(expr, local_ctx, default=False)

        return False

    def _resolve_condition_path(
        self,
        path: str,
        *,
        payload: dict[str, Any],
        node_results: dict[str, Any],
        task: TaskInstance,
    ) -> Any:
        path = path.strip()
        if not path:
            return None
        if path.startswith("payload."):
            return self._resolve_context_path(payload, path.removeprefix("payload."))
        if path.startswith("nodes."):
            return self._resolve_context_path(node_results, path.removeprefix("nodes."))
        if path.startswith("task."):
            return self._resolve_context_path(
                {
                    "task_id": task.task_id,
                    "domain": task.domain,
                    "method": task.method,
                    "status": task.status,
                    "metadata": task.metadata,
                },
                path.removeprefix("task."),
            )

        value = self._resolve_context_path(payload, path)
        if value is not None:
            return value
        return self._resolve_context_path(node_results, path)

    def _resolve_context_path(self, context: Any, path: str) -> Any:
        current = context
        for part in path.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list) and part.isdigit():
                idx = int(part)
                if idx < 0 or idx >= len(current):
                    return None
                current = current[idx]
            else:
                return None
            if current is None:
                return None
        return current

    def _resolve_workflow_placeholders(
        self,
        value: Any,
        *,
        payload: dict[str, Any],
        node_results: dict[str, Any],
        task: TaskInstance,
    ) -> Any:
        if isinstance(value, dict):
            return {
                k: self._resolve_workflow_placeholders(v, payload=payload, node_results=node_results, task=task)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [
                self._resolve_workflow_placeholders(v, payload=payload, node_results=node_results, task=task)
                for v in value
            ]
        if not isinstance(value, str):
            return value

        pattern = re.compile(r"\$\{([^}]+)\}")
        matches = list(pattern.finditer(value))
        if not matches:
            return value

        if len(matches) == 1 and matches[0].span() == (0, len(value)):
            return self._resolve_workflow_reference(matches[0].group(1), payload=payload, node_results=node_results, task=task)

        resolved_text = value
        for match in reversed(matches):
            token = match.group(1)
            resolved = self._resolve_workflow_reference(token, payload=payload, node_results=node_results, task=task)
            resolved_text = (
                resolved_text[: match.start()]
                + str(resolved if resolved is not None else "")
                + resolved_text[match.end() :]
            )
        return resolved_text

    def _resolve_workflow_reference(
        self,
        token: str,
        *,
        payload: dict[str, Any],
        node_results: dict[str, Any],
        task: TaskInstance,
    ) -> Any:
        key = token.strip()
        if key.startswith("ENV:"):
            env_name = key.split(":", 1)[1].strip()
            return os.getenv(env_name, "")

        if key.startswith("payload."):
            return self._resolve_context_path(payload, key.removeprefix("payload."))
        if key.startswith("nodes."):
            return self._resolve_context_path(node_results, key.removeprefix("nodes."))
        if key.startswith("task."):
            task_map = {
                "task_id": task.task_id,
                "domain": task.domain,
                "method": task.method,
                "status": task.status,
                "metadata": task.metadata,
            }
            return self._resolve_context_path(task_map, key.removeprefix("task."))

        value = self._resolve_context_path(payload, key)
        if value is not None:
            return value
        return self._resolve_context_path(node_results, key)

    def _resolve_method_spec(self, capability: str) -> MethodSpec | None:
        registry = getattr(self.orchestrator, "domain_registry", None)
        if registry is None:
            return None
        resolve_method_spec = getattr(registry, "resolve_method_spec", None)
        if not callable(resolve_method_spec):
            return None
        try:
            spec = resolve_method_spec(capability)
            if isinstance(spec, MethodSpec):
                return spec
            if isinstance(spec, dict):
                try:
                    return MethodSpec(**spec)
                except Exception:
                    return None
            return None
        except Exception:
            return None

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._json_safe(item) for item in value]
        if hasattr(value, "model_dump"):
            try:
                dumped = value.model_dump(mode="json")
                return self._json_safe(dumped)
            except Exception:
                return str(value)
        try:
            json.dumps(value)
            return value
        except Exception:
            return str(value)

    def _resolve_placeholders(self, value: Any, outputs: dict[int, DomainOutput]) -> Any:
        """
        Resolve placeholders in params:
        - ${1.explanation}
        - ${1.result.price}
        - ${ENV:TELEGRAM_DEFAULT_CHAT_ID}
        """
        if isinstance(value, dict):
            return {k: self._resolve_placeholders(v, outputs) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve_placeholders(v, outputs) for v in value]
        if not isinstance(value, str):
            return value

        pattern = re.compile(r"\$\{([^}]+)\}")
        matches = list(pattern.finditer(value))
        if not matches:
            return value

        # If full string is a single placeholder and value is not text, preserve type.
        if len(matches) == 1 and matches[0].span() == (0, len(value)):
            return self._resolve_reference(matches[0].group(1), outputs)

        resolved_text = value
        for match in reversed(matches):
            token = match.group(1)
            resolved = self._resolve_reference(token, outputs)
            resolved_text = (
                resolved_text[:match.start()]
                + str(resolved if resolved is not None else "")
                + resolved_text[match.end():]
            )
        return resolved_text

    def _resolve_reference(self, token: str, outputs: dict[int, DomainOutput]) -> Any:
        token = token.strip()
        if token.startswith("ENV:"):
            env_name = token.split(":", 1)[1].strip()
            return os.getenv(env_name, "")

        head, _, tail = token.partition(".")
        if not head.isdigit():
            return ""

        step_id = int(head)
        output = outputs.get(step_id)
        if not output:
            return ""

        if not tail:
            return output.model_dump(mode="json")

        current: Any = output
        for part in tail.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            else:
                current = getattr(current, part, None)
            if current is None:
                return ""
        return current
