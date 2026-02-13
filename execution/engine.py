"""
Execution Engine — Runs the Execution Plan.

Responsibility:
- Execute ordered steps
- Manage timeouts (global & per-step)
- Handle dependencies (DAG)
- Stop on critical failure
"""

import asyncio
import logging
import os
import re
from typing import Any

from shared.models import ExecutionPlan, ExecutionStep, DomainOutput, IntentOutput
from orchestrator.orchestrator import Orchestrator
from execution.result_combiner import ResultCombiner

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Engine to execute plans using the Orchestrator."""

    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator
        self.result_combiner = ResultCombiner()

    async def execute_plan(self, plan: ExecutionPlan, original_intent: IntentOutput) -> DomainOutput:
        """
        Execute the plan step-by-step.
        
        Args:
            plan: The decomposition of tasks
            original_intent: The original intent context (for domain fallback if needed)
            
        Returns:
            The result of the FINAL step in the plan.
        """
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
                step for step in pending.values()
                if all(dep_id in outputs for dep_id in step.depends_on)
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
                *(self._execute_step(step, outputs, original_intent) for step in batch)
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
    ) -> tuple[int, DomainOutput]:
        """Execute a single step with parameter interpolation."""
        logger.info("Executing Step %d: %s (%s)", step.id, step.capability, step.domain or original_intent.domain)
        try:
            resolved_params = self._resolve_placeholders(step.params, previous_outputs)
            step_intent = IntentOutput(
                domain=step.domain or original_intent.domain,
                capability=step.capability,
                confidence=original_intent.confidence,
                parameters=resolved_params,
                original_query=original_intent.original_query,
            )

            if step.timeout_seconds and step.timeout_seconds > 0:
                output = await asyncio.wait_for(
                    self.orchestrator.process(step_intent),
                    timeout=step.timeout_seconds,
                )
            else:
                output = await self.orchestrator.process(step_intent)
            return step.id, output
        except Exception as e:
            logger.critical("Engine crash on Step %d: %s", step.id, e)
            return step.id, DomainOutput(
                status="failure",
                result={},
                explanation=f"Execution Engine crashed at step {step.id}: {e}",
                confidence=0.0,
                metadata={"error": str(e)},
            )

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
