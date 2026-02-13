"""
Execution Engine — Runs the Execution Plan.

Responsibility:
- Execute ordered steps
- Manage timeouts (global & per-step)
- Handle dependencies (future)
- Stop on critical failure
"""

import logging
from typing import Any

from shared.models import ExecutionPlan, ExecutionStep, DomainOutput, IntentOutput
from orchestrator.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Engine to execute plans using the Orchestrator."""

    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator

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
        
        final_output = None
        
        # Context for sharing data between steps (future feature)
        execution_context: dict[int, Any] = {}

        for step in plan.steps:
            logger.info("Executing Step %d: %s", step.id, step.capability)
            
            # Construct a transient IntentOutput for this step
            # We reuse the domain from original intent for now, or let Orchestrator resolve by capability
            step_intent = IntentOutput(
                domain=original_intent.domain, # Inherit broad domain
                capability=step.capability,
                confidence=original_intent.confidence, # Inherit confidence
                parameters=dict(step.params)
            )

            try:
                # TODO: Implement step timeout using context manager or threads
                output = await self.orchestrator.process(step_intent)
                
                execution_context[step.id] = output
                final_output = output
                
                if output.status in ("failure", "clarification"):
                    logger.info("Step %d stopped execution: status=%s", step.id, output.status)
                    # Stop execution on failure or clarification
                    return output

            except Exception as e:
                logger.critical("Engine crash on Step %d: %s", step.id, e)
                return DomainOutput(
                    status="failure",
                    result={},
                    explanation=f"Execution Engine crashed at step {step.id}: {e}",
                    confidence=0.0,
                    metadata={"error": str(e)}
                )

        if final_output is None:
             return DomainOutput(
                status="failure",
                result={},
                explanation="Plan executed but produced no output.",
                confidence=0.0
            )

        return final_output
