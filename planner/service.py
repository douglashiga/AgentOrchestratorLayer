"""
Planner Layer — Decomposes Intent into an Execution Plan.

Responsibility:
- Transform IntentOutput (Capability) → ExecutionPlan (Steps)
- Initially: 1-to-1 mapping (Atomic Plan)
- Future: Chain of Thought decomposition for complex queries
"""

from typing import Any
import logging

from shared.models import IntentOutput, ExecutionPlan, ExecutionStep

logger = logging.getLogger(__name__)


class PlannerService:
    """Service to generate execution plans from intents."""

    def generate_plan(self, intent: IntentOutput) -> ExecutionPlan:
        """
        Convert an intent into an actionable execution plan.
        For now, this is a direct mapping (1 Intent = 1 Step).
        """
        logger.info("Generating plan for intent: %s", intent.capability)

        # Simple 1-step plan
        step = ExecutionStep(
            id=1,
            capability=intent.capability,
            params=intent.parameters,
            depends_on=[]
        )

        plan = ExecutionPlan(
            execution_mode="sequential",
            steps=[step]
        )
        
        logger.debug("Plan generated with %d steps", len(plan.steps))
        return plan
