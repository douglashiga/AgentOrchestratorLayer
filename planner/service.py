"""
Planner Layer — Decomposes Intent into an Execution Plan.

Responsibility:
- Transform IntentOutput → ExecutionPlan (1..N steps)
- Support deterministic multi-domain decomposition
"""

import logging

from shared.models import IntentOutput, ExecutionPlan
from planner.task_decomposer import TaskDecomposer

logger = logging.getLogger(__name__)


class PlannerService:
    """Service to generate execution plans from intents."""

    def __init__(self, known_capabilities: list[str] | None = None):
        self.task_decomposer = TaskDecomposer(known_capabilities=known_capabilities or [])

    def update_capabilities(self, capabilities: list[str]) -> None:
        self.task_decomposer.update_capabilities(capabilities)

    def generate_plan(self, intent: IntentOutput) -> ExecutionPlan:
        """
        Convert an intent into an actionable execution plan.
        For now, this is a direct mapping (1 Intent = 1 Step).
        """
        logger.info("Generating plan for intent: %s", intent.capability)
        plan = self.task_decomposer.decompose(intent)
        
        logger.debug("Plan generated with %d steps", len(plan.steps))
        return plan
