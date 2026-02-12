"""
Skill Gateway — Controls access to skills.

Responsibility:
- Validate skill availability via SkillRegistry
- Execute skills through controlled access

Prohibitions:
- No strategy decisions
- No fiscal rule application
"""

import logging
from typing import Any

from skills.registry import SkillRegistry

logger = logging.getLogger(__name__)


class SkillGateway:
    """Controlled access layer to skill implementations."""

    def __init__(self, skill_registry: SkillRegistry):
        self.skill_registry = skill_registry

    def execute(self, skill_name: str, parameters: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a skill by name with given parameters.
        Returns the raw skill result (dict with 'success' and 'data' or 'error').
        """
        skill = self.skill_registry.resolve(skill_name)
        if skill is None:
            logger.warning("Skill not found: %s", skill_name)
            return {
                "success": False,
                "error": f"Skill '{skill_name}' not registered.",
            }

        try:
            result = skill.execute(parameters)
            logger.info("Skill '%s' executed successfully", skill_name)
            return result
        except Exception as e:
            logger.exception("Skill '%s' execution failed", skill_name)
            return {
                "success": False,
                "error": f"Skill execution error: {e}",
            }
