"""
Skill Registry — Maps skill_name → implementation.

Pure lookup, no logic.
"""

from __future__ import annotations

import logging
from typing import Protocol, Any

logger = logging.getLogger(__name__)


class SkillImplementation(Protocol):
    """Protocol that all skill implementations must follow."""

    def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Execute the skill with given parameters.
        Must return: {"success": bool, "data": {...}} or {"success": False, "error": "..."}
        """
        ...


class SkillRegistry:
    """Registry mapping skill names to their implementations."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillImplementation] = {}

    def register(self, skill_name: str, implementation: SkillImplementation) -> None:
        """Register a skill implementation."""
        logger.info("Registered skill: %s → %s", skill_name, type(implementation).__name__)
        self._skills[skill_name] = implementation

    def resolve(self, skill_name: str) -> SkillImplementation | None:
        """Resolve a skill by name. Returns None if not found."""
        return self._skills.get(skill_name)

    @property
    def registered_skills(self) -> list[str]:
        """List all registered skill names."""
        return list(self._skills.keys())
