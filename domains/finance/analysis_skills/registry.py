"""
Finance Skill Registry — Discovery and resolution for analysis skills.

Manages the mapping between capabilities and analysis skills,
and provides a discovery interface for LLM-based capability selection.
"""

import logging
from typing import Any

from domains.finance.analysis_skills.base import AnalysisSkill

logger = logging.getLogger(__name__)


class FinanceSkillRegistry:
    """
    Registry for finance analysis skills.

    Each skill can be mapped to one or more capabilities.
    Provides discovery metadata for LLM-based routing.
    """

    def __init__(self):
        self._skills: dict[str, AnalysisSkill] = {}  # skill.name → skill
        self._capability_map: dict[str, str] = {}  # capability → skill.name

    def register(self, skill: AnalysisSkill, capabilities: list[str]) -> None:
        """
        Register a skill and map it to capabilities.

        Args:
            skill: The AnalysisSkill implementation
            capabilities: List of capability names this skill handles
        """
        self._skills[skill.name] = skill
        for cap in capabilities:
            self._capability_map[cap] = skill.name
            logger.info("Registered analysis skill: %s → %s", cap, skill.name)

    def resolve(self, capability: str) -> AnalysisSkill | None:
        """
        Find the analysis skill for a capability.

        Args:
            capability: Capability name (e.g. 'analyze_technical')

        Returns:
            The AnalysisSkill or None if not registered
        """
        skill_name = self._capability_map.get(capability)
        if skill_name is None:
            return None
        return self._skills.get(skill_name)

    def discover(self) -> list[dict[str, Any]]:
        """
        List all registered skills with metadata for discovery.

        Returns:
            List of dicts with name, description, capabilities, data_requirements
        """
        result = []
        # Invert capability_map to get capabilities per skill
        skill_capabilities: dict[str, list[str]] = {}
        for cap, skill_name in self._capability_map.items():
            skill_capabilities.setdefault(skill_name, []).append(cap)

        for skill_name, skill in self._skills.items():
            result.append({
                "name": skill.name,
                "description": skill.description,
                "capabilities": skill_capabilities.get(skill_name, []),
                "data_requirements": [
                    {
                        "capability": req.capability,
                        "tier": req.tier,
                        "output_key": req.output_key,
                        "required": req.required,
                    }
                    for req in skill.data_requirements
                ],
                "has_model_policy": skill.model_policy is not None,
            })

        return result

    def list_skill_names(self) -> list[str]:
        """List all registered skill names."""
        return list(self._skills.keys())
