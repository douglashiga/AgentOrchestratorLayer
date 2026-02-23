"""
Analysis Skill Base — Protocol and models for analysis skills.

An AnalysisSkill defines HOW to analyze financial data using an LLM:
- What data to gather (data_requirements)
- What system prompt to use (persona/expertise)
- How to build the user prompt from gathered data
- Optional model policy overrides
"""

from __future__ import annotations

import logging
import re
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from shared.models import ModelPolicy

logger = logging.getLogger(__name__)


class DataRequirement(BaseModel):
    """
    A single data dependency for an analysis skill.

    Specifies what data to fetch before running the LLM analysis.
    Supports parameter interpolation: ${symbol} is replaced from user params.
    """
    model_config = {"frozen": True}

    capability: str = Field(
        ..., description="Capability to call (e.g. 'get_historical_data', 'get_technical_analysis')"
    )
    tier: str = Field(
        default="facts", description="Which tier to use: 'facts' or 'calculator'"
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the capability. Use ${param_name} for interpolation from user params.",
    )
    output_key: str = Field(
        ..., description="Key to store this data under in the gathered_data dict"
    )
    required: bool = Field(
        default=True, description="If True, analysis fails when this data is unavailable"
    )

    def resolve_params(self, user_params: dict[str, Any]) -> dict[str, Any]:
        """
        Resolve parameter interpolation: ${symbol} → user_params["symbol"].

        Args:
            user_params: The user's resolved parameters

        Returns:
            New dict with interpolated values
        """
        resolved = {}
        for key, value in self.params.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                param_name = value[2:-1]
                resolved[key] = user_params.get(param_name, value)
            else:
                resolved[key] = value
        return resolved


@runtime_checkable
class AnalysisSkill(Protocol):
    """
    Protocol for finance analysis skills.

    Each skill is a specialized analyst persona with:
    - Knowledge of what data it needs
    - A system prompt that defines its expertise
    - A method to build the analysis prompt from gathered data
    """

    @property
    def name(self) -> str:
        """Unique skill identifier (e.g. 'technical_analysis')."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description for discovery."""
        ...

    @property
    def system_prompt(self) -> str:
        """System prompt that specializes the LLM for this analysis type."""
        ...

    @property
    def data_requirements(self) -> list[DataRequirement]:
        """List of data dependencies to gather before analysis."""
        ...

    @property
    def model_policy(self) -> ModelPolicy | None:
        """Optional model configuration override. None = use default."""
        ...

    def build_user_prompt(
        self,
        params: dict[str, Any],
        gathered_data: dict[str, Any],
        original_query: str = "",
    ) -> str:
        """
        Build the user-facing prompt from gathered data.

        Args:
            params: Resolved user parameters
            gathered_data: Data fetched per data_requirements, keyed by output_key
            original_query: Original user query text

        Returns:
            Complete user prompt string for the LLM
        """
        ...
