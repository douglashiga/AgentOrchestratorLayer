"""
Tier Base — Contracts for the 3-tier finance architecture.

Defines:
- Tier enum (facts, calculator, analysis)
- TierContext (unified input for all tiers)
- TierProcessor protocol (common interface)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from shared.models import DomainOutput, ExecutionIntent, IntentOutput

logger = logging.getLogger(__name__)


class Tier(str, Enum):
    """Processing tier classification."""
    FACTS = "facts"
    CALCULATOR = "calculator"
    ANALYSIS = "analysis"


class TierContext(BaseModel):
    """
    Unified context passed to any tier processor.

    Contains everything a tier needs to execute:
    - intent: the resolved execution intent
    - params: validated + pre-flow resolved parameters
    - metadata: METADATA_OVERRIDES for this capability
    - original_query: raw user input for LLM-based tiers
    """
    model_config = {"frozen": True}

    intent: ExecutionIntent
    params: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    original_query: str = ""


@runtime_checkable
class TierProcessor(Protocol):
    """Protocol that all tier processors must implement."""

    async def process(self, context: TierContext) -> DomainOutput:
        """
        Process the given context and return a DomainOutput.

        Args:
            context: TierContext with intent, params, metadata, and original_query

        Returns:
            DomainOutput with status, result, explanation, confidence, metadata
        """
        ...
