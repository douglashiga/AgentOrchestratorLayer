from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IntentInput(BaseModel):
    domain: str = Field(default="communication")
    capability: str
    confidence: float = Field(default=1.0)
    parameters: dict[str, Any] = Field(default_factory=dict)
    original_query: str = Field(default="")


class DomainOutput(BaseModel):
    status: str = Field(..., description="'success', 'failure', or 'clarification'")
    result: dict[str, Any] = Field(default_factory=dict)
    explanation: str = Field(default="")
    confidence: float = Field(default=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
