"""
Shared Pydantic models for all layers.
All contexts are immutable (frozen) after creation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ─── Entry Layer ───────────────────────────────────────────────

class EntryRequest(BaseModel):
    """Normalized input from any entry adapter."""
    model_config = {"frozen": True}

    session_id: str
    input_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── Goal Layer ───────────────────────────────────────────────

class GoalDefinition(BaseModel):
    """A business-level goal within a domain.
    Goals sit between Domain and Capability:
      Domain → Goals (business objectives) → Capabilities (executable)
    """
    model_config = {"frozen": True}
    goal: str = Field(..., description="Goal identifier, e.g. 'GET_QUOTE', 'IMPACT_ANALYSIS'")
    description: str = Field(default="", description="Human-readable goal description")
    capabilities: list[str] = Field(default_factory=list, description="Capabilities this goal maps to")
    requires_domains: list[str] = Field(default_factory=list, description="Cross-domain dependencies, e.g. ['NEWS', 'PORTFOLIO']")
    hints: dict[str, Any] = Field(default_factory=dict, description="Intent matching hints: {keywords: [...], examples: [...]}")
    entities_schema: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Schema for entities the Intent LLM should extract for this goal. "
            "Keys are entity names, values describe type/enum/required. "
            "Example: {'direction': {'type': 'enum', 'values': ['GAINERS','LOSERS','BOTH'], 'required': true}}"
        ),
    )


# ─── Intent Layer (Goal-based) ────────────────────────────────

class IntentOutput(BaseModel):
    """Minimal goal-based intent output from the Intent Layer.
    Intent identifies WHAT the user wants (domain + goal + entities).
    Never references capabilities or cross-domain relationships.
    """
    model_config = {"frozen": True}
    primary_domain: str = Field(..., description="Target domain, e.g. 'finance', 'communication'")
    goal: str = Field(..., description="Business goal within the domain, e.g. 'GET_QUOTE', 'SEND_NOTIFICATION'")
    entities: dict[str, Any] = Field(default_factory=dict, description="Extracted entities from user text (symbols, company names, periods, etc.)")
    confidence: float = Field(..., description="Extraction confidence 0.0-1.0")
    original_query: str = Field(default="", description="Raw user input")

    def to_execution_intent(self, resolved_capability: str) -> ExecutionIntent:
        """Convert to execution-ready intent with a resolved capability."""
        return ExecutionIntent(
            domain=self.primary_domain,
            capability=resolved_capability,
            confidence=self.confidence,
            parameters=dict(self.entities),
            original_query=self.original_query,
        )


class ExecutionIntent(BaseModel):
    """Resolved intent ready for orchestrator/execution.
    Produced by the Planner after resolving goal → capability.
    """
    model_config = {"frozen": True}
    domain: str = Field(..., description="Target domain")
    capability: str = Field(..., description="Resolved capability to execute")
    confidence: float = Field(..., description="Confidence from original intent")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Resolved parameters for execution")
    original_query: str = Field(default="", description="Raw user input")

# ─── Planner Layer (New v3) ────────────────────────────────────

class ExecutionStep(BaseModel):
    """Atomic step in an execution plan."""
    model_config = {"frozen": True}
    id: int
    domain: str | None = Field(default=None, description="Target domain for this step")
    capability: str
    params: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[int] = Field(default_factory=list)
    required: bool = Field(default=True, description="If true, stop/mark failure when this step fails")
    output_key: str | None = Field(default=None, description="Optional key used in combined output")
    timeout_seconds: float | None = Field(default=None, description="Optional per-step timeout")


class ExecutionPlan(BaseModel):
    """Structured plan for the Execution Engine."""
    model_config = {"frozen": True}
    execution_mode: str = Field(..., description="'sequential', 'parallel', or 'dag'")
    steps: list[ExecutionStep]
    combine_mode: str = Field(default="last", description="'last', 'report', or 'merge'")
    max_concurrency: int = Field(default=4, description="Maximum number of concurrent steps")
# ─── Model Layer (Policy) ──────────────────────────────────────

class ModelPolicy(BaseModel):
    """Configuration for Model Layer execution."""
    model_config = {"frozen": True}
    model_name: str
    temperature: float = 0.0
    timeout_seconds: float = 30.0
    max_retries: int = 3
    json_mode: bool = True


# ─── Domain Output (New v2) ────────────────────────────────────

class DomainOutput(BaseModel):
    """Standardized output from any Domain Handler."""
    model_config = {"frozen": True}
    status: str = Field(..., description="'success', 'failure', or 'clarification'")
    result: dict[str, Any] = Field(default_factory=dict)
    explanation: str = Field(default="")
    confidence: float = Field(default=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── Domain Context (Finance) ──────────────────────────────────

class DomainContext(BaseModel):
    """Deterministic context resolved from ticker/symbol analysis."""
    model_config = {"frozen": True}

    # Market identification
    market: str = Field(..., description="Market code: 'US', 'SE', 'BR'")
    country: str = Field(default="", description="Full country name")
    currency: str = Field(..., description="Currency code: 'USD', 'SEK', 'BRL'")
    currency_symbol: str = Field(default="$", description="Currency display symbol")

    # Exchange
    exchange: str = Field(default="", description="Primary exchange: 'NYSE', 'NASDAQ', 'B3', 'OMX'")
    exchange_suffix: str = Field(default="", description="Yahoo/IB symbol suffix: '.SA', '.ST', ''")
    exchange_timezone: str = Field(default="America/New_York", description="Exchange timezone")

    # Trading rules
    lot_size: int = Field(default=100, description="Standard lot size for options")
    tick_size: float = Field(default=0.01, description="Minimum price increment")
    trading_hours: str = Field(default="09:30-16:00", description="Local trading hours")
    settlement_days: int = Field(default=2, description="T+N settlement cycle")
    has_options: bool = Field(default=True, description="Whether options are traded")
    has_fractional: bool = Field(default=False, description="Supports fractional shares")

    # Tax & fiscal
    tax_model: str = Field(default="standard", description="Tax model: 'ISK', 'standard', 'ISA'")
    tax_rate_gains: float = Field(default=0.0, description="Capital gains tax rate (0-1)")
    tax_notes: str = Field(default="", description="Important fiscal notes for this market")


# ─── Execution Context ─────────────────────────────────────────

class ExecutionContext(BaseModel):
    """Combined context: domain + skill data. Never persisted."""
    model_config = {"frozen": True}

    domain_context: DomainContext
    skill_data: dict[str, Any] = Field(default_factory=dict)
    multi_contexts: dict[str, DomainContext] | None = Field(
        default=None,
        description="Per-symbol contexts for multi-symbol queries (symbol → DomainContext mapping)"
    )


# ─── Decision (final output) ───────────────────────────────────

class Decision(BaseModel):
    """Structured output from Strategy Core."""
    model_config = {"frozen": True}

    action: str = Field(..., description="Action performed")
    result: dict[str, Any] = Field(default_factory=dict, description="Result data")
    risk_metrics: dict[str, Any] = Field(default_factory=dict, description="Risk metrics if applicable")
    explanation: str = Field(default="", description="Human-readable explanation")
    success: bool = Field(default=True)
    error: str | None = Field(default=None)
