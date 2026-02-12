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


# ─── Intent (LLM output, schema-validated) ─────────────────────

class Intent(BaseModel):
    """Structured intent extracted by the LLM Intent Adapter."""
    model_config = {"frozen": True}

    domain: str = Field(..., description="Target domain, e.g. 'finance'")
    action: str = Field(..., description="Action to perform, e.g. 'get_stock_price'")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# ─── Domain Context (deterministic) ────────────────────────────

class DomainContext(BaseModel):
    """Deterministic context resolved from ticker/symbol analysis.
    All fields are fixed per market — no inference, no LLM.
    """
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
