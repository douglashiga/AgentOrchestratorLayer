"""
Strategy Core — Deterministic finance logic.

Responsibility:
- Execute deterministic calculations
- Structure execution results into Decision

Critical Rules:
- No LLM
- No probabilistic heuristics
- No direct infrastructure access
- 100% testable
"""

import logging

from shared.models import Decision, DomainContext, ExecutionContext, IntentOutput

logger = logging.getLogger(__name__)


class StrategyCore:
    """Deterministic strategy engine. Structures skill data into Decisions."""

    def execute(self, intent: IntentOutput, execution_context: ExecutionContext, registry: Any = None) -> Decision:
        """
        Process intent + execution context into a Decision.
        For v1: structures skill data into a clean Decision response.
        """
        skill_data = execution_context.skill_data
        domain_ctx = execution_context.domain_context

        # Check if skill returned an error
        if not skill_data.get("success", False):
            return Decision(
                action=intent.capability,
                success=False,
                error=skill_data.get("error", "Unknown skill error"),
                explanation=f"Failed to execute '{intent.capability}'.",
            )

        # Build result with market context enrichment
        raw_result = skill_data.get("data", {})
        
        # Flatten inner envelope if it exists
        if isinstance(raw_result, dict) and "data" in raw_result:
            inner_data = raw_result["data"]
            if isinstance(inner_data, dict):
                 # Flatten: Keep outer keys (metadata) and merge inner keys (data)
                 raw_result = {**raw_result, **inner_data}
            # If list, keep structure as-is to preserve metadata (like 'market') in the root

        # Normalize result to dictionary
        if isinstance(raw_result, list):
            result = {"items": raw_result, "count": len(raw_result)}
        elif isinstance(raw_result, dict):
            result = raw_result
        else:
            result = {"value": raw_result}

        enriched_result = {
            **result,
            "_market_context": {
                "market": domain_ctx.market,
                "country": domain_ctx.country,
                "currency": domain_ctx.currency,
                "currency_symbol": domain_ctx.currency_symbol,
                "exchange": domain_ctx.exchange,
                "exchange_suffix": domain_ctx.exchange_suffix,
                "timezone": domain_ctx.exchange_timezone,
                "trading_hours": domain_ctx.trading_hours,
                "lot_size": domain_ctx.lot_size,
                "settlement": f"T+{domain_ctx.settlement_days}",
                "has_options": domain_ctx.has_options,
                "tax_model": domain_ctx.tax_model,
                "tax_rate_gains": f"{domain_ctx.tax_rate_gains:.0%}",
                "tax_notes": domain_ctx.tax_notes,
            },
        }

        # Generate human-readable explanation
        explanation = self._generate_explanation(intent, result, domain_ctx, registry)

        return Decision(
            action=intent.capability,
            result=enriched_result,
            risk_metrics={},
            explanation=explanation,
            success=True,
        )

    def _generate_explanation(
        self, intent: IntentOutput, result: dict, domain_ctx: DomainContext, registry: Any = None
    ) -> str:
        """Generate deterministic human-readable explanation."""
        capability = intent.capability
        params = intent.parameters
        symbol = params.get("symbol", params.get("query", "N/A"))

        # ─── Metadata-Driven Generation ──────────────────────────────
        metadata = {}
        if registry:
            metadata = registry.get_metadata(capability)
        
        template = metadata.get("explanation_template")
        
        # ─── List Handling & Count Injection ─────────────────────────
        # If result is a list (or normalized list), we inject 'count' into the context
        # and checking if the template logic handles it or if we append it.
        
        items_count = 0
        if isinstance(result, dict):
            if "items" in result and isinstance(result["items"], list):
                items_count = len(result["items"])
            elif "data" in result and isinstance(result["data"], list):
                items_count = len(result["data"])
        elif isinstance(result, list):
            items_count = len(result)

        if template:
             try:
                  # Robust Formatting: Use .format() but handle missing keys gracefully?
                  # Actually, we expect the template to use known keys: 
                  # {symbol}, {market}, {currency}, {params}, {result}
                  
                  # Flatten params for easier access in template (e.g. {params[period]} -> {period} if we want, 
                  # but let's stick to strict {params[key]} or pre-calculated context)
                  
                  # Context for template
                  ctx = {
                      "symbol": symbol,
                      "market": domain_ctx.market,
                      "currency": domain_ctx.currency,
                      "params": params,
                      "result": result,
                      "count": items_count
                  }
                  
                  explanation = template.format(**ctx)
                  
                  if items_count > 0 and "{count}" not in template:
                      explanation += f" Found {items_count} items."
                      
                  return explanation

             except Exception as e:
                  logger.warning(f"Failed to format template '{template}': {e}")
                  # Fallback to generic if template fails
        
        # ─── Generic Fallback ────────────────────────────────────────
        return f"Executed '{capability}' for {symbol}."
