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

from shared.models import Decision, DomainContext, ExecutionContext, Intent

logger = logging.getLogger(__name__)


class StrategyCore:
    """Deterministic strategy engine. Structures skill data into Decisions."""

    def execute(self, intent: Intent, execution_context: ExecutionContext) -> Decision:
        """
        Process intent + execution context into a Decision.
        For v1: structures skill data into a clean Decision response.
        """
        skill_data = execution_context.skill_data
        domain_ctx = execution_context.domain_context

        # Check if skill returned an error
        if not skill_data.get("success", False):
            return Decision(
                action=intent.action,
                success=False,
                error=skill_data.get("error", "Unknown skill error"),
                explanation=f"Failed to execute '{intent.action}'.",
            )

        # Build result with market context enrichment
        result = skill_data.get("data", {})
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
        explanation = self._generate_explanation(intent, result, domain_ctx)

        return Decision(
            action=intent.action,
            result=enriched_result,
            risk_metrics={},
            explanation=explanation,
            success=True,
        )

    def _generate_explanation(
        self, intent: Intent, result: dict, domain_ctx: DomainContext
    ) -> str:
        """Generate deterministic human-readable explanation."""
        action = intent.action
        params = intent.parameters
        symbol = params.get("symbol", params.get("query", "N/A"))

        match action:
            case "get_stock_price":
                price = result.get("price", "N/A")
                return f"{symbol} is currently trading at {price} {domain_ctx.currency}."
            case "get_fundamentals":
                return f"Fundamentals data for {symbol} ({domain_ctx.market} market)."
            case "get_dividends":
                return f"Dividend history for {symbol} ({domain_ctx.currency})."
            case "get_company_info":
                name = result.get("name", symbol)
                return f"Company information for {name}."
            case "get_historical_data":
                duration = params.get("duration", "N/A")
                return f"Historical data for {symbol} over {duration}."
            case "get_option_chain":
                return f"Option chain for {symbol}."
            case "get_option_greeks":
                return f"Option Greeks for {symbol}."
            case "get_financial_statements":
                return f"Financial statements for {symbol}."
            case "get_exchange_info":
                return f"Exchange information for {symbol}."
            case "get_account_summary":
                return "Account summary retrieved."
            case "search_symbol" | "yahoo_search":
                query = params.get("query", symbol)
                return f"Search results for '{query}'."
            case _:
                return f"Executed '{action}' for {symbol}."
