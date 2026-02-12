"""
Finance Domain Handler.

Responsibility:
- Orchestrate internally: Context Resolver → Skill Gateway → Strategy Core
- Return Decision

Prohibitions:
- No LLM usage
- No bypassing gateway
- No global state
"""

import logging

from shared.models import Decision, DomainOutput, ExecutionContext, IntentOutput
from domains.finance.context import ContextResolver
from domains.finance.core import StrategyCore
from skills.gateway import SkillGateway
from domains.finance.schemas import (
    TopGainersInput, TopLosersInput, StockPriceInput, 
    HistoricalDataInput, StockScreenerInput, 
    TechnicalSignalsInput, CompareFundamentalsInput
)
from pydantic import ValidationError
from typing import get_type_hints
import inspect

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class FinanceDomainHandler:
    """Finance domain handler — orchestrates context, skills, and strategy."""

    def __init__(self, skill_gateway: SkillGateway, registry: Any = None):
        self.context_resolver = ContextResolver()
        self.strategy_core = StrategyCore()
        self.skill_gateway = skill_gateway
        self.registry = registry

    async def execute(self, intent: IntentOutput) -> DomainOutput:
        """
        Execute finance capability with type-safe dispatch.
        Attempts to find a specific handler method (e.g. get_top_gainers) 
        matching the intent capability.
        """
        try:
            # 1. Dynamic Dispatch to Typed Methods
            method_name = intent.capability
            if hasattr(self, method_name):
                method = getattr(self, method_name)
                # Ensure it's a bound method, not a property
                if callable(method):
                    try:
                        # Inspect method hints to find the Pydantic model
                        type_hints = get_type_hints(method)
                        input_model = type_hints.get("params")
                        
                        if input_model:
                            # Validate inputs against schema
                            validated_params = input_model(**intent.parameters)
                            # Call specific method with Intent AND Validated Params
                            return await method(intent, validated_params)
                    except ValidationError as ve:
                         # Pydantic validation error -> Ask for clarification
                         error_msg = str(ve).replace("\n", "; ")
                         return DomainOutput(
                            status="clarification",
                            result={},
                            explanation=f"I need more information to proceed. {error_msg}",
                            confidence=1.0,
                            metadata={"validation_error": str(ve)}
                        )
                    except Exception as e:
                         # Other execution error -> Failure
                         return DomainOutput(
                            status="failure",
                            explanation=f"Error executing {method_name}: {e}",
                            metadata={"error": str(e)}
                        )

            # 2. Fallback to Generic Execution (Legacy)
            return await self._generic_execute(intent)

        except Exception as e:
            logger.error("Finance execution failed: %s", e, exc_info=True)
            return DomainOutput(
                status="failure",
                result={},
                explanation=f"Error executing {intent.capability}: {str(e)}",
                confidence=0.0
            )

    # ─── Typed Capabilities ──────────────────────────────────────────

    async def get_top_gainers(self, intent: IntentOutput, params: TopGainersInput) -> DomainOutput:
        return await self._run_pipeline(intent, params.model_dump())

    async def get_top_losers(self, intent: IntentOutput, params: TopLosersInput) -> DomainOutput:
        return await self._run_pipeline(intent, params.model_dump())

    async def get_stock_price(self, intent: IntentOutput, params: StockPriceInput) -> DomainOutput:
        return await self._run_pipeline(intent, params.model_dump())

    async def get_historical_data(self, intent: IntentOutput, params: HistoricalDataInput) -> DomainOutput:
        return await self._run_pipeline(intent, params.model_dump())

    async def get_stock_screener(self, intent: IntentOutput, params: StockScreenerInput) -> DomainOutput:
        return await self._run_pipeline(intent, params.model_dump())

    async def get_technical_signals(self, intent: IntentOutput, params: TechnicalSignalsInput) -> DomainOutput:
        return await self._run_pipeline(intent, params.model_dump())
        
    async def compare_fundamentals(self, intent: IntentOutput, params: CompareFundamentalsInput) -> DomainOutput:
        return await self._run_pipeline(intent, params.model_dump())

    # ─── Unified Execution Pipeline ──────────────────────────────────

    async def _run_pipeline(self, intent: IntentOutput, params: dict) -> DomainOutput:
        """
        Unified pipeline for ALL finance executions (Typed & Generic).
        1. Metadata Check (Clarification overlap?) - Skipped if Typed (Pydantic handles valid structure)
        2. Context Resolution
        3. Skill Execution
        4. Strategy Core
        """
        capability = intent.capability
        
        # ─── 1. Context Resolution ───────────────────────────────────
        try:
            # Try to resolve from specific params first
            if params.get("symbol"):
                domain_context = self.context_resolver.resolve(params["symbol"])
            elif params.get("symbols") and isinstance(params["symbols"], list) and params["symbols"]:
                 domain_context = self.context_resolver.resolve(params["symbols"][0])
            elif params.get("market"):
                 domain_context = self.context_resolver.get_market_profile(params["market"])
                 if not domain_context:
                      domain_context = self.context_resolver.resolve("DEFAULT") 
            else:
                 domain_context = self.context_resolver.resolve("DEFAULT")

            logger.info("Context resolved: %s (%s)", domain_context.market, domain_context.currency)
            
            # ─── 2. Skill Execution ──────────────────────────────────
            
            # Prepare Skill Params
            skill_params = {
                **params, 
                "_action": capability,
            }
            
            # Inject Tool Map from Metadata
            metadata = {}
            if self.registry:
                 metadata = self.registry.get_metadata(capability)
            if metadata:
                 skill_params["_tool_map"] = metadata.get("tool_map", {})
            
            # Inject Context if needed (e.g. market code from symbol resolution)
            if not skill_params.get("market") and domain_context.market != "US":
                skill_params["market"] = domain_context.market

            # Execute via Gateway
            skill_data = self.skill_gateway.execute("mcp_finance", skill_params)
            logger.debug("Skill data received: success=%s", skill_data.get("success"))

            # ─── 3. Strategy Execution ───────────────────────────────
            execution_context = ExecutionContext(
                domain_context=domain_context,
                skill_data=skill_data,
            )

            decision = self.strategy_core.execute(intent, execution_context, registry=self.registry)
            
            # ─── 4. Output Generation ────────────────────────────────
            output_metadata = {"risk_metrics": decision.risk_metrics}
            if decision.error:
                output_metadata["error"] = decision.error

            return DomainOutput(
                status="success" if decision.success else "failure",
                result=decision.result,
                explanation=decision.explanation,
                confidence=1.0,
                metadata=output_metadata
            )

        except Exception as e:
            logger.error(f"Pipeline failed for {capability}: {e}", exc_info=True)
            return DomainOutput(
                status="failure",
                result={},
                explanation=f"Error executing finance action: {str(e)}",
                confidence=0.0,
                metadata={"error": str(e)}
            )

    async def _generic_execute(self, intent: IntentOutput) -> DomainOutput:
        """Legacy generic execution path - delegates to unified pipeline."""
        if intent.capability == "chat":
            return DomainOutput(
                status="success",
                result={},
                explanation="This looks like a general question. Please ask a finance question."
            )
            
        # For generic execution, run manual metadata checks that Pydantic would have caught
        # (Legacy clarification logic could go here if we wanted to keep purely string-based checks)
        
        return await self._run_pipeline(intent, intent.parameters)
