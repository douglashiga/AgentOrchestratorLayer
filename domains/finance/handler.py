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

logger = logging.getLogger(__name__)


class FinanceDomainHandler:
    """Finance domain handler — orchestrates context, skills, and strategy."""

    def __init__(self, skill_gateway: SkillGateway):
        self.context_resolver = ContextResolver()
        self.strategy_core = StrategyCore()
        self.skill_gateway = skill_gateway

    def execute(self, intent: IntentOutput) -> DomainOutput:
        """
        Execute the full finance domain flow:
        1. Clarify ambiguity (check missing params)
        2. Resolve DomainContext
        3. Fetch data via Skill Gateway
        4. Run Strategy Core
        5. Return DomainOutput
        """
        logger.info("Finance handler executing: capability=%s", intent.capability)

        if intent.capability == "chat":
            return DomainOutput(
                status="success",
                result={},
                explanation="This looks like a general question. Please ask a finance question."
            )

        # ─── 1. Clarification Logic ──────────────────────────────────
        if intent.capability in ("get_top_gainers", "get_top_losers", "get_top_dividends", "get_market_performance"):
            market = intent.parameters.get("market")
            if not market:
                return DomainOutput(
                    status="clarification",
                    result={},
                    explanation="Which market are you interested in? (e.g., US, Brazil, Sweden)",
                    confidence=1.0
                )

        if intent.capability == "compare_fundamentals":
            symbols = intent.parameters.get("symbols")
            if not symbols or len(symbols) < 2:
                 return DomainOutput(
                    status="clarification",
                    result={},
                    explanation="Please provide at least two companies to compare (e.g., 'Petrobras and Vale').",
                    confidence=1.0
                )

        # ─── 2. Context Resolution ───────────────────────────────────
        try:
            symbol = intent.parameters.get("symbol")
            if symbol:
                domain_context = self.context_resolver.resolve(symbol)
            elif intent.parameters.get("symbols"):
                 # Resolve context from first symbol for multi-symbol actions
                 domain_context = self.context_resolver.resolve(intent.parameters["symbols"][0])
            elif intent.parameters.get("market"):
                 # Resolve context from market code
                 domain_context = self.context_resolver.get_market_profile(intent.parameters["market"])
                 if not domain_context:
                      domain_context = self.context_resolver.resolve("DEFAULT") 
            else:
                 domain_context = self.context_resolver.resolve("DEFAULT")

            logger.info("Context resolved: %s (%s)", domain_context.market, domain_context.currency)

            # ─── 3. Skill Execution ──────────────────────────────────
            # Map capability (intent) -> action (skill)
            skill_params = {**intent.parameters, "_action": intent.capability}
            
            # Inject context into params if needed (e.g. market code)
            if not skill_params.get("market") and domain_context.market != "US":
                skill_params["market"] = domain_context.market

            skill_data = self.skill_gateway.execute("mcp_finance", skill_params)
            logger.debug("Skill data received: success=%s", skill_data.get("success"))

            # ─── 4. Strategy Execution ───────────────────────────────
            execution_context = ExecutionContext(
                domain_context=domain_context,
                skill_data=skill_data,
            )

            decision = self.strategy_core.execute(intent, execution_context)
            
            # ─── 5. Output Generation ────────────────────────────────
            metadata = {"risk_metrics": decision.risk_metrics}
            if decision.error:
                metadata["error"] = decision.error

            return DomainOutput(
                status="success" if decision.success else "failure",
                result=decision.result,
                explanation=decision.explanation,
                confidence=1.0,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Finance handler failed: {e}", exc_info=True)
            return DomainOutput(
                status="failure",
                result={},
                explanation=f"Error executing finance action: {str(e)}",
                confidence=0.0,
                metadata={"error": str(e)}
            )
