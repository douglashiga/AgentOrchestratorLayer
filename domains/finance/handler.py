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
        1. Resolve DomainContext (deterministic)
        2. Fetch data via Skill Gateway
        3. Run Strategy Core (deterministic)
        4. Return DomainOutput
        """
        logger.info("Finance handler executing: capability=%s", intent.capability)

        # Step 1: Resolve domain context from symbol
        symbol = intent.parameters.get("symbol")
        domain_context = self.context_resolver.resolve(symbol)
        logger.debug("Resolved context: market=%s currency=%s", domain_context.market, domain_context.currency)

        # Step 2: Fetch data via skill gateway
        # Map capability (intent) -> action (skill)
        skill_params = {**intent.parameters, "_action": intent.capability}
        skill_data = self.skill_gateway.execute("mcp_finance", skill_params)
        logger.debug("Skill data received: success=%s", skill_data.get("success"))

        # Step 3: Build execution context (immutable, never persisted)
        execution_context = ExecutionContext(
            domain_context=domain_context,
            skill_data=skill_data,
        )

        # Step 4: Run strategy core (deterministic)
        decision = self.strategy_core.execute(intent, execution_context)
        
        # Step 5: Convert Decision -> DomainOutput
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
