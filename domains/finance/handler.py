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

from shared.models import Decision, ExecutionContext, Intent
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

    def execute(self, intent: Intent) -> Decision:
        """
        Execute the full finance domain flow:
        1. Resolve DomainContext (deterministic)
        2. Fetch data via Skill Gateway
        3. Run Strategy Core (deterministic)
        4. Return Decision
        """
        logger.info("Finance handler executing: action=%s", intent.action)

        # Step 1: Resolve domain context from symbol
        symbol = intent.parameters.get("symbol")
        domain_context = self.context_resolver.resolve(symbol)
        logger.debug("Resolved context: market=%s currency=%s", domain_context.market, domain_context.currency)

        # Step 2: Fetch data via skill gateway
        skill_params = {**intent.parameters, "_action": intent.action}
        skill_data = self.skill_gateway.execute("mcp_finance", skill_params)
        logger.debug("Skill data received: success=%s", skill_data.get("success"))

        # Step 3: Build execution context (immutable, never persisted)
        execution_context = ExecutionContext(
            domain_context=domain_context,
            skill_data=skill_data,
        )

        # Step 4: Run strategy core (deterministic)
        decision = self.strategy_core.execute(intent, execution_context)
        return decision
