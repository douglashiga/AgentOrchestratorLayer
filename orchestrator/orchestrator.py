"""
Orchestrator — Stateless request router.

Responsibility:
- Resolve domain from Intent
- Delegate to DomainHandler
- Aggregate response

Prohibitions:
- No business logic
- No skill calls
- No state
- No MCP calls
"""

import logging

from shared.models import Decision, Intent

from registry.domain_registry import DomainRegistry

logger = logging.getLogger(__name__)


class Orchestrator:
    """Stateless orchestrator that resolves domains and delegates execution."""

    def __init__(self, domain_registry: DomainRegistry):
        self.domain_registry = domain_registry

    def process(self, intent: Intent) -> Decision:
        """
        Resolve the domain from the intent and delegate execution.
        Returns a Decision — never raises for business errors.
        """
        logger.info("Orchestrator processing: domain=%s action=%s", intent.domain, intent.action)

        # Resolve domain handler
        handler = self.domain_registry.resolve(intent.domain)
        if handler is None:
            return Decision(
                action=intent.action,
                success=False,
                error=f"Unknown domain: '{intent.domain}'",
                explanation=f"No handler registered for domain '{intent.domain}'.",
            )

        # Delegate to domain handler
        try:
            decision = handler.execute(intent)
            logger.info("Orchestrator received decision: action=%s success=%s",
                        decision.action, decision.success)
            return decision
        except Exception as e:
            logger.exception("Domain handler error for %s", intent.domain)
            return Decision(
                action=intent.action,
                success=False,
                error=str(e),
                explanation=f"Error in domain '{intent.domain}': {e}",
            )
