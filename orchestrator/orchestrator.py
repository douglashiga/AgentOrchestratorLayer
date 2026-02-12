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

from shared.models import DomainOutput, IntentOutput

from registry.domain_registry import HandlerRegistry

logger = logging.getLogger(__name__)


class Orchestrator:
    """Stateless orchestrator that resolves domains and delegates execution."""

    def __init__(self, domain_registry: HandlerRegistry):
        self.domain_registry = domain_registry

    def process(self, intent: IntentOutput) -> DomainOutput:
        """
        Resolve the domain from the intent and delegate execution.
        Returns a DomainOutput — never raises for business errors.
        """
        logger.info("Orchestrator processing: domain=%s capability=%s", intent.domain, intent.capability)

        # 1. Try to resolve by specific Capability
        handler = self.domain_registry.resolve_capability(intent.capability)
        
        # 2. Fallback: Resolve by Domain (legacy/broad)
        if handler is None:
            handler = self.domain_registry.resolve_domain(intent.domain)

        if handler is None:
            return DomainOutput(
                status="failure",
                result={},
                explanation=f"No handler registered for capability '{intent.capability}' or domain '{intent.domain}'.",
                confidence=1.0,
                metadata={"error": f"Unknown capability/domain: {intent.capability}/{intent.domain}"}
            )

        # Delegate to domain handler
        try:
            domain_output = handler.execute(intent)
            logger.info("Orchestrator received output: status=%s", domain_output.status)
            return domain_output
        except Exception as e:
            logger.exception("Domain handler error for %s", intent.domain)
            return DomainOutput(
                status="failure",
                result={},
                explanation=f"Error in domain '{intent.domain}': {e}",
                confidence=0.0,
                metadata={"error": str(e)}
            )
