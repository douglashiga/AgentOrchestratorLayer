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
import inspect
import os

from shared.models import DomainOutput, ExecutionIntent, IntentOutput
from registry.domain_registry import HandlerRegistry
from models.selector import ModelSelector

logger = logging.getLogger(__name__)


class Orchestrator:
    """Stateless orchestrator that resolves domains and delegates execution."""

    def __init__(self, domain_registry: HandlerRegistry, model_selector: ModelSelector):
        self.domain_registry = domain_registry
        self.model_selector = model_selector
        self.confidence_threshold = float(
            os.getenv(
                "ORCHESTRATOR_CONFIDENCE_THRESHOLD",
                os.getenv("SOFT_CONFIRM_THRESHOLD", "0.94"),
            )
        )
        self.test_mode_enabled = os.getenv("ORCHESTRATOR_TEST_MODE", "false").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    async def process(self, intent: IntentOutput | ExecutionIntent) -> DomainOutput:
        """
        Resolve the domain from the intent and delegate execution.
        Returns a DomainOutput — never raises for business errors.
        """
        logger.info("Orchestrator processing: domain=%s capability=%s (conf=%.2f)", 
                    intent.domain, intent.capability, intent.confidence)
        test_mode = self._is_test_mode(intent)

        # 1. Confidence Gating
        # Keep general-domain responses responsive even when fallback confidence is low.
        is_general_domain = intent.domain == "general"
        if intent.confidence < self.confidence_threshold and not is_general_domain and not test_mode:
            logger.warning(
                "Intent confidence too low: %.2f < %.2f",
                intent.confidence,
                self.confidence_threshold,
            )

            return DomainOutput(
                status="clarification",
                result={},
                explanation=(
                    f"Não tenho confiança suficiente para executar '{intent.capability}'. "
                    "Pode confirmar os parâmetros principais?"
                ),
                confidence=intent.confidence
            )

        # 2. Try to resolve by specific Capability
        handler = self.domain_registry.resolve_capability(intent.capability)
        resolution_path = "capability"
        
        # 3. Fallback: Resolve by Domain (legacy/broad)
        if handler is None:
            handler = self.domain_registry.resolve_domain(intent.domain)
            resolution_path = "domain"

        if handler is None:
            return DomainOutput(
                status="failure",
                result={},
                explanation=f"No handler registered for capability '{intent.capability}' or domain '{intent.domain}'.",
                confidence=1.0,
                metadata={"error": f"Unknown capability/domain: {intent.capability}/{intent.domain}"}
            )

        if test_mode:
            handler_name = handler.__class__.__name__
            return DomainOutput(
                status="success",
                result={
                    "test_mode": True,
                    "route": {
                        "resolution_path": resolution_path,
                        "domain": intent.domain,
                        "capability": intent.capability,
                        "handler_class": handler_name,
                    },
                    "intent": intent.model_dump(mode="json"),
                },
                explanation=(
                    f"Test mode: roteamento válido até domain handler '{handler_name}' "
                    f"via {resolution_path}. Execução de domínio foi pulada."
                ),
                confidence=1.0,
                metadata={
                    "test_mode": True,
                    "test_mode_stage": "domain_routed",
                    "resolution_path": resolution_path,
                },
            )

        # Delegate to domain handler
        try:
            handler_result = handler.execute(intent)
            if inspect.isawaitable(handler_result):
                domain_output = await handler_result
            else:
                domain_output = handler_result
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

    def _is_test_mode(self, intent: IntentOutput | ExecutionIntent) -> bool:
        if self.test_mode_enabled:
            return True
        params = intent.parameters or {}
        return bool(params.get("_test_mode") is True)
