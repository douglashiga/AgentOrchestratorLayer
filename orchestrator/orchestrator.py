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

from shared.models import DomainOutput, IntentOutput, ModelPolicy
from registry.domain_registry import HandlerRegistry
from models.selector import ModelSelector

logger = logging.getLogger(__name__)


class Orchestrator:
    """Stateless orchestrator that resolves domains and delegates execution."""

    def __init__(self, domain_registry: HandlerRegistry, model_selector: ModelSelector):
        self.domain_registry = domain_registry
        self.model_selector = model_selector
        self.clarification_policy = ModelPolicy(
            model_name="llama3.1:8b",
            temperature=0.7,
            timeout_seconds=5.0,
            max_retries=1,
            json_mode=False
        )

    async def process(self, intent: IntentOutput) -> DomainOutput:
        """
        Resolve the domain from the intent and delegate execution.
        Returns a DomainOutput — never raises for business errors.
        """
        logger.info("Orchestrator processing: domain=%s capability=%s (conf=%.2f)", 
                    intent.domain, intent.capability, intent.confidence)

        # 1. Confidence Gating (User Requirement: 98% threshold)
        CONFIDENCE_THRESHOLD = 0.98
        if intent.confidence < CONFIDENCE_THRESHOLD:
            logger.warning("Intent confidence too low: %.2f < %.2f", intent.confidence, CONFIDENCE_THRESHOLD)
            
            question = self._generate_clarification_question(intent)
            
            return DomainOutput(
                status="clarification",
                result={},
                explanation=question,
                confidence=intent.confidence
            )

        # 2. Try to resolve by specific Capability
        handler = self.domain_registry.resolve_capability(intent.capability)
        
        # 3. Fallback: Resolve by Domain (legacy/broad)
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

    def _generate_clarification_question(self, intent: IntentOutput) -> str:
        """Generate a natural language clarification question."""
        try:
            # Fetch capability metadata/schema for context
            metadata = self.domain_registry.get_metadata(intent.capability)
            schema = metadata.get("schema", {})
            description = metadata.get("description", "")
            valid_values = metadata.get("valid_values", {})

            prompt = (
                f"You are a helpful financial assistant. The user wants to perform '{intent.capability}'.\n"
                f"Context:\n"
                f"- User's Original Input: '{intent.original_query}'\n"
                f"- Extracted Params: {intent.parameters}\n"
                f"- Schema: {schema}\n"
                f"- Valid Choices: {valid_values}\n"
                f"- Current Confidence: {intent.confidence:.0%} (Target: >98%)\n\n"
                "OBJECTIVE: Ask a SHORT clarification question to get the missing information.\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. **LANGUAGE**: Output the question IN THE SAME LANGUAGE as the 'User's Original Input'.\n"
                "   - If input is Portuguese -> Question MUST be Portuguese.\n"
                "   - If input is English -> Question MUST be English.\n"
                "2. **NO PREAMBLE**: Output ONLY the question text. Do not say 'Here is the question'.\n"
                "3. **BE SPECIFIC**: Ask for the missing parameter explicitly using the Valid Choices.\n"
                "4. Example (if input is PT): 'Para listar as maiores altas, qual mercado você prefere? (US, BR, SE)'\n"
                "Question:"
            )
            
            messages = [{"role": "user", "content": prompt}]
            response = self.model_selector.generate(messages, self.clarification_policy)
            return str(response).strip()
        except Exception as e:
            logger.warning("Failed to generate clarification: %s", e)
            return (
                f"I'm not quite sure I understood (Confidence: {intent.confidence:.1%}). "
                "Could you please be more specific?"
            )
