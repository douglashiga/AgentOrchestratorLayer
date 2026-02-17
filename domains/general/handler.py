"""
General Domain Handler — Conversational responses.

Responsibility:
- Handle non-finance interactions (greetings, help, general chat)
- Use Model Layer for natural conversation

Performance:
- Uses ModelSelector for connection pooling, retries, and reliability
"""

import logging
from collections import defaultdict
from typing import Any, Callable

from models.selector import ModelSelector
from shared.models import Decision, DomainOutput, ExecutionIntent, IntentOutput, ModelPolicy

logger = logging.getLogger(__name__)


class GeneralDomainHandler:
    """Handles general conversation using Ollama LLM via Model Layer."""

    SYSTEM_PROMPT = (
        "You are a friendly financial assistant. "
        "You can help users with stock market questions, financial data, and general conversation. "
        "Be concise, helpful, and friendly. "
        "If the user greets you, greet them back and briefly explain what you can do. "
        "Respond in the same language the user uses."
    )

    def __init__(
        self,
        model_selector: ModelSelector,
        model_name: str = "qwen2.5-coder:32b",
        capability_catalog_provider: Callable[[], list[dict[str, Any]]] | None = None,
    ):
        self.model_selector = model_selector
        self.capability_catalog_provider = capability_catalog_provider
        self.max_tokens = 256
        self.policy = ModelPolicy(
            model_name=model_name,
            temperature=0.7,
            timeout_seconds=60.0,
            max_retries=2,
            json_mode=False, # Chat returns text, not JSON
        )

    async def execute(self, intent: IntentOutput | ExecutionIntent) -> DomainOutput:
        """Generate a conversational response."""
        if intent.capability == "list_capabilities":
            response_text = self._build_capability_listing()
            return DomainOutput(
                status="success",
                result={"response": response_text},
                explanation=response_text,
                confidence=1.0,
            )

        user_message = intent.parameters.get("message", "")

        try:
            # We generate a session_id on the fly or could assume one from context if available
            response_text = self._generate_response(user_message)
            return DomainOutput(
                status="success",
                result={"response": response_text},
                explanation=response_text,
                confidence=1.0,
            )
        except Exception as e:
            logger.error("General domain error: %s", e)
            fallback_text = self._fallback_response(user_message)
            return DomainOutput(
                status="success",
                result={"response": fallback_text},
                explanation=fallback_text,
                confidence=0.5,
                metadata={"error": str(e)}
            )

    def _fallback_response(self, user_message: str) -> str:
        question = (user_message or "").strip()
        if question:
            return (
                "Estou com instabilidade no chat geral agora. "
                "Tente novamente em instantes, ou reformule sua pergunta em modo objetivo."
            )
        return "Estou com instabilidade no chat geral agora. Tente novamente em instantes."

    def _build_capability_listing(self) -> str:
        """Build deterministic domain/capability listing from runtime registry data."""
        if not self.capability_catalog_provider:
            return "No capability catalog available right now."

        try:
            catalog = self.capability_catalog_provider() or []
        except Exception as e:
            logger.error("Failed to fetch capability catalog: %s", e)
            return "I couldn't load the capability catalog right now."

        if not catalog:
            return "No capabilities are currently registered."

        grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for item in catalog:
            domain = str(item.get("domain", "")).strip() or "unknown"
            capability = str(item.get("capability", "")).strip()
            description = str(item.get("description", "")).strip()
            if capability:
                grouped[domain].append((capability, description))

        lines = ["Here is what I can do by domain:"]
        for domain in sorted(grouped.keys()):
            lines.append(f"- {domain}:")
            for capability, description in sorted(grouped[domain], key=lambda x: x[0]):
                if description:
                    lines.append(f"  - {capability}: {description}")
                else:
                    lines.append(f"  - {capability}")

        return "\n".join(lines)

    def _generate_response(self, user_message: str) -> str:
        """Call Model Selector."""
        messages = self.build_chat_messages(user_message)
        
        # Determine strict output type (str)
        response = self.model_selector.generate(
            messages=messages,
            policy=self.policy
        )
        
        if isinstance(response, str):
            return response.strip()
        else:
            # Should not happen as json_mode=False
            return str(response)

    def build_chat_messages(self, user_message: str) -> list[dict[str, str]]:
        """Expose the exact conversational prompt/messages used by this handler."""
        return [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT,
            },
            {"role": "user", "content": user_message},
        ]
