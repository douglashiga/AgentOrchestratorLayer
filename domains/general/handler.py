"""
General Domain Handler — Conversational responses.

Responsibility:
- Handle non-finance interactions (greetings, help, general chat)
- Use Model Layer for natural conversation

Performance:
- Uses ModelSelector for connection pooling, retries, and reliability
"""

import logging

from models.selector import ModelSelector
from shared.models import Decision, DomainOutput, IntentOutput, ModelPolicy

logger = logging.getLogger(__name__)


class GeneralDomainHandler:
    """Handles general conversation using Ollama LLM via Model Layer."""

    def __init__(self, model_selector: ModelSelector, model_name: str = "qwen2.5-coder:32b"):
        self.model_selector = model_selector
        self.max_tokens = 256
        self.policy = ModelPolicy(
            model_name=model_name,
            temperature=0.7,
            timeout_seconds=60.0,
            max_retries=2,
            json_mode=False, # Chat returns text, not JSON
        )

    def execute(self, intent: IntentOutput) -> DomainOutput:
        """Generate a conversational response."""
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
            return DomainOutput(
                status="failure",
                result={},
                explanation="Failed to generate response.",
                confidence=0.0,
                metadata={"error": str(e)}
            )

    def _generate_response(self, user_message: str) -> str:
        """Call Model Selector."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a friendly financial assistant. "
                    "You can help users with stock market questions, financial data, and general conversation. "
                    "Be concise, helpful, and friendly. "
                    "If the user greets you, greet them back and briefly explain what you can do. "
                    "Respond in the same language the user uses."
                ),
            },
            {"role": "user", "content": user_message},
        ]
        
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
