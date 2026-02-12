"""
General Domain Handler — Conversational responses.

Responsibility:
- Handle non-finance interactions (greetings, help, general chat)
- Use Ollama LLM for natural conversation

Performance:
- Persistent httpx.Client with connection pooling
- keep_alive to keep model loaded in Ollama memory

This domain is an exception to the "no LLM in domains" rule because
general conversation IS the core logic here (no calculations involved).
"""

import logging

import httpx

from shared.models import Decision, Intent

logger = logging.getLogger(__name__)


class GeneralDomainHandler:
    """Handles general conversation using Ollama LLM."""

    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "qwen2.5-coder:32b"):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        # Persistent HTTP client with connection pooling
        self._client = httpx.Client(
            base_url=self.ollama_url,
            timeout=60.0,
            limits=httpx.Limits(max_keepalive_connections=2),
        )

    def execute(self, intent: Intent) -> Decision:
        """Generate a conversational response."""
        user_message = intent.parameters.get("message", "")

        try:
            response_text = self._generate_response(user_message)
            return Decision(
                action="chat",
                result={"response": response_text},
                explanation=response_text,
                success=True,
            )
        except Exception as e:
            logger.error("General domain error: %s", e)
            return Decision(
                action="chat",
                success=False,
                error=str(e),
                explanation="Failed to generate response.",
            )

    def _generate_response(self, user_message: str) -> str:
        """Call Ollama with persistent connection and keep_alive."""
        try:
            response = self._client.post(
                "/api/chat",
                json={
                    "model": self.model,
                    "messages": [
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
                    ],
                    "stream": False,
                    "keep_alive": "10m",
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 256,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "").strip()
        except httpx.HTTPError as e:
            raise ValueError(f"Ollama API error: {e}") from e
