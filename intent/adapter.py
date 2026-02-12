"""
Intent Adapter — LLM-powered intent extraction.

Responsibility:
- Convert input_text + history → structured IntentOutput
- Temperature = 0, JSON-only output
- Pydantic schema validation via Model Layer
- No external calls beyond ModelSelector
- No calculations

Failure → abort flow.
"""

import logging
import json
from typing import Any

from observability.logger import Observability
from models.selector import ModelSelector
from shared.models import IntentOutput, ModelPolicy

logger = logging.getLogger(__name__)

class IntentAdapter:
    """Extracts structured Intent from user input using Model Layer."""

    def __init__(self, model_selector: ModelSelector, initial_capabilities: list[str] | None = None):
        self.model_selector = model_selector
        self.capabilities = initial_capabilities or []
        self.policy = ModelPolicy(
            model_name="llama3.1:8b",
            temperature=0.0,
            timeout_seconds=10.0,  # Fast timeout for intent
            max_retries=3,
            json_mode=True,
        )

    def update_capabilities(self, capabilities: list[str]) -> None:
        """Update the list of capabilities known to the LLM."""
        self.capabilities = capabilities
        logger.info("IntentAdapter updated with %d capabilities", len(capabilities))

    def extract(self, input_text: str, history: list[dict] | None = None, session_id: str | None = None) -> IntentOutput:
        """
        Extract Intent from input_text + optional history.
        Uses Model Layer for reliable execution.
        """
        messages = self._build_messages(input_text, history)
        
        try:
            # We pass session_id to generate for observability
            raw_data = self.model_selector.generate(
                messages=messages,
                policy=self.policy,
                session_id=session_id
            )
            
            # Additional strict validation
            if not isinstance(raw_data, dict):
                raise ValueError("Model returned non-dict JSON")

            # Map to internal IntentOutput schema
            return IntentOutput(
                domain=raw_data.get("domain", "general"),
                capability=raw_data.get("action", "chat"), 
                confidence=float(raw_data.get("confidence", 0.0)),
                parameters=raw_data.get("parameters", {}),
                original_query=input_text
            )
            
        except Exception as e:
            # Fallback for critical failures
            logger.error("Intent extraction failed: %s", e)
            return IntentOutput(
                domain="general",
                capability="chat",
                confidence=0.0,
                parameters={"message": input_text},
                original_query=input_text
            )

    def _build_system_prompt(self) -> str:
        """Build the system prompt dynamically based on current capabilities."""
        return f"""You are an intent extraction engine. Your ONLY job is to analyze the user's message and return a structured JSON object.

You MUST respond with ONLY a valid JSON object. No explanations, no markdown, no extra text.

The JSON must follow this exact schema:
{{
  "domain": "<domain>",
  "action": "<action>",
  "parameters": {{}},
  "confidence": <float 0.0-1.0>
}}

There are TWO domains:

1. "general" — for greetings, casual conversation, questions about yourself, help requests, or anything NOT related to finance/stocks/markets.
   - action: "chat"
   - parameters: {{"message": "<the user's message>"}}
   - Examples: "oi", "hello", "como funciona?", "me ajuda", "quem é você?", "obrigado"

2. "finance" — for anything related to stocks, markets, prices, options, fundamentals, dividends, company info, financial data.
   Available actions:
{json.dumps(self.capabilities, indent=2)}

Rules:
1. FIRST decide the domain: if the message is about stocks, markets, prices, finance → "finance". Otherwise → "general".
2. Symbol detection:
   - For Brazilian stocks (Bovespa), ALWAYS append ".SA" to the symbol (e.g., "PETR4.SA", "VALE3.SA").
   - For Swedish stocks (Stockholm), ALWAYS append ".ST" (e.g., "NDA.ST", "VOLV-B.ST").
   - For US stocks, use the symbol as is (e.g., "AAPL", "TSLA").
   - If a company is Brazilian (like Vale, Petrobras, Itaú) but the user didn't specify the market, use ".SA".
3. Confidence Scenarios:
   - Set "confidence": 0.99 ONLY if you are absolutely certain about the action and symbols.
   - Set "confidence": 0.90 or lower if there is ANY ambiguity (e.g., you are guessing a ticker name, or the user's query is vague).
   - If the query is "qual o preco da vale?", domain is "finance", action is "get_stock_price", symbol is "VALE3.SA", confidence is 0.99.
4. If the domain is finance but unsure about the action, use "yahoo_search" with the user's query.
5. FOLLOW-UP / CONTEXT AWARENESS (CRITICAL):
   - Look at the CHAT HISTORY.
   - If the LAST message from the Assistant was a question (e.g., "Which market?", "Qual mercado?"), the User's input is likely the ANSWER to that question.
   - You MUST combine the answer with the implied intent from the previous turn.
   - Example:
     Assistant: "Qual mercado você prefere? (US, BR, SE)"
     User: "BR"
     -> Domain: "finance", Action: "get_top_gainers" (inferred from context), Parameters: {{"market": "BR"}}
     -> DO NOT return "chat" domain for "BR" if it answers a finance question.

6. NEVER add explanations. Return ONLY the JSON object.
"""

    def _build_messages(self, input_text: str, history: list[dict] | None = None) -> list[dict]:
        """Build message list for Ollama chat API."""
        messages = [{"role": "system", "content": self._build_system_prompt()}]

        # Add history context (last few turns only)
        if history:
            for turn in history[-6:]:  # Limit to last 6 turns
                messages.append({
                    "role": turn.get("role", "user"),
                    "content": turn.get("content", ""),
                })

        messages.append({"role": "user", "content": input_text})
        return messages
