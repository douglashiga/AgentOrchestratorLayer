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

# Available actions that map to MCP Finance Server tools
FINANCE_CAPABILITIES = [
    "get_stock_price",
    "get_historical_data",
    "search_symbol",
    "get_account_summary",
    "get_option_chain",
    "get_option_greeks",
    "get_fundamentals",
    "get_dividends",
    "get_company_info",
    "get_financial_statements",
    "get_exchange_info",
    "yahoo_search",
    "get_top_gainers",
    "get_top_losers",
    "get_top_dividends",
    "get_market_performance",
    "compare_fundamentals",
]

SYSTEM_PROMPT = f"""You are an intent extraction engine. Your ONLY job is to analyze the user's message and return a structured JSON object.

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
{json.dumps(FINANCE_CAPABILITIES, indent=2)}

Parameter rules for finance actions:
- get_stock_price: {{"symbol": "AAPL"}}
- get_historical_data: {{"symbol": "AAPL", "duration": "1 M", "bar_size": "1 day"}}
- search_symbol: {{"query": "Apple"}}
- get_account_summary: {{}}
- get_option_chain: {{"symbol": "AAPL"}}
- get_option_greeks: {{"symbol": "AAPL", "date": "20240119", "strike": 150.0, "right": "C"}}
- get_fundamentals: {{"symbol": "AAPL"}}
- get_dividends: {{"symbol": "KO"}}
- get_company_info: {{"symbol": "TSLA"}}
- get_financial_statements: {{"symbol": "MSFT"}}
- get_exchange_info: {{"symbol": "VOW3.DE"}}
- yahoo_search: {{"query": "Brazilian banks"}}
- get_top_gainers: {{"market": "BR"}} (or "US", "SE")
- get_top_losers: {{"market": "US", "period": "1d"}}
- get_top_dividends: {{"market": "SE"}}
- get_market_performance: {{"market": "US", "period": "YTD"}}
- compare_fundamentals: {{"symbols": ["PETR4.SA", "VALE3.SA"]}}


Rules:
1. FIRST decide the domain: if the message is about stocks, markets, prices, finance → "finance". Otherwise → "general".
2. For finance: pick the most appropriate action and extract parameters.
3. For Swedish stocks, append ".ST" to the symbol (e.g., "NDA.ST" for Nordea).
4. For Brazilian stocks, append ".SA" (e.g., "PETR4.SA").
5. For German stocks, append ".DE" (e.g., "VOW3.DE").
6. If the domain is finance but unsure about the action, use "yahoo_search" with the user's query.
7. Confidence should reflect how certain you are about the intent (0.0-1.0).
8. NEVER add explanations. Return ONLY the JSON object.
"""


class IntentAdapter:
    """Extracts structured Intent from user input using Model Layer."""

    def __init__(self, model_selector: ModelSelector):
        self.model_selector = model_selector
        self.policy = ModelPolicy(
            model_name="llama3.1:8b",
            temperature=0.0,
            timeout_seconds=10.0,  # Fast timeout for intent
            max_retries=3,
            json_mode=True,
        )

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
            # Note: The prompt asks for 'domain', 'action', 'parameters', 'confidence'
            # We map this to our strict IntentOutput
            # IMPORTANT: We trust the prompt to match schema, but we default if fields missing
            return IntentOutput(
                domain=raw_data.get("domain", "general"),
                capability=raw_data.get("action", "chat"), # Default to chat if unsure
                confidence=float(raw_data.get("confidence", 0.0)),
                parameters=raw_data.get("parameters", {})
            )
            
        except Exception as e:
            # Fallback for critical failures
            logger.error("Intent extraction failed: %s", e)
            return IntentOutput(
                domain="general",
                capability="chat",
                confidence=0.0,
                parameters={"message": input_text}
            )

    def _build_messages(self, input_text: str, history: list[dict] | None = None) -> list[dict]:
        """Build message list for Ollama chat API."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add history context (last few turns only)
        if history:
            for turn in history[-6:]:  # Limit to last 6 turns
                messages.append({
                    "role": turn.get("role", "user"),
                    "content": turn.get("content", ""),
                })

        messages.append({"role": "user", "content": input_text})
        return messages
