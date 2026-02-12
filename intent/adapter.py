"""
Intent Adapter — LLM-powered intent extraction via Ollama.

Responsibility:
- Convert input_text + history → structured Intent
- Temperature = 0, JSON-only output
- Pydantic schema validation
- No external calls beyond Ollama
- No calculations

Failure → abort flow.
"""

import json
import logging

import httpx
from pydantic import ValidationError

from shared.models import Intent

logger = logging.getLogger(__name__)

# Available actions that map to MCP Finance Server tools
AVAILABLE_ACTIONS = [
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
{json.dumps(AVAILABLE_ACTIONS, indent=2)}

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
    """Extracts structured Intent from user input using Ollama LLM."""

    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "llama3.1:8b"):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        # Persistent HTTP client with connection pooling
        self._client = httpx.Client(
            base_url=self.ollama_url,
            timeout=60.0,
            limits=httpx.Limits(max_keepalive_connections=2),
        )

    def extract(self, input_text: str, history: list[dict] | None = None) -> Intent:
        """
        Extract Intent from input_text + optional history.
        Raises ValueError on LLM failure or schema validation failure.
        """
        messages = self._build_messages(input_text, history)
        raw_response = self._call_ollama(messages)
        return self._parse_and_validate(raw_response)

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

    def _call_ollama(self, messages: list[dict]) -> str:
        """Call Ollama chat API with temperature=0, persistent connection."""
        try:
            response = self._client.post(
                "/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "keep_alive": "10m",
                    "options": {
                        "temperature": 0,
                        "num_predict": 512,
                    },
                    "format": "json",
                },
            )
            response.raise_for_status()
            data = response.json()
            content = data.get("message", {}).get("content", "")
            logger.debug("Ollama raw response: %s", content)
            return content
        except httpx.HTTPError as e:
            raise ValueError(f"Ollama API error: {e}") from e

    def _parse_and_validate(self, raw_response: str) -> Intent:
        """Parse JSON and validate against Intent schema."""
        # Try to extract JSON from response
        raw = raw_response.strip()

        # Handle potential markdown code blocks
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1])

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM returned invalid JSON: {e}\nRaw response: {raw_response}"
            ) from e

        try:
            intent = Intent(**parsed)
        except ValidationError as e:
            raise ValueError(
                f"LLM output failed schema validation: {e}\nParsed data: {parsed}"
            ) from e

        logger.info("Extracted intent: domain=%s action=%s confidence=%.2f",
                     intent.domain, intent.action, intent.confidence)
        return intent
