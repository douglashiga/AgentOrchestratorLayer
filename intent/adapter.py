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
import re
from typing import Any

from models.selector import ModelSelector
from shared.models import IntentOutput, ModelPolicy

logger = logging.getLogger(__name__)

class IntentAdapter:
    """Extracts structured Intent from user input using Model Layer."""

    def __init__(
        self,
        model_selector: ModelSelector,
        initial_capabilities: list[str] | None = None,
        capability_catalog: list[dict[str, Any]] | None = None,
    ):
        self.model_selector = model_selector
        self.capabilities = initial_capabilities or []
        self.capability_catalog = capability_catalog or []
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

    def update_capability_catalog(self, capability_catalog: list[dict[str, Any]]) -> None:
        """Update runtime domain/capability catalog for dynamic prompt generation."""
        self.capability_catalog = capability_catalog
        logger.info("IntentAdapter updated with %d catalog entries", len(capability_catalog))

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
            intent = IntentOutput(
                domain=raw_data.get("domain", "general"),
                capability=raw_data.get("action", "chat"), 
                confidence=float(raw_data.get("confidence", 0.0)),
                parameters=raw_data.get("parameters", {}),
                original_query=input_text
            )
            return self._apply_post_extraction_guard(intent, input_text)
            
        except Exception as e:
            # Fallback for critical failures
            logger.error("Intent extraction failed: %s", e)
            deterministic = self._deterministic_fallback(input_text)
            if deterministic:
                return self._enrich_intent_from_query(deterministic, input_text)
            fallback_intent = IntentOutput(
                domain="general",
                capability="chat",
                confidence=0.0,
                parameters={"message": input_text},
                original_query=input_text
            )
            return self._enrich_intent_from_query(fallback_intent, input_text)

    def _deterministic_fallback(self, input_text: str) -> IntentOutput | None:
        """
        Deterministic rescue path when model extraction fails.
        Keeps orchestrator clean and avoids dropping obvious finance requests to general chat.
        """
        text = (input_text or "").strip()
        text_l = text.lower()
        finance_keywords = ("preco", "preço", "cotacao", "cotação", "valor", "price", "ticker", "acao", "ação")
        if not any(k in text_l for k in finance_keywords):
            return None

        if not self._has_capability("finance", "get_stock_price"):
            return None

        symbol = self._extract_symbol_from_text(text)
        params: dict[str, Any] = {}
        confidence = 0.85
        if symbol:
            params["symbol"] = symbol
            confidence = 0.95

        return IntentOutput(
            domain="finance",
            capability="get_stock_price",
            confidence=confidence,
            parameters=params,
            original_query=input_text,
        )

    def _apply_post_extraction_guard(self, intent: IntentOutput, input_text: str) -> IntentOutput:
        """
        If model output drifts to general/chat for obvious price queries,
        force deterministic finance intent.
        """
        if intent.domain == "general" and intent.capability == "chat":
            deterministic = self._deterministic_fallback(input_text)
            if deterministic:
                return deterministic
        return self._enrich_intent_from_query(intent, input_text)

    def _enrich_intent_from_query(self, intent: IntentOutput, input_text: str) -> IntentOutput:
        """
        Deterministically enrich intent with cross-domain execution hints inferred
        from the user sentence (e.g. notify/send/share after primary action).
        """
        params = dict(intent.parameters or {})
        notify_detected = self._infer_notify_from_text(input_text)
        if notify_detected:
            params["notify"] = True
            # For explicit send/share requests, avoid unnecessary soft confirmation.
            if intent.domain != "general":
                return intent.model_copy(update={"parameters": params, "confidence": max(intent.confidence, 0.95)})
        return intent.model_copy(update={"parameters": params})

    def _infer_notify_from_text(self, input_text: str) -> bool:
        text = (input_text or "").strip().lower()
        if not text:
            return False

        send_keywords = (
            "envie",
            "enviar",
            "manda",
            "mandar",
            "notifique",
            "notificar",
            "me avise",
            "compartilhe",
            "compartilhar",
            "telegram",
        )
        return any(token in text for token in send_keywords)

    def _has_capability(self, domain: str, capability: str) -> bool:
        if self.capability_catalog:
            return any(
                str(item.get("domain", "")).strip() == domain
                and str(item.get("capability", "")).strip() == capability
                for item in self.capability_catalog
            )
        return capability in self.capabilities

    def _extract_symbol_from_text(self, text: str) -> str | None:
        upper = (text or "").upper()
        if not upper:
            return None

        explicit = re.findall(r"\b([A-Z0-9-]{1,12}\.[A-Z]{1,4})\b", upper)
        if explicit:
            return explicit[0]

        b3 = re.findall(r"\b([A-Z]{4}(?:3|4|5|6|11)(?:F)?)\b", upper)
        if b3:
            return f"{b3[0]}.SA"

        quasi_ticker = re.findall(r"\b([A-Z]{3,8}\d{1,2}[A-Z]?)\b", upper)
        if quasi_ticker:
            token = quasi_ticker[0]
            m = re.fullmatch(r"([A-Z]{5,8})(\d{1,2}[A-Z]?)", token)
            if m:
                letters, suffix = m.groups()
                if suffix.startswith(("3", "4", "5", "6", "11")):
                    compact = f"{letters[:4]}{suffix}"
                    if re.fullmatch(r"[A-Z]{4}(?:3|4|5|6|11)(?:F)?", compact):
                        return f"{compact}.SA"
            return token

        us = re.findall(r"\b([A-Z]{1,5})\b", text)
        for token in us:
            if token.isupper() and token.lower() not in {"qual", "valor", "preco", "price"}:
                return token
        return None

    def _render_capability_catalog(self) -> str:
        """Render domain/action catalog from runtime registry data."""
        if not self.capability_catalog:
            return ""

        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in self.capability_catalog:
            domain = str(item.get("domain", "")).strip() or "unknown"
            grouped.setdefault(domain, []).append(item)

        lines: list[str] = []
        for domain in sorted(grouped.keys()):
            lines.append(f"- Domain '{domain}':")
            for cap in sorted(grouped[domain], key=lambda x: str(x.get("capability", ""))):
                cap_name = str(cap.get("capability", "")).strip()
                cap_desc = str(cap.get("description", "")).strip() or "No description provided."
                lines.append(f"  - action '{cap_name}': {cap_desc}")
        return "\n".join(lines)

    def _build_system_prompt(self) -> str:
        """Build the system prompt dynamically from the runtime catalog when available."""
        catalog_block = self._render_capability_catalog()
        if catalog_block:
            has_list_capability = any(
                str(item.get("domain", "")).strip() == "general"
                and str(item.get("capability", "")).strip() == "list_capabilities"
                for item in self.capability_catalog
            )
            discovery_rule = (
                '2. For greetings/casual conversation/help: use domain "general" and action "chat".\n'
                '3. If user asks "what can you do", "list capabilities", or "which domains": use domain "general" and action "list_capabilities".\n'
                if has_list_capability
                else '2. For greetings, casual conversation, help, or "what can you do?" questions: use domain "general" and action "chat".\n'
            )
            return f"""You are an intent extraction engine. Your ONLY job is to analyze the user's message and return a structured JSON object.

You MUST respond with ONLY a valid JSON object. No explanations, no markdown, no extra text.

The JSON must follow this exact schema:
{{
  "domain": "<domain>",
  "action": "<action>",
  "parameters": {{}},
  "confidence": <float 0.0-1.0>
}}

Runtime domain/action catalog:
{catalog_block}

Rules:
1. Use only domains/actions listed above.
{discovery_rule}4. If confidence is not high, return confidence <= 0.90.
5. If the user answer is a short follow-up to an assistant question, infer the intended domain/action from chat history.
6. If the user asks to deliver/send/notify the result after executing an action, keep the primary action and set parameters.notify = true.
7. Return ONLY the JSON object.
"""

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

6. If the user asks to send/notify the output after running an action, keep the primary finance action and include parameters.notify = true.
7. NEVER add explanations. Return ONLY the JSON object.
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
