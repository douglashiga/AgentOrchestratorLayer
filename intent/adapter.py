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
import os
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
        intent_timeout_seconds = float(os.getenv("INTENT_MODEL_TIMEOUT_SECONDS", "20"))
        intent_max_retries = int(os.getenv("INTENT_MODEL_MAX_RETRIES", "3"))
        intent_model_name = os.getenv(
            "INTENT_MODEL_NAME",
            os.getenv("OLLAMA_INTENT_MODEL", "llama3.1:8b"),
        ).strip() or "llama3.1:8b"
        self.policy = ModelPolicy(
            model_name=intent_model_name,
            temperature=0.0,
            timeout_seconds=max(1.0, intent_timeout_seconds),
            max_retries=max(1, intent_max_retries),
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

        symbols = self._extract_symbols_from_text(text)
        params: dict[str, Any] = {}
        confidence = 0.85
        if symbols:
            if len(symbols) == 1:
                params["symbol"] = symbols[0]
            else:
                params["symbols"] = symbols
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
                return self._enrich_intent_from_query(deterministic, input_text)
        return self._enrich_intent_from_query(intent, input_text)

    def _enrich_intent_from_query(self, intent: IntentOutput, input_text: str) -> IntentOutput:
        """
        Deterministically enrich intent with cross-domain execution hints inferred
        from the user sentence (e.g. notify/send/share after primary action).
        """
        params = dict(intent.parameters or {})
        confidence = float(intent.confidence)
        if intent.domain == "finance" and intent.capability == "get_stock_price":
            inferred_symbols = self._extract_symbols_from_text(input_text)
            existing_symbols: list[str] = []

            existing_single = params.get("symbol")
            if isinstance(existing_single, str) and existing_single.strip():
                existing_symbols.append(existing_single.strip().upper())

            existing_list = params.get("symbols")
            if isinstance(existing_list, list):
                for item in existing_list:
                    if isinstance(item, str) and item.strip():
                        existing_symbols.append(item.strip().upper())

            merged: list[str] = []
            seen: set[str] = set()
            for item in inferred_symbols + existing_symbols:
                key = item.strip().upper()
                if not key or key in seen:
                    continue
                if not self._is_likely_symbol(key):
                    continue
                seen.add(key)
                merged.append(key)

            if merged:
                params["symbol"] = merged[0]
                if len(merged) > 1:
                    params["symbols"] = merged
                else:
                    params.pop("symbols", None)
                confidence = max(confidence, 0.95)

        params = self._apply_parameter_contracts(
            domain=intent.domain,
            capability=intent.capability,
            params=params,
        )

        notify_detected = self._infer_notify_from_text(input_text)
        if notify_detected:
            params["notify"] = True
            # For explicit send/share requests, avoid unnecessary soft confirmation.
            if intent.domain != "general":
                return intent.model_copy(update={"parameters": params, "confidence": max(confidence, 0.95)})
        return intent.model_copy(update={"parameters": params, "confidence": confidence})

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

    def _parse_json_object(self, raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {}
        return {}

    def _capability_catalog_entry(self, domain: str, capability: str) -> dict[str, Any] | None:
        for item in self.capability_catalog:
            cap = str(item.get("capability") or item.get("name") or "").strip()
            dom = str(item.get("domain") or "").strip()
            if dom == domain and cap == capability:
                return item
        return None

    def _schema_parameter_specs(self, schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
        if not isinstance(schema, dict):
            return {}

        out: dict[str, dict[str, Any]] = {}
        properties = schema.get("properties")
        raw_required = schema.get("required")
        required_items = raw_required if isinstance(raw_required, list) else []
        required_set = {str(item).strip() for item in required_items if str(item).strip()}
        if not isinstance(properties, dict):
            return out

        for param_name, prop in properties.items():
            if not isinstance(prop, dict):
                continue
            name = str(param_name).strip()
            if not name:
                continue
            spec: dict[str, Any] = {}
            if isinstance(prop.get("type"), str):
                spec["type"] = prop["type"]
            if name in required_set:
                spec["required"] = True
            if isinstance(prop.get("description"), str) and prop["description"].strip():
                spec["description"] = prop["description"].strip()
            if isinstance(prop.get("examples"), list):
                spec["examples"] = [v for v in prop["examples"] if isinstance(v, (str, int, float, bool))]
            if "default" in prop:
                spec["default"] = prop["default"]
            if isinstance(prop.get("enum"), list):
                spec["enum"] = prop["enum"]
            if isinstance(prop.get("pattern"), str) and prop["pattern"].strip():
                spec["pattern"] = prop["pattern"].strip()
            if isinstance(prop.get("format"), str) and prop["format"].strip():
                spec["format"] = prop["format"].strip()
            out[name] = spec
        return out

    def _parameter_specs_for_capability(self, domain: str, capability: str) -> dict[str, dict[str, Any]]:
        entry = self._capability_catalog_entry(domain=domain, capability=capability)
        if not entry:
            return {}

        metadata = self._parse_json_object(entry.get("metadata"))
        raw_specs = metadata.get("parameter_specs")
        specs: dict[str, dict[str, Any]] = {}
        if isinstance(raw_specs, dict):
            for param_name, value in raw_specs.items():
                key = str(param_name).strip()
                if key and isinstance(value, dict):
                    specs[key] = dict(value)
        elif isinstance(raw_specs, list):
            for item in raw_specs:
                if not isinstance(item, dict):
                    continue
                key = str(item.get("name", "")).strip()
                if not key:
                    continue
                specs[key] = {k: v for k, v in item.items() if k != "name"}

        schema = self._parse_json_object(entry.get("schema"))
        if not schema:
            schema = self._parse_json_object(entry.get("input_schema"))
        for key, schema_spec in self._schema_parameter_specs(schema).items():
            current = specs.setdefault(key, {})
            for prop_name, prop_value in schema_spec.items():
                current.setdefault(prop_name, prop_value)

        return specs

    def _normalize_parameter_value(self, value: Any, spec: dict[str, Any]) -> Any:
        if isinstance(value, list):
            item_spec = spec.get("items") if isinstance(spec.get("items"), dict) else spec
            return [self._normalize_parameter_value(item, item_spec) for item in value]

        if not isinstance(value, str):
            return value

        text = value.strip()
        if not text:
            return text

        aliases = spec.get("aliases")
        if isinstance(aliases, dict):
            direct = aliases.get(text)
            if direct is None:
                direct = aliases.get(text.upper())
            if direct is None:
                direct = aliases.get(text.lower())
            if isinstance(direct, str) and direct.strip():
                text = direct.strip()

        normalization = spec.get("normalization")
        if isinstance(normalization, dict):
            case_mode = str(normalization.get("case", "")).strip().lower()
            if case_mode == "upper":
                text = text.upper()
            elif case_mode == "lower":
                text = text.lower()
            suffix = normalization.get("suffix")
            if isinstance(suffix, str) and suffix.strip():
                suffix_value = suffix.strip()
                if not text.upper().endswith(suffix_value.upper()):
                    text = f"{text}{suffix_value}"

        enum = spec.get("enum")
        if isinstance(enum, list):
            for item in enum:
                if str(item).strip().lower() == text.lower():
                    text = str(item).strip()
                    break

        return text

    def _infer_value_from_symbol_suffix(self, params: dict[str, Any], spec: dict[str, Any]) -> Any:
        infer_map = spec.get("infer_from_symbol_suffix")
        if not isinstance(infer_map, dict):
            return None

        symbol_value: str | None = None
        raw_symbol = params.get("symbol")
        if isinstance(raw_symbol, str) and raw_symbol.strip():
            symbol_value = raw_symbol.strip().upper()
        else:
            raw_symbols = params.get("symbols")
            if isinstance(raw_symbols, list):
                for item in raw_symbols:
                    if isinstance(item, str) and item.strip():
                        symbol_value = item.strip().upper()
                        break

        if not symbol_value:
            return None

        for suffix_raw, inferred in infer_map.items():
            suffix = str(suffix_raw).strip().upper()
            if not suffix:
                continue
            if symbol_value.endswith(suffix):
                return inferred
        return None

    def _apply_parameter_contracts(
        self,
        *,
        domain: str,
        capability: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        specs = self._parameter_specs_for_capability(domain=domain, capability=capability)
        if not specs:
            return params

        normalized = dict(params)
        for param_name, spec in specs.items():
            if not isinstance(spec, dict):
                continue
            value = normalized.get(param_name)
            if value in (None, ""):
                inferred = self._infer_value_from_symbol_suffix(normalized, spec)
                if inferred not in (None, ""):
                    normalized[param_name] = inferred
                    value = inferred
                elif "default" in spec:
                    normalized[param_name] = spec.get("default")
                    value = normalized.get(param_name)
            if value is None:
                continue
            normalized[param_name] = self._normalize_parameter_value(value, spec)
        return normalized

    def _extract_symbol_from_text(self, text: str) -> str | None:
        symbols = self._extract_symbols_from_text(text)
        return symbols[0] if symbols else None

    def _is_likely_symbol(self, value: str) -> bool:
        token = (value or "").strip().upper()
        if not token:
            return False

        if re.fullmatch(r"[A-Z0-9-]{1,12}\.[A-Z]{1,4}", token):
            return True
        if re.fullmatch(r"[A-Z]{4}(?:3|4|5|6|11)(?:F)?(?:\.SA)?", token):
            return True

        blocked = {
            "QUAL",
            "VALOR",
            "PRECO",
            "PRICE",
            "ACAO",
            "AÇÃO",
            "BOLSA",
            "SUECIA",
            "SUÉCIA",
            "DA",
            "DE",
            "DO",
            "E",
            "NA",
            "NO",
            "EM",
            "PARA",
            "ME",
            "MANDA",
            "MANDAR",
            "ENVIA",
            "ENVIAR",
            "ENVIE",
            "NOTIFIQUE",
            "NOTIFICAR",
            "TELEGRAM",
            "COMPARTILHE",
            "COMPARTILHAR",
            "AVISE",
        }
        if token in blocked:
            return False

        return bool(re.fullmatch(r"[A-Z]{1,5}", token))

    def _extract_symbols_from_text(self, text: str) -> list[str]:
        upper = (text or "").upper()
        if not upper:
            return []

        aliases = {
            "PETRO": "PETR4.SA",
            "PETROBRAS": "PETR4.SA",
            "VALE": "VALE3.SA",
            "NORDEA": "NDA-SE.ST",
        }
        collected: list[str] = []

        def add_symbol(value: str) -> None:
            symbol = value.strip().upper()
            if not symbol:
                return
            if not self._is_likely_symbol(symbol):
                return
            if symbol not in collected:
                collected.append(symbol)

        explicit = re.findall(r"\b([A-Z0-9-]{1,12}\.[A-Z]{1,4})\b", upper)
        for item in explicit:
            add_symbol(item)

        b3 = re.findall(r"\b([A-Z]{4}(?:3|4|5|6|11)(?:F)?)\b", upper)
        for item in b3:
            add_symbol(f"{item}.SA")

        quasi_ticker = re.findall(r"\b([A-Z]{3,8}\d{1,2}[A-Z]?)\b", upper)
        for token in quasi_ticker:
            if re.fullmatch(r"[A-Z]{4}(?:3|4|5|6|11)(?:F)?", token):
                add_symbol(f"{token}.SA")
                continue
            m = re.fullmatch(r"([A-Z]{5,8})(\d{1,2}[A-Z]?)", token)
            if m:
                letters, suffix = m.groups()
                if suffix.startswith(("3", "4", "5", "6", "11")):
                    compact = f"{letters[:4]}{suffix}"
                    if re.fullmatch(r"[A-Z]{4}(?:3|4|5|6|11)(?:F)?", compact):
                        add_symbol(f"{compact}.SA")
                        continue
            add_symbol(token)

        for alias_name, canonical in aliases.items():
            if re.search(rf"\b{re.escape(alias_name)}\b", upper):
                add_symbol(canonical)

        us = re.findall(r"\b([A-Z]{1,5})\b", upper)
        stopwords = {
            "QUAL",
            "VALOR",
            "PRECO",
            "PRICE",
            "ACAO",
            "AÇÃO",
            "BOLSA",
            "SUECIA",
            "SUÉCIA",
            "DA",
            "DE",
            "DO",
            "E",
            "NA",
            "NO",
            "EM",
            "PARA",
            "ME",
            "MANDA",
            "MANDAR",
            "ENVIA",
            "ENVIAR",
            "ENVIE",
            "NOTIFIQUE",
            "NOTIFICAR",
            "TELEGRAM",
            "COMPARTILHE",
            "COMPARTILHAR",
            "AVISE",
        }
        for token in us:
            if token in stopwords:
                continue
            if len(token) < 2:
                continue
            if token in aliases:
                # Alias names are already mapped to canonical tickers.
                continue
            if token.isupper():
                add_symbol(token)

        return collected

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
                param_specs = self._parameter_specs_for_capability(domain=domain, capability=cap_name)
                for idx, (param_name, spec) in enumerate(sorted(param_specs.items()), start=1):
                    if idx > 6:
                        lines.append("    - ...")
                        break
                    if not isinstance(spec, dict):
                        continue
                    type_name = str(spec.get("type", "any")).strip() or "any"
                    required = bool(spec.get("required"))
                    default_value = spec.get("default")
                    examples = spec.get("examples") if isinstance(spec.get("examples"), list) else []
                    example_text = ""
                    if examples:
                        example_text = f" ex: {examples[0]}"
                    default_text = ""
                    if default_value not in (None, ""):
                        default_text = f" default: {default_value}"
                    req_text = "required" if required else "optional"
                    lines.append(
                        f"    - param '{param_name}' ({type_name}, {req_text}){default_text}{example_text}"
                    )
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
