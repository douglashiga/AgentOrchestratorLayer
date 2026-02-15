"""
Intent Adapter — LLM-powered intent extraction.

Responsibility:
- Convert input_text + history -> structured IntentOutput
- Temperature = 0, JSON-only output
- Pydantic schema validation via Model Layer
- No domain business rules or deterministic routing logic
"""

import json
import logging
import os
from typing import Any

from models.selector import ModelSelector
from shared.models import IntentOutput, ModelPolicy

logger = logging.getLogger(__name__)


class IntentAdapter:
    """Extract structured intent from user input using the model layer only."""

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
        """Update legacy capability list used by fallback system prompt."""
        self.capabilities = capabilities
        logger.info("IntentAdapter updated with %d capabilities", len(capabilities))

    def update_capability_catalog(self, capability_catalog: list[dict[str, Any]]) -> None:
        """Update runtime domain/capability catalog used by dynamic system prompt."""
        self.capability_catalog = capability_catalog
        logger.info("IntentAdapter updated with %d catalog entries", len(capability_catalog))

    def extract(self, input_text: str, history: list[dict] | None = None, session_id: str | None = None) -> IntentOutput:
        """Extract intent via model output without deterministic enrichment."""
        messages = self._build_messages(input_text, history)

        try:
            raw_data = self.model_selector.generate(
                messages=messages,
                policy=self.policy,
                session_id=session_id,
            )
            if not isinstance(raw_data, dict):
                raise ValueError("Model returned non-dict JSON")

            raw_parameters = raw_data.get("parameters", {})
            parameters = raw_parameters if isinstance(raw_parameters, dict) else {}

            return IntentOutput(
                domain=str(raw_data.get("domain", "general") or "general"),
                capability=str(raw_data.get("action", "chat") or "chat"),
                confidence=float(raw_data.get("confidence", 0.0)),
                parameters=parameters,
                original_query=input_text,
            )
        except Exception as e:
            logger.error("Intent extraction failed: %s", e)
            return IntentOutput(
                domain="general",
                capability="chat",
                confidence=0.0,
                parameters={"message": input_text},
                original_query=input_text,
            )

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

                metadata = self._parse_json_object(cap.get("metadata"))
                intent_hints = metadata.get("intent_hints") if isinstance(metadata, dict) else None
                if isinstance(intent_hints, dict):
                    hint_keywords = intent_hints.get("keywords")
                    if isinstance(hint_keywords, list):
                        valid_keywords = [str(v).strip() for v in hint_keywords if str(v).strip()]
                        if valid_keywords:
                            lines.append(f"    - intent keywords: {', '.join(valid_keywords[:8])}")
                    hint_examples = intent_hints.get("examples")
                    if isinstance(hint_examples, list):
                        valid_examples = [str(v).strip() for v in hint_examples if str(v).strip()]
                        if valid_examples:
                            lines.append(f"    - intent examples: {valid_examples[0]}")

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
                    example_text = f" ex: {examples[0]}" if examples else ""
                    default_text = f" default: {default_value}" if default_value not in (None, "") else ""
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
5. Prefer capability descriptions, intent hints, and parameter examples from the runtime catalog when choosing action/parameters.
6. Return ONLY the JSON object.
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

2. "finance" — for anything related to stocks, markets, prices, options, fundamentals, dividends, company info, financial data.
   Available actions:
{json.dumps(self.capabilities, indent=2)}

Rules:
1. FIRST decide the domain: if the message is about stocks, markets, prices, finance -> "finance". Otherwise -> "general".
2. If finance action is unclear, select the closest finance action and reduce confidence.
3. Return ONLY the JSON object.
"""

    def _build_messages(self, input_text: str, history: list[dict] | None = None) -> list[dict]:
        """Build model messages for intent extraction."""
        messages = [{"role": "system", "content": self._build_system_prompt()}]

        if history:
            for turn in history[-6:]:
                messages.append(
                    {
                        "role": turn.get("role", "user"),
                        "content": turn.get("content", ""),
                    }
                )

        messages.append({"role": "user", "content": input_text})
        return messages
