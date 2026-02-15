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
import re
import unicodedata
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
        self.force_domain_routing = os.getenv("INTENT_FORCE_DOMAIN_ROUTING", "true").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.multi_pass_enabled = os.getenv("INTENT_MULTI_PASS_ENABLED", "true").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.multi_pass_shortlist_size = int(os.getenv("INTENT_MULTI_PASS_SHORTLIST_SIZE", "6"))

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
        fallback_domain, fallback_capability, fallback_confidence = self._fallback_intent_target(input_text)

        if self.multi_pass_enabled and self.capability_catalog:
            multi_pass_intent = self._extract_multi_pass(
                input_text=input_text,
                history=history,
                session_id=session_id,
                fallback_domain=fallback_domain,
                fallback_capability=fallback_capability,
            )
            if multi_pass_intent is not None:
                return multi_pass_intent

        messages = self._build_messages(input_text, history)

        try:
            raw_data = self.model_selector.generate(
                messages=messages,
                policy=self.policy,
                session_id=session_id,
            )
            intent = self._intent_from_payload(
                payload=raw_data,
                input_text=input_text,
                fallback_domain=fallback_domain,
                fallback_capability=fallback_capability,
            )
            return self._finalize_intent(intent=intent, shortlist=[])
        except Exception as e:
            logger.error("Intent extraction failed: %s", e)
            fallback_params: dict[str, Any] = {}
            if fallback_domain == "general" and fallback_capability == "chat":
                fallback_params["message"] = input_text
            fallback_intent = IntentOutput(
                domain=fallback_domain,
                capability=fallback_capability,
                confidence=fallback_confidence,
                parameters=fallback_params,
                original_query=input_text,
            )
            return self._finalize_intent(intent=fallback_intent, shortlist=[])

    def _extract_multi_pass(
        self,
        *,
        input_text: str,
        history: list[dict] | None,
        session_id: str | None,
        fallback_domain: str,
        fallback_capability: str,
    ) -> IntentOutput | None:
        """
        Multi-pass extraction:
        1) Analyze relevant domains/capabilities/obvious parameters.
        2) Select final capability from a scoped shortlist and normalize obvious params.
        """
        analysis_payload = self._run_analysis_pass(input_text=input_text, history=history, session_id=session_id)
        if analysis_payload is None:
            return None

        shortlist = self._build_capability_shortlist(analysis_payload=analysis_payload, input_text=input_text)
        selection_payload = self._run_selection_pass(
            input_text=input_text,
            history=history,
            session_id=session_id,
            shortlist=shortlist,
        )

        candidate_payload = selection_payload if self._payload_looks_like_intent(selection_payload) else None
        if candidate_payload is None and self._payload_looks_like_intent(analysis_payload):
            candidate_payload = analysis_payload
        if candidate_payload is None:
            return None

        intent = self._intent_from_payload(
            payload=candidate_payload,
            input_text=input_text,
            fallback_domain=fallback_domain,
            fallback_capability=fallback_capability,
        )
        obvious_parameters = analysis_payload.get("obvious_parameters")
        params = dict(intent.parameters or {})
        if isinstance(obvious_parameters, dict):
            for key, value in obvious_parameters.items():
                name = str(key).strip()
                if not name:
                    continue
                if name not in params or params.get(name) in (None, "", []):
                    params[name] = value
        intent = intent.model_copy(update={"parameters": params})
        return self._finalize_intent(intent=intent, shortlist=shortlist)

    def _finalize_intent(self, *, intent: IntentOutput, shortlist: list[dict[str, Any]]) -> IntentOutput:
        params = self._normalize_parameters_for_capability(
            domain=intent.domain,
            capability=intent.capability,
            params=dict(intent.parameters or {}),
        )
        # NOTE: Execution step composition is handled by TaskDecomposer, not here
        # The adapter's job is to extract intent structure, not create execution plans
        # (Removing _execution_steps to let TaskDecomposer handle decomposition via metadata)
        return intent.model_copy(update={"parameters": params})

    def _run_analysis_pass(
        self,
        *,
        input_text: str,
        history: list[dict] | None,
        session_id: str | None,
    ) -> dict[str, Any] | None:
        messages = self._build_analysis_messages(input_text=input_text, history=history)
        try:
            raw = self.model_selector.generate(
                messages=messages,
                policy=self.policy,
                session_id=session_id,
            )
        except Exception as exc:
            logger.warning("Intent analysis pass failed: %s", exc)
            return None

        payload = self._parse_json_object(raw)
        if not payload:
            return None
        return payload

    def _run_selection_pass(
        self,
        *,
        input_text: str,
        history: list[dict] | None,
        session_id: str | None,
        shortlist: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not shortlist:
            return None
        messages = self._build_selection_messages(input_text=input_text, history=history, shortlist=shortlist)
        try:
            raw = self.model_selector.generate(
                messages=messages,
                policy=self.policy,
                session_id=session_id,
            )
        except Exception as exc:
            logger.warning("Intent selection pass failed: %s", exc)
            return None
        payload = self._parse_json_object(raw)
        return payload if payload else None

    def _payload_looks_like_intent(self, payload: dict[str, Any] | None) -> bool:
        if not isinstance(payload, dict):
            return False
        domain = str(payload.get("domain", "")).strip()
        action = str(payload.get("action", "")).strip()
        return bool(domain and action)

    def _intent_from_payload(
        self,
        *,
        payload: Any,
        input_text: str,
        fallback_domain: str,
        fallback_capability: str,
    ) -> IntentOutput:
        if not isinstance(payload, dict):
            raise ValueError("Model returned non-dict JSON")
        raw_parameters = payload.get("parameters", {})
        parameters = raw_parameters if isinstance(raw_parameters, dict) else {}
        return IntentOutput(
            domain=str(payload.get("domain", fallback_domain) or fallback_domain),
            capability=str(payload.get("action", fallback_capability) or fallback_capability),
            confidence=float(payload.get("confidence", 0.0)),
            parameters=parameters,
            original_query=input_text,
        )

    def _build_capability_shortlist(
        self,
        *,
        analysis_payload: dict[str, Any],
        input_text: str,
    ) -> list[dict[str, Any]]:
        candidate_map: dict[tuple[str, str], float] = {}

        def _upsert(domain: str, action: str, confidence: float) -> None:
            key = (domain, action)
            prev = candidate_map.get(key, 0.0)
            candidate_map[key] = max(prev, confidence)

        relevant_caps = analysis_payload.get("relevant_capabilities")
        if isinstance(relevant_caps, list):
            for item in relevant_caps:
                if not isinstance(item, dict):
                    continue
                domain = str(item.get("domain", "")).strip()
                action = str(item.get("action", "")).strip()
                if not domain or not action:
                    continue
                try:
                    confidence = float(item.get("confidence", 0.0))
                except Exception:
                    confidence = 0.0
                _upsert(domain, action, confidence)

        if not candidate_map and self._payload_looks_like_intent(analysis_payload):
            domain = str(analysis_payload.get("domain", "")).strip()
            action = str(analysis_payload.get("action", "")).strip()
            confidence = float(analysis_payload.get("confidence", 0.0) or 0.0)
            if domain and action:
                _upsert(domain, action, confidence)

        # Deterministic catalog evidence expansion (metadata-driven).
        query_norm = self._normalize_text(input_text)
        query_tokens = self._tokenize_text(input_text)
        for item in self.capability_catalog:
            if not isinstance(item, dict):
                continue
            domain = str(item.get("domain", "")).strip()
            action = str(item.get("capability") or item.get("name") or "").strip()
            if not domain or not action:
                continue
            score = self._catalog_entry_intent_score(
                query_norm=query_norm,
                query_tokens=query_tokens,
                item=item,
            )
            if score <= 0:
                continue
            # Normalize score to confidence-like range while preserving ranking.
            evidence_confidence = min(0.99, max(0.01, score / 10.0))
            _upsert(domain, action, evidence_confidence)

        if not candidate_map:
            return []

        shortlist: list[dict[str, Any]] = []
        for (domain, action), confidence in sorted(candidate_map.items(), key=lambda item: item[1], reverse=True):
            entry = self._capability_catalog_entry(domain=domain, capability=action)
            if entry is None:
                continue
            description = str(entry.get("description", "")).strip()
            metadata = self._parse_json_object(entry.get("metadata"))
            shortlist.append(
                {
                    "domain": domain,
                    "action": action,
                    "confidence": confidence,
                    "description": description,
                    "metadata": metadata,
                }
            )
            if len(shortlist) >= max(1, self.multi_pass_shortlist_size):
                break
        return shortlist

    def _compose_execution_steps(
        self,
        *,
        domain: str,
        capability: str,
        params: dict[str, Any],
        shortlist: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        clean_params = {
            str(key): value
            for key, value in (params or {}).items()
            if str(key) and not str(key).startswith("_")
        }
        steps: list[dict[str, Any]] = [
            {
                "id": 1,
                "domain": domain,
                "capability": capability,
                "params": clean_params,
                "required": True,
                "output_key": "primary",
            }
        ]

        if clean_params.get("notify") is not True:
            return steps

        notifier_entry = self._select_notifier_entry(shortlist=shortlist)
        if notifier_entry is None:
            return steps

        notifier_domain = str(notifier_entry.get("domain", "")).strip()
        notifier_action = str(notifier_entry.get("action") or notifier_entry.get("capability") or "").strip()
        notifier_meta = notifier_entry.get("metadata")
        if not notifier_domain or not notifier_action or not isinstance(notifier_meta, dict):
            return steps

        notifier_params = self._build_notifier_params(source_params=clean_params, notifier_meta=notifier_meta)
        steps.append(
            {
                "id": 2,
                "domain": notifier_domain,
                "capability": notifier_action,
                "params": notifier_params,
                "depends_on": [1],
                "required": False,
                "output_key": "notification",
            }
        )
        return steps

    def _select_notifier_entry(self, *, shortlist: list[dict[str, Any]]) -> dict[str, Any] | None:
        candidates: list[dict[str, Any]] = []
        for item in shortlist:
            metadata = item.get("metadata")
            if not isinstance(metadata, dict):
                continue
            composition = metadata.get("composition")
            if isinstance(composition, dict) and str(composition.get("role", "")).strip().lower() == "notifier":
                candidates.append(item)

        if not candidates:
            for item in self.capability_catalog:
                if not isinstance(item, dict):
                    continue
                metadata = self._parse_json_object(item.get("metadata"))
                composition = metadata.get("composition")
                if not isinstance(composition, dict):
                    continue
                if str(composition.get("role", "")).strip().lower() != "notifier":
                    continue
                candidates.append(
                    {
                        "domain": str(item.get("domain", "")).strip(),
                        "action": str(item.get("capability", "")).strip(),
                        "metadata": metadata,
                    }
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda entry: int(
                (
                    entry.get("metadata", {})
                    if isinstance(entry.get("metadata"), dict)
                    else {}
                ).get("composition", {}).get("priority", 0)
            ),
            reverse=True,
        )
        return candidates[0]

    def _build_notifier_params(self, *, source_params: dict[str, Any], notifier_meta: dict[str, Any]) -> dict[str, Any]:
        composition = notifier_meta.get("composition")
        if not isinstance(composition, dict):
            return {}
        param_map = composition.get("param_map")
        if not isinstance(param_map, dict):
            return {}

        resolved: dict[str, Any] = {}
        for target_param, source_spec in param_map.items():
            if isinstance(source_spec, dict):
                from_params = source_spec.get("from_parameters")
                copied: Any = None
                if isinstance(from_params, list):
                    for source_name in from_params:
                        key = str(source_name).strip()
                        if not key:
                            continue
                        if key in source_params and source_params.get(key) not in (None, ""):
                            copied = source_params.get(key)
                            break
                if copied not in (None, ""):
                    resolved[str(target_param)] = copied
                    continue
                if "default" in source_spec:
                    resolved[str(target_param)] = source_spec.get("default")
                    continue
            else:
                resolved[str(target_param)] = source_spec
        return resolved

    def _fallback_intent_target(self, input_text: str) -> tuple[str, str, float]:
        """Choose fallback target from runtime catalog using metadata evidence."""
        text_norm = self._normalize_text(input_text)
        text_tokens = self._tokenize_text(input_text)

        best: tuple[float, str, str] | None = None
        parsed_catalog: list[tuple[str, str]] = []
        if isinstance(self.capability_catalog, list):
            for item in self.capability_catalog:
                if not isinstance(item, dict):
                    continue
                domain = str(item.get("domain", "")).strip()
                capability = str(item.get("capability", "")).strip()
                if not domain or not capability:
                    continue
                parsed_catalog.append((domain, capability))
                score = self._catalog_entry_intent_score(
                    query_norm=text_norm,
                    query_tokens=text_tokens,
                    item=item,
                )
                if score <= 0:
                    continue

                if best is None or score > best[0]:
                    best = (score, domain, capability)

        if best is not None:
            score, domain, capability = best
            confidence = min(0.7, max(0.25, score / 10.0))
            return domain, capability, confidence

        for domain, capability in parsed_catalog:
            if domain == "general" and capability == "chat":
                return domain, capability, 0.0

        if parsed_catalog:
            domain, capability = parsed_catalog[0]
            return domain, capability, 0.0
        return "general", "chat", 0.0

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""
        lowered = text.lower()
        normalized = unicodedata.normalize("NFKD", lowered)
        return "".join(ch for ch in normalized if not unicodedata.combining(ch))

    def _tokenize_text(self, text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9]{2,}", self._normalize_text(text))}

    def _hint_match_score(self, query_norm: str, query_tokens: set[str], hint: str) -> float:
        hint_norm = self._normalize_text(str(hint))
        if not hint_norm:
            return 0.0

        if hint_norm in query_norm:
            return 8.0

        hint_tokens = {token for token in re.findall(r"[a-z0-9]{2,}", hint_norm)}
        if not hint_tokens or not query_tokens:
            return 0.0

        overlap = hint_tokens & query_tokens
        if not overlap:
            return 0.0
        return 2.0 + (len(overlap) / max(1, len(hint_tokens))) * 4.0

    def _hint_values(self, payload: Any) -> list[str]:
        if not isinstance(payload, dict):
            return []
        values: list[str] = []
        for key in ("keywords", "examples"):
            raw = payload.get(key)
            if not isinstance(raw, list):
                continue
            for item in raw:
                text = str(item).strip()
                if text:
                    values.append(text)
        return values

    def _catalog_entry_intent_score(
        self,
        *,
        query_norm: str,
        query_tokens: set[str],
        item: dict[str, Any],
    ) -> float:
        capability = str(item.get("capability") or item.get("name") or "").strip()
        description = str(item.get("description", "")).strip()
        metadata = self._parse_json_object(item.get("metadata"))

        specific_hints: list[str] = []
        if capability:
            specific_hints.append(capability.replace("_", " "))
        if description:
            specific_hints.append(description)
        metadata_description = str(metadata.get("description", "")).strip()
        if metadata_description:
            specific_hints.append(metadata_description)
        specific_hints.extend(self._hint_values(metadata.get("intent_hints")))

        domain_hints: list[str] = []
        domain_description = str(metadata.get("domain_description", "")).strip()
        if domain_description:
            domain_hints.append(domain_description)
        domain_hints.extend(self._hint_values(metadata.get("domain_intent_hints")))

        specific_score = 0.0
        for hint in specific_hints:
            specific_score = max(specific_score, self._hint_match_score(query_norm, query_tokens, hint))

        domain_score = 0.0
        for hint in domain_hints:
            domain_score = max(domain_score, self._hint_match_score(query_norm, query_tokens, hint))

        # Domain hints are useful to identify area, but must not dominate capability selection.
        return specific_score + (domain_score * 0.1)

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
            if suffix and symbol_value.endswith(suffix):
                return inferred
        return None

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
            if direct is None:
                text_norm = self._normalize_text(text)
                for raw_alias, mapped in aliases.items():
                    if self._normalize_text(str(raw_alias)) == text_norm:
                        direct = mapped
                        break
            if direct is not None:
                return direct

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
                    text = item
                    break
        return text

    def _normalize_parameters_for_capability(
        self,
        *,
        domain: str,
        capability: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        specs = self._parameter_specs_for_capability(domain=domain, capability=capability)
        if not specs:
            return params

        normalized = dict(params or {})
        for param_name, spec in specs.items():
            if not isinstance(spec, dict):
                continue
            key = str(param_name).strip()
            if not key:
                continue
            value = normalized.get(key)
            if value in (None, ""):
                inferred = self._infer_value_from_symbol_suffix(normalized, spec)
                if inferred not in (None, ""):
                    normalized[key] = inferred
                    value = inferred
                elif "default" in spec:
                    normalized[key] = spec.get("default")
                    value = normalized.get(key)
            if value is None:
                continue
            normalized[key] = self._normalize_parameter_value(value, spec)
        return normalized

    def _render_capability_catalog(self) -> str:
        """Render domain/action catalog from runtime registry data."""
        if not self.capability_catalog:
            return ""

        grouped = self._domain_catalog()

        lines: list[str] = []
        for domain in sorted(grouped.keys()):
            domain_payload = grouped[domain]
            domain_description = str(domain_payload.get("description", "")).strip()
            if domain_description:
                lines.append(f"- Domain '{domain}': {domain_description}")
            else:
                lines.append(f"- Domain '{domain}':")

            domain_hints = domain_payload.get("intent_hints")
            if isinstance(domain_hints, dict):
                hint_keywords = domain_hints.get("keywords")
                if isinstance(hint_keywords, list):
                    valid_keywords = [str(v).strip() for v in hint_keywords if str(v).strip()]
                    if valid_keywords:
                        lines.append(f"  - domain intent keywords: {', '.join(valid_keywords[:8])}")
                hint_examples = domain_hints.get("examples")
                if isinstance(hint_examples, list):
                    valid_examples = [str(v).strip() for v in hint_examples if str(v).strip()]
                    if valid_examples:
                        lines.append(f"  - domain intent examples: {valid_examples[0]}")

            capabilities = domain_payload.get("capabilities")
            if not isinstance(capabilities, list):
                capabilities = []
            for cap in sorted(capabilities, key=lambda x: str(x.get("capability", ""))):
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

    def _domain_catalog(self) -> dict[str, dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for item in self.capability_catalog:
            domain = str(item.get("domain", "")).strip() or "unknown"
            payload = grouped.setdefault(
                domain,
                {
                    "description": "",
                    "intent_hints": {},
                    "capabilities": [],
                },
            )
            metadata = self._parse_json_object(item.get("metadata"))
            if not payload.get("description"):
                domain_description = str(metadata.get("domain_description", "")).strip()
                if domain_description:
                    payload["description"] = domain_description
            if not payload.get("intent_hints") and isinstance(metadata.get("domain_intent_hints"), dict):
                payload["intent_hints"] = metadata.get("domain_intent_hints")
            payload["capabilities"].append(item)
        return grouped

    def _render_domain_catalog(self) -> str:
        grouped = self._domain_catalog()
        lines: list[str] = []
        for domain in sorted(grouped.keys()):
            payload = grouped[domain]
            description = str(payload.get("description", "")).strip()
            if description:
                lines.append(f"- domain '{domain}': {description}")
            else:
                lines.append(f"- domain '{domain}'")
            intent_hints = payload.get("intent_hints")
            if isinstance(intent_hints, dict):
                keywords = intent_hints.get("keywords")
                if isinstance(keywords, list):
                    valid = [str(v).strip() for v in keywords if str(v).strip()]
                    if valid:
                        lines.append(f"  - keywords: {', '.join(valid[:8])}")
                examples = intent_hints.get("examples")
                if isinstance(examples, list):
                    valid_examples = [str(v).strip() for v in examples if str(v).strip()]
                    if valid_examples:
                        lines.append(f"  - example: {valid_examples[0]}")
            capabilities = payload.get("capabilities")
            if isinstance(capabilities, list):
                cap_names = [str(cap.get("capability", "")).strip() for cap in capabilities if str(cap.get("capability", "")).strip()]
                if cap_names:
                    lines.append(f"  - capabilities: {', '.join(sorted(cap_names)[:12])}")
        return "\n".join(lines)

    def _render_shortlist_catalog(self, shortlist: list[dict[str, Any]]) -> str:
        if not shortlist:
            return ""
        lines: list[str] = []
        for item in shortlist:
            domain = str(item.get("domain", "")).strip()
            action = str(item.get("action", "")).strip()
            description = str(item.get("description", "")).strip()
            confidence = item.get("confidence")
            confidence_txt = ""
            if isinstance(confidence, (int, float)):
                confidence_txt = f" (analysis_conf={float(confidence):.2f})"
            lines.append(f"- {domain}.{action}{confidence_txt}: {description}")
            metadata = item.get("metadata")
            if isinstance(metadata, dict):
                specs = metadata.get("parameter_specs")
                if isinstance(specs, dict):
                    for idx, (param_name, spec) in enumerate(sorted(specs.items()), start=1):
                        if idx > 6:
                            lines.append("  - ...")
                            break
                        if not isinstance(spec, dict):
                            continue
                        ptype = str(spec.get("type", "any")).strip() or "any"
                        req = "required" if bool(spec.get("required")) else "optional"
                        lines.append(f"  - param '{param_name}' ({ptype}, {req})")
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
            has_non_general_capability = any(
                str(item.get("domain", "")).strip() not in {"", "general"}
                for item in self.capability_catalog
            )
            if self.force_domain_routing and has_non_general_capability:
                discovery_rule = (
                    '2. Prefer non-general domains/actions from the catalog.\n'
                    '3. Use domain "general"/action "chat" only for explicit assistant/chit-chat questions.\n'
                )
            else:
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

    def _build_analysis_prompt(self) -> str:
        domain_catalog = self._render_domain_catalog()
        capability_catalog = self._render_capability_catalog()
        return f"""You are an intent analysis engine.

Your job is to analyze the user message and produce a relevance map across domains, capabilities, and obvious parameters.
Use ONLY the domains/actions from the runtime catalog.

Return ONLY a valid JSON object with this schema:
{{
  "relevant_domains": [{{"domain": "<domain>", "confidence": <0.0-1.0>, "reason": "<short>"}}],
  "relevant_capabilities": [{{"domain": "<domain>", "action": "<action>", "confidence": <0.0-1.0>, "reason": "<short>"}}],
  "obvious_parameters": {{}}
}}

Rules:
1. Add only domains/capabilities that are relevant to the user request.
2. "obvious_parameters" should include only values explicit or very clear from user text.
3. Do not invent unknown values.
4. Keep confidence conservative when ambiguous.

Domain catalog:
{domain_catalog}

Capability catalog:
{capability_catalog}
"""

    def _build_selection_prompt(self, shortlist: list[dict[str, Any]]) -> str:
        shortlist_block = self._render_shortlist_catalog(shortlist)
        return f"""You are an intent selector.

Choose exactly ONE final action from the provided shortlist and return ONLY valid JSON.

Required output schema:
{{
  "domain": "<domain>",
  "action": "<action>",
  "parameters": {{}},
  "confidence": <float 0.0-1.0>
}}

Rules:
1. domain/action MUST be one of the shortlist entries.
2. Fill parameters only when explicit/obvious from user message or aliases.
3. Keep confidence lower when missing required parameters.
4. Return ONLY JSON.

Shortlist:
{shortlist_block}
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

    def _build_analysis_messages(self, input_text: str, history: list[dict] | None = None) -> list[dict]:
        messages = [{"role": "system", "content": self._build_analysis_prompt()}]
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

    def _build_selection_messages(
        self,
        *,
        input_text: str,
        history: list[dict] | None = None,
        shortlist: list[dict[str, Any]],
    ) -> list[dict]:
        messages = [{"role": "system", "content": self._build_selection_prompt(shortlist=shortlist)}]
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
