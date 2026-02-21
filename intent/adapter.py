"""
Intent Adapter — LLM-powered goal-based intent extraction.

Responsibility:
- Convert input_text + history -> structured IntentOutput (goal-based)
- Temperature = 0, JSON-only output
- Intent sees ONLY domains and goals, NEVER capabilities or tools
- Pydantic schema validation via Model Layer
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
    """Extract structured goal-based intent from user input using the model layer."""

    def __init__(
        self,
        model_selector: ModelSelector,
        goal_catalog: list[dict[str, Any]] | None = None,
    ):
        self.model_selector = model_selector
        self.goal_catalog = goal_catalog or []
        self._goal_lookup: dict[str, dict[str, Any]] | None = None  # lazy cache, reset on catalog update

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

    def update_goal_catalog(self, goal_catalog: list[dict[str, Any]]) -> None:
        """Update runtime goal catalog used by dynamic system prompt."""
        self.goal_catalog = goal_catalog or []
        self._goal_lookup = None  # invalidate cached lookup
        logger.info("IntentAdapter updated with %d goal catalog entries", len(self.goal_catalog))

    def extract(
        self,
        input_text: str,
        history: list[dict] | None = None,
        session_id: str | None = None,
    ) -> IntentOutput:
        """Extract goal-based intent via single-pass LLM extraction."""
        fallback_domain, fallback_goal, fallback_confidence = self._fallback_intent_target(input_text)

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
                fallback_goal=fallback_goal,
            )
            return intent
        except Exception as e:
            logger.error("Intent extraction failed: %s", e)
            return IntentOutput(
                primary_domain=fallback_domain,
                goal=fallback_goal,
                entities={},
                confidence=fallback_confidence,
                original_query=input_text,
            )

    # ─── Payload parsing ──────────────────────────────────────

    def _intent_from_payload(
        self,
        *,
        payload: Any,
        input_text: str,
        fallback_domain: str,
        fallback_goal: str,
    ) -> IntentOutput:
        if not isinstance(payload, dict):
            raise ValueError("Model returned non-dict JSON")

        raw_entities = payload.get("entities", {})
        entities = raw_entities if isinstance(raw_entities, dict) else {}

        domain = str(payload.get("primary_domain", fallback_domain) or fallback_domain).strip()
        goal = str(payload.get("goal", fallback_goal) or fallback_goal).strip()

        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        # Clamp confidence to valid range
        confidence = max(0.0, min(1.0, confidence))

        # Validate domain/goal against the catalog (guard: skip if catalog is empty)
        goal_lookup = self._build_goal_lookup()
        key = f"{domain}:{goal}"
        goal_item: dict[str, Any] | None = goal_lookup.get(key) if goal_lookup else None
        if goal_item is None and goal_lookup:
            logger.warning(
                "IntentAdapter: LLM returned unknown goal '%s', falling back to '%s:%s'",
                key,
                fallback_domain,
                fallback_goal,
            )
            domain, goal = fallback_domain, fallback_goal
            confidence = min(confidence, 0.5)
            goal_item = goal_lookup.get(f"{domain}:{goal}")

        # Sanitize entities against the goal's entities_schema
        entities_schema = goal_item.get("entities_schema", {}) if goal_item else {}
        if isinstance(entities_schema, dict) and entities_schema:
            entities = self._sanitize_entities(entities, entities_schema)

        return IntentOutput(
            primary_domain=domain,
            goal=goal,
            entities=entities,
            confidence=confidence,
            original_query=input_text,
        )

    # ─── Goal catalog lookup and entity sanitization ─────────

    def _build_goal_lookup(self) -> dict[str, dict[str, Any]]:
        """Build (and cache) a ``domain:goal`` → goal_item lookup from the current catalog.

        Returns an empty dict when the catalog is empty so callers can skip validation safely.
        """
        if self._goal_lookup is None:
            self._goal_lookup = {
                f"{item.get('domain', '')}:{item.get('goal', '')}": item
                for item in self.goal_catalog
                if isinstance(item, dict) and item.get("domain") and item.get("goal")
            }
        return self._goal_lookup

    def _sanitize_entities(
        self,
        entities: dict[str, Any],
        entities_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Sanitize entities against the goal's entities_schema.

        Strategy:
        - For keys **declared in the schema** as ``enum``: drop the value if it is not in
          the declared ``values`` list (prevent bad enum values from reaching GoalResolver).
        - All other keys (string-typed declared fields, and keys *not* in the schema) are
          kept as-is — ``entities_schema`` is a minimum declaration, not an exhaustive
          allowlist. Cross-cutting operational params like ``notify`` or ``compose`` are
          intentionally absent from schemas but must pass through to the planner.
        - Never raises; always returns a dict.
        """
        clean: dict[str, Any] = dict(entities)  # start with all keys
        for key, spec in entities_schema.items():
            if key not in clean:
                continue
            if not isinstance(spec, dict):
                continue
            field_type = str(spec.get("type", "string")).strip().lower()
            if field_type == "enum":
                value = clean[key]
                allowed_values = spec.get("values")
                if isinstance(allowed_values, list) and value not in allowed_values:
                    logger.debug(
                        "IntentAdapter: dropping entity '%s'='%s' — not in enum values %s",
                        key,
                        value,
                        allowed_values,
                    )
                    del clean[key]
        return clean

    # ─── Fallback: deterministic goal hint matching ───────────

    def _fallback_intent_target(self, input_text: str) -> tuple[str, str, float]:
        """Choose fallback target from goal catalog using hint evidence."""
        text_norm = self._normalize_text(input_text)
        text_tokens = self._tokenize_text(input_text)

        best: tuple[float, str, str] | None = None

        for item in self.goal_catalog:
            if not isinstance(item, dict):
                continue
            domain = str(item.get("domain", "")).strip()
            goal = str(item.get("goal", "")).strip()
            if not domain or not goal:
                continue
            score = self._goal_entry_intent_score(
                query_norm=text_norm,
                query_tokens=text_tokens,
                item=item,
            )
            if score <= 0:
                continue
            if best is None or score > best[0]:
                best = (score, domain, goal)

        if best is not None:
            score, domain, goal = best
            confidence = min(0.7, max(0.25, score / 10.0))
            return domain, goal, confidence

        return "general", "CHAT", 0.0

    def _goal_entry_intent_score(
        self,
        *,
        query_norm: str,
        query_tokens: set[str],
        item: dict[str, Any],
    ) -> float:
        """Score a goal catalog entry against user query using hints."""
        goal = str(item.get("goal", "")).strip()
        description = str(item.get("description", "")).strip()
        hints = item.get("hints") if isinstance(item.get("hints"), dict) else {}

        goal_hints: list[str] = []
        if goal:
            goal_hints.append(goal.replace("_", " "))
        if description:
            goal_hints.append(description)
        goal_hints.extend(self._hint_values(hints))

        # Domain-level hints (lower weight)
        domain_description = str(item.get("domain_description", "")).strip()
        domain_hints_data = item.get("domain_hints") if isinstance(item.get("domain_hints"), dict) else {}
        domain_hints: list[str] = []
        if domain_description:
            domain_hints.append(domain_description)
        domain_hints.extend(self._hint_values(domain_hints_data))

        goal_score = 0.0
        for hint in goal_hints:
            goal_score = max(goal_score, self._hint_match_score(query_norm, query_tokens, hint))

        domain_score = 0.0
        for hint in domain_hints:
            domain_score = max(domain_score, self._hint_match_score(query_norm, query_tokens, hint))

        return goal_score + (domain_score * 0.1)

    # ─── Text normalization and hint matching utilities ────────

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

    # ─── Goal catalog rendering for LLM prompt ────────────────

    def _render_goal_catalog(self) -> str:
        """Render domains, goals, and entities_schema — NO capabilities, NO parameter specs."""
        if not self.goal_catalog:
            return ""

        # Group goals by domain
        grouped: dict[str, dict[str, Any]] = {}
        for item in self.goal_catalog:
            if not isinstance(item, dict):
                continue
            domain = str(item.get("domain", "")).strip()
            if not domain:
                continue
            payload = grouped.setdefault(domain, {"description": "", "goals": []})
            if not payload["description"]:
                desc = str(item.get("domain_description", "")).strip()
                if desc:
                    payload["description"] = desc
            payload["goals"].append(item)

        lines: list[str] = []
        for domain in sorted(grouped.keys()):
            domain_payload = grouped[domain]
            desc = domain_payload.get("description", "")
            lines.append(f"- Domain '{domain}': {desc}" if desc else f"- Domain '{domain}'")

            for goal_item in domain_payload.get("goals", []):
                goal_name = str(goal_item.get("goal", "")).strip()
                goal_desc = str(goal_item.get("description", "")).strip()
                lines.append(f"  - goal '{goal_name}': {goal_desc}")

                hints = goal_item.get("hints") if isinstance(goal_item.get("hints"), dict) else {}
                keywords = hints.get("keywords")
                if isinstance(keywords, list):
                    valid = [str(v).strip() for v in keywords if str(v).strip()]
                    if valid:
                        lines.append(f"    keywords: {', '.join(valid[:8])}")
                examples = hints.get("examples")
                if isinstance(examples, list):
                    valid_ex = [str(v).strip() for v in examples if str(v).strip()]
                    if valid_ex:
                        lines.append(f"    example: {valid_ex[0]}")

                # Render entities_schema so LLM knows what entities to extract
                entities_schema = goal_item.get("entities_schema")
                if isinstance(entities_schema, dict) and entities_schema:
                    lines.append(f"    entities to extract:")
                    for ent_name, ent_def in entities_schema.items():
                        if not isinstance(ent_def, dict):
                            continue
                        ent_type = str(ent_def.get("type", "string")).strip()
                        ent_desc = str(ent_def.get("description", "")).strip()
                        values = ent_def.get("values")
                        required = ent_def.get("required", False)
                        parts = [f"      - {ent_name}"]
                        if ent_type == "enum" and isinstance(values, list):
                            parts.append(f"(enum: {', '.join(str(v) for v in values)})")
                        else:
                            parts.append(f"({ent_type})")
                        if required:
                            parts.append("[required]")
                        if ent_desc:
                            parts.append(f": {ent_desc}")
                        lines.append(" ".join(parts))

        return "\n".join(lines)

    # ─── Prompt construction ──────────────────────────────────

    def _build_system_prompt(self) -> str:
        """Build the goal-based system prompt. Intent never sees capabilities."""
        catalog_block = self._render_goal_catalog()

        return f"""You are an intent extraction engine. Return ONLY a valid JSON object. No explanations, no markdown.

Schema:
{{
  "primary_domain": "<DOMAIN>",
  "goal": "<GOAL_NAME>",
  "entities": {{}},
  "confidence": <float 0.0-1.0>
}}

Available domains and goals:
{catalog_block}

Rules:
1. Use ONLY domains/goals listed above. Do NOT invent domains or goals.
2. "entities" = human-friendly values extracted directly from user text.
   - Use *_text keys for names/descriptions the user actually said (company names, market names, periods).
   - Use enum keys (like "direction", "focus") with the exact enum values listed in the goal schema.
   - NEVER guess technical identifiers (tickers, API codes, job IDs). The domain will resolve those.
   - Example: user says "Nordea" → entities.symbol_text = "Nordea" (NOT "NDA-SE.ST")
   - Example: user says "maiores altas" → entities.direction = "GAINERS"
3. If a goal has "entities to extract" listed, include those entity keys when the user provides them.
4. Do NOT guess entity values. Omit the entity key if unsure.
5. If confidence is not high, return confidence <= 0.90.
6. For greetings or casual conversation, use domain 'general' and goal 'CHAT'.
7. Return ONLY the JSON object."""

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
