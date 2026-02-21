"""
Goal Resolver — Deterministic goal-to-capability resolution.

Responsibility:
- Convert IntentOutput (goal-based) → ExecutionIntent (capability-based)
- Use entities_schema to deterministically select capability when a goal maps to multiple capabilities
- No LLM calls — purely rule-based resolution

Flow:
  IntentOutput(primary_domain='finance', goal='TOP_MOVERS', entities={'direction': 'GAINERS'})
  → GoalResolver resolves goal → capability using entities + goal_def
  → ExecutionIntent(domain='finance', capability='get_top_gainers', parameters={...})
"""

from __future__ import annotations

import logging
from typing import Any

from shared.models import ExecutionIntent, IntentOutput

logger = logging.getLogger(__name__)


class GoalResolver:
    """Deterministic resolver: IntentOutput (goal) → ExecutionIntent (capability).

    Uses the goal catalog (from HandlerRegistry) to map goals to capabilities.
    When a goal has multiple capabilities, uses entities_schema to pick the right one.
    """

    def __init__(self, goal_catalog_dict: dict[str, dict[str, Any]] | None = None):
        self._goals = goal_catalog_dict or {}

    def update_goals(self, goal_catalog_dict: dict[str, dict[str, Any]]) -> None:
        """Update the goal catalog from runtime registry."""
        self._goals = goal_catalog_dict or {}
        logger.info("GoalResolver updated with %d goals", len(self._goals))

    def resolve(self, intent: IntentOutput) -> ExecutionIntent:
        """Resolve a goal-based IntentOutput into a capability-based ExecutionIntent.

        Resolution strategy:
        1. Look up goal definition by domain:goal_name
        2. If single capability → use it directly
        3. If multiple capabilities → use entities to pick deterministically
        4. Fallback: first capability in the list
        """
        goal_key = f"{intent.primary_domain}:{intent.goal}"
        goal_def = self._goals.get(goal_key)

        if goal_def is None:
            logger.warning(
                "GoalResolver: no goal definition for '%s'. Using goal as capability fallback.",
                goal_key,
            )
            return intent.to_execution_intent(resolved_capability=intent.goal.lower())

        capabilities = goal_def.get("capabilities", [])
        if not isinstance(capabilities, list) or not capabilities:
            logger.warning(
                "GoalResolver: goal '%s' has no capabilities. Using goal as capability.",
                goal_key,
            )
            return intent.to_execution_intent(resolved_capability=intent.goal.lower())

        # Single capability — trivial case
        if len(capabilities) == 1:
            resolved = str(capabilities[0]).strip()
            logger.info("GoalResolver: %s → %s (single capability)", goal_key, resolved)
            return intent.to_execution_intent(resolved_capability=resolved)

        # Multiple capabilities — use entities_schema to disambiguate
        entities_schema = goal_def.get("entities_schema")
        if isinstance(entities_schema, dict) and entities_schema:
            resolved = self._resolve_multi_capability(
                capabilities=capabilities,
                entities=dict(intent.entities or {}),
                entities_schema=entities_schema,
                goal_key=goal_key,
            )
            if resolved is not None:
                if isinstance(resolved, list) and len(resolved) > 1:
                    # Multi-capability: inject _execution_steps for the TaskDecomposer
                    logger.info(
                        "GoalResolver: %s → %s (multi-capability via _execution_steps)",
                        goal_key,
                        resolved,
                    )
                    return self._build_multi_step_intent(intent, resolved)
                else:
                    single = resolved if isinstance(resolved, str) else str(resolved[0]).strip()
                    logger.info(
                        "GoalResolver: %s → %s (resolved via entities_schema)",
                        goal_key,
                        single,
                    )
                    return intent.to_execution_intent(resolved_capability=single)

        # Fallback: first capability
        fallback = str(capabilities[0]).strip()
        logger.info(
            "GoalResolver: %s → %s (fallback to first capability, %d available)",
            goal_key,
            fallback,
            len(capabilities),
        )
        return intent.to_execution_intent(resolved_capability=fallback)

    def _resolve_multi_capability(
        self,
        *,
        capabilities: list[str],
        entities: dict[str, Any],
        entities_schema: dict[str, Any],
        goal_key: str,
    ) -> list[str] | str | None:
        """Use entities_schema rules to pick the right capability from multiple options.

        Returns:
        - ``str`` — a single capability name (no multi-step needed)
        - ``list[str]`` — ordered list of capabilities to run as parallel steps
        - ``None`` — no mapping found; caller falls back to first capability
        """
        for entity_name, schema_def in entities_schema.items():
            if not isinstance(schema_def, dict):
                continue

            capability_map = schema_def.get("capability_map")
            if not isinstance(capability_map, dict):
                continue

            entity_value = entities.get(entity_name)
            if entity_value is None:
                # Use default if specified
                default = schema_def.get("default")
                if default is not None:
                    entity_value = default
                else:
                    continue

            # Normalize to uppercase string for matching
            normalized = str(entity_value).strip().upper()

            # Check if this value maps to a specific capability
            mapped = capability_map.get(normalized)
            if mapped is None:
                # Try case-insensitive match
                for map_key, map_val in capability_map.items():
                    if map_key.upper() == normalized:
                        mapped = map_val
                        break

            if mapped is None:
                continue

            if isinstance(mapped, str):
                resolved = str(mapped).strip()
                if resolved == "_all":
                    # Return ALL capabilities in the goal's declared list
                    return [str(c).strip() for c in capabilities if str(c).strip()]
                if resolved in capabilities:
                    return resolved
            elif isinstance(mapped, list) and mapped:
                # Explicit ordered list from capability_map
                valid = [str(c).strip() for c in mapped if str(c).strip()]
                if valid:
                    return valid  # may be single- or multi-item

        return None

    def _build_multi_step_intent(self, intent: IntentOutput, resolved_capabilities: list[str]) -> ExecutionIntent:
        """Build an ExecutionIntent with ``_execution_steps`` for multi-capability resolution.

        The primary ``capability`` is set to the first in the list so that downstream
        metadata lookups (e.g. composition rules) still have a reference point.
        ``_execution_steps`` is injected into ``parameters`` so that
        ``TaskDecomposer._plan_from_execution_steps_hint`` builds the full multi-step plan.
        """
        extra_params = dict(intent.entities or {})
        extra_params["_execution_steps"] = [
            {"capability": cap, "domain": intent.primary_domain}
            for cap in resolved_capabilities
        ]
        return ExecutionIntent(
            domain=intent.primary_domain,
            capability=resolved_capabilities[0],
            confidence=intent.confidence,
            parameters=extra_params,
            original_query=intent.original_query,
        )
