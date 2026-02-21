"""
Planner Layer — Decomposes Intent into an Execution Plan.

Responsibility:
- Transform IntentOutput → ExecutionIntent → ExecutionPlan (1..N steps)
- GoalResolver: deterministic goal → capability resolution
- Support metadata-driven multi-domain decomposition
- Inject deterministic structured memory before planning
"""

import json
import logging
import os
from typing import Any

from memory.store import MemoryStore
from models.selector import ModelSelector
from shared.models import ExecutionIntent, IntentOutput, ExecutionPlan
from planner.function_calling_planner import FunctionCallingPlanner, normalize_capability_catalog
from planner.goal_resolver import GoalResolver
from planner.task_decomposer import TaskDecomposer

logger = logging.getLogger(__name__)


class PlannerService:
    """Service to generate execution plans from intents."""

    def __init__(
        self,
        capability_catalog: list[dict[str, Any]] | None = None,
        model_selector: ModelSelector | None = None,
        memory_store: MemoryStore | None = None,
        goal_catalog_dict: dict[str, dict[str, Any]] | None = None,
    ):
        _normalized = normalize_capability_catalog(capability_catalog or [])
        self.capability_catalog = _normalized
        self.goal_resolver = GoalResolver(goal_catalog_dict=goal_catalog_dict)
        self.task_decomposer = TaskDecomposer(capability_catalog=_normalized)
        self.function_planner = FunctionCallingPlanner(
            model_selector=model_selector,
            capability_catalog=_normalized,
        )
        self.memory_store = memory_store
        self.last_memory_context: dict[str, Any] = {"enabled": False}
        self.last_execution_intent: ExecutionIntent | None = None
        self.memory_enabled = os.getenv("PLANNER_MEMORY_ENABLED", "true").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.memory_allow_without_schema = os.getenv("PLANNER_MEMORY_ALLOW_WITHOUT_SCHEMA", "false").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.memory_param_mappings = self._load_memory_param_mappings()

    def update_capability_catalog(self, capability_catalog: list[dict[str, Any]]) -> None:
        normalized = normalize_capability_catalog(capability_catalog or [])
        self.capability_catalog = normalized
        self.task_decomposer.update_capability_catalog(normalized)
        self.function_planner.update_capability_catalog(normalized)

    def update_goal_catalog(self, goal_catalog_dict: dict[str, dict[str, Any]]) -> None:
        """Update the goal resolver with fresh goal definitions."""
        self.goal_resolver.update_goals(goal_catalog_dict)

    def resolve_intent(self, intent: IntentOutput) -> ExecutionIntent:
        """Resolve a goal-based IntentOutput into a capability-based ExecutionIntent.

        This is the bridge between the Intent Layer (goals) and the Execution Layer (capabilities).
        """
        return self.goal_resolver.resolve(intent)

    def generate_plan(
        self,
        intent: IntentOutput | ExecutionIntent,
        session_id: str | None = None,
    ) -> ExecutionPlan:
        """
        Convert an intent into an actionable execution plan.

        Accepts either:
        - IntentOutput (goal-based) → resolves goal → capability first
        - ExecutionIntent (already resolved) → skips goal resolution

        Flow: IntentOutput → GoalResolver → ExecutionIntent → TaskDecomposer → ExecutionPlan
        """
        # Step 1: Resolve goal → capability (if needed)
        if isinstance(intent, ExecutionIntent):
            execution_intent = intent
            logger.info(
                "Generating plan: capability=%s (pre-resolved)",
                execution_intent.capability,
            )
        else:
            execution_intent = self.goal_resolver.resolve(intent)
            logger.info(
                "Generating plan: goal=%s → capability=%s",
                intent.goal,
                execution_intent.capability,
            )
        self.last_execution_intent = execution_intent

        # Step 2: Inject memory into execution intent
        enriched_intent, memory_context = self._inject_memory(execution_intent, session_id=session_id)
        self.last_memory_context = memory_context

        # Step 3: Decompose into execution steps
        base_plan = self.task_decomposer.decompose(enriched_intent)

        # Step 4: Optionally expand with LLM-based function calling
        plan = self.function_planner.expand_plan(
            intent=enriched_intent,
            base_plan=base_plan,
            memory_context=memory_context,
        )

        logger.debug("Plan generated with %d steps", len(plan.steps))
        return plan

    def close(self) -> None:
        if self.memory_store and hasattr(self.memory_store, "close"):
            try:
                self.memory_store.close()  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning("PlannerService memory close failed: %s", e)

    def _inject_memory(self, intent: ExecutionIntent, session_id: str | None) -> tuple[ExecutionIntent, dict[str, Any]]:
        if not self.memory_enabled or self.memory_store is None:
            return intent, {"enabled": False}

        namespaces: list[str] = []
        if session_id:
            namespaces.append(f"session:{session_id}")
        namespaces.append("global")

        params = dict(intent.parameters or {})
        allowed_params = self._allowed_parameter_names(intent.domain, intent.capability)

        applied: dict[str, Any] = {}
        resolved_values: dict[str, Any] = {}
        for source_key, target_param, safe_without_schema in self.memory_param_mappings:
            source_value = self._get_from_namespaces(source_key, namespaces)
            if source_value is None:
                continue
            normalized = self._normalize_memory_value(target_param, source_value)
            resolved_values[source_key] = normalized

            if target_param in params and params[target_param] not in (None, ""):
                continue
            if allowed_params is None:
                # No schema available for this capability.
                # Inject only when the mapping is marked safe or the global override is on.
                if not safe_without_schema and not self.memory_allow_without_schema:
                    continue
            elif target_param not in allowed_params:
                continue
            params[target_param] = normalized
            applied[target_param] = normalized

        search_hits: list[dict[str, Any]] = []
        query_text = (intent.original_query or "").strip()
        if query_text:
            for namespace in namespaces:
                try:
                    hits = self.memory_store.search(query_text, namespace=namespace, limit=3)
                except Exception as e:
                    logger.warning("Memory search failed for namespace %s: %s", namespace, e)
                    continue
                if hits:
                    search_hits.extend(hits)

        if not applied:
            return intent, {
                "enabled": True,
                "namespaces": namespaces,
                "applied": {},
                "values": resolved_values,
                "search_hits": search_hits,
            }

        updated_intent = intent.model_copy(update={"parameters": params})
        logger.info("Planner memory injected params: %s", applied)
        return updated_intent, {
            "enabled": True,
            "namespaces": namespaces,
            "applied": applied,
            "values": resolved_values,
            "search_hits": search_hits,
        }

    def _load_memory_param_mappings(self) -> list[tuple[str, str, bool]]:
        """Load memory-key → parameter mappings.

        Each entry is ``(source_key, target_param, safe_without_schema)``.
        When ``safe_without_schema=True``, the value is injected even when the
        capability has no JSON-Schema ``properties`` block — appropriate for
        broadly-applicable context values (market, risk mode, capital).
        When ``False``, injection requires the capability schema to explicitly
        list the parameter (conservative default).

        The JSON env format supports an optional ``"safe_without_schema"`` field
        (dict or list format); missing field defaults to ``False`` (backward compat).
        """
        default_map: list[tuple[str, str, bool]] = [
            ("preferred_market",      "market",               True),
            ("risk_mode",             "risk_mode",            True),
            ("wheel_active",          "wheel_active",         True),
            ("last_wheel_operation",  "last_wheel_operation", False),
            ("wheel_last_operation",  "last_wheel_operation", False),
            ("available_capital",     "capital",              True),
            ("capital_available",     "capital",              True),
        ]
        raw = os.getenv("PLANNER_MEMORY_PARAM_MAP_JSON", "").strip()
        if not raw:
            return default_map
        try:
            payload = json.loads(raw)
        except Exception:
            logger.warning("Invalid PLANNER_MEMORY_PARAM_MAP_JSON, using defaults.")
            return default_map

        mappings: list[tuple[str, str, bool]] = []
        if isinstance(payload, dict):
            for source, target in payload.items():
                source_k = str(source).strip()
                target_k = str(target).strip()
                if source_k and target_k:
                    mappings.append((source_k, target_k, False))
        elif isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                source_k = str(item.get("memory_key", "")).strip()
                target_k = str(item.get("param", "")).strip()
                safe = bool(item.get("safe_without_schema", False))
                if source_k and target_k:
                    mappings.append((source_k, target_k, safe))

        if not mappings:
            return default_map
        return mappings

    def _get_from_namespaces(self, key: str, namespaces: list[str]) -> Any:
        for namespace in namespaces:
            try:
                value = self.memory_store.get(key, namespace=namespace) if self.memory_store else None
            except Exception as e:
                logger.warning("Memory get failed for %s/%s: %s", namespace, key, e)
                continue
            if value is not None:
                return value
        return None

    def _allowed_parameter_names(self, domain: str, capability: str) -> set[str] | None:
        """Return the set of allowed parameter names for a capability, or None if schema has no properties.

        The catalog is pre-normalized (see normalize_capability_catalog), so ``schema`` is always
        a dict and ``capability`` is always the canonical key — no fallback parsing needed.
        """
        for entry in self.capability_catalog:
            if entry.get("domain") != domain or entry.get("capability") != capability:
                continue
            schema = entry.get("schema") or {}
            properties = schema.get("properties")
            if isinstance(properties, dict):
                return set(properties.keys())
            return None
        return None

    def _normalize_memory_value(self, target_param: str, value: Any) -> Any:
        if target_param == "market":
            text = str(value).strip().upper()
            alias = {
                "SWEDEN": "SE",
                "SUECIA": "SE",
                "SE": "SE",
                "BRAZIL": "BR",
                "BRASIL": "BR",
                "BR": "BR",
                "US": "US",
                "USA": "US",
                "UNITED STATES": "US",
            }
            return alias.get(text, text)
        if target_param in ("wheel_active",):
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in ("1", "true", "yes", "on", "sim", "s")
        if target_param in ("capital",):
            if isinstance(value, (int, float)):
                return value
            try:
                return float(str(value).replace(",", "."))
            except Exception:
                return value
        return value
