"""
Registry — Maps domains and capabilities to handlers.

Responsibility:
- Maintain mapping of domain_name -> Handler
- Maintain mapping of capability -> Handler
- Store metadata for capabilities (criticality, execution_mode)
"""

import logging
import json
from typing import Any

from shared.workflow_contracts import MethodSpec

logger = logging.getLogger(__name__)


class HandlerRegistry:
    """Registry for domain handlers and capabilities."""

    def __init__(self):
        self._domains: dict[str, Any] = {}
        self._capabilities: dict[str, Any] = {}
        self._metadata: dict[str, dict] = {}
        self._method_specs: dict[str, MethodSpec] = {}
        self._goals: dict[str, dict[str, Any]] = {}  # "domain:GOAL_NAME" → goal_def

    def register_domain(self, domain_name: str, handler: Any) -> None:
        """Register a handler for a high-level domain."""
        self._domains[domain_name] = handler
        logger.info("Registered domain: %s → %s", domain_name, handler.__class__.__name__)

    def register_capability(
        self, 
        capability: str, 
        handler: Any, 
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Register a specific capability to a handler."""
        self._capabilities[capability] = handler
        if metadata:
            self._metadata[capability] = metadata
            method_spec = self._parse_method_spec(capability=capability, metadata=metadata)
            if method_spec is not None:
                self._method_specs[capability] = method_spec
        logger.info("Registered capability: %s → %s", capability, handler.__class__.__name__)

    def register_method_spec(self, capability: str, method_spec: MethodSpec | dict[str, Any]) -> None:
        """Register/override declarative method contract for a capability."""
        spec = method_spec if isinstance(method_spec, MethodSpec) else MethodSpec(**method_spec)
        self._method_specs[capability] = spec

    def resolve_method_spec(self, capability: str) -> MethodSpec | None:
        """Resolve a declarative method contract by capability name."""
        return self._method_specs.get(capability)

    def resolve_domain(self, domain_name: str) -> Any | None:
        """Resolve handler by domain name."""
        return self._domains.get(domain_name)

    def resolve_capability(self, capability: str) -> Any | None:
        """Resolve handler by capability."""
        return self._capabilities.get(capability)

    def get_metadata(self, capability: str) -> dict[str, Any]:
        """Get metadata for a capability."""
        return self._metadata.get(capability, {})

    @property
    def registered_domains(self) -> list[str]:
        return list(self._domains.keys())

    @property
    def registered_capabilities(self) -> list[str]:
        return list(self._capabilities.keys())

    # ─── Goals ─────────────────────────────────────────────────

    def register_goal(self, domain: str, goal_name: str, goal_def: dict[str, Any]) -> None:
        """Register a goal for a domain."""
        key = f"{domain}:{goal_name}"
        self._goals[key] = goal_def
        logger.info("Registered goal: %s", key)

    def resolve_goal(self, domain: str, goal_name: str) -> dict[str, Any] | None:
        """Resolve a goal definition by domain and goal name."""
        return self._goals.get(f"{domain}:{goal_name}")

    def list_goals_for_domain(self, domain: str) -> list[dict[str, Any]]:
        """List all goals registered for a domain."""
        prefix = f"{domain}:"
        return [v for k, v in self._goals.items() if k.startswith(prefix)]

    def build_goal_catalog(self) -> list[dict[str, Any]]:
        """Build a catalog of domains with their goals for the intent prompt.
        Returns list of dicts: [{domain, description, goals: [{goal, description, hints, entities_schema}]}]
        Includes entities_schema so the Intent LLM knows what entities to extract.
        NO capabilities or parameter specs — intent never sees those.
        """
        domain_map: dict[str, dict[str, Any]] = {}
        for key, goal_def in self._goals.items():
            domain = key.split(":", 1)[0]
            if domain not in domain_map:
                domain_map[domain] = {"domain": domain, "description": "", "goals": []}
            goal_entry: dict[str, Any] = {
                "goal": goal_def.get("goal", key.split(":", 1)[1]),
                "description": goal_def.get("description", ""),
                "hints": goal_def.get("hints", {}),
            }
            entities_schema = goal_def.get("entities_schema")
            if isinstance(entities_schema, dict) and entities_schema:
                goal_entry["entities_schema"] = entities_schema
            domain_map[domain]["goals"].append(goal_entry)
        return list(domain_map.values())

    def build_goal_catalog_dict(self) -> dict[str, dict[str, Any]]:
        """Build a dict of goals keyed by 'domain:GOAL_NAME' for planner resolution."""
        return dict(self._goals)

    def _parse_method_spec(self, capability: str, metadata: dict[str, Any]) -> MethodSpec | None:
        """Best-effort parser for method contracts embedded in capability metadata."""
        if not isinstance(metadata, dict):
            return None

        explicit = metadata.get("method_spec")
        if isinstance(explicit, str):
            try:
                explicit = json.loads(explicit)
            except Exception:
                explicit = None
        if isinstance(explicit, dict):
            try:
                return MethodSpec(**explicit)
            except Exception as exc:
                logger.warning("Invalid method_spec for capability '%s': %s", capability, exc)
                return None

        workflow = metadata.get("workflow")
        if isinstance(workflow, str):
            try:
                workflow = json.loads(workflow)
            except Exception:
                workflow = None
        if not isinstance(workflow, dict):
            return None

        raw_policy = metadata.get("policy", {})
        if isinstance(raw_policy, str):
            try:
                raw_policy = json.loads(raw_policy)
            except Exception:
                raw_policy = {}
        if not isinstance(raw_policy, dict):
            raw_policy = {}

        raw_input_schema = metadata.get("schema", {})
        if isinstance(raw_input_schema, str):
            try:
                raw_input_schema = json.loads(raw_input_schema)
            except Exception:
                raw_input_schema = {}
        if not isinstance(raw_input_schema, dict):
            raw_input_schema = {}

        raw_output_schema = metadata.get("output_schema", {})
        if isinstance(raw_output_schema, str):
            try:
                raw_output_schema = json.loads(raw_output_schema)
            except Exception:
                raw_output_schema = {}
        if not isinstance(raw_output_schema, dict):
            raw_output_schema = {}

        contract_payload = {
            "domain": str(metadata.get("domain", "")).strip() or "unknown",
            "method": capability,
            "version": str(metadata.get("version", "1.0.0")).strip() or "1.0.0",
            "description": str(metadata.get("description", "")).strip(),
            "input_schema": raw_input_schema,
            "output_schema": raw_output_schema,
            "workflow": workflow,
            "policy": raw_policy,
            "tags": metadata.get("tags", []) if isinstance(metadata.get("tags"), list) else [],
            "metadata": metadata.get("contract_metadata", {})
            if isinstance(metadata.get("contract_metadata"), dict)
            else {},
        }

        try:
            return MethodSpec(**contract_payload)
        except Exception as exc:
            logger.warning(
                "Invalid workflow contract for capability '%s' (metadata.workflow): %s",
                capability,
                exc,
            )
            return None
