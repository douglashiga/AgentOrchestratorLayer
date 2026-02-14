"""
Task Decomposer — converts an Intent into multi-step plans.

The decomposer is metadata-driven:
- Builds a default one-step plan for any intent.
- Expands into DAG plans only when capability metadata declares composition rules.
"""

from __future__ import annotations

import json
from typing import Any

from shared.models import ExecutionPlan, ExecutionStep, IntentOutput


class TaskDecomposer:
    """Deterministic decomposition driven by runtime capability metadata."""

    def __init__(self, capability_catalog: list[dict[str, Any]] | None = None):
        self.capability_catalog = capability_catalog or []

    def update_capability_catalog(self, capability_catalog: list[dict[str, Any]]) -> None:
        self.capability_catalog = capability_catalog or []

    def decompose(self, intent: IntentOutput) -> ExecutionPlan:
        """Decompose intent into 1..N execution steps."""
        base_step = self._build_primary_step(intent)
        source_entry = self._find_capability_entry(intent.domain, intent.capability)
        source_meta = self._metadata_of(source_entry)
        composition_cfg = source_meta.get("composition")
        if not isinstance(composition_cfg, dict):
            return self._single_step_plan(base_step)

        if not self._composition_enabled(intent, composition_cfg):
            return self._single_step_plan(base_step)

        followup_roles = composition_cfg.get("followup_roles") or []
        if not isinstance(followup_roles, list) or not followup_roles:
            return self._single_step_plan(base_step)

        followup_entry = self._select_followup_capability(followup_roles)
        if not followup_entry:
            return self._single_step_plan(base_step)

        followup_step = self._build_followup_step(
            intent=intent,
            followup_entry=followup_entry,
            source_config=composition_cfg,
        )
        if not followup_step:
            return self._single_step_plan(base_step)

        return ExecutionPlan(
            execution_mode="dag",
            combine_mode="report",
            max_concurrency=4,
            steps=[base_step, followup_step],
        )

    def _build_primary_step(self, intent: IntentOutput) -> ExecutionStep:
        return ExecutionStep(
            id=1,
            domain=intent.domain,
            capability=intent.capability,
            params=dict(intent.parameters),
            depends_on=[],
            required=True,
            output_key="primary",
        )

    def _single_step_plan(self, primary_step: ExecutionStep) -> ExecutionPlan:
        return ExecutionPlan(
            execution_mode="sequential",
            combine_mode="last",
            steps=[primary_step],
        )

    def _find_capability_entry(self, domain: str, capability: str) -> dict[str, Any] | None:
        for entry in self.capability_catalog:
            if str(entry.get("domain", "")) == domain and str(entry.get("capability", "")) == capability:
                return entry
        return None

    def _metadata_of(self, entry: dict[str, Any] | None) -> dict[str, Any]:
        if not entry:
            return {}
        metadata = entry.get("metadata")
        if isinstance(metadata, str):
            try:
                parsed = json.loads(metadata)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return metadata if isinstance(metadata, dict) else {}

    def _composition_enabled(self, intent: IntentOutput, composition_cfg: dict[str, Any]) -> bool:
        enabled_if = composition_cfg.get("enabled_if")
        if enabled_if is None:
            return True
        context = {
            "parameters": dict(intent.parameters or {}),
            "query": intent.original_query,
            "domain": intent.domain,
            "capability": intent.capability,
        }
        return self._eval_condition(enabled_if, context)

    def _eval_condition(self, condition: Any, context: dict[str, Any]) -> bool:
        if isinstance(condition, bool):
            return condition
        if not isinstance(condition, dict):
            return False

        if "all" in condition and isinstance(condition["all"], list):
            return all(self._eval_condition(item, context) for item in condition["all"])
        if "any" in condition and isinstance(condition["any"], list):
            return any(self._eval_condition(item, context) for item in condition["any"])
        if "not" in condition:
            return not self._eval_condition(condition["not"], context)

        path = condition.get("path")
        if not isinstance(path, str) or not path.strip():
            return False
        value = self._resolve_path(context, path)

        if condition.get("exists") is True:
            return value is not None
        if condition.get("truthy") is True:
            return bool(value)
        if "equals" in condition:
            return value == condition.get("equals")
        if "in" in condition and isinstance(condition["in"], list):
            return value in condition["in"]
        return False

    def _resolve_path(self, payload: Any, path: str) -> Any:
        current = payload
        for part in path.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    def _select_followup_capability(self, roles: list[str]) -> dict[str, Any] | None:
        candidates: list[dict[str, Any]] = []
        role_set = {str(role).strip() for role in roles if str(role).strip()}
        for entry in self.capability_catalog:
            metadata = self._metadata_of(entry)
            comp = metadata.get("composition")
            if not isinstance(comp, dict):
                continue
            role = str(comp.get("role", "")).strip()
            if role and role in role_set:
                candidates.append(entry)

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: int(
                self._metadata_of(item).get("composition", {}).get("priority", 0)
            ),
            reverse=True,
        )
        return candidates[0]

    def _build_followup_step(
        self,
        intent: IntentOutput,
        followup_entry: dict[str, Any],
        source_config: dict[str, Any],
    ) -> ExecutionStep | None:
        followup_meta = self._metadata_of(followup_entry)
        followup_comp = followup_meta.get("composition")
        if not isinstance(followup_comp, dict):
            return None

        param_map = followup_comp.get("param_map")
        if not isinstance(param_map, dict) or not param_map:
            return None

        resolved_params: dict[str, Any] = {}
        for target_param, source_spec in param_map.items():
            if isinstance(source_spec, dict):
                copied = self._copy_from_intent_params(intent.parameters, source_spec.get("from_parameters"))
                if copied is not None and copied != "":
                    resolved_params[target_param] = copied
                    continue
                if "default" in source_spec:
                    resolved_params[target_param] = source_spec["default"]
                    continue
                if source_spec.get("required"):
                    return None
            else:
                resolved_params[target_param] = source_spec

        extra_params = followup_comp.get("static_params")
        if isinstance(extra_params, dict):
            resolved_params.update(extra_params)

        return ExecutionStep(
            id=2,
            domain=str(followup_entry.get("domain", "")).strip() or None,
            capability=str(followup_entry.get("capability", "")).strip(),
            params=resolved_params,
            depends_on=[1],
            required=bool(source_config.get("followup_required", False)),
            output_key=str(source_config.get("followup_output_key", "followup")),
        )

    def _copy_from_intent_params(self, params: dict[str, Any], keys: Any) -> Any:
        if not isinstance(keys, list):
            return None
        for key in keys:
            if not isinstance(key, str):
                continue
            value = params.get(key)
            if value is not None and value != "":
                return value
        return None
