"""
Function-Calling Planner inspired by Semantic Kernel auto tool invocation.

This planner is optional and reliability-first:
- Starts from deterministic TaskDecomposer output.
- Uses the model to decide whether to add more capability steps.
- Enforces strict validation against runtime capability catalog.
- Falls back to deterministic plan on any planner error.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from models.selector import ModelSelector
from shared.models import ExecutionPlan, ExecutionStep, IntentOutput, ModelPolicy

logger = logging.getLogger(__name__)


def _parse_csv_set(raw: str) -> set[str]:
    if not raw:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


class FunctionCallingPlanner:
    """Model-guided planning loop with strict catalog validation."""

    def __init__(
        self,
        model_selector: ModelSelector | None = None,
        capability_catalog: list[dict[str, Any]] | None = None,
    ):
        self.model_selector = model_selector
        self.capability_catalog = self._normalize_catalog(capability_catalog or [])
        self.enabled = os.getenv("PLANNER_FUNCTION_CALLING_ENABLED", "true").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.mode = os.getenv("PLANNER_FUNCTION_CHOICE_MODE", "auto").strip().lower()  # auto|required|none
        if self.mode not in {"auto", "required", "none"}:
            self.mode = "auto"
        self.max_iterations = max(1, int(os.getenv("PLANNER_MAX_ITERATIONS", "2")))
        self.model_name = os.getenv("PLANNER_MODEL", "llama3.1:8b").strip() or "llama3.1:8b"
        self.included_domains = _parse_csv_set(os.getenv("PLANNER_INCLUDED_DOMAINS", ""))
        self.excluded_domains = _parse_csv_set(os.getenv("PLANNER_EXCLUDED_DOMAINS", ""))
        self.included_capabilities = _parse_csv_set(os.getenv("PLANNER_INCLUDED_CAPABILITIES", ""))
        self.excluded_capabilities = _parse_csv_set(os.getenv("PLANNER_EXCLUDED_CAPABILITIES", ""))
        self.policy = ModelPolicy(
            model_name=self.model_name,
            temperature=0.0,
            timeout_seconds=float(os.getenv("PLANNER_TIMEOUT_SECONDS", "8")),
            max_retries=2,
            json_mode=True,
        )

    def update_capability_catalog(self, capability_catalog: list[dict[str, Any]]) -> None:
        self.capability_catalog = self._normalize_catalog(capability_catalog or [])

    def expand_plan(
        self,
        intent: IntentOutput,
        base_plan: ExecutionPlan,
        memory_context: dict[str, Any] | None = None,
    ) -> ExecutionPlan:
        """Try to extend a deterministic plan with additional validated steps."""
        if not self._should_run(intent, base_plan):
            return base_plan

        steps = list(base_plan.steps)
        combine_mode = base_plan.combine_mode

        for _ in range(self.max_iterations):
            proposal = self._propose_next_step(intent, steps, combine_mode, memory_context=memory_context)
            if not proposal:
                break

            decision = str(proposal.get("decision", "stop")).strip().lower()
            if decision != "add_step":
                break

            new_step = self._proposal_to_step(proposal, steps)
            if not new_step:
                break

            if any(self._step_signature(existing) == self._step_signature(new_step) for existing in steps):
                logger.info(
                    "FunctionCallingPlanner: duplicate step rejected (%s.%s)",
                    new_step.domain,
                    new_step.capability,
                )
                break

            steps.append(new_step)
            combine_mode = self._safe_combine_mode(proposal.get("combine_mode"), default="report")

        if len(steps) == len(base_plan.steps) and self.mode == "required":
            fallback_step = self._required_fallback_step(intent, steps)
            if fallback_step:
                steps.append(fallback_step)
                combine_mode = "report"

        if len(steps) == len(base_plan.steps):
            return base_plan

        return ExecutionPlan(
            execution_mode="dag",
            combine_mode=combine_mode if len(steps) > 1 else base_plan.combine_mode,
            max_concurrency=max(base_plan.max_concurrency, 4),
            steps=steps,
        )

    def _required_fallback_step(
        self,
        intent: IntentOutput,
        current_steps: list[ExecutionStep],
    ) -> ExecutionStep | None:
        candidates = []
        for entry in self._eligible_candidates(current_steps):
            metadata = entry.get("metadata")
            if not isinstance(metadata, dict):
                continue
            composition = metadata.get("composition")
            if not isinstance(composition, dict):
                continue
            role = str(composition.get("role", "")).strip()
            if not role:
                continue
            candidates.append(entry)

        if not candidates:
            return None

        candidates.sort(
            key=lambda row: int(
                (row.get("metadata") or {}).get("composition", {}).get("priority", 0)
            ),
            reverse=True,
        )
        chosen = candidates[0]
        composition = (chosen.get("metadata") or {}).get("composition", {})
        params = self._build_params_from_param_map(intent.parameters, composition.get("param_map"))
        if not params:
            return None

        next_id = max(step.id for step in current_steps) + 1
        return ExecutionStep(
            id=next_id,
            domain=chosen["domain"],
            capability=chosen["capability"],
            params=params,
            depends_on=[current_steps[-1].id],
            required=False,
            output_key=f"step_{next_id}",
        )

    def _should_run(self, intent: IntentOutput, base_plan: ExecutionPlan) -> bool:
        if not self.enabled:
            return False
        if self.mode == "none":
            return False
        if self.model_selector is None:
            return False
        if intent.domain == "general":
            return False
        if not self.capability_catalog:
            return False
        if len(base_plan.steps) > 1 and self.mode != "required":
            return False
        if len(base_plan.steps) >= 4:
            return False
        return True

    def _propose_next_step(
        self,
        intent: IntentOutput,
        current_steps: list[ExecutionStep],
        combine_mode: str,
        memory_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        candidates = self._eligible_candidates(current_steps)
        if not candidates:
            return None

        messages = self._build_messages(
            intent,
            current_steps,
            combine_mode,
            candidates,
            memory_context=memory_context,
        )
        try:
            raw = self.model_selector.generate(messages=messages, policy=self.policy)
        except Exception as e:
            logger.warning("FunctionCallingPlanner model call failed: %s", e)
            return None

        if not isinstance(raw, dict):
            logger.warning("FunctionCallingPlanner received non-dict proposal")
            return None
        return raw

    def _build_messages(
        self,
        intent: IntentOutput,
        current_steps: list[ExecutionStep],
        combine_mode: str,
        candidates: list[dict[str, Any]],
        memory_context: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        system_prompt = (
            "You are a planning engine.\n"
            "Return ONLY JSON with this schema:\n"
            "{\n"
            '  "decision": "stop" | "add_step",\n'
            '  "combine_mode": "last" | "report" | "merge",\n'
            '  "step": {\n'
            '    "domain": "string",\n'
            '    "capability": "string",\n'
            '    "params": {},\n'
            '    "depends_on": [1],\n'
            '    "required": false,\n'
            '    "output_key": "string"\n'
            "  }\n"
            "}\n"
            "Rules:\n"
            "1. Add a step ONLY if it clearly improves the user outcome.\n"
            "2. If user requested send/notify/share, add a communication/notifier capability when available.\n"
            "3. Use memory_context values when they are relevant and deterministic.\n"
            "4. You may reference previous step output with placeholders like ${1.explanation}.\n"
            "5. If no extra action is needed, return decision=stop.\n"
            "6. Never output capabilities not present in candidates.\n"
            "7. Never include text outside JSON.\n"
        )

        payload = {
            "original_query": intent.original_query,
            "intent": intent.model_dump(mode="json"),
            "memory_context": memory_context or {},
            "current_plan": {
                "combine_mode": combine_mode,
                "steps": [step.model_dump(mode="json") for step in current_steps],
            },
            "candidates": candidates,
        }
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    def _proposal_to_step(
        self,
        proposal: dict[str, Any],
        current_steps: list[ExecutionStep],
    ) -> ExecutionStep | None:
        step_obj = proposal.get("step")
        if not isinstance(step_obj, dict):
            return None

        domain = str(step_obj.get("domain", "")).strip()
        capability = str(step_obj.get("capability", "")).strip()
        if not domain or not capability:
            return None

        catalog_entry = self._resolve_catalog_entry(domain, capability)
        if not catalog_entry:
            logger.info("FunctionCallingPlanner rejected unknown capability: %s.%s", domain, capability)
            return None

        params = step_obj.get("params", {})
        if not isinstance(params, dict):
            params = {}
        params = self._sanitize_params(catalog_entry, params)

        valid_ids = {step.id for step in current_steps}
        depends_on_raw = step_obj.get("depends_on")
        depends_on: list[int] = []
        if isinstance(depends_on_raw, list):
            for item in depends_on_raw:
                if isinstance(item, int) and item in valid_ids:
                    depends_on.append(item)
        if not depends_on:
            depends_on = [current_steps[-1].id]

        next_id = max(step.id for step in current_steps) + 1
        output_key = str(step_obj.get("output_key", f"step_{next_id}")).strip() or f"step_{next_id}"

        return ExecutionStep(
            id=next_id,
            domain=domain,
            capability=capability,
            params=params,
            depends_on=depends_on,
            required=bool(step_obj.get("required", False)),
            output_key=output_key,
        )

    def _safe_combine_mode(self, value: Any, default: str = "last") -> str:
        mode = str(value or "").strip().lower()
        if mode in {"last", "report", "merge"}:
            return mode
        return default

    def _eligible_candidates(self, current_steps: list[ExecutionStep]) -> list[dict[str, Any]]:
        used = {(step.domain or "", step.capability) for step in current_steps}
        candidates: list[dict[str, Any]] = []
        for entry in self.capability_catalog:
            domain = str(entry.get("domain", "")).strip()
            capability = str(entry.get("capability", "")).strip()
            metadata = entry.get("metadata", {})
            if isinstance(metadata, dict) and metadata.get("planner_available") is False:
                continue
            if not domain or not capability:
                continue
            if (domain, capability) in used:
                continue
            if not self._is_allowed_by_filters(domain, capability):
                continue
            candidates.append(
                {
                    "domain": domain,
                    "capability": capability,
                    "description": entry.get("description", ""),
                    "schema": entry.get("schema", {}),
                    "metadata": metadata,
                }
            )
        return candidates

    def _is_allowed_by_filters(self, domain: str, capability: str) -> bool:
        if self.included_domains and domain not in self.included_domains:
            return False
        if domain in self.excluded_domains:
            return False
        if self.included_capabilities and capability not in self.included_capabilities:
            return False
        if capability in self.excluded_capabilities:
            return False
        return True

    def _resolve_catalog_entry(self, domain: str, capability: str) -> dict[str, Any] | None:
        for entry in self.capability_catalog:
            if entry.get("domain") == domain and entry.get("capability") == capability:
                return entry
        return None

    def _sanitize_params(self, catalog_entry: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        schema = catalog_entry.get("schema") or {}
        if not isinstance(schema, dict):
            return params
        properties = schema.get("properties")
        if not isinstance(properties, dict) or not properties:
            return params
        allowed = set(properties.keys())
        return {key: value for key, value in params.items() if key in allowed}

    def _build_params_from_param_map(
        self,
        intent_params: dict[str, Any],
        param_map: Any,
    ) -> dict[str, Any]:
        if not isinstance(param_map, dict):
            return {}
        resolved: dict[str, Any] = {}
        for target, spec in param_map.items():
            if isinstance(spec, dict):
                copied = self._copy_from_params(intent_params, spec.get("from_parameters"))
                if copied is not None and copied != "":
                    resolved[target] = copied
                    continue
                if "default" in spec:
                    resolved[target] = spec["default"]
                    continue
                if spec.get("required"):
                    return {}
            else:
                resolved[target] = spec
        return resolved

    def _copy_from_params(self, params: dict[str, Any], source_keys: Any) -> Any:
        if not isinstance(source_keys, list):
            return None
        for key in source_keys:
            if not isinstance(key, str):
                continue
            value = params.get(key)
            if value is not None and value != "":
                return value
        return None

    def _normalize_catalog(self, raw_catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for row in raw_catalog:
            if not isinstance(row, dict):
                continue
            domain = str(row.get("domain", "")).strip()
            capability = str(row.get("capability") or row.get("name") or "").strip()
            if not domain or not capability:
                continue

            description = str(row.get("description", "")).strip()
            schema = row.get("schema")
            if not isinstance(schema, dict):
                schema = row.get("input_schema")
            if isinstance(schema, str):
                try:
                    parsed_schema = json.loads(schema)
                    schema = parsed_schema if isinstance(parsed_schema, dict) else {}
                except Exception:
                    schema = {}
            if not isinstance(schema, dict):
                schema = {}

            metadata = row.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    parsed_meta = json.loads(metadata)
                    metadata = parsed_meta if isinstance(parsed_meta, dict) else {}
                except Exception:
                    metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}

            normalized.append(
                {
                    "domain": domain,
                    "capability": capability,
                    "description": description,
                    "schema": schema,
                    "metadata": metadata,
                }
            )
        return normalized

    def _step_signature(self, step: ExecutionStep) -> tuple[str, str, str]:
        domain = step.domain or ""
        payload = json.dumps(step.params, ensure_ascii=False, sort_keys=True)
        return domain, step.capability, payload
