#!/usr/bin/env python3
"""
Simulate iterative parameter completion for a capability.

This probe validates two things per stage:
1) route-only orchestration (test mode)
2) generic contract validation from capability schema/parameter_specs
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import build_pipeline
from shared.models import ExecutionIntent


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _load_stages(args: argparse.Namespace) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    for raw in args.stage_json:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("--stage-json must be a JSON object")
        stages.append(payload)

    if args.stages_file:
        raw = Path(args.stages_file).read_text(encoding="utf-8")
        payload = json.loads(raw)
        if not isinstance(payload, list):
            raise ValueError("--stages-file must contain a JSON array of objects")
        for item in payload:
            if isinstance(item, dict):
                stages.append(item)
            else:
                raise ValueError("Each stage must be a JSON object")
    return stages


def _parameter_specs(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = metadata.get("parameter_specs")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = {}

    specs: dict[str, dict[str, Any]] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            name = str(key).strip()
            if name and isinstance(value, dict):
                specs[name] = dict(value)
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            specs[name] = {k: v for k, v in item.items() if k != "name"}
    return specs


def _human_param_label(param_name: str, spec: dict[str, Any]) -> str:
    custom = spec.get("label_human")
    if isinstance(custom, str) and custom.strip():
        return custom.strip()
    return param_name.replace("_", " ")


def _human_value_label(value: Any, spec: dict[str, Any]) -> str:
    if value in (None, ""):
        return ""
    value_text = str(value).strip()
    if not value_text:
        return ""

    value_labels = spec.get("value_labels")
    if isinstance(value_labels, dict):
        for key in (value_text, value_text.upper(), value_text.lower()):
            mapped = value_labels.get(key)
            if isinstance(mapped, str) and mapped.strip():
                return mapped.strip()
    return value_text


def _default_value(param_name: str, spec: dict[str, Any], metadata: dict[str, Any]) -> Any:
    if "default" in spec:
        return spec.get("default")
    key = f"default_{param_name}"
    if key in metadata:
        return metadata.get(key)
    return None


def _required_fields(
    *,
    params: dict[str, Any],
    metadata: dict[str, Any],
    required_override: list[str],
) -> list[str]:
    if required_override:
        required = required_override
    else:
        schema = _to_dict(metadata.get("schema"))
        raw_required = schema.get("required", [])
        required = [str(item).strip() for item in raw_required if str(item).strip()] if isinstance(raw_required, list) else []

    missing: list[str] = []
    for field in required:
        if params.get(field) in (None, ""):
            missing.append(field)
    return missing


def _build_validation_payload(
    *,
    params: dict[str, Any],
    metadata: dict[str, Any],
    missing_required: list[str],
) -> tuple[str, dict[str, Any]]:
    specs = _parameter_specs(metadata)
    provided_params = sorted(
        [str(k) for k, v in params.items() if not str(k).startswith("_") and v not in (None, "")]
    )
    assumed_defaults: dict[str, Any] = {}
    questions: list[dict[str, Any]] = []

    for name in missing_required:
        spec = specs.get(name, {}) if isinstance(specs.get(name), dict) else {}
        default_value = _default_value(name, spec, metadata)
        if default_value not in (None, ""):
            assumed_defaults[name] = default_value

        questions.append(
            {
                "param": name,
                "label": _human_param_label(name, spec),
                "required": True,
                "default": default_value,
                "default_label": _human_value_label(default_value, spec) if default_value not in (None, "") else "",
            }
        )

    if missing_required and len(assumed_defaults) == len(missing_required):
        defaults_text = ", ".join(
            _human_value_label(assumed_defaults[name], specs.get(name, {}))
            for name in missing_required
            if _human_value_label(assumed_defaults[name], specs.get(name, {}))
        )
        if not defaults_text:
            defaults_text = ", ".join(f"{k}={v}" for k, v in assumed_defaults.items())
        explanation = (
            f"Posso seguir com {defaults_text} por padrão para continuar? "
            "Se preferir, me diga os valores que você quer."
        )
    else:
        labels = ", ".join(_human_param_label(name, specs.get(name, {})) for name in missing_required)
        explanation = (
            f"Para continuar, faltam: {labels}. "
            "Se quiser, posso assumir os padrões disponíveis."
        )

    payload = {
        "missing_required_params": missing_required,
        "provided_params": provided_params,
        "assumed_defaults": assumed_defaults,
        "questions": questions,
        "confirmation_mode": "defaults_first_human_confirmation",
    }
    return explanation, payload


async def _run(args: argparse.Namespace) -> int:
    stages = _load_stages(args)
    if not stages:
        print("No stages provided. Use --stage-json or --stages-file.", file=sys.stderr)
        return 2

    # Route-only mode in orchestrator.
    os.environ.setdefault("ORCHESTRATOR_TEST_MODE", "true")

    (
        _cli,
        _conversation,
        _intent_adapter,
        _planner,
        engine,
        model_selector,
        mcp_adapter,
    ) = build_pipeline()

    report: list[dict[str, Any]] = []
    required_fields = [item.strip() for item in args.required if str(item).strip()]
    registry = engine.orchestrator.domain_registry
    try:
        for idx, params in enumerate(stages, start=1):
            stage_params = dict(params)
            intent = ExecutionIntent(
                domain=args.domain,
                capability=args.capability,
                confidence=1.0,
                parameters=stage_params,
                original_query=f"[loop-probe:{idx}]",
            )
            route_output = await engine.orchestrator.process(intent)

            metadata = _to_dict(registry.get_metadata(args.capability))
            missing_required = _required_fields(
                params=stage_params,
                metadata=metadata,
                required_override=required_fields,
            )

            if missing_required:
                status = "clarification"
                explanation, validation_payload = _build_validation_payload(
                    params=stage_params,
                    metadata=metadata,
                    missing_required=missing_required,
                )
            else:
                status = route_output.status
                explanation = route_output.explanation
                validation_payload = {
                    "missing_required_params": [],
                    "provided_params": sorted(
                        [str(k) for k, v in stage_params.items() if not str(k).startswith("_") and v not in (None, "")]
                    ),
                    "assumed_defaults": {},
                    "questions": [],
                    "confirmation_mode": "defaults_first_human_confirmation",
                }

            report.append(
                {
                    "stage": idx,
                    "params": params,
                    "status": status,
                    "explanation": explanation,
                    "metadata": route_output.metadata,
                    "result": route_output.result,
                    "route_status": route_output.status,
                    "validation": validation_payload,
                }
            )

            if args.stop_on_success and status == "success" and route_output.status == "success":
                break
    finally:
        try:
            engine.close()
        except Exception:
            pass
        try:
            model_selector.close()
        except Exception:
            pass
        try:
            mcp_adapter.close()
        except Exception:
            pass

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Iterative required-params probe.")
    parser.add_argument("--domain", required=True, help="Intent domain")
    parser.add_argument("--capability", required=True, help="Intent capability")
    parser.add_argument("--stage-json", action="append", default=[], help="Stage params as JSON object")
    parser.add_argument("--stages-file", default="", help="JSON array file with stage param objects")
    parser.add_argument("--required", action="append", default=[], help="Required field name override (repeatable)")
    parser.add_argument("--stop-on-success", action="store_true", help="Stop when first success happens")
    parser.add_argument("--output-json", default="", help="Optional output report JSON path")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
