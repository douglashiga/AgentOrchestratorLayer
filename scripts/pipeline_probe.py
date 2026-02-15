#!/usr/bin/env python3
"""
Pipeline probe for fast validation of user questions.

Default behavior:
- Parse intent
- Normalize params
- Generate plan
- Validate route-to-domain for each step (without executing domain logic)

Optional:
- Execute plan (--execute) for end-to-end checks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import _normalize_intent_parameters, build_pipeline


def _load_questions(args: argparse.Namespace) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    if args.question:
        for item in args.question:
            text = str(item).strip()
            if text:
                questions.append({"question": text, "expected": {}})

    if args.questions_file:
        raw = Path(args.questions_file).read_text(encoding="utf-8")
        if args.questions_file.endswith(".json"):
            payload = json.loads(raw)
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, str):
                        text = item.strip()
                        if text:
                            questions.append({"question": text, "expected": {}})
                        continue
                    if isinstance(item, dict):
                        text = str(item.get("question", "")).strip()
                        if not text:
                            continue
                        expected = item.get("expected", {})
                        if not isinstance(expected, dict):
                            expected = {}
                        questions.append({"question": text, "expected": expected})
            else:
                raise ValueError("--questions-file .json must contain a JSON array")
        else:
            for line in raw.splitlines():
                text = line.strip()
                if not text or text.startswith("#"):
                    continue
                questions.append({"question": text, "expected": {}})

    dedup: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in questions:
        key = str(item.get("question", "")).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        dedup.append(item)
    return dedup


def _listify(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _evaluate_expected(
    *,
    expected: dict[str, Any],
    intent_payload: dict[str, Any],
    plan_payload: dict[str, Any],
    route: list[dict[str, Any]],
    output_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    params = intent_payload.get("parameters", {}) if isinstance(intent_payload.get("parameters"), dict) else {}
    plan_steps = plan_payload.get("steps", []) if isinstance(plan_payload.get("steps"), list) else []

    def add_check(name: str, ok: bool, actual: Any, wanted: Any) -> None:
        checks.append({"name": name, "ok": ok, "actual": actual, "expected": wanted})
        if not ok:
            failures.append(name)

    exp_domain = expected.get("intent_domain")
    if isinstance(exp_domain, str) and exp_domain.strip():
        add_check(
            "intent_domain",
            str(intent_payload.get("domain", "")).strip() == exp_domain.strip(),
            intent_payload.get("domain"),
            exp_domain,
        )

    exp_cap = expected.get("intent_capability")
    if isinstance(exp_cap, str) and exp_cap.strip():
        add_check(
            "intent_capability",
            str(intent_payload.get("capability", "")).strip() == exp_cap.strip(),
            intent_payload.get("capability"),
            exp_cap,
        )

    exp_conf = expected.get("intent_confidence_gte")
    if isinstance(exp_conf, (int, float)):
        conf = float(intent_payload.get("confidence", 0.0))
        add_check("intent_confidence_gte", conf >= float(exp_conf), conf, float(exp_conf))

    exp_params_contains = expected.get("intent_parameters_contains")
    if isinstance(exp_params_contains, dict):
        for key, wanted_value in exp_params_contains.items():
            actual_value = params.get(key)
            add_check(f"intent_parameters_contains.{key}", actual_value == wanted_value, actual_value, wanted_value)

    exp_params_has = _listify(expected.get("intent_parameters_has_keys"))
    for key in [str(v).strip() for v in exp_params_has if str(v).strip()]:
        add_check(f"intent_parameters_has_key.{key}", key in params, key in params, True)

    exp_params_not_has = _listify(expected.get("intent_parameters_not_has_keys"))
    for key in [str(v).strip() for v in exp_params_not_has if str(v).strip()]:
        add_check(f"intent_parameters_not_has_key.{key}", key not in params, key in params, False)

    exp_symbols = expected.get("intent_symbols_exact")
    if isinstance(exp_symbols, list):
        actual_symbols = params.get("symbols")
        if not isinstance(actual_symbols, list):
            if isinstance(params.get("symbol"), str) and params.get("symbol"):
                actual_symbols = [params.get("symbol")]
            else:
                actual_symbols = []
        add_check("intent_symbols_exact", actual_symbols == exp_symbols, actual_symbols, exp_symbols)

    exp_min_steps = expected.get("plan_min_steps")
    if isinstance(exp_min_steps, int):
        add_check("plan_min_steps", len(plan_steps) >= exp_min_steps, len(plan_steps), exp_min_steps)

    exp_caps = expected.get("plan_contains_step_capabilities")
    if isinstance(exp_caps, list):
        step_caps = [str(step.get("capability", "")).strip() for step in plan_steps if isinstance(step, dict)]
        for cap in [str(v).strip() for v in exp_caps if str(v).strip()]:
            add_check(f"plan_contains_step_capability.{cap}", cap in step_caps, step_caps, cap)

    exp_route_ok = expected.get("route_ok")
    if isinstance(exp_route_ok, bool):
        actual_route_ok = bool(route) and all(bool(step.get("reachable")) for step in route)
        add_check("route_ok", actual_route_ok == exp_route_ok, actual_route_ok, exp_route_ok)

    exp_exec_status = expected.get("execute_status_in")
    if isinstance(exp_exec_status, list):
        actual_status = output_payload.get("status") if isinstance(output_payload, dict) else None
        add_check("execute_status_in", actual_status in exp_exec_status, actual_status, exp_exec_status)

    return {
        "ok": not failures,
        "failed_checks": failures,
        "checks": checks,
    }


def _route_steps(engine: Any, plan_payload: dict[str, Any], default_domain: str) -> list[dict[str, Any]]:
    steps = plan_payload.get("steps")
    if not isinstance(steps, list):
        return []

    registry = engine.orchestrator.domain_registry
    routed: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        capability = str(step.get("capability", "")).strip()
        domain = str(step.get("domain", "")).strip() or default_domain
        handler = registry.resolve_capability(capability)
        resolution = "capability"
        if handler is None:
            handler = registry.resolve_domain(domain)
            resolution = "domain"

        routed.append(
            {
                "id": step.get("id"),
                "domain": domain,
                "capability": capability,
                "reachable": handler is not None,
                "resolution": resolution if handler is not None else "none",
                "handler_class": handler.__class__.__name__ if handler is not None else None,
                "depends_on": step.get("depends_on", []),
                "required": bool(step.get("required", True)),
            }
        )
    return routed


async def _run_probe(args: argparse.Namespace) -> int:
    questions = _load_questions(args)
    if not questions:
        print("No questions provided. Use --question or --questions-file.", file=sys.stderr)
        return 2

    (
        _cli,
        _conversation,
        intent_adapter,
        planner,
        engine,
        model_selector,
        mcp_adapter,
    ) = build_pipeline()

    had_failures = False
    report: list[dict[str, Any]] = []

    try:
        for index, row in enumerate(questions, start=1):
            question = str(row.get("question", "")).strip()
            expected = row.get("expected", {})
            if not isinstance(expected, dict):
                expected = {}
            session_id = f"{args.session_prefix}-{index}"
            item: dict[str, Any] = {
                "index": index,
                "question": question,
                "expected": expected,
            }
            try:
                intent = intent_adapter.extract(question, history=[], session_id=session_id)
                normalized = _normalize_intent_parameters(
                    intent,
                    engine.orchestrator.domain_registry,
                    entry_request=None,
                )
                plan = planner.generate_plan(normalized, session_id=session_id)

                intent_payload = normalized.model_dump(mode="json")
                plan_payload = plan.model_dump(mode="json")
                route = _route_steps(engine=engine, plan_payload=plan_payload, default_domain=normalized.domain)
                route_ok = bool(route) and all(bool(step.get("reachable")) for step in route)

                item["intent"] = intent_payload
                item["plan"] = plan_payload
                item["route"] = route
                item["route_ok"] = route_ok

                if args.execute:
                    output = await engine.execute_plan(plan, original_intent=normalized)
                    output_payload = output.model_dump(mode="json")
                    item["output"] = output_payload
                    item["execute_ok"] = output.status in {"success", "clarification"}
                else:
                    output_payload = None
                    item["execute_ok"] = None

                expected_eval = _evaluate_expected(
                    expected=expected,
                    intent_payload=intent_payload,
                    plan_payload=plan_payload,
                    route=route,
                    output_payload=output_payload,
                )
                item["expected_ok"] = expected_eval["ok"]
                item["expected_eval"] = expected_eval

                if not route_ok:
                    had_failures = True
                if expected and not expected_eval["ok"]:
                    had_failures = True
            except Exception as exc:
                had_failures = True
                item["error"] = str(exc)
            report.append(item)
    finally:
        # Close resources created by build_pipeline
        try:
            planner.close()
        except Exception:
            pass
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
    return 1 if had_failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe pipeline stages for multiple questions.")
    parser.add_argument("--question", action="append", default=[], help="Question text (can be repeated).")
    parser.add_argument("--questions-file", default="", help="Path to .txt or .json with questions.")
    parser.add_argument("--execute", action="store_true", help="Also execute the plan end-to-end.")
    parser.add_argument("--session-prefix", default="probe", help="Session id prefix.")
    parser.add_argument("--output-json", default="", help="Optional path to write report JSON.")
    args = parser.parse_args()

    try:
        return asyncio.run(_run_probe(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
