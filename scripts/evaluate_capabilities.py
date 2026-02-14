"""
Capability evaluator for remote domains.

Measures "answerability" per capability:
- success
- clarification
as pass

Usage:
  python3 scripts/evaluate_capabilities.py
  FINANCE_DOMAIN_URL=http://localhost:8003 COMMUNICATION_DOMAIN_URL=http://localhost:8002 \
    python3 scripts/evaluate_capabilities.py
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class CapabilityResult:
    domain: str
    capability: str
    status: str
    explanation: str
    error: str


def _domain_urls() -> dict[str, str]:
    return {
        "finance": os.getenv("FINANCE_DOMAIN_URL", "http://localhost:8003").rstrip("/"),
        "communication": os.getenv("COMMUNICATION_DOMAIN_URL", "http://localhost:8002").rstrip("/"),
    }


def _load_manifest(client: httpx.Client, base_url: str) -> dict[str, Any]:
    response = client.get(f"{base_url}/manifest", timeout=30.0)
    response.raise_for_status()
    return response.json()


def _sample_array_item(field_name: str) -> Any:
    name = field_name.lower()
    if "symbol" in name:
        return "PETR4.SA"
    if "metric" in name:
        return "pe_ratio"
    return "sample"


def _sample_value(domain: str, capability: str, field_name: str, field_schema: dict[str, Any]) -> Any:
    if "default" in field_schema and field_schema["default"] is not None:
        return field_schema["default"]

    enum_values = field_schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        return enum_values[0]

    name = field_name.lower()
    ftype = str(field_schema.get("type", "string")).lower()

    if name in {"symbol", "ticker", "stock_symbol"}:
        if "wheel" in capability.lower():
            return "PETR4"
        return "PETR4.SA"
    if name in {"symbols"}:
        return ["PETR4.SA", "VALE3.SA"]
    if name in {"query"}:
        return "PETR4"
    if name in {"market"}:
        return "BR"
    if name in {"exchange"}:
        return "BOVESPA"
    if name in {"currency"}:
        return "BRL"
    if name in {"signal_type"}:
        return "rsi_oversold"
    if name in {"period"}:
        return "1mo"
    if name in {"duration"}:
        return "1 D"
    if name in {"interval"}:
        return "1d"
    if name in {"bar_size"}:
        return "1 hour"
    if name in {"message", "text"}:
        return f"Teste capability {domain}.{capability}"
    if name in {"chat_id", "group_id"}:
        return os.getenv("TELEGRAM_DEFAULT_CHAT_ID", "0")
    if name in {"job_name"}:
        return "daily_market_sync"
    if name in {"tool_name", "tool"}:
        return "get_stock_price"
    if name in {"metrics"}:
        return ["pe_ratio"]

    if ftype == "integer":
        return 10
    if ftype == "number":
        return 1.0
    if ftype == "boolean":
        return True
    if ftype == "array":
        return [_sample_array_item(name)]
    if ftype == "object":
        return {}
    return "sample"


def _build_params(domain: str, capability: str, schema: dict[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(properties, dict):
        properties = {}
    if not isinstance(required, list):
        required = []

    params: dict[str, Any] = {}
    for field in required:
        field_name = str(field).strip()
        if not field_name:
            continue
        field_schema = properties.get(field_name, {})
        if not isinstance(field_schema, dict):
            field_schema = {}
        sample = _sample_value(domain, capability, field_name, field_schema)
        if sample is not None:
            params[field_name] = sample

    # Helpful optional params for known communication actions
    if capability in {"send_telegram_message", "send_telegram_group_message"}:
        params.setdefault("message", f"Teste {capability}")
        if capability == "send_telegram_message":
            params.setdefault("chat_id", os.getenv("TELEGRAM_DEFAULT_CHAT_ID", "0"))
        else:
            params.setdefault("group_id", os.getenv("TELEGRAM_DEFAULT_CHAT_ID", "0"))

    return params


def _call_execute(
    client: httpx.Client,
    base_url: str,
    domain: str,
    capability: str,
    params: dict[str, Any],
) -> CapabilityResult:
    payload = {
        "domain": domain,
        "capability": capability,
        "confidence": 1.0,
        "parameters": params,
        "original_query": f"eval::{domain}.{capability}",
    }
    try:
        response = client.post(f"{base_url}/execute", json=payload, timeout=120.0)
        response.raise_for_status()
        data = response.json()
        status = str(data.get("status", "failure"))
        explanation = str(data.get("explanation", ""))
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        error = str(metadata.get("error", ""))
        return CapabilityResult(domain, capability, status, explanation, error)
    except Exception as e:
        return CapabilityResult(domain, capability, "failure", "execute_error", str(e))


def main() -> int:
    urls = _domain_urls()
    results: list[CapabilityResult] = []

    with httpx.Client() as client:
        for domain, base_url in urls.items():
            try:
                manifest = _load_manifest(client, base_url)
            except Exception as e:
                print(f"[ERROR] manifest {domain} @ {base_url}: {e}")
                results.append(CapabilityResult(domain, "__manifest__", "failure", "manifest_error", str(e)))
                continue

            capabilities = manifest.get("capabilities", [])
            if not isinstance(capabilities, list):
                print(f"[WARN] invalid capabilities list for domain={domain}")
                continue

            for cap in capabilities:
                name = str(cap.get("name", "")).strip()
                if not name:
                    continue
                schema = cap.get("schema", {})
                if not isinstance(schema, dict):
                    schema = {}
                params = _build_params(domain, name, schema)
                result = _call_execute(client, base_url, domain, name, params)
                results.append(result)

    total = len(results)
    if total == 0:
        print("No capabilities evaluated.")
        return 1

    pass_statuses = {"success", "clarification"}
    passed = [r for r in results if r.status in pass_statuses]
    failed = [r for r in results if r.status not in pass_statuses]
    pass_rate = (len(passed) / total) * 100.0

    print(f"Capability evaluation: {len(passed)}/{total} pass ({pass_rate:.1f}%)")
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    print("Status breakdown:", json.dumps(by_status, ensure_ascii=False))

    if failed:
        print("\nFailures:")
        for r in failed[:30]:
            short_error = (r.error or r.explanation or "")[:220]
            print(f"- {r.domain}.{r.capability}: {r.status} | {short_error}")

    # Require strong answerability by default
    target = float(os.getenv("CAPABILITY_TARGET_PASS_RATE", "90"))
    print(f"Target pass rate: {target:.1f}%")
    return 0 if pass_rate >= target else 2


if __name__ == "__main__":
    raise SystemExit(main())
