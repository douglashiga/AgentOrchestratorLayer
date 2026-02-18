"""
Agent Orchestrator — Main CLI Entrypoint.

Wires all layers and runs the interactive CLI loop.
"""

import argparse
import json
import logging
import sys
import os
import asyncio
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from entry.cli import CLIAdapter
from entry.telegram import TelegramEntryAdapter
from conversation.manager import ConversationManager
from intent.adapter import IntentAdapter
from planner.service import PlannerService
from execution.engine import ExecutionEngine
from orchestrator.orchestrator import Orchestrator
from registry.domain_registry import HandlerRegistry
from registry.db import RegistryDB
from registry.loader import RegistryLoader
from skills.gateway import SkillGateway
from skills.registry import SkillRegistry
from skills.implementations.mcp_adapter import MCPAdapter
from memory.store import SQLiteMemoryStore
from models.selector import ModelSelector
from shared.models import EntryRequest, ExecutionIntent, IntentOutput
from shared.response_formatter import format_domain_output
from shared.workflow_contracts import ClarificationAnswer

# ─── Configuration ──────────────────────────────────────────────

logger = logging.getLogger(__name__)

# Ensure local .env is loaded before reading runtime configuration.
load_dotenv(override=False)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_INTENT_MODEL = os.getenv("OLLAMA_INTENT_MODEL", "llama3.1:8b")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", OLLAMA_INTENT_MODEL)
MCP_URL = os.getenv("MCP_URL", "http://localhost:8000/sse")
DB_PATH = os.getenv("DB_PATH", "agent.db")
REGISTRY_DB_PATH = os.getenv("REGISTRY_DB_PATH", "registry.db")
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "memory.db")
MEMORY_BOOTSTRAP_JSON = os.getenv("MEMORY_BOOTSTRAP_JSON", "").strip()
LOG_LEVEL = logging.INFO
BOOTSTRAP_DOMAINS_JSON = os.getenv("BOOTSTRAP_DOMAINS_JSON", "").strip()
BOOTSTRAP_DOMAINS_FILE = os.getenv("BOOTSTRAP_DOMAINS_FILE", "").strip()
AUTO_SYNC_REMOTE_CAPABILITIES = os.getenv("AUTO_SYNC_REMOTE_CAPABILITIES", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
SEED_CORE_DEFAULTS = os.getenv("SEED_CORE_DEFAULTS", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
SOFT_CONFIRMATION_ENABLED = os.getenv("SOFT_CONFIRMATION_ENABLED", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
SOFT_CONFIRM_THRESHOLD = float(os.getenv("SOFT_CONFIRM_THRESHOLD", "0.94"))
ORCHESTRATOR_CONFIDENCE_THRESHOLD = float(
    os.getenv("ORCHESTRATOR_CONFIDENCE_THRESHOLD", str(SOFT_CONFIRM_THRESHOLD))
)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_ENTRY_POLL_TIMEOUT_SECONDS = int(os.getenv("TELEGRAM_ENTRY_POLL_TIMEOUT_SECONDS", "20"))
TELEGRAM_ENTRY_REQUEST_TIMEOUT_SECONDS = float(os.getenv("TELEGRAM_ENTRY_REQUEST_TIMEOUT_SECONDS", "35"))
TELEGRAM_ENTRY_ALLOWED_CHAT_IDS = {
    item.strip()
    for item in os.getenv("TELEGRAM_ENTRY_ALLOWED_CHAT_IDS", "").split(",")
    if item.strip()
}
TELEGRAM_ENTRY_DEBUG = os.getenv("TELEGRAM_ENTRY_DEBUG", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
TELEGRAM_ENTRY_DEBUG_MAX_CHARS = int(os.getenv("TELEGRAM_ENTRY_DEBUG_MAX_CHARS", "3200"))
INTENT_FORCE_DOMAIN_ROUTING = os.getenv("INTENT_FORCE_DOMAIN_ROUTING", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# ─── Rich Console ───────────────────────────────────────────────

console = Console()


def setup_logging() -> None:
    """Configure logging."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def preload_models() -> None:
    """Warm up local Ollama models so first query is fast."""
    import httpx
    provider = os.getenv("MODEL_PROVIDER", "auto").strip().lower()
    base_url = os.getenv("MODEL_BASE_URL", OLLAMA_URL).strip() or OLLAMA_URL
    base_url_l = base_url.lower()
    if provider in {"anthropic", "openai_compatible"}:
        return
    if provider == "auto" and ("anthropic.com" in base_url_l or "openai.com" in base_url_l):
        return
    for model in (OLLAMA_INTENT_MODEL, OLLAMA_CHAT_MODEL):
        try:
            httpx.post(
                f"{base_url.rstrip('/')}/api/chat",
                json={"model": model, "messages": [], "keep_alive": "10m"},
                timeout=30.0,
            )
            logger.info("Preloaded model: %s", model)
        except Exception:
            logger.warning("Could not preload model: %s", model)


def _load_bootstrap_domains() -> list[dict]:
    """
    Load bootstrap domain config from env JSON or file path.
    Priority:
    1. BOOTSTRAP_DOMAINS_JSON
    2. BOOTSTRAP_DOMAINS_FILE
    """
    if BOOTSTRAP_DOMAINS_JSON:
        try:
            payload = json.loads(BOOTSTRAP_DOMAINS_JSON)
            if isinstance(payload, list):
                return payload
            logger.warning("BOOTSTRAP_DOMAINS_JSON must be a JSON array.")
        except json.JSONDecodeError as e:
            logger.warning("Invalid BOOTSTRAP_DOMAINS_JSON: %s", e)

    if BOOTSTRAP_DOMAINS_FILE:
        path = Path(BOOTSTRAP_DOMAINS_FILE)
        if not path.exists():
            logger.warning("BOOTSTRAP_DOMAINS_FILE not found: %s", path)
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return payload
            logger.warning("BOOTSTRAP_DOMAINS_FILE must contain a JSON array: %s", path)
        except Exception as e:
            logger.warning("Failed to parse BOOTSTRAP_DOMAINS_FILE '%s': %s", path, e)

    return []


def _bootstrap_memory(memory_store: SQLiteMemoryStore) -> None:
    """
    Seed memory store from env JSON.

    Accepted formats:
    - {"preferred_market":"SE","risk_mode":"moderate"}
    - {"global":{"preferred_market":"SE"},"session:abc":{"risk_mode":"high"}}
    """
    if not MEMORY_BOOTSTRAP_JSON:
        return

    try:
        payload = json.loads(MEMORY_BOOTSTRAP_JSON)
    except json.JSONDecodeError as e:
        logger.warning("Invalid MEMORY_BOOTSTRAP_JSON: %s", e)
        return

    if not isinstance(payload, dict):
        logger.warning("MEMORY_BOOTSTRAP_JSON must be a JSON object.")
        return

    direct_global = all(not isinstance(v, dict) for v in payload.values())
    if direct_global:
        for key, value in payload.items():
            memory_store.save(str(key), value, namespace="global")
        logger.info("Bootstrapped %d memory entries in global namespace", len(payload))
        return

    total = 0
    for namespace, values in payload.items():
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            memory_store.save(str(key), value, namespace=str(namespace))
            total += 1
    logger.info("Bootstrapped %d memory entries across namespaces", total)


def _is_yes(text: str) -> bool:
    return text.strip().lower() in {"sim", "s", "yes", "y", "ok", "confirmo", "confirmar"}


def _is_no(text: str) -> bool:
    return text.strip().lower() in {"nao", "não", "n", "no", "cancelar", "cancela"}


def _looks_like_chat_id(value: str) -> bool:
    return bool(re.fullmatch(r"-?\d+", value.strip()))


def _normalize_text(text: str) -> str:
    raw = (text or "").strip().lower()
    if not raw:
        return ""
    normalized = unicodedata.normalize("NFKD", raw)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_only).strip()


def _tokenize_text(text: str) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    return re.findall(r"[a-z0-9]+", normalized)


def _token_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


def _hint_match_score(query_norm: str, query_tokens: list[str], hint_text: str) -> float:
    hint_norm = _normalize_text(hint_text)
    if not hint_norm:
        return 0.0
    score = 0.0
    if hint_norm in query_norm:
        score += 5.0
    hint_tokens = [token for token in _tokenize_text(hint_norm) if len(token) >= 3]
    if not hint_tokens or not query_tokens:
        return score

    matched = 0.0
    for hint_token in hint_tokens:
        best = max((_token_similarity(hint_token, query_token) for query_token in query_tokens), default=0.0)
        if best >= 0.86:
            matched += 1.0
        elif best >= 0.75 and len(hint_token) >= 5:
            matched += 0.5

    coverage = matched / max(1, len(hint_tokens))
    score += coverage * 3.0
    if coverage >= 0.8 and len(hint_tokens) >= 2:
        score += 1.0
    return score


def _is_notifier_capability(metadata: dict[str, Any]) -> bool:
    composition_meta = metadata.get("composition")
    if not isinstance(composition_meta, dict):
        return False
    return str(composition_meta.get("role", "")).strip().lower() == "notifier"


def _hint_values(payload: Any) -> list[str]:
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


def _capability_intent_score(
    query_norm: str,
    query_tokens: list[str],
    capability: str,
    metadata: dict[str, Any],
) -> float:
    specific_hints: list[str] = []
    cap_hint = str(capability).replace("_", " ").strip()
    if cap_hint:
        specific_hints.append(cap_hint)
    description = str(metadata.get("description", "")).strip()
    if description:
        specific_hints.append(description)
    specific_hints.extend(_hint_values(metadata.get("intent_hints")))

    domain_hints: list[str] = []
    domain_description = str(metadata.get("domain_description", "")).strip()
    if domain_description:
        domain_hints.append(domain_description)
    domain_hints.extend(_hint_values(metadata.get("domain_intent_hints")))

    specific_score = 0.0
    for hint in specific_hints:
        specific_score = max(specific_score, _hint_match_score(query_norm, query_tokens, hint))

    domain_score = 0.0
    for hint in domain_hints:
        domain_score = max(domain_score, _hint_match_score(query_norm, query_tokens, hint))

    # Domain hints identify the area; capability hints must dominate method choice.
    return specific_score + (domain_score * 0.1)


def _intent_match_min_score(force_domain_routing: bool) -> float:
    return 1.0 if force_domain_routing else 3.5


def _looks_like_notify_request(
    query: str,
    registry: HandlerRegistry | None,
    *,
    force_domain_routing: bool = False,
) -> bool:
    if registry is None:
        return False

    query_norm = _normalize_text(query)
    query_tokens = _tokenize_text(query)
    if not query_norm or not query_tokens:
        return False

    best_score = 0.0
    for capability in registry.registered_capabilities:
        metadata = registry.get_metadata(capability)
        if not isinstance(metadata, dict) or not _is_notifier_capability(metadata):
            continue
        score = _capability_intent_score(
            query_norm=query_norm,
            query_tokens=query_tokens,
            capability=capability,
            metadata=metadata,
        )
        best_score = max(best_score, score)

    return best_score >= _intent_match_min_score(force_domain_routing)


def _reroute_general_chat_from_registry(intent, registry, params: dict[str, Any], force_domain_routing: bool = False):
    if registry is None:
        return None
    query = intent.original_query or ""
    query_norm = _normalize_text(query)
    query_tokens = _tokenize_text(query)
    if not query_norm or not query_tokens:
        return None
    best: tuple[float, str, str, dict[str, Any]] | None = None
    for capability in registry.registered_capabilities:
        metadata = registry.get_metadata(capability) if registry else {}
        if not isinstance(metadata, dict):
            continue
        domain = str(metadata.get("domain", "")).strip()
        if not domain or domain == "general":
            continue

        if _is_notifier_capability(metadata):
            continue

        score = _capability_intent_score(
            query_norm=query_norm,
            query_tokens=query_tokens,
            capability=capability,
            metadata=metadata,
        )
        if score <= 0:
            continue
        if best is None or score > best[0]:
            best = (score, domain, capability, metadata)

    if best is None:
        return None

    score, domain, capability, metadata = best
    min_score = _intent_match_min_score(force_domain_routing)
    if score < min_score:
        return None

    rerouted_params: dict[str, Any] = {}
    parameter_specs = metadata.get("parameter_specs")
    if isinstance(parameter_specs, dict):
        query_norm = _normalize_text(query)
        for param_name, spec in parameter_specs.items():
            if not isinstance(spec, dict):
                continue
            aliases = spec.get("aliases")
            if not isinstance(aliases, dict):
                continue
            for alias_raw, value in aliases.items():
                alias = _normalize_text(str(alias_raw))
                if alias and re.search(rf"\b{re.escape(alias)}\b", query_norm):
                    rerouted_params[str(param_name)] = value
                    break

    if params.get("notify") is True or _looks_like_notify_request(
        query,
        registry,
        force_domain_routing=force_domain_routing,
    ):
        rerouted_params["notify"] = True

    confidence_floor = 0.85
    if force_domain_routing:
        confidence_floor = min(0.75, 0.5 + (max(0.0, score) * 0.06))

    return intent.model_copy(
        update={
            "domain": domain,
            "capability": capability,
            "parameters": rerouted_params,
            "confidence": max(float(intent.confidence), confidence_floor),
        }
    )


def _normalize_parameter_value(value: Any, spec: dict[str, Any]) -> Any:
    if isinstance(value, list):
        item_spec = spec.get("items") if isinstance(spec.get("items"), dict) else spec
        return [_normalize_parameter_value(item, item_spec) for item in value]

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
            text_norm = _normalize_text(text)
            for alias_key, alias_value in aliases.items():
                if _normalize_text(str(alias_key)) == text_norm:
                    direct = alias_value
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

    type_name = str(spec.get("type", "")).strip().lower()
    if type_name in {"boolean", "bool"}:
        lowered = text.lower()
        if lowered in {"true", "1", "yes", "y", "sim", "s", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "nao", "não", "off"}:
            return False
    if type_name in {"integer", "int"}:
        try:
            return int(text)
        except Exception:
            return value
    if type_name in {"number", "float"}:
        try:
            return float(text)
        except Exception:
            return value

    return text


def _infer_value_from_symbol_suffix(params: dict[str, Any], spec: dict[str, Any]) -> Any:
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
        if not suffix:
            continue
        if symbol_value.endswith(suffix):
            return inferred
    return None


def _apply_parameter_contracts(params: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    parameter_specs = metadata.get("parameter_specs")
    if not isinstance(parameter_specs, dict):
        return params

    normalized = dict(params)
    for param_name, spec in parameter_specs.items():
        if not isinstance(spec, dict):
            continue
        key = str(param_name).strip()
        if not key:
            continue
        value = normalized.get(key)
        # If field is missing, check for a *_text variant supplied by the intent extractor
        # (e.g. market_text → market, period_text → period).
        # The raw text value will be alias-resolved by _normalize_parameter_value below.
        if value in (None, ""):
            text_raw = normalized.get(f"{key}_text")
            if text_raw not in (None, ""):
                value = text_raw
        if value in (None, ""):
            inferred = _infer_value_from_symbol_suffix(normalized, spec)
            if inferred not in (None, ""):
                normalized[key] = inferred
                value = inferred
            elif "default" in spec:
                normalized[key] = spec.get("default")
                value = normalized.get(key)
        if value is None:
            continue
        normalized[key] = _normalize_parameter_value(value, spec)

    return normalized


def _required_parameter_names(metadata: dict[str, Any]) -> set[str]:
    required: set[str] = set()
    parameter_specs = metadata.get("parameter_specs")
    if isinstance(parameter_specs, dict):
        for param_name, spec in parameter_specs.items():
            if not isinstance(spec, dict):
                continue
            if bool(spec.get("required")):
                name = str(param_name).strip()
                if name:
                    required.add(name)

    schema = metadata.get("schema")
    if isinstance(schema, dict):
        raw_required = schema.get("required")
        if isinstance(raw_required, list):
            for item in raw_required:
                name = str(item).strip()
                if name:
                    required.add(name)
    return required


def _capability_declares_parameter(metadata: dict[str, Any], param_name: str) -> bool:
    name = str(param_name).strip()
    if not name:
        return False

    parameter_specs = metadata.get("parameter_specs")
    if isinstance(parameter_specs, dict) and name in parameter_specs:
        return True

    schema = metadata.get("schema")
    if isinstance(schema, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict) and name in properties:
            return True
    return False


def _has_required_parameters(params: dict[str, Any], metadata: dict[str, Any]) -> bool:
    required = _required_parameter_names(metadata)
    if not required:
        return False

    for key in required:
        value = params.get(key)
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
        if isinstance(value, list) and len(value) == 0:
            return False
    return True


def _can_resolve_required_params_via_flow(metadata: dict[str, Any]) -> bool:
    required = _required_parameter_names(metadata)
    if not required:
        return False

    flow = metadata.get("flow")
    if isinstance(flow, str):
        try:
            flow = json.loads(flow)
        except Exception:
            flow = None
    if not isinstance(flow, dict):
        return False

    pre_steps = flow.get("pre")
    if not isinstance(pre_steps, list):
        return False

    resolvable: set[str] = set()
    for step in pre_steps:
        if not isinstance(step, dict):
            continue
        step_type = str(step.get("type", "")).strip().lower()
        param_name = str(step.get("param", "")).strip()
        if not step_type or not param_name:
            continue
        if step_type.startswith("resolve_"):
            resolvable.add(param_name)

    return required.issubset(resolvable)


def _normalize_intent_parameters(intent, registry, entry_request: EntryRequest | None = None):
    """Normalize alias params and apply metadata defaults before planning/execution."""
    params = dict(intent.parameters or {})
    capability = intent.capability

    # Canonicalize intent domain when capability is registered with explicit domain metadata.
    if registry is not None and registry.resolve_capability(capability) is not None:
        cap_metadata = registry.get_metadata(capability)
        if isinstance(cap_metadata, dict):
            cap_domain = str(cap_metadata.get("domain", "")).strip()
            if cap_domain and cap_domain != intent.domain:
                intent = intent.model_copy(update={"domain": cap_domain})

    if intent.domain == "general" and capability == "chat":
        rerouted = _reroute_general_chat_from_registry(
            intent,
            registry,
            params,
            force_domain_routing=INTENT_FORCE_DOMAIN_ROUTING,
        )
        if rerouted is not None:
            intent = rerouted
            params = dict(intent.parameters or {})
            capability = intent.capability

    metadata = registry.get_metadata(capability) if registry else {}
    if not isinstance(metadata, dict):
        metadata = {}

    is_notifier_capability = _is_notifier_capability(metadata)
    if not is_notifier_capability:
        if params.get("notify") is not True and _looks_like_notify_request(
            intent.original_query,
            registry,
            force_domain_routing=INTENT_FORCE_DOMAIN_ROUTING,
        ):
            params["notify"] = True

    # Communication fixes: chat target normalization for telegram capabilities.
    if str(metadata.get("channel", "")).strip().lower() == "telegram":
        default_chat_id = os.getenv("TELEGRAM_DEFAULT_CHAT_ID", "").strip()
        source_chat_id = ""
        if entry_request and entry_request.metadata.get("source") == "telegram":
            source_chat_id = str(entry_request.metadata.get("chat_id", "")).strip()

        has_group_id = _capability_declares_parameter(metadata, "group_id")
        has_chat_id = _capability_declares_parameter(metadata, "chat_id")

        if has_group_id:
            if not params.get("group_id") and params.get("chat_id"):
                params["group_id"] = params.get("chat_id")
            if not params.get("group_id") and source_chat_id:
                params["group_id"] = source_chat_id
            if not params.get("group_id") and default_chat_id:
                params["group_id"] = default_chat_id
            if params.get("group_id") and not _looks_like_chat_id(str(params["group_id"])):
                if source_chat_id:
                    params["group_id"] = source_chat_id
                elif default_chat_id:
                    params["group_id"] = default_chat_id
        if has_chat_id and not has_group_id:
            if not params.get("chat_id") and source_chat_id:
                params["chat_id"] = source_chat_id
            if not params.get("chat_id") and default_chat_id:
                params["chat_id"] = default_chat_id
            if params.get("chat_id") and not _looks_like_chat_id(str(params["chat_id"])):
                if source_chat_id:
                    params["chat_id"] = source_chat_id
                elif default_chat_id:
                    params["chat_id"] = default_chat_id

    # Apply metadata defaults for missing/null params.
    for key, value in metadata.items():
        if key.startswith("default_"):
            param_name = key.replace("default_", "")
            if param_name not in params or params[param_name] in (None, ""):
                params[param_name] = value

    params = _apply_parameter_contracts(params=params, metadata=metadata)

    confidence = float(intent.confidence)
    if (
        registry is not None
        and intent.domain != "general"
        and capability
        and registry.resolve_capability(capability) is not None
    ):
        query_norm = _normalize_text(intent.original_query or "")
        query_tokens = _tokenize_text(intent.original_query or "")
        capability_score = _capability_intent_score(
            query_norm=query_norm,
            query_tokens=query_tokens,
            capability=capability,
            metadata=metadata,
        )
        min_intent_score = _intent_match_min_score(INTENT_FORCE_DOMAIN_ROUTING)

        if _has_required_parameters(params, metadata):
            confidence = max(confidence, ORCHESTRATOR_CONFIDENCE_THRESHOLD)
        elif _can_resolve_required_params_via_flow(metadata) and capability_score >= min_intent_score:
            confidence = max(confidence, ORCHESTRATOR_CONFIDENCE_THRESHOLD)

    if params.get("notify") is True and intent.domain != "general":
        confidence = max(confidence, max(0.95, ORCHESTRATOR_CONFIDENCE_THRESHOLD))

    return intent.model_copy(update={"parameters": params, "confidence": confidence})


def _should_soft_confirm(intent) -> bool:
    if not SOFT_CONFIRMATION_ENABLED:
        return False
    if intent.domain == "general":
        return False
    if (intent.parameters or {}).get("notify") is True:
        return False
    return intent.confidence < SOFT_CONFIRM_THRESHOLD


def _build_soft_confirmation_message(intent, registry) -> str:
    metadata = registry.get_metadata(intent.capability) if registry else {}
    params = dict(intent.parameters or {})
    defaults_used: list[str] = []
    for key, value in metadata.items():
        if key.startswith("default_"):
            param_name = key.replace("default_", "")
            if params.get(param_name) == value:
                defaults_used.append(f"{param_name}={value}")

    params_preview = json.dumps(params, ensure_ascii=False)
    lines = [
        "Entendi sua solicitação assim:",
        f"- Domínio: {intent.domain}",
        f"- Ação: {intent.capability}",
        f"- Parâmetros: {params_preview}",
    ]
    if defaults_used:
        lines.append(f"- Valores padrão aplicados: {', '.join(defaults_used)}")
    lines.append("Confirmo a execução com esses valores? (sim/não)")
    return "\n".join(lines)


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


def _split_text_for_telegram(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # Try to split on newline for readability.
        if end < len(text):
            newline_pos = text.rfind("\n", start, end)
            if newline_pos > start:
                end = newline_pos
        chunks.append(text[start:end])
        start = end + (1 if end < len(text) and text[end] == "\n" else 0)
    return chunks


async def _send_telegram_debug_json(
    telegram_entry: TelegramEntryAdapter,
    chat_id: str,
    stage: str,
    payload: Any,
) -> None:
    if not TELEGRAM_ENTRY_DEBUG:
        return

    try:
        body = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
    except Exception:
        body = str(payload)

    chunks = _split_text_for_telegram(body, TELEGRAM_ENTRY_DEBUG_MAX_CHARS)
    total = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        header = f"[debug:{stage} {idx}/{total}]"
        await telegram_entry.send_message(chat_id, f"{header}\n{chunk}")


def build_pipeline() -> tuple[
    CLIAdapter,
    ConversationManager,
    IntentAdapter,
    PlannerService,
    ExecutionEngine,
    ModelSelector,
    MCPAdapter,
]:
    """Wire all layers together using Dynamic Registry."""
    # Shared
    model_selector = ModelSelector(ollama_url=OLLAMA_URL)
    memory_store = SQLiteMemoryStore(db_path=MEMORY_DB_PATH)
    _bootstrap_memory(memory_store)

    # Skills (still needed for local domains)
    mcp_adapter = MCPAdapter(mcp_url=MCP_URL)
    skill_registry = SkillRegistry()
    skill_registry.register("mcp_finance", mcp_adapter)
    skill_gateway = SkillGateway(skill_registry)

    # Dynamic Registry
    db = RegistryDB(db_path=REGISTRY_DB_PATH)
    runtime_registry = HandlerRegistry()
    loader = RegistryLoader(db, runtime_registry)

    # Context for Local Handlers
    loader_context = {
        "skill_gateway": skill_gateway,
        "model_selector": model_selector,
        "capability_catalog_provider": db.list_capabilities,
    }

    # Optional core defaults; can be disabled for fully table-driven configuration.
    if SEED_CORE_DEFAULTS:
        loader.sync_core_defaults()

    # Inject external domains from configuration into the registry DB.
    bootstrap_domains = _load_bootstrap_domains()
    if bootstrap_domains:
        logger.info("Applying bootstrap domains from configuration...")
        loader.bootstrap_domains(
            bootstrap_domains,
            sync_remote_capabilities=AUTO_SYNC_REMOTE_CAPABILITIES,
        )

    # Always refresh currently enabled remote domains from live services on startup.
    if AUTO_SYNC_REMOTE_CAPABILITIES:
        loader.sync_all_remote_capabilities()

    # Load All Domains
    loader.load_all(loader_context)

    # Core
    orchestrator = Orchestrator(domain_registry=runtime_registry, model_selector=model_selector)

    # Build goal catalog for Intent Adapter (goals only, no capabilities)
    goal_catalog = runtime_registry.build_goal_catalog()
    # Flatten goal catalog into per-goal items for the intent adapter
    goal_catalog_flat: list[dict[str, Any]] = []
    for domain_entry in goal_catalog:
        domain_name = domain_entry.get("domain", "")
        domain_desc = domain_entry.get("description", "")
        for goal_item in domain_entry.get("goals", []):
            flat_item = dict(goal_item)
            flat_item["domain"] = domain_name
            flat_item["domain_description"] = domain_desc
            goal_catalog_flat.append(flat_item)

    capability_catalog = db.list_capabilities()
    goal_catalog_dict = runtime_registry.build_goal_catalog_dict()

    intent_adapter = IntentAdapter(
        model_selector=model_selector,
        goal_catalog=goal_catalog_flat,
    )

    planner_service = PlannerService(
        capability_catalog=capability_catalog,
        model_selector=model_selector,
        memory_store=memory_store,
        goal_catalog_dict=goal_catalog_dict,
    )
    execution_engine = ExecutionEngine(orchestrator=orchestrator)
    conversation_manager = ConversationManager(db_path=DB_PATH)
    cli_adapter = CLIAdapter()

    return (
        cli_adapter,
        conversation_manager,
        intent_adapter,
        planner_service,
        execution_engine,
        model_selector,
        mcp_adapter,
    )


def render_result(intent, output) -> None:
    """Render DomainOutput to the CLI using Rich."""
    if output.status == "success":
        # Chat responses — just show the text
        if intent.capability == "chat":
            console.print()
            console.print(Panel(
                Text(output.explanation, style="white"),
                title="🤖 Assistant",
                border_style="cyan",
                box=box.ROUNDED,
            ))
            return

        # Header
        console.print()
        console.print(Panel(
            Text(output.explanation, style="bold green"),
            title="✅ Result",
            border_style="green",
            box=box.ROUNDED,
        ))

        # Result data table
        result = output.result
        # Copy to avoid mutating frozen pydantic if meaningful, but result is a dict
        result_view = result.copy()
        market_ctx = result_view.pop("_market_context", None)

        if result_view:
            table = Table(
                title="📊 Data",
                box=box.SIMPLE_HEAVY,
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Field", style="bold white")
            table.add_column("Value", style="white")

            for key, value in result_view.items():
                if isinstance(value, dict):
                    # Check if it's a comparison dictionary (like symbol -> metrics)
                    # Simple heuristic: if value is dict, maybe flatten or JSON dump
                    table.add_row(key, json.dumps(value, indent=2))
                elif isinstance(value, list):
                    # Check if list of dicts (like gainers) to show specialized table?
                    # For now just summary
                     table.add_row(key, f"[{len(value)} items] " + str(value)[:50] + "...")
                else:
                    table.add_row(key, str(value))

            console.print(table)

        # Market context
        if market_ctx:
            ctx_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
            ctx_table.add_column("", style="bold dim")
            ctx_table.add_column("", style="dim")
            ctx_table.add_row("🏦 Exchange", f"{market_ctx.get('exchange', '')} ({market_ctx.get('country', market_ctx.get('market', ''))})")
            ctx_table.add_row("💱 Currency", f"{market_ctx.get('currency_symbol', '')} {market_ctx.get('currency', '')}")
            ctx_table.add_row("🕐 Hours", f"{market_ctx.get('trading_hours', '')} ({market_ctx.get('timezone', '')})")
            ctx_table.add_row("📦 Lot/Settlement", f"{market_ctx.get('lot_size', '')} | {market_ctx.get('settlement', 'T+2')}")
            ctx_table.add_row("📋 Tax", f"{market_ctx.get('tax_model', '')} ({market_ctx.get('tax_rate_gains', '')})")
            if market_ctx.get("tax_notes"):
                ctx_table.add_row("💡 Notes", market_ctx["tax_notes"])
            console.print(Panel(ctx_table, title="📍 Market Context", border_style="dim", box=box.ROUNDED))

    elif output.status == "clarification":
        # Clarification needed
        console.print()
        console.print(Panel(
            Text(output.explanation, style="bold yellow"),
            title="❓ Clarification Needed",
            border_style="yellow",
            box=box.ROUNDED,
        ))

    else:
        # Failure case
        error_msg = output.metadata.get("error", "Unknown error")
        console.print()
        console.print(Panel(
            Text(f"Error: {error_msg}", style="bold red"),
            title="❌ Failed",
            border_style="red",
            box=box.ROUNDED,
        ))
        if output.explanation:
            console.print(Text(f"  ℹ️  {output.explanation}", style="dim"))


def render_intent_debug(intent) -> None:
    """Show extracted intent in debug panel."""
    table = Table(box=box.MINIMAL, show_header=False, padding=(0, 1))
    table.add_column("Key", style="bold yellow")
    table.add_column("Value", style="white")
    table.add_row("Domain", intent.domain)
    table.add_row("Capability", intent.capability)
    table.add_row("Original Query", intent.original_query)
    table.add_row("Parameters", str(intent.parameters))
    table.add_row("Confidence", f"{intent.confidence:.0%}")

    console.print(Panel(table, title="🧠 Intent (LLM)", border_style="yellow", box=box.ROUNDED))


async def run_telegram_loop() -> None:
    """Telegram polling loop using the same orchestration pipeline."""
    conversation: ConversationManager | None = None
    planner: PlannerService | None = None
    engine: ExecutionEngine | None = None
    model_selector: ModelSelector | None = None
    mcp_adapter: MCPAdapter | None = None
    pending_confirmation_by_session: dict[str, Any] = {}
    pending_workflow_by_session: dict[str, dict[str, str]] = {}

    if not TELEGRAM_BOT_TOKEN:
        console.print("[bold red]TELEGRAM_BOT_TOKEN is not configured.[/]")
        return

    console.print(Panel(
        Text.from_markup(
            "[bold cyan]Agent Orchestrator (Telegram Entry)[/bold cyan]\n"
            f"[dim]Intent: {OLLAMA_INTENT_MODEL} • Chat: {OLLAMA_CHAT_MODEL}[/dim]\n"
            f"[dim]Polling timeout: {TELEGRAM_ENTRY_POLL_TIMEOUT_SECONDS}s[/dim]"
        ),
        title="📨",
        border_style="cyan",
        box=box.DOUBLE,
    ))

    with console.status("[yellow]Loading models...[/yellow]", spinner="dots"):
        preload_models()

    try:
        (
            _cli,
            conversation,
            intent_adapter,
            planner,
            engine,
            model_selector,
            mcp_adapter,
        ) = build_pipeline()
    except Exception as e:
        console.print(f"[bold red]Failed to initialize pipeline:[/] {e}")
        return

    telegram_entry = TelegramEntryAdapter(
        bot_token=TELEGRAM_BOT_TOKEN,
        poll_timeout_seconds=TELEGRAM_ENTRY_POLL_TIMEOUT_SECONDS,
        request_timeout_seconds=TELEGRAM_ENTRY_REQUEST_TIMEOUT_SECONDS,
        allowed_chat_ids=TELEGRAM_ENTRY_ALLOWED_CHAT_IDS,
    )
    update_offset: int | None = None
    console.print("[dim]Telegram entry loop started.[/dim]")

    try:
        while True:
            entries, update_offset = await telegram_entry.poll_updates(update_offset)
            if not entries:
                continue

            for entry_request in entries:
                chat_id = str(entry_request.metadata.get("chat_id", "")).strip()
                if not chat_id:
                    continue

                try:
                    await _send_telegram_debug_json(
                        telegram_entry,
                        chat_id,
                        "entry_request",
                        entry_request.model_dump(mode="json"),
                    )
                    history = conversation.get_history(entry_request.session_id)
                    tg_intent_output: IntentOutput | None = None  # Set during extraction
                    pending_intent = pending_confirmation_by_session.get(entry_request.session_id)
                    pending_workflow = pending_workflow_by_session.get(entry_request.session_id)

                    if pending_workflow and pending_intent is None:
                        answer_value = entry_request.input_text.strip()
                        resume_answer = ClarificationAnswer(
                            question_id=pending_workflow["question_id"],
                            task_id=pending_workflow["task_id"],
                            selected_option=answer_value,
                            confirmed=not _is_no(answer_value),
                        )
                        output = await engine.resume_task(resume_answer)
                        await _send_telegram_debug_json(
                            telegram_entry,
                            chat_id,
                            "workflow_resumed",
                            output.model_dump(mode="json"),
                        )

                        if output.status == "clarification":
                            next_task_id = str(output.metadata.get("task_id", "")).strip()
                            next_question_id = str(output.metadata.get("question_id", "")).strip()
                            if next_task_id and next_question_id:
                                pending_workflow_by_session[entry_request.session_id] = {
                                    "task_id": next_task_id,
                                    "question_id": next_question_id,
                                }
                            else:
                                pending_workflow_by_session.pop(entry_request.session_id, None)
                        else:
                            pending_workflow_by_session.pop(entry_request.session_id, None)

                        conversation.save(entry_request.session_id, "user", entry_request.input_text)
                        response_text = format_domain_output(output, channel="telegram")
                        conversation.save(entry_request.session_id, "assistant", response_text)
                        await telegram_entry.send_message(chat_id, response_text)
                        continue

                    if pending_intent is not None:
                        if _is_yes(entry_request.input_text):
                            intent = pending_intent.model_copy(update={"confidence": 1.0})
                            pending_confirmation_by_session.pop(entry_request.session_id, None)
                            await telegram_entry.send_message(chat_id, "Confirmado. Executando...")
                            await _send_telegram_debug_json(
                                telegram_entry,
                                chat_id,
                                "confirmation_result",
                                {"confirmed": True, "intent": intent.model_dump(mode="json")},
                            )
                        elif _is_no(entry_request.input_text):
                            pending_confirmation_by_session.pop(entry_request.session_id, None)
                            msg = "Perfeito. Me diga novamente como deseja executar."
                            conversation.save(entry_request.session_id, "user", entry_request.input_text)
                            conversation.save(entry_request.session_id, "assistant", msg)
                            await _send_telegram_debug_json(
                                telegram_entry,
                                chat_id,
                                "confirmation_result",
                                {"confirmed": False, "reason": "user_declined"},
                            )
                            await telegram_entry.send_message(chat_id, msg)
                            continue
                        else:
                            msg = "Responda apenas 'sim' ou 'não' para confirmar."
                            conversation.save(entry_request.session_id, "user", entry_request.input_text)
                            conversation.save(entry_request.session_id, "assistant", msg)
                            await _send_telegram_debug_json(
                                telegram_entry,
                                chat_id,
                                "confirmation_result",
                                {"confirmed": None, "reason": "invalid_confirmation_text"},
                            )
                            await telegram_entry.send_message(chat_id, msg)
                            continue
                    else:
                        tg_intent_output = intent_adapter.extract(
                            entry_request.input_text,
                            history,
                            session_id=entry_request.session_id,
                        )
                        await _send_telegram_debug_json(
                            telegram_entry,
                            chat_id,
                            "intent_extracted",
                            tg_intent_output.model_dump(mode="json"),
                        )
                        # Resolve goal → capability
                        intent = planner.resolve_intent(tg_intent_output)
                        intent = _normalize_intent_parameters(
                            intent,
                            engine.orchestrator.domain_registry,
                            entry_request=entry_request,
                        )
                        await _send_telegram_debug_json(
                            telegram_entry,
                            chat_id,
                            "intent_normalized",
                            intent.model_dump(mode="json"),
                        )

                        if _should_soft_confirm(intent):
                            question = _build_soft_confirmation_message(intent, engine.orchestrator.domain_registry)
                            pending_confirmation_by_session[entry_request.session_id] = intent
                            conversation.save(entry_request.session_id, "user", entry_request.input_text)
                            conversation.save(entry_request.session_id, "assistant", question)
                            await _send_telegram_debug_json(
                                telegram_entry,
                                chat_id,
                                "confirmation_prompt",
                                {"question": question, "intent": intent.model_dump(mode="json")},
                            )
                            await telegram_entry.send_message(chat_id, question)
                            continue

                    plan_input = tg_intent_output if tg_intent_output is not None else intent
                    plan = planner.generate_plan(plan_input, session_id=entry_request.session_id)
                    await _send_telegram_debug_json(
                        telegram_entry,
                        chat_id,
                        "execution_plan",
                        {
                            "plan": plan.model_dump(mode="json"),
                            "memory_context": getattr(planner, "last_memory_context", {}),
                        },
                    )
                    output = await engine.execute_plan(plan, original_intent=intent)
                    await _send_telegram_debug_json(
                        telegram_entry,
                        chat_id,
                        "execution_output",
                        output.model_dump(mode="json"),
                    )

                    if output.status == "clarification":
                        task_id = str(output.metadata.get("task_id", "")).strip()
                        question_id = str(output.metadata.get("question_id", "")).strip()
                        if task_id and question_id:
                            pending_workflow_by_session[entry_request.session_id] = {
                                "task_id": task_id,
                                "question_id": question_id,
                            }
                    else:
                        pending_workflow_by_session.pop(entry_request.session_id, None)

                    conversation.save(entry_request.session_id, "user", entry_request.input_text)
                    if output.status in ("success", "clarification"):
                        response_text = format_domain_output(output, channel="telegram")
                    else:
                        response_text = format_domain_output(output, channel="telegram")
                    conversation.save(entry_request.session_id, "assistant", response_text)

                    await telegram_entry.send_message(chat_id, response_text)

                except Exception as e:
                    logger.exception("Telegram entry processing failed for chat_id=%s", chat_id)
                    await _send_telegram_debug_json(
                        telegram_entry,
                        chat_id,
                        "entry_error",
                        {"error": str(e)},
                    )
                    await telegram_entry.send_message(chat_id, f"Erro ao processar sua solicitação: {e}")

    except KeyboardInterrupt:
        console.print("\n[dim]Telegram loop interrupted. Goodbye! 👋[/dim]")
    finally:
        if engine:
            engine.close()
        if conversation:
            conversation.close()
        if planner:
            planner.close()
        if model_selector:
            model_selector.close()
        if mcp_adapter:
            mcp_adapter.close()


async def run_agent_loop() -> None:
    """Interactive Agent Loop."""
    conversation: ConversationManager | None = None
    planner: PlannerService | None = None
    engine: ExecutionEngine | None = None
    model_selector: ModelSelector | None = None
    mcp_adapter: MCPAdapter | None = None
    pending_confirmation_intent = None
    pending_workflow: dict[str, str] | None = None

    console.print(Panel(
        Text.from_markup(
            "[bold cyan]Agent Orchestrator[/bold cyan]\n"
            f"[dim]Intent: {OLLAMA_INTENT_MODEL} • Chat: {OLLAMA_CHAT_MODEL}[/dim]\n"
            "[dim]Type your question or 'exit' to quit[/dim]"
        ),
        title="🤖",
        border_style="cyan",
        box=box.DOUBLE,
    ))

    # Preload models into GPU memory
    with console.status("[yellow]Loading models...[/yellow]", spinner="dots"):
        preload_models()

    try:
        (
            cli,
            conversation,
            intent_adapter,
            planner,
            engine,
            model_selector,
            mcp_adapter,
        ) = build_pipeline()
    except Exception as e:
        console.print(f"[bold red]Failed to initialize pipeline:[/] {e}")
        sys.exit(1)

    console.print(f"[dim]Session: {cli.session_id}[/dim]")
    console.print(f"[dim]Domains: {engine.orchestrator.domain_registry.registered_domains}[/dim]")
    # console.print(f"[dim]Capabilities: {len(engine.orchestrator.domain_registry.registered_capabilities)}[/dim]")
    console.print()

    try:
        while True:
            try:
                # Step 1: Entry — read input
                # Use run_in_executor for blocking input if needed, but console.input is robust enough for CLI
                raw_input = console.input("[bold cyan]You → [/]")

                if raw_input.strip().lower() in ("exit", "quit", "q"):
                    console.print("[dim]Goodbye! 👋[/dim]")
                    break

                if not raw_input.strip():
                    continue

                entry_request = cli.read_input(raw_input)
                intent_output: IntentOutput | None = None  # Set during extraction

                # Step 2: Conversation — get history
                history = conversation.get_history(entry_request.session_id)

                if pending_workflow is not None and pending_confirmation_intent is None:
                    answer_text = entry_request.input_text.strip()
                    resume_answer = ClarificationAnswer(
                        question_id=pending_workflow["question_id"],
                        task_id=pending_workflow["task_id"],
                        selected_option=answer_text,
                        confirmed=not _is_no(answer_text),
                    )
                    with console.status("[green]Resuming workflow...[/green]", spinner="dots"):
                        output = await engine.resume_task(resume_answer)
                    render_result(
                        intent=ExecutionIntent(
                            domain="workflow",
                            capability="resume_task",
                            confidence=1.0,
                            parameters={},
                            original_query=entry_request.input_text,
                        ),
                        output=output,
                    )

                    if output.status == "clarification":
                        next_task_id = str(output.metadata.get("task_id", "")).strip()
                        next_question_id = str(output.metadata.get("question_id", "")).strip()
                        if next_task_id and next_question_id:
                            pending_workflow = {"task_id": next_task_id, "question_id": next_question_id}
                        else:
                            pending_workflow = None
                    else:
                        pending_workflow = None

                    conversation.save(entry_request.session_id, "user", entry_request.input_text)
                    response_text = format_domain_output(output, channel="frontend")
                    conversation.save(entry_request.session_id, "assistant", response_text)
                    console.print()
                    continue

                # Handle one-shot confirmation flow first.
                if pending_confirmation_intent is not None:
                    if _is_yes(entry_request.input_text):
                        intent = pending_confirmation_intent.model_copy(update={"confidence": 1.0})
                        pending_confirmation_intent = None
                        console.print("[dim]Confirmação recebida. Executando...[/dim]")
                    elif _is_no(entry_request.input_text):
                        pending_confirmation_intent = None
                        msg = "Perfeito. Me diga novamente como deseja executar."
                        console.print(Panel(Text(msg, style="bold yellow"), title="❓ Confirmation", border_style="yellow", box=box.ROUNDED))
                        conversation.save(entry_request.session_id, "user", entry_request.input_text)
                        conversation.save(entry_request.session_id, "assistant", msg)
                        console.print()
                        continue
                    else:
                        msg = "Responda apenas 'sim' ou 'não' para confirmar."
                        console.print(Panel(Text(msg, style="bold yellow"), title="❓ Confirmation", border_style="yellow", box=box.ROUNDED))
                        conversation.save(entry_request.session_id, "user", entry_request.input_text)
                        conversation.save(entry_request.session_id, "assistant", msg)
                        console.print()
                        continue
                else:
                    # Step 3: Intent Adapter — extract goal-based intent (LLM)
                    with console.status("[yellow]Thinking...[/yellow]", spinner="dots"):
                        try:
                            intent_output = intent_adapter.extract(
                                entry_request.input_text,
                                history,
                                session_id=entry_request.session_id,
                            )
                        except ValueError as e:
                            console.print(f"\n[bold red]Intent extraction failed:[/] {e}")
                            conversation.save(entry_request.session_id, "user", entry_request.input_text)
                            conversation.save(entry_request.session_id, "assistant", f"Error: {e}")
                            continue

                    # Step 3b: Resolve goal → capability (deterministic)
                    intent = planner.resolve_intent(intent_output)

                    intent = _normalize_intent_parameters(
                        intent,
                        engine.orchestrator.domain_registry,
                        entry_request=entry_request,
                    )

                    # Show intent debug
                    render_intent_debug(intent)

                    # Soft confirmation once, then execute on explicit "sim".
                    if _should_soft_confirm(intent):
                        question = _build_soft_confirmation_message(intent, engine.orchestrator.domain_registry)
                        pending_confirmation_intent = intent
                        console.print()
                        console.print(Panel(
                            Text(question, style="bold yellow"),
                            title="❓ Confirm Action",
                            border_style="yellow",
                            box=box.ROUNDED,
                        ))
                        conversation.save(entry_request.session_id, "user", entry_request.input_text)
                        conversation.save(entry_request.session_id, "assistant", question)
                        console.print()
                        continue

                # Step 4: Planner (Decomposition)
                # intent_output is set when going through normal extraction (IntentOutput from LLM).
                # When coming from a confirmed pending intent, intent_output is None and intent
                # is already a resolved ExecutionIntent — planner accepts both types.
                plan_input = intent_output if intent_output is not None else intent
                with console.status("[blue]Planning...[/blue]", spinner="dots"):
                    plan = planner.generate_plan(plan_input, session_id=entry_request.session_id)
                    logger.info("Plan generated: %d steps", len(plan.steps))

                # Step 5-8: Execution Engine (Orchestrator + Domain + Model)
                with console.status("[green]Executing...[/green]", spinner="dots"):
                    output = await engine.execute_plan(plan, original_intent=intent)

                # Step 9: Render output
                render_result(intent, output)

                if output.status == "clarification":
                    task_id = str(output.metadata.get("task_id", "")).strip()
                    question_id = str(output.metadata.get("question_id", "")).strip()
                    if task_id and question_id:
                        pending_workflow = {"task_id": task_id, "question_id": question_id}
                    else:
                        pending_workflow = None
                else:
                    pending_workflow = None

                # Step 9: Persist conversation
                conversation.save(entry_request.session_id, "user", entry_request.input_text)
                if output.status in ("success", "clarification"):
                    response_text = format_domain_output(output, channel="frontend")
                else:
                    response_text = format_domain_output(output, channel="frontend")
                conversation.save(entry_request.session_id, "assistant", response_text)

                console.print()

            except KeyboardInterrupt:
                console.print("\n[dim]Interrupted. Goodbye! 👋[/dim]")
                break
            except EOFError:
                console.print("\n[dim]Goodbye! 👋[/dim]")
                break
    finally:
        if engine:
            engine.close()
        if conversation:
            conversation.close()
        if planner:
            planner.close()
        if model_selector:
            model_selector.close()
        if mcp_adapter:
            mcp_adapter.close()


# ─── Admin Commands ─────────────────────────────────────────────

def admin_list_domains():
    db = RegistryDB(db_path=REGISTRY_DB_PATH)
    domains = db.list_domains()
    
    table = Table(title="Registered Domains")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Config", style="dim")
    table.add_column("Enabled", style="green")
    
    for d in domains:
        table.add_row(d["name"], d["type"], d["config"], str(bool(d["enabled"])))
        
    console.print(table)


def admin_add_domain(name, dtype, config_str):
    try:
        config = json.loads(config_str)
    except json.JSONDecodeError:
        console.print("[bold red]Error:[/] Config must be valid JSON.")
        return

    db = RegistryDB(db_path=REGISTRY_DB_PATH)
    db.register_domain(name, dtype, config)
    console.print(f"[bold green]Registered domain:[/] {name}")


def admin_sync_domain(name):
    db = RegistryDB(db_path=REGISTRY_DB_PATH)
    runtime_registry = HandlerRegistry() # Dummy for loader
    loader = RegistryLoader(db, runtime_registry)
    
    if loader.sync_capabilities(name):
        console.print(f"[bold green]Successfully synced capabilities for:[/] {name}")
    else:
        console.print(f"[bold red]Failed to sync capabilities for:[/] {name}")


def _parse_json_or_text(raw_value: str) -> Any:
    value = raw_value.strip()
    if not value:
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def admin_memory_set(key: str, value: str, namespace: str = "global") -> None:
    store = SQLiteMemoryStore(db_path=MEMORY_DB_PATH)
    try:
        parsed = _parse_json_or_text(value)
        store.save(key, parsed, namespace=namespace)
        console.print(f"[bold green]Memory saved:[/] {namespace}/{key} = {json.dumps(parsed, ensure_ascii=False)}")
    finally:
        store.close()


def admin_memory_get(key: str, namespace: str = "global") -> None:
    store = SQLiteMemoryStore(db_path=MEMORY_DB_PATH)
    try:
        value = store.get(key, namespace=namespace)
        if value is None:
            console.print(f"[bold yellow]Not found:[/] {namespace}/{key}")
            return
        console.print(f"[bold cyan]{namespace}/{key}[/]: {json.dumps(value, ensure_ascii=False)}")
    finally:
        store.close()


def admin_memory_search(query: str, namespace: str | None = None, limit: int = 10) -> None:
    store = SQLiteMemoryStore(db_path=MEMORY_DB_PATH)
    try:
        rows = store.search(query, namespace=namespace, limit=limit)
        if not rows:
            console.print("[bold yellow]No memory entries matched your query.[/]")
            return

        table = Table(title="Memory Search Results")
        table.add_column("Namespace", style="cyan")
        table.add_column("Key", style="magenta")
        table.add_column("Value", style="white")
        table.add_column("Updated", style="dim")
        for row in rows:
            table.add_row(
                str(row.get("namespace", "")),
                str(row.get("key", "")),
                json.dumps(row.get("value"), ensure_ascii=False),
                str(row.get("updated_at", "")),
            )
        console.print(table)
    finally:
        store.close()


def main() -> None:
    """Entrypoint with CLI args."""
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Agent Orchestrator")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run Agent
    subparsers.add_parser("run", help="Run interactive agent")
    subparsers.add_parser("run-telegram", help="Run Telegram polling entry channel")
    
    # Domain Management
    subparsers.add_parser("domain-list", help="List registered domains")
    
    add_parser = subparsers.add_parser("domain-add", help="Register a new domain")
    add_parser.add_argument("name", help="Domain name")
    add_parser.add_argument("type", choices=["local", "remote_http"], help="Domain type")
    add_parser.add_argument("config", help="JSON config string (e.g. '{\"url\": \"...\"}')")
    
    sync_parser = subparsers.add_parser("domain-sync", help="Sync capabilities from remote domain")
    sync_parser.add_argument("name", help="Domain name")

    memory_set_parser = subparsers.add_parser("memory-set", help="Set memory key/value")
    memory_set_parser.add_argument("key", help="Memory key")
    memory_set_parser.add_argument("value", help="Memory value (JSON or text)")
    memory_set_parser.add_argument("--namespace", default="global", help="Memory namespace (default: global)")

    memory_get_parser = subparsers.add_parser("memory-get", help="Get memory key")
    memory_get_parser.add_argument("key", help="Memory key")
    memory_get_parser.add_argument("--namespace", default="global", help="Memory namespace (default: global)")

    memory_search_parser = subparsers.add_parser("memory-search", help="Search memory entries")
    memory_search_parser.add_argument("query", help="Search query")
    memory_search_parser.add_argument("--namespace", default=None, help="Optional namespace filter")
    memory_search_parser.add_argument("--limit", type=int, default=10, help="Max results")
    
    args = parser.parse_args()
    
    if args.command == "domain-list":
        admin_list_domains()
    elif args.command == "domain-add":
        admin_add_domain(args.name, args.type, args.config)
    elif args.command == "domain-sync":
        admin_sync_domain(args.name)
    elif args.command == "memory-set":
        admin_memory_set(args.key, args.value, namespace=args.namespace)
    elif args.command == "memory-get":
        admin_memory_get(args.key, namespace=args.namespace)
    elif args.command == "memory-search":
        admin_memory_search(args.query, namespace=args.namespace, limit=args.limit)
    elif args.command == "run-telegram":
        try:
            asyncio.run(run_telegram_loop())
        except KeyboardInterrupt:
            pass
    elif args.command == "run" or args.command is None:
        try:
            asyncio.run(run_agent_loop())
        except KeyboardInterrupt:
            pass
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
