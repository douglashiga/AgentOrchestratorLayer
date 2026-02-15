"""
OpenAI-compatible API server for Agent Orchestrator.

Implements minimal endpoints required by Open WebUI:
- GET /v1/models
- POST /v1/chat/completions
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import httpx
from pydantic import BaseModel, ConfigDict, Field

from main import build_pipeline, _normalize_intent_parameters
from shared.delivery_layer import build_delivery_payload
from shared.models import IntentOutput
from shared.workflow_contracts import ClarificationAnswer

OPENAI_API_DEBUG_TRACE = os.getenv("OPENAI_API_DEBUG_TRACE", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
OPENAI_API_INCLUDE_SUGGESTIONS = os.getenv("OPENAI_API_INCLUDE_SUGGESTIONS", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
OPENAI_API_SUGGESTIONS_IN_CONTENT = os.getenv("OPENAI_API_SUGGESTIONS_IN_CONTENT", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
OPENAI_API_SUGGESTION_ACTIONS_IN_CONTENT = os.getenv(
    "OPENAI_API_SUGGESTION_ACTIONS_IN_CONTENT",
    "false",
).strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
OPENAI_API_STREAM_STATUS_UPDATES = os.getenv("OPENAI_API_STREAM_STATUS_UPDATES", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
OPENAI_API_STREAM_PROGRESS_EVENTS = os.getenv("OPENAI_API_STREAM_PROGRESS_EVENTS", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
OPENAI_API_STREAM_PROGRESS_FORMAT = os.getenv("OPENAI_API_STREAM_PROGRESS_FORMAT", "human").strip().lower()
OPENAI_API_STREAM_CHUNK_SIZE = int(os.getenv("OPENAI_API_STREAM_CHUNK_SIZE", "160"))
OPENAI_API_SIMULATED_STREAM_DELAY_MS = float(os.getenv("OPENAI_API_SIMULATED_STREAM_DELAY_MS", "12"))
GENERAL_FASTPATH_ENABLED = os.getenv("GENERAL_FASTPATH_ENABLED", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MODEL_ID_DEFAULT = "agent-orchestrator"
MODEL_ID_FASTPATH = "agent-orchestrator-fastpath"
OPENAI_API_PRELOAD_MODELS = os.getenv("OPENAI_API_PRELOAD_MODELS", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
OPENAI_API_PRELOAD_TIMEOUT_SECONDS = float(os.getenv("OPENAI_API_PRELOAD_TIMEOUT_SECONDS", "40"))
_TICKER_PATTERN = re.compile(r"\b[A-Z]{4}(?:3|4|5|6|11)(?:F)?(?:\.[A-Z]{1,4})?\b")
_SYMBOL_SUFFIX_PATTERN = re.compile(r"\b[A-Z0-9-]{1,12}\.(?:SA|ST|US|L)\b")
_GENERAL_CHAT_HINTS = (
    "oi",
    "olá",
    "ola",
    "bom dia",
    "boa tarde",
    "boa noite",
    "hello",
    "hi",
    "hey",
    "obrigado",
    "obrigada",
    "valeu",
    "thanks",
    "thank you",
    "como você está",
    "como voce esta",
    "como vc está",
    "como vc esta",
    "quem é você",
    "quem e voce",
    "como funciona",
    "what can you do",
    "help",
)
_FINANCE_MARKERS = (
    "ação",
    "acoes",
    "ação",
    "acoes",
    "stock",
    "stocks",
    "ticker",
    "preço",
    "preco",
    "cotação",
    "cotacao",
    "dividendo",
    "dividend",
    "mercado",
    "market",
    "rsi",
    "macd",
    "option",
    "opção",
    "opcao",
    "fundamental",
    "fundamento",
    "ibov",
    "nasdaq",
    "nyse",
    "b3",
)
_ACTION_MARKERS = (
    "envie",
    "enviar",
    "manda",
    "mandar",
    "notifique",
    "notificar",
    "compare",
    "listar",
    "liste",
    "busque",
    "buscar",
    "quero saber",
)


def _models_to_preload() -> list[str]:
    names = [
        os.getenv("OLLAMA_INTENT_MODEL", "llama3.1:8b").strip(),
        os.getenv("OLLAMA_CHAT_MODEL", os.getenv("OLLAMA_INTENT_MODEL", "llama3.1:8b")).strip(),
    ]
    seen: set[str] = set()
    ordered: list[str] = []
    for model_name in names:
        if not model_name or model_name in seen:
            continue
        seen.add(model_name)
        ordered.append(model_name)
    return ordered


def _preload_models_sync(base_url: str, provider: str = "ollama") -> None:
    if not OPENAI_API_PRELOAD_MODELS:
        return
    if provider != "ollama":
        return
    base_url = base_url.rstrip("/")
    for model_name in _models_to_preload():
        try:
            httpx.post(
                f"{base_url}/api/chat",
                json={"model": model_name, "messages": [], "keep_alive": "15m"},
                timeout=OPENAI_API_PRELOAD_TIMEOUT_SECONDS,
            )
        except Exception:
            # Keep startup resilient even if warmup fails.
            continue


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = Field(default=MODEL_ID_DEFAULT)
    messages: list[ChatMessage]
    stream: bool = False
    user: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    x_general_fastpath: bool | None = None


def _message_content_to_text(content: str | list[dict[str, Any]] | None) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return " ".join(parts).strip()
    return ""


def _extract_user_text(messages: list[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role != "user":
            continue
        return _message_content_to_text(msg.content)
    return ""


def _request_messages_to_turn_history(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for msg in messages:
        role = str(msg.role or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = _message_content_to_text(msg.content)
        if not content:
            continue
        turns.append({"role": role, "content": content})
    # Last user message is the current request payload and will be appended by the adapter.
    if turns and str(turns[-1].get("role", "")) == "user":
        turns.pop()
    return turns[-20:]


def _normalize_session_token(value: Any) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    token = re.sub(r"\s+", "-", token)
    token = re.sub(r"[^a-zA-Z0-9._:-]", "", token)
    return token[:96]


def _resolve_message_fingerprint(messages: list[ChatMessage]) -> str:
    # Use the first user utterance: stable through later turns in the same chat.
    first_user = ""
    for msg in messages:
        if msg.role != "user":
            continue
        first_user = _message_content_to_text(msg.content)
        if first_user:
            break
    if not first_user:
        return ""
    normalized = re.sub(r"\s+", " ", first_user.strip().lower())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def _resolve_session_id(request: ChatCompletionRequest) -> str:
    extras = request.model_extra if isinstance(request.model_extra, dict) else {}
    nested_metadata = extras.get("metadata")
    metadata = nested_metadata if isinstance(nested_metadata, dict) else {}
    nested_openwebui = extras.get("x_openwebui")
    openwebui = nested_openwebui if isinstance(nested_openwebui, dict) else {}

    primary_user = _normalize_session_token(request.user)
    extra_candidates = (
        extras.get("session_id"),
        extras.get("x_session_id"),
        extras.get("chat_id"),
        extras.get("conversation_id"),
        extras.get("thread_id"),
        metadata.get("session_id"),
        metadata.get("chat_id"),
        metadata.get("conversation_id"),
        openwebui.get("session_id"),
        openwebui.get("chat_id"),
        openwebui.get("conversation_id"),
    )
    explicit_chat = ""
    for candidate in extra_candidates:
        token = _normalize_session_token(candidate)
        if token:
            explicit_chat = token
            break

    if explicit_chat and primary_user:
        return f"{primary_user}:{explicit_chat}"
    if explicit_chat:
        return f"chat:{explicit_chat}"

    fingerprint = _resolve_message_fingerprint(request.messages)
    if primary_user:
        return f"user:{primary_user}"
    if fingerprint:
        return f"anon:{fingerprint}"
    return str(uuid.uuid4())[:8]


def _extract_symbol_from_text(text: str) -> str:
    upper = (text or "").strip().upper()
    if not upper:
        return ""

    explicit = re.search(r"\b([A-Z0-9-]{1,12}\.[A-Z]{1,4})\b", upper)
    if explicit:
        return explicit.group(1)

    b3_base = re.search(r"\b([A-Z]{4}(?:3|4|5|6|11)(?:F)?)\b", upper)
    if b3_base:
        return f"{b3_base.group(1)}.SA"

    alpha_only = re.fullmatch(r"\s*([A-Z]{1,8})\s*", upper)
    if alpha_only:
        return alpha_only.group(1)
    return ""


def _extract_ambiguous_raw_symbol(explanation: str) -> str:
    text = (explanation or "").strip()
    if not text:
        return ""
    match = re.search(r"ticker para '([^']+)'", text, flags=re.IGNORECASE)
    if not match:
        return ""
    return str(match.group(1)).strip().upper()


def _is_non_specific_symbol_token(value: str) -> bool:
    token = (value or "").strip().upper()
    if not token:
        return False
    if "." in token:
        return False
    if re.fullmatch(r"[A-Z]{4}(?:3|4|5|6|11)(?:F)?", token):
        return False
    return bool(re.fullmatch(r"[A-Z]{1,8}", token))


def _apply_symbol_selection_to_params(
    params: dict[str, Any],
    selected_symbol: str,
    *,
    ambiguous_raw_symbol: str = "",
) -> dict[str, Any]:
    selected = selected_symbol.strip().upper()
    if not selected:
        return dict(params)

    updated = dict(params)
    symbols = updated.get("symbols")

    if isinstance(symbols, list) and symbols:
        new_symbols: list[str] = []
        replaced = False
        for item in symbols:
            text = str(item).strip()
            if not text:
                continue
            token = text.upper()
            if ambiguous_raw_symbol and token == ambiguous_raw_symbol and not replaced:
                new_symbols.append(selected)
                replaced = True
                continue
            new_symbols.append(text)

        if not replaced:
            for idx, item in enumerate(new_symbols):
                if _is_non_specific_symbol_token(item):
                    new_symbols[idx] = selected
                    replaced = True
                    break

        if selected not in {str(item).strip().upper() for item in new_symbols}:
            new_symbols.append(selected)

        updated["symbols"] = new_symbols
        single_symbol = str(updated.get("symbol", "")).strip()
        if not single_symbol or _is_non_specific_symbol_token(single_symbol):
            updated["symbol"] = selected
        return updated

    updated["symbol"] = selected
    return updated


def _is_supported_clarification_metadata(metadata: dict[str, Any]) -> bool:
    resolution = str(metadata.get("resolution", "")).strip().lower()
    if resolution in {"ambiguous_symbol", "symbol_not_found", "symbol_lookup_failed"}:
        return True
    missing = metadata.get("missing_required_params")
    return isinstance(missing, list) and any(str(field).strip() for field in missing)


def _extract_clarification_payload(output: Any) -> dict[str, Any] | None:
    top_metadata = getattr(output, "metadata", {})
    top_result = getattr(output, "result", {})
    top_explanation = getattr(output, "explanation", "")

    if isinstance(top_metadata, dict) and _is_supported_clarification_metadata(top_metadata):
        return {
            "metadata": top_metadata,
            "result": top_result if isinstance(top_result, dict) else {},
            "explanation": str(top_explanation or ""),
        }

    if not isinstance(top_result, dict):
        return None

    steps = top_result.get("steps")
    if not isinstance(steps, dict):
        return None

    candidate_rows: list[tuple[int, dict[str, Any]]] = []
    for value in steps.values():
        if not isinstance(value, dict):
            continue
        if str(value.get("status", "")).strip().lower() != "clarification":
            continue
        metadata = value.get("metadata")
        if not isinstance(metadata, dict):
            continue
        if not _is_supported_clarification_metadata(metadata):
            continue
        priority = 0 if bool(value.get("required")) else 1
        candidate_rows.append((priority, value))

    if not candidate_rows:
        return None

    candidate_rows.sort(key=lambda item: item[0])
    chosen = candidate_rows[0][1]
    metadata = chosen.get("metadata")
    result = chosen.get("result")
    explanation = chosen.get("explanation", top_explanation)
    return {
        "metadata": metadata if isinstance(metadata, dict) else {},
        "result": result if isinstance(result, dict) else {},
        "explanation": str(explanation or ""),
    }


def _should_store_pending_clarification(output: Any) -> bool:
    if str(getattr(output, "status", "")).strip().lower() != "clarification":
        return False
    extracted = _extract_clarification_payload(output)
    return extracted is not None


def _build_pending_clarification_entry(intent: IntentOutput, output: Any) -> dict[str, Any]:
    extracted = _extract_clarification_payload(output)
    if extracted is None:
        metadata = getattr(output, "metadata", {})
        result = getattr(output, "result", {})
        explanation = getattr(output, "explanation", "")
    else:
        metadata = extracted.get("metadata", {})
        result = extracted.get("result", {})
        explanation = extracted.get("explanation", "")

    return {
        "intent": intent.model_dump(mode="json"),
        "metadata": metadata if isinstance(metadata, dict) else {},
        "result": result if isinstance(result, dict) else {},
        "explanation": str(explanation or ""),
    }


def _resolve_pending_clarification_intent(pending_entry: dict[str, Any], user_text: str) -> IntentOutput | None:
    if _is_decline_message(user_text):
        return None

    payload = pending_entry.get("intent")
    if not isinstance(payload, dict):
        return None
    try:
        original_intent = IntentOutput(**payload)
    except Exception:
        return None

    updated_params = dict(original_intent.parameters or {})
    metadata = pending_entry.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    resolution = str(metadata.get("resolution", "")).strip().lower()

    applied = False
    if resolution in {"ambiguous_symbol", "symbol_not_found", "symbol_lookup_failed"}:
        selected = _extract_symbol_from_text(user_text)
        if selected:
            ambiguous_raw = _extract_ambiguous_raw_symbol(str(pending_entry.get("explanation", "")))
            updated_params = _apply_symbol_selection_to_params(
                updated_params,
                selected,
                ambiguous_raw_symbol=ambiguous_raw,
            )
            applied = True

    missing = metadata.get("missing_required_params")
    if not applied and isinstance(missing, list):
        fields = [str(field).strip() for field in missing if str(field).strip()]
        if len(fields) == 1 and user_text.strip():
            updated_params[fields[0]] = user_text.strip()
            applied = True

    if not applied:
        return None

    return IntentOutput(
        domain=original_intent.domain,
        capability=original_intent.capability,
        confidence=1.0,
        parameters=updated_params,
        original_query=original_intent.original_query or user_text,
    )


def _build_suggestions_text(intent_capability: str) -> str:
    if not OPENAI_API_INCLUDE_SUGGESTIONS:
        return ""
    suggestions = [
        "Mostre o mesmo resultado para outro símbolo.",
        "Explique rapidamente o resultado em linguagem simples.",
    ]
    if intent_capability.startswith("get_"):
        suggestions.insert(0, "Envie esse resultado no Telegram.")
    return "\n\nSugestões:\n- " + "\n- ".join(suggestions)


def _build_suggestion_prompts(intent_capability: str, user_text: str) -> list[dict[str, str]]:
    prompts = [
        {
            "label": "Explicar simples",
            "prompt": "Explique esse resultado em linguagem simples e objetiva.",
        },
        {
            "label": "Rodar novamente",
            "prompt": user_text,
        },
    ]
    if intent_capability.startswith("get_"):
        prompts.insert(
            0,
            {
                "label": "Enviar no Telegram",
                "prompt": "Envie esse resultado no Telegram usando os valores padrão.",
            },
        )
    return prompts


def _suggestion_texts(suggestions: list[dict[str, str]] | list[str] | None) -> list[str]:
    if not isinstance(suggestions, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in suggestions:
        text = ""
        if isinstance(item, dict):
            text = str(item.get("prompt") or item.get("label") or "").strip()
        elif isinstance(item, str):
            text = item.strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _build_debug_trace(entry: dict[str, Any]) -> str:
    if not OPENAI_API_DEBUG_TRACE:
        return ""
    trace_json = json.dumps(entry, ensure_ascii=False, indent=2)
    return f"\n\n[debug]\n```json\n{trace_json}\n```"


def _build_clickable_actions_block(suggestions: list[dict[str, str]]) -> str:
    if not suggestions:
        return ""
    lines = ["\n\nAções rápidas (clique):"]
    for item in suggestions:
        label = item["label"]
        prompt = item["prompt"]
        lines.append(f"- [{label}](#prompt={quote(prompt)})")
    return "\n".join(lines)


def _format_response_text(
    base_text: str,
    intent_capability: str,
    user_text: str,
    debug_payload: dict[str, Any],
) -> tuple[str, list[dict[str, str]]]:
    tail_text, suggestions = _format_response_tail(
        intent_capability=intent_capability,
        user_text=user_text,
        debug_payload=debug_payload,
    )
    return f"{base_text}{tail_text}", suggestions


def _format_response_tail(
    *,
    intent_capability: str,
    user_text: str,
    debug_payload: dict[str, Any],
) -> tuple[str, list[dict[str, str]]]:
    suggestions = _build_suggestion_prompts(intent_capability, user_text) if OPENAI_API_INCLUDE_SUGGESTIONS else []
    sections: list[str] = []
    if OPENAI_API_INCLUDE_SUGGESTIONS and OPENAI_API_SUGGESTIONS_IN_CONTENT:
        sections.append(_build_suggestions_text(intent_capability))
        if OPENAI_API_SUGGESTION_ACTIONS_IN_CONTENT:
            sections.append(_build_clickable_actions_block(suggestions))
    if OPENAI_API_DEBUG_TRACE:
        sections.append(_build_debug_trace(debug_payload))
    return "".join(sections), suggestions


def _sse_line(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _chunk_payload(
    completion_id: str,
    model: str,
    created: int,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def _chunk_text(text: str, size: int) -> list[str]:
    if not text:
        return []
    chunk_size = max(32, size)
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _simulated_stream_delay_seconds() -> float:
    ms = OPENAI_API_SIMULATED_STREAM_DELAY_MS
    if ms < 0:
        ms = 0
    if ms > 2000:
        ms = 2000
    return ms / 1000.0


async def _iter_stream_text_chunks(text: str, size: int) -> Any:
    delay = _simulated_stream_delay_seconds()
    for piece in _chunk_text(text, size):
        yield piece
        if delay > 0:
            await asyncio.sleep(delay)
        else:
            await asyncio.sleep(0)


def _compact_json(payload: Any, *, max_chars: int = 420) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        text = str(payload)
    if len(text) <= max(64, max_chars):
        return text
    limit = max(64, max_chars) - 3
    return text[:limit] + "..."


def _compact_text(value: Any, *, max_chars: int = 140) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= max(24, max_chars):
        return text
    return text[: max(24, max_chars) - 3] + "..."


def _inline_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return _compact_text(value, max_chars=72)
    return _compact_json(value, max_chars=72)


def _inline_fields(payload: Any, *, max_items: int = 4) -> str:
    if not isinstance(payload, dict) or not payload:
        return ""
    items = list(payload.items())
    parts: list[str] = []
    for idx, (key, value) in enumerate(items):
        if idx >= max(1, max_items):
            parts.append("...")
            break
        parts.append(f"{key}={_inline_value(value)}")
    return ", ".join(parts)


def _summarize_progress_result(payload: Any) -> str:
    if not isinstance(payload, dict):
        return _compact_text(_compact_json(payload), max_chars=100)

    data = payload.get("data")
    core = data if isinstance(data, dict) else payload

    if isinstance(core, dict):
        symbol = core.get("symbol")
        price = core.get("price")
        currency = core.get("currency")
        if symbol not in (None, "") and price not in (None, ""):
            text = f"symbol={symbol}, price={price}"
            if currency not in (None, ""):
                text += f" {currency}"
            return _compact_text(text, max_chars=96)

    success = payload.get("success")
    error = payload.get("error")
    if isinstance(error, dict):
        error_message = _compact_text(error.get("message"), max_chars=60)
    else:
        error_message = _compact_text(error, max_chars=60)

    parts: list[str] = []
    if isinstance(success, bool):
        parts.append(f"success={'true' if success else 'false'}")
    if error_message:
        parts.append(f"error={error_message}")
    if parts:
        return ", ".join(parts)

    return _inline_fields(payload, max_items=3) or _compact_text(_compact_json(payload), max_chars=96)


def _format_progress_event_json(event: dict[str, Any], event_type: str) -> str:
    pretty = OPENAI_API_STREAM_PROGRESS_FORMAT == "json_pretty"

    def _render(tag: str, payload: Any) -> str:
        if pretty:
            try:
                pretty_text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
            except Exception:
                pretty_text = _compact_json(payload, max_chars=1000)
            return f"{tag}\n```json\n{pretty_text}\n```\n"
        return f"{tag} {_compact_json(payload)}\n"

    if event_type in {"intent_extracted", "intent_normalized", "resume_started"}:
        return _render("[progress]", event)

    if event_type == "plan_generated":
        plan = event.get("plan")
        if isinstance(plan, dict):
            mode = str(plan.get("execution_mode", "")).strip()
            steps = plan.get("steps")
            steps_count = len(steps) if isinstance(steps, list) else 0
            summary = {
                "type": event_type,
                "execution_mode": mode,
                "steps": steps_count,
            }
            return _render("[progress]", summary)
        return _render("[progress]", event)

    if event_type in {"step_started", "step_completed", "resume_completed"}:
        compact: dict[str, Any] = {"type": event_type}
        for key in (
            "step_id",
            "domain",
            "capability",
            "status",
            "confidence",
            "intent_confidence",
            "params",
            "result",
            "explanation",
        ):
            if key in event:
                compact[key] = event.get(key)
        if event_type == "step_completed":
            compact["result_summary"] = _summarize_progress_result(event.get("result"))
        return _render("[step]", compact)

    return _render("[progress]", event)


def _format_progress_event_human(event: dict[str, Any], event_type: str) -> str:
    if event_type in {"intent_extracted", "intent_normalized"}:
        intent = event.get("intent") if isinstance(event.get("intent"), dict) else {}
        domain = str(intent.get("domain", "")).strip() or "unknown"
        capability = str(intent.get("capability", "")).strip() or "unknown"
        params = _inline_fields(intent.get("parameters"), max_items=5)
        conf_raw = intent.get("confidence")
        conf_text = ""
        if isinstance(conf_raw, (int, float)):
            conf_text = f" | conf={float(conf_raw):.2f}"
        label = "Intent extraido" if event_type == "intent_extracted" else "Intent normalizado"
        suffix = f"{conf_text} | params: {params}" if params else conf_text
        return f"[progress] {label}: {domain}.{capability}{suffix}\n"

    if event_type == "plan_generated":
        plan = event.get("plan") if isinstance(event.get("plan"), dict) else {}
        mode = str(plan.get("execution_mode", "")).strip() or "sequential"
        steps = plan.get("steps")
        steps_count = len(steps) if isinstance(steps, list) else 0
        return f"[progress] Plano gerado: mode={mode}, steps={steps_count}\n"

    if event_type == "resume_started":
        task_id = _compact_text(event.get("task_id"), max_chars=24) or "n/a"
        question_id = _compact_text(event.get("question_id"), max_chars=24) or "n/a"
        return f"[progress] Retomando fluxo: task={task_id}, question={question_id}\n"

    if event_type == "resume_completed":
        status = str(event.get("status", "")).strip() or "unknown"
        explanation = _compact_text(event.get("explanation"), max_chars=120)
        suffix = f" | {explanation}" if explanation else ""
        return f"[step] Retomada concluida: status={status}{suffix}\n"

    if event_type == "step_started":
        step_id = str(event.get("step_id", "?"))
        domain = str(event.get("domain", "")).strip() or "unknown"
        capability = str(event.get("capability", "")).strip() or "unknown"
        params = _inline_fields(event.get("params"), max_items=5)
        conf = event.get("confidence")
        conf_text = f" | conf={float(conf):.2f}" if isinstance(conf, (int, float)) else ""
        suffix = f"{conf_text} | params: {params}" if params else conf_text
        return f"[step] Iniciando step {step_id}: {domain}.{capability}{suffix}\n"

    if event_type == "step_completed":
        step_id = str(event.get("step_id", "?"))
        domain = str(event.get("domain", "")).strip() or "unknown"
        capability = str(event.get("capability", "")).strip() or "unknown"
        status = str(event.get("status", "")).strip() or "unknown"
        explanation = _compact_text(event.get("explanation"), max_chars=120)
        result = _summarize_progress_result(event.get("result"))
        conf = event.get("confidence")
        intent_conf = event.get("intent_confidence")
        details: list[str] = []
        if isinstance(conf, (int, float)):
            details.append(f"conf={float(conf):.2f}")
        if isinstance(intent_conf, (int, float)):
            details.append(f"intent_conf={float(intent_conf):.2f}")
        if result:
            details.append(f"result: {result}")
        if explanation:
            details.append(explanation)
        suffix = f" | {' | '.join(details)}" if details else ""
        return f"[step] Step {step_id} concluido ({status}): {domain}.{capability}{suffix}\n"

    return f"[progress] {_compact_text(_compact_json(event), max_chars=200)}\n"


def _format_progress_event(event: dict[str, Any]) -> str:
    event_type = str(event.get("type", "")).strip().lower()
    if not event_type:
        return ""
    if OPENAI_API_STREAM_PROGRESS_FORMAT in {"json", "json_pretty"}:
        return _format_progress_event_json(event, event_type)
    return _format_progress_event_human(event, event_type)


async def _emit_progress_event(progress_callback: Any, event: dict[str, Any]) -> None:
    if progress_callback is None:
        return
    try:
        maybe_result = progress_callback(event)
        if inspect.isawaitable(maybe_result):
            await maybe_result
    except Exception:
        return


def _is_fastpath_model(model_name: str) -> bool:
    return model_name.strip().lower() == MODEL_ID_FASTPATH


def _resolve_fastpath_requested(request: ChatCompletionRequest) -> bool:
    if request.x_general_fastpath is not None:
        return request.x_general_fastpath
    if _is_fastpath_model(request.model):
        return True
    return GENERAL_FASTPATH_ENABLED


def _last_assistant_text(history: list[dict[str, Any]]) -> str:
    for item in reversed(history):
        if str(item.get("role", "")).lower() == "assistant":
            return str(item.get("content", "")).strip().lower()
    return ""


def _is_confirmation_like(text: str) -> bool:
    normalized = text.strip().lower()
    return normalized in {"sim", "s", "não", "nao", "n", "yes", "y", "no", "ok", "br", "us", "se"}


def _is_decline_message(text: str) -> bool:
    normalized = text.strip().lower()
    return normalized in {"não", "nao", "n", "no", "cancel", "cancelar", "stop"}


def _general_fastpath_candidate(user_text: str, history: list[dict[str, Any]]) -> tuple[bool, str]:
    text = user_text.strip()
    if not text:
        return False, "empty_text"
    lower = text.lower()

    if _is_confirmation_like(lower):
        return False, "confirmation_like"

    if _TICKER_PATTERN.search(text.upper()) or _SYMBOL_SUFFIX_PATTERN.search(text.upper()):
        return False, "ticker_detected"

    if any(marker in lower for marker in _FINANCE_MARKERS):
        return False, "finance_marker_detected"

    if any(marker in lower for marker in _ACTION_MARKERS):
        return False, "action_marker_detected"

    last_assistant = _last_assistant_text(history)
    if any(
        token in last_assistant
        for token in ("qual mercado", "which market", "qual símbolo", "which ticker", "enviar no telegram")
    ):
        return False, "followup_context"

    if any(hint in lower for hint in _GENERAL_CHAT_HINTS):
        return True, "general_hint_detected"

    return False, "no_general_hint"


def _fastpath_info(*, requested: bool, candidate: bool, used: bool, reason: str, mode: str) -> dict[str, Any]:
    return {
        "requested": requested,
        "candidate": candidate,
        "used": used,
        "reason": reason,
        "mode": mode,
    }


def _resolve_general_handler(engine: Any) -> Any | None:
    registry = engine.orchestrator.domain_registry
    handler = registry.resolve_capability("chat")
    if handler is not None:
        return handler
    return registry.resolve_domain("general")


def _general_prompt_messages(handler: Any, user_text: str) -> list[dict[str, str]]:
    if hasattr(handler, "build_chat_messages") and callable(handler.build_chat_messages):
        messages = handler.build_chat_messages(user_text)
        if isinstance(messages, list) and messages:
            return messages
    return [
        {
            "role": "system",
            "content": (
                "You are a friendly financial assistant. "
                "You can help users with stock market questions, financial data, and general conversation. "
                "Be concise, helpful, and friendly. "
                "If the user greets you, greet them back and briefly explain what you can do. "
                "Respond in the same language the user uses."
            ),
        },
        {"role": "user", "content": user_text},
    ]


def _resolve_general_model(handler: Any) -> str:
    policy = getattr(handler, "policy", None)
    model_name = getattr(policy, "model_name", "")
    if isinstance(model_name, str) and model_name.strip():
        return model_name.strip()
    return os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")


async def _execute_handler(handler: Any, intent: IntentOutput) -> Any:
    result = handler.execute(intent)
    if inspect.isawaitable(result):
        return await result
    return result


def _build_fastpath_debug_payload(
    *,
    session_id: str,
    user_text: str,
    model_name: str,
    fastpath: dict[str, Any],
    output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "entry": {
            "session_id": session_id,
            "user_text": user_text,
        },
        "fast_path": fastpath,
        "intent_extracted": {
            "domain": "general",
            "action": "chat",
            "confidence": 1.0,
            "parameters": {"message": user_text},
        },
        "intent_normalized": {
            "domain": "general",
            "action": "chat",
            "confidence": 1.0,
            "parameters": {"message": user_text},
        },
        "plan": {
            "execution_mode": "fast_path",
            "steps": [],
        },
        "fast_path_model": model_name,
    }
    if output is not None:
        payload["output"] = output
    return payload


async def _stream_ollama_chat_tokens(
    *,
    ollama_url: str,
    model_name: str,
    messages: list[dict[str, str]],
    temperature: float | None,
    max_tokens: int | None,
):
    options: dict[str, Any] = {
        "num_ctx": 4096,
        "num_predict": 512,
    }
    if temperature is not None:
        options["temperature"] = temperature
    if isinstance(max_tokens, int) and max_tokens > 0:
        options["num_predict"] = max(32, min(max_tokens, 2048))

    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        "keep_alive": "10m",
        "options": options,
    }

    async with httpx.AsyncClient(base_url=ollama_url, timeout=None) as client:
        async with client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if data.get("error"):
                    raise RuntimeError(str(data.get("error")))

                piece = str((data.get("message") or {}).get("content", ""))
                if piece:
                    yield piece

                if data.get("done"):
                    break


async def _run_agent_turn(
    *,
    session_id: str,
    user_text: str,
    conversation: Any,
    intent_adapter: Any,
    planner: Any,
    engine: Any,
    history: list[dict[str, Any]] | None = None,
    pending_workflow_by_session: dict[str, dict[str, str]] | None = None,
    pending_clarification_by_session: dict[str, dict[str, Any]] | None = None,
    progress_callback: Any = None,
) -> tuple[str, str, dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    pending_map = pending_workflow_by_session if isinstance(pending_workflow_by_session, dict) else {}
    pending_clarification_map = (
        pending_clarification_by_session if isinstance(pending_clarification_by_session, dict) else {}
    )
    pending_workflow = pending_map.get(session_id)
    if pending_workflow:
        answer_text = user_text.strip()
        answer = ClarificationAnswer(
            question_id=str(pending_workflow.get("question_id", "")).strip(),
            task_id=str(pending_workflow.get("task_id", "")).strip(),
            selected_option=answer_text,
            confirmed=not _is_decline_message(answer_text),
        )
        await _emit_progress_event(
            progress_callback,
            {
                "type": "resume_started",
                "task_id": answer.task_id,
                "question_id": answer.question_id,
            },
        )
        output = await engine.resume_task(answer)
        await _emit_progress_event(
            progress_callback,
            {
                "type": "resume_completed",
                "status": output.status,
                "result": output.result,
                "explanation": output.explanation,
            },
        )
        if output.status == "clarification":
            next_task_id = str(output.metadata.get("task_id", "")).strip()
            next_question_id = str(output.metadata.get("question_id", "")).strip()
            if next_task_id and next_question_id:
                pending_map[session_id] = {"task_id": next_task_id, "question_id": next_question_id}
            else:
                pending_map.pop(session_id, None)
        else:
            pending_map.pop(session_id, None)
        pending_clarification_map.pop(session_id, None)

        delivery = build_delivery_payload(output)
        base_text = delivery.content
        clean_response_text = base_text.strip()
        debug_payload = {
            "entry": {
                "session_id": session_id,
                "user_text": user_text,
            },
            "resume": {
                "task_id": answer.task_id,
                "question_id": answer.question_id,
                "answer": answer_text,
            },
            "output": output.model_dump(mode="json"),
        }
        response_text, suggestions = _format_response_text(
            base_text=base_text,
            intent_capability="resume_task",
            user_text=user_text,
            debug_payload=debug_payload,
        )
        conversation.save(session_id, "user", user_text)
        conversation.save(session_id, "assistant", clean_response_text)
        return response_text, "resume_task", debug_payload, suggestions, {
            "kind": delivery.kind,
            "content": delivery.content,
            "data": delivery.data,
        }

    turn_history = history if history is not None else conversation.get_history(session_id)
    pending_clarification = pending_clarification_map.get(session_id)
    resumed_from_clarification = False
    if isinstance(pending_clarification, dict):
        resumed_intent = _resolve_pending_clarification_intent(pending_clarification, user_text)
        if resumed_intent is not None:
            intent = resumed_intent
            resumed_from_clarification = True
            await _emit_progress_event(
                progress_callback,
                {
                    "type": "clarification_followup_applied",
                    "intent": {
                        "domain": intent.domain,
                        "capability": intent.capability,
                        "parameters": intent.parameters,
                    },
                },
            )
        else:
            pending_clarification_map.pop(session_id, None)

    if not resumed_from_clarification:
        intent = intent_adapter.extract(user_text, turn_history, session_id=session_id)
    intent_extracted = intent.model_dump(mode="json")
    await _emit_progress_event(
        progress_callback,
        {
            "type": "intent_extracted",
            "intent": {
                "domain": intent_extracted.get("domain"),
                "capability": intent_extracted.get("capability"),
                "confidence": intent_extracted.get("confidence"),
                "parameters": intent_extracted.get("parameters", {}),
            },
        },
    )
    intent = _normalize_intent_parameters(intent, engine.orchestrator.domain_registry, entry_request=None)
    intent_normalized = intent.model_dump(mode="json")
    await _emit_progress_event(
        progress_callback,
        {
            "type": "intent_normalized",
            "intent": {
                "domain": intent_normalized.get("domain"),
                "capability": intent_normalized.get("capability"),
                "confidence": intent_normalized.get("confidence"),
                "parameters": intent_normalized.get("parameters", {}),
            },
        },
    )
    plan = planner.generate_plan(intent, session_id=session_id)
    plan_dump = plan.model_dump(mode="json")
    await _emit_progress_event(
        progress_callback,
        {
            "type": "plan_generated",
            "plan": {
                "execution_mode": plan_dump.get("execution_mode"),
                "combine_mode": plan_dump.get("combine_mode"),
                "steps": plan_dump.get("steps", []),
            },
        },
    )
    output = await engine.execute_plan(
        plan,
        original_intent=intent,
        progress_callback=progress_callback,
    )
    if output.status == "clarification":
        task_id = str(output.metadata.get("task_id", "")).strip()
        question_id = str(output.metadata.get("question_id", "")).strip()
        if task_id and question_id:
            pending_map[session_id] = {"task_id": task_id, "question_id": question_id}
            pending_clarification_map.pop(session_id, None)
        elif _should_store_pending_clarification(output):
            pending_map.pop(session_id, None)
            pending_clarification_map[session_id] = _build_pending_clarification_entry(intent, output)
        else:
            pending_map.pop(session_id, None)
            pending_clarification_map.pop(session_id, None)
    else:
        pending_map.pop(session_id, None)
        pending_clarification_map.pop(session_id, None)

    delivery = build_delivery_payload(output)
    base_text = delivery.content
    clean_response_text = base_text.strip()
    debug_payload = {
        "entry": {
            "session_id": session_id,
            "user_text": user_text,
        },
        "intent_extracted": intent_extracted,
        "intent_normalized": intent_normalized,
        "memory_context": getattr(planner, "last_memory_context", {}),
        "plan": plan_dump,
        "output": output.model_dump(mode="json"),
    }
    response_text, suggestions = _format_response_text(
        base_text=base_text,
        intent_capability=intent.capability,
        user_text=user_text,
        debug_payload=debug_payload,
    )
    conversation.save(session_id, "user", user_text)
    # Keep history clean for better context quality in subsequent turns.
    conversation.save(session_id, "assistant", clean_response_text)
    return response_text, intent.capability, debug_payload, suggestions, {
        "kind": delivery.kind,
        "content": delivery.content,
        "data": delivery.data,
    }


async def _run_general_fastpath_turn(
    *,
    session_id: str,
    user_text: str,
    conversation: Any,
    engine: Any,
    fastpath: dict[str, Any],
) -> tuple[str, str, dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    handler = _resolve_general_handler(engine)
    if handler is None:
        raise RuntimeError("General handler is not available for fast-path mode.")

    intent = IntentOutput(
        domain="general",
        capability="chat",
        confidence=1.0,
        parameters={"message": user_text},
        original_query=user_text,
    )
    output = await _execute_handler(handler, intent)
    delivery = build_delivery_payload(output)
    base_text = delivery.content
    clean_response_text = base_text.strip()
    model_name = _resolve_general_model(handler)
    debug_payload = _build_fastpath_debug_payload(
        session_id=session_id,
        user_text=user_text,
        model_name=model_name,
        fastpath=fastpath,
        output=output.model_dump(mode="json") if hasattr(output, "model_dump") else None,
    )
    response_text, suggestions = _format_response_text(
        base_text=base_text,
        intent_capability="chat",
        user_text=user_text,
        debug_payload=debug_payload,
    )
    conversation.save(session_id, "user", user_text)
    conversation.save(session_id, "assistant", clean_response_text)
    return response_text, "chat", debug_payload, suggestions, {
        "kind": delivery.kind,
        "content": delivery.content,
        "data": delivery.data,
    }


@asynccontextmanager
async def lifespan(_app: FastAPI):
    (
        _cli,
        conversation,
        intent_adapter,
        planner,
        engine,
        model_selector,
        mcp_adapter,
    ) = build_pipeline()
    _preload_models_sync(
        base_url=str(getattr(model_selector, "ollama_url", os.getenv("OLLAMA_URL", "http://host.docker.internal:11434"))),
        provider=str(getattr(model_selector, "provider", "ollama")),
    )
    _app.state.conversation = conversation
    _app.state.intent_adapter = intent_adapter
    _app.state.planner = planner
    _app.state.engine = engine
    _app.state.model_selector = model_selector
    _app.state.mcp_adapter = mcp_adapter
    _app.state.pending_workflow_by_session = {}
    _app.state.pending_clarification_by_session = {}
    yield
    engine.close()
    conversation.close()
    planner.close()
    model_selector.close()
    mcp_adapter.close()


app = FastAPI(
    title="Agent Orchestrator OpenAI API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID_DEFAULT,
                "object": "model",
                "created": now,
                "owned_by": "agent",
            },
            {
                "id": MODEL_ID_FASTPATH,
                "object": "model",
                "created": now,
                "owned_by": "agent",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
    user_text = _extract_user_text(request.messages)
    if not user_text:
        user_text = "Hello"

    session_id = _resolve_session_id(request)
    conversation = app.state.conversation
    intent_adapter = app.state.intent_adapter
    planner = app.state.planner
    engine = app.state.engine
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    stored_history = conversation.get_history(session_id)
    request_history = _request_messages_to_turn_history(request.messages)
    history = request_history if request_history else stored_history
    fastpath_requested = _resolve_fastpath_requested(request)
    fastpath_candidate, fastpath_reason = _general_fastpath_candidate(user_text, history)
    model_provider = str(getattr(app.state.model_selector, "provider", "ollama")).strip().lower()
    stream_fastpath_supported = model_provider == "ollama"
    use_fastpath = fastpath_requested and fastpath_candidate and (not request.stream or stream_fastpath_supported)

    if request.stream:
        async def event_stream():
            yield _sse_line(
                _chunk_payload(
                    completion_id=completion_id,
                    model=request.model,
                    created=created,
                    delta={"role": "assistant"},
                )
            )

            if use_fastpath:
                streamed_any = False
                try:
                    if OPENAI_API_STREAM_STATUS_UPDATES:
                        yield _sse_line(
                            _chunk_payload(
                                completion_id=completion_id,
                                model=request.model,
                                created=created,
                                delta={"content": "⚡ Fast-path de chat geral ativo.\n"},
                            )
                        )

                    handler = _resolve_general_handler(engine)
                    if handler is None:
                        raise RuntimeError("General handler is not available.")

                    model_name = _resolve_general_model(handler)
                    messages = _general_prompt_messages(handler, user_text)
                    ollama_url = str(
                        getattr(app.state.model_selector, "ollama_url", os.getenv("OLLAMA_URL", "http://localhost:11434"))
                    )
                    pieces: list[str] = []
                    async for piece in _stream_ollama_chat_tokens(
                        ollama_url=ollama_url,
                        model_name=model_name,
                        messages=messages,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                    ):
                        streamed_any = True
                        pieces.append(piece)
                        yield _sse_line(
                            _chunk_payload(
                                completion_id=completion_id,
                                model=request.model,
                                created=created,
                                delta={"content": piece},
                            )
                        )
                        await asyncio.sleep(0)

                    clean_response_text = "".join(pieces).strip()
                    fastpath = _fastpath_info(
                        requested=fastpath_requested,
                        candidate=fastpath_candidate,
                        used=True,
                        reason=fastpath_reason,
                        mode="stream",
                    )
                    debug_payload = _build_fastpath_debug_payload(
                        session_id=session_id,
                        user_text=user_text,
                        model_name=model_name,
                        fastpath=fastpath,
                        output={
                            "status": "success",
                            "result": {"response": clean_response_text},
                            "explanation": clean_response_text,
                            "confidence": 1.0,
                            "metadata": {},
                        },
                    )
                    tail_text, suggestions = _format_response_tail(
                        intent_capability="chat",
                        user_text=user_text,
                        debug_payload=debug_payload,
                    )
                    suggestion_texts = _suggestion_texts(suggestions)
                    async for piece in _iter_stream_text_chunks(tail_text, OPENAI_API_STREAM_CHUNK_SIZE):
                        yield _sse_line(
                            _chunk_payload(
                                completion_id=completion_id,
                                model=request.model,
                                created=created,
                                delta={"content": piece},
                            )
                        )

                    conversation.save(session_id, "user", user_text)
                    conversation.save(session_id, "assistant", clean_response_text)
                    delivery = {"kind": "text", "content": clean_response_text, "data": {}}

                    if suggestion_texts:
                        yield _sse_line(
                            _chunk_payload(
                                completion_id=completion_id,
                                model=request.model,
                                created=created,
                                delta={"suggestions": suggestion_texts},
                            )
                        )
                    yield _sse_line(
                        _chunk_payload(
                            completion_id=completion_id,
                            model=request.model,
                            created=created,
                            delta={},
                            finish_reason="stop",
                        )
                    )
                    yield _sse_line(
                        {
                            "id": completion_id,
                            "object": "chat.completion.meta",
                            "created": created,
                            "model": request.model,
                            "x_openwebui": {
                                "suggestions": suggestion_texts,
                                "suggestion_actions": suggestions,
                                "delivery": delivery,
                                "debug": debug_payload if OPENAI_API_DEBUG_TRACE else {},
                            },
                        }
                    )
                except Exception as e:
                    if streamed_any:
                        error_text = f"\n\n❌ Erro no fast-path: {e}"
                        yield _sse_line(
                            _chunk_payload(
                                completion_id=completion_id,
                                model=request.model,
                                created=created,
                                delta={"content": error_text},
                            )
                        )
                        yield _sse_line(
                            _chunk_payload(
                                completion_id=completion_id,
                                model=request.model,
                                created=created,
                                delta={},
                                finish_reason="stop",
                            )
                        )
                    else:
                        if OPENAI_API_STREAM_STATUS_UPDATES:
                            yield _sse_line(
                                _chunk_payload(
                                    completion_id=completion_id,
                                    model=request.model,
                                    created=created,
                                    delta={"content": "↩️ Fast-path indisponível, usando pipeline completo...\n"},
                                )
                            )
                        response_text, _cap, debug_payload, suggestions, delivery = await _run_agent_turn(
                            session_id=session_id,
                            user_text=user_text,
                            conversation=conversation,
                            intent_adapter=intent_adapter,
                            planner=planner,
                            engine=engine,
                            history=history,
                            pending_workflow_by_session=app.state.pending_workflow_by_session,
                            pending_clarification_by_session=app.state.pending_clarification_by_session,
                        )
                        debug_payload["fast_path"] = _fastpath_info(
                            requested=fastpath_requested,
                            candidate=fastpath_candidate,
                            used=False,
                            reason=f"fallback: {e}",
                            mode="pipeline",
                        )
                        suggestion_texts = _suggestion_texts(suggestions)
                        async for piece in _iter_stream_text_chunks(response_text, OPENAI_API_STREAM_CHUNK_SIZE):
                            yield _sse_line(
                                _chunk_payload(
                                    completion_id=completion_id,
                                    model=request.model,
                                    created=created,
                                    delta={"content": piece},
                                )
                            )
                        if suggestion_texts:
                            yield _sse_line(
                                _chunk_payload(
                                    completion_id=completion_id,
                                    model=request.model,
                                    created=created,
                                    delta={"suggestions": suggestion_texts},
                                )
                            )
                        yield _sse_line(
                            _chunk_payload(
                                completion_id=completion_id,
                                model=request.model,
                                created=created,
                                delta={},
                                finish_reason="stop",
                            )
                        )
                        yield _sse_line(
                            {
                                "id": completion_id,
                                "object": "chat.completion.meta",
                                "created": created,
                                "model": request.model,
                                "x_openwebui": {
                                    "suggestions": suggestion_texts,
                                    "suggestion_actions": suggestions,
                                    "delivery": delivery,
                                    "debug": debug_payload if OPENAI_API_DEBUG_TRACE else {},
                                },
                            }
                        )
            else:
                if OPENAI_API_STREAM_STATUS_UPDATES:
                    for status_text in (
                        "🔄 Recebi sua solicitação.\n",
                        "🧠 Analisando intenção e plano...\n",
                        "⚙️ Executando tarefas...\n",
                    ):
                        yield _sse_line(
                            _chunk_payload(
                                completion_id=completion_id,
                                model=request.model,
                                created=created,
                                delta={"content": status_text},
                            )
                        )
                        await asyncio.sleep(0.01)

                try:
                    if OPENAI_API_STREAM_PROGRESS_EVENTS:
                        progress_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

                        async def _progress_callback(event: dict[str, Any]) -> None:
                            await progress_queue.put(event)

                        run_task = asyncio.create_task(
                            _run_agent_turn(
                                session_id=session_id,
                                user_text=user_text,
                                conversation=conversation,
                                intent_adapter=intent_adapter,
                                planner=planner,
                                engine=engine,
                                history=history,
                                pending_workflow_by_session=app.state.pending_workflow_by_session,
                                pending_clarification_by_session=app.state.pending_clarification_by_session,
                                progress_callback=_progress_callback,
                            )
                        )

                        while True:
                            if run_task.done() and progress_queue.empty():
                                break
                            try:
                                event = await asyncio.wait_for(progress_queue.get(), timeout=0.15)
                            except asyncio.TimeoutError:
                                continue
                            progress_text = _format_progress_event(event)
                            if not progress_text:
                                continue
                            yield _sse_line(
                                _chunk_payload(
                                    completion_id=completion_id,
                                    model=request.model,
                                    created=created,
                                    delta={"content": progress_text},
                                )
                            )
                            await asyncio.sleep(0)

                        response_text, _cap, debug_payload, suggestions, delivery = await run_task
                    else:
                        response_text, _cap, debug_payload, suggestions, delivery = await _run_agent_turn(
                            session_id=session_id,
                            user_text=user_text,
                            conversation=conversation,
                            intent_adapter=intent_adapter,
                            planner=planner,
                            engine=engine,
                            history=history,
                            pending_workflow_by_session=app.state.pending_workflow_by_session,
                            pending_clarification_by_session=app.state.pending_clarification_by_session,
                        )
                    debug_payload["fast_path"] = _fastpath_info(
                        requested=fastpath_requested,
                        candidate=fastpath_candidate,
                        used=False,
                        reason=fastpath_reason,
                        mode="pipeline",
                    )
                    suggestion_texts = _suggestion_texts(suggestions)
                    async for piece in _iter_stream_text_chunks(response_text, OPENAI_API_STREAM_CHUNK_SIZE):
                        yield _sse_line(
                            _chunk_payload(
                                completion_id=completion_id,
                                model=request.model,
                                created=created,
                                delta={"content": piece},
                            )
                        )

                    if suggestion_texts:
                        yield _sse_line(
                            _chunk_payload(
                                completion_id=completion_id,
                                model=request.model,
                                created=created,
                                delta={"suggestions": suggestion_texts},
                            )
                        )
                    yield _sse_line(
                        _chunk_payload(
                            completion_id=completion_id,
                            model=request.model,
                            created=created,
                            delta={},
                            finish_reason="stop",
                        )
                    )
                    yield _sse_line(
                        {
                            "id": completion_id,
                            "object": "chat.completion.meta",
                            "created": created,
                            "model": request.model,
                            "x_openwebui": {
                                "suggestions": suggestion_texts,
                                "suggestion_actions": suggestions,
                                "delivery": delivery,
                                "debug": debug_payload if OPENAI_API_DEBUG_TRACE else {},
                            },
                        }
                    )
                except Exception as e:
                    error_text = f"❌ Erro ao processar: {e}"
                    yield _sse_line(
                        _chunk_payload(
                            completion_id=completion_id,
                            model=request.model,
                            created=created,
                            delta={"content": error_text},
                        )
                    )
                    yield _sse_line(
                        _chunk_payload(
                            completion_id=completion_id,
                            model=request.model,
                            created=created,
                            delta={},
                            finish_reason="stop",
                        )
                    )
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    if use_fastpath:
        try:
            fastpath = _fastpath_info(
                requested=fastpath_requested,
                candidate=fastpath_candidate,
                used=True,
                reason=fastpath_reason,
                mode="non_stream",
            )
            response_text, intent_capability, debug_payload, suggestions, delivery = await _run_general_fastpath_turn(
                session_id=session_id,
                user_text=user_text,
                conversation=conversation,
                engine=engine,
                fastpath=fastpath,
            )
        except Exception as e:
            response_text, intent_capability, debug_payload, suggestions, delivery = await _run_agent_turn(
                session_id=session_id,
                user_text=user_text,
                conversation=conversation,
                intent_adapter=intent_adapter,
                planner=planner,
                engine=engine,
                history=history,
                pending_workflow_by_session=app.state.pending_workflow_by_session,
                pending_clarification_by_session=app.state.pending_clarification_by_session,
            )
            debug_payload["fast_path"] = _fastpath_info(
                requested=fastpath_requested,
                candidate=fastpath_candidate,
                used=False,
                reason=f"fallback: {e}",
                mode="pipeline",
            )
    else:
        response_text, intent_capability, debug_payload, suggestions, delivery = await _run_agent_turn(
            session_id=session_id,
            user_text=user_text,
            conversation=conversation,
            intent_adapter=intent_adapter,
            planner=planner,
            engine=engine,
            history=history,
            pending_workflow_by_session=app.state.pending_workflow_by_session,
            pending_clarification_by_session=app.state.pending_clarification_by_session,
        )
        debug_payload["fast_path"] = _fastpath_info(
            requested=fastpath_requested,
            candidate=fastpath_candidate,
            used=False,
            reason=fastpath_reason,
            mode="pipeline",
        )
    suggestion_texts = _suggestion_texts(suggestions)

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text,
                    "suggestions": suggestion_texts,
                    "x_suggestions": suggestions,
                    "suggestion_actions": suggestions,
                },
                "finish_reason": "stop",
            }
        ],
        "suggestions": suggestion_texts,
        "x_openwebui": {
            "suggestions": suggestion_texts,
            "suggestion_actions": suggestions,
            "intent_capability": intent_capability,
            "delivery": delivery,
            "debug": debug_payload if OPENAI_API_DEBUG_TRACE else {},
        },
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "system_fingerprint": "agent-orchestrator-v1",
    }
