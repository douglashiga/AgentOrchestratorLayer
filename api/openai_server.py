"""
OpenAI-compatible API server for Agent Orchestrator.

Implements minimal endpoints required by Open WebUI:
- GET /v1/models
- POST /v1/chat/completions
"""

from __future__ import annotations

import asyncio
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
from pydantic import BaseModel, Field

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
OPENAI_API_STREAM_STATUS_UPDATES = os.getenv("OPENAI_API_STREAM_STATUS_UPDATES", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
OPENAI_API_STREAM_CHUNK_SIZE = int(os.getenv("OPENAI_API_STREAM_CHUNK_SIZE", "160"))
GENERAL_FASTPATH_ENABLED = os.getenv("GENERAL_FASTPATH_ENABLED", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MODEL_ID_DEFAULT = "agent-orchestrator"
MODEL_ID_FASTPATH = "agent-orchestrator-fastpath"
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


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=MODEL_ID_DEFAULT)
    messages: list[ChatMessage]
    stream: bool = False
    user: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    x_general_fastpath: bool | None = None


def _extract_user_text(messages: list[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role != "user":
            continue
        content = msg.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return " ".join(parts).strip()
    return ""


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
    if OPENAI_API_INCLUDE_SUGGESTIONS:
        sections.append(_build_suggestions_text(intent_capability))
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
) -> tuple[str, str, dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    pending_map = pending_workflow_by_session if isinstance(pending_workflow_by_session, dict) else {}
    pending_workflow = pending_map.get(session_id)
    if pending_workflow:
        answer_text = user_text.strip()
        answer = ClarificationAnswer(
            question_id=str(pending_workflow.get("question_id", "")).strip(),
            task_id=str(pending_workflow.get("task_id", "")).strip(),
            selected_option=answer_text,
            confirmed=not _is_decline_message(answer_text),
        )
        output = await engine.resume_task(answer)
        if output.status == "clarification":
            next_task_id = str(output.metadata.get("task_id", "")).strip()
            next_question_id = str(output.metadata.get("question_id", "")).strip()
            if next_task_id and next_question_id:
                pending_map[session_id] = {"task_id": next_task_id, "question_id": next_question_id}
            else:
                pending_map.pop(session_id, None)
        else:
            pending_map.pop(session_id, None)

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
    intent = intent_adapter.extract(user_text, turn_history, session_id=session_id)
    intent_extracted = intent.model_dump(mode="json")
    intent = _normalize_intent_parameters(intent, engine.orchestrator.domain_registry, entry_request=None)
    intent_normalized = intent.model_dump(mode="json")
    plan = planner.generate_plan(intent, session_id=session_id)
    plan_dump = plan.model_dump(mode="json")
    output = await engine.execute_plan(plan, original_intent=intent)
    if output.status == "clarification":
        task_id = str(output.metadata.get("task_id", "")).strip()
        question_id = str(output.metadata.get("question_id", "")).strip()
        if task_id and question_id:
            pending_map[session_id] = {"task_id": task_id, "question_id": question_id}
    else:
        pending_map.pop(session_id, None)

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
    _app.state.conversation = conversation
    _app.state.intent_adapter = intent_adapter
    _app.state.planner = planner
    _app.state.engine = engine
    _app.state.model_selector = model_selector
    _app.state.mcp_adapter = mcp_adapter
    _app.state.pending_workflow_by_session = {}
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

    session_id = request.user or str(uuid.uuid4())[:8]
    conversation = app.state.conversation
    intent_adapter = app.state.intent_adapter
    planner = app.state.planner
    engine = app.state.engine
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    history = conversation.get_history(session_id)
    fastpath_requested = _resolve_fastpath_requested(request)
    fastpath_candidate, fastpath_reason = _general_fastpath_candidate(user_text, history)
    use_fastpath = fastpath_requested and fastpath_candidate

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
                    for piece in _chunk_text(tail_text, OPENAI_API_STREAM_CHUNK_SIZE):
                        yield _sse_line(
                            _chunk_payload(
                                completion_id=completion_id,
                                model=request.model,
                                created=created,
                                delta={"content": piece},
                            )
                        )
                        await asyncio.sleep(0)

                    conversation.save(session_id, "user", user_text)
                    conversation.save(session_id, "assistant", clean_response_text)
                    delivery = {"kind": "text", "content": clean_response_text, "data": {}}

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
                                "suggestions": suggestions,
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
                        )
                        debug_payload["fast_path"] = _fastpath_info(
                            requested=fastpath_requested,
                            candidate=fastpath_candidate,
                            used=False,
                            reason=f"fallback: {e}",
                            mode="pipeline",
                        )
                        for piece in _chunk_text(response_text, OPENAI_API_STREAM_CHUNK_SIZE):
                            yield _sse_line(
                                _chunk_payload(
                                    completion_id=completion_id,
                                    model=request.model,
                                    created=created,
                                    delta={"content": piece},
                                )
                            )
                            await asyncio.sleep(0)
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
                                    "suggestions": suggestions,
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
                    response_text, _cap, debug_payload, suggestions, delivery = await _run_agent_turn(
                        session_id=session_id,
                        user_text=user_text,
                        conversation=conversation,
                        intent_adapter=intent_adapter,
                        planner=planner,
                        engine=engine,
                        history=history,
                        pending_workflow_by_session=app.state.pending_workflow_by_session,
                    )
                    debug_payload["fast_path"] = _fastpath_info(
                        requested=fastpath_requested,
                        candidate=fastpath_candidate,
                        used=False,
                        reason=fastpath_reason,
                        mode="pipeline",
                    )
                    for piece in _chunk_text(response_text, OPENAI_API_STREAM_CHUNK_SIZE):
                        yield _sse_line(
                            _chunk_payload(
                                completion_id=completion_id,
                                model=request.model,
                                created=created,
                                delta={"content": piece},
                            )
                        )
                        await asyncio.sleep(0)

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
                                "suggestions": suggestions,
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
        )
        debug_payload["fast_path"] = _fastpath_info(
            requested=fastpath_requested,
            candidate=fastpath_candidate,
            used=False,
            reason=fastpath_reason,
            mode="pipeline",
        )

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
                    "x_suggestions": suggestions,
                },
                "finish_reason": "stop",
            }
        ],
        "x_openwebui": {
            "suggestions": suggestions,
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
