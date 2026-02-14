"""
OpenAI-compatible API server for Agent Orchestrator.

Implements minimal endpoints required by Open WebUI:
- GET /v1/models
- POST /v1/chat/completions
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from main import build_pipeline, _normalize_intent_parameters

OPENAI_API_DEBUG_TRACE = os.getenv("OPENAI_API_DEBUG_TRACE", "true").strip().lower() in (
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


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="agent-orchestrator")
    messages: list[ChatMessage]
    stream: bool = False
    user: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


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
    suggestions = _build_suggestion_prompts(intent_capability, user_text) if OPENAI_API_INCLUDE_SUGGESTIONS else []
    sections = [base_text]
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


async def _run_agent_turn(
    *,
    session_id: str,
    user_text: str,
    conversation: Any,
    intent_adapter: Any,
    planner: Any,
    engine: Any,
) -> tuple[str, str, dict[str, Any], list[dict[str, str]]]:
    history = conversation.get_history(session_id)
    intent = intent_adapter.extract(user_text, history, session_id=session_id)
    intent_extracted = intent.model_dump(mode="json")
    intent = _normalize_intent_parameters(intent, engine.orchestrator.domain_registry, entry_request=None)
    intent_normalized = intent.model_dump(mode="json")
    plan = planner.generate_plan(intent, session_id=session_id)
    plan_dump = plan.model_dump(mode="json")
    output = await engine.execute_plan(plan, original_intent=intent)
    base_text = output.explanation if output.explanation else str(output.result)
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
    conversation.save(session_id, "assistant", response_text)
    return response_text, intent.capability, debug_payload, suggestions


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
    yield
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
                "id": "agent-orchestrator",
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
                response_text, _cap, debug_payload, suggestions = await _run_agent_turn(
                    session_id=session_id,
                    user_text=user_text,
                    conversation=conversation,
                    intent_adapter=intent_adapter,
                    planner=planner,
                    engine=engine,
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
                # non-standard extension for clients that can use structured suggestions.
                yield _sse_line(
                    {
                        "id": completion_id,
                        "object": "chat.completion.meta",
                        "created": created,
                        "model": request.model,
                        "x_openwebui": {
                            "suggestions": suggestions,
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

    response_text, intent_capability, debug_payload, suggestions = await _run_agent_turn(
        session_id=session_id,
        user_text=user_text,
        conversation=conversation,
        intent_adapter=intent_adapter,
        planner=planner,
        engine=engine,
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
            "debug": debug_payload if OPENAI_API_DEBUG_TRACE else {},
        },
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "system_fingerprint": "agent-orchestrator-v1",
    }
