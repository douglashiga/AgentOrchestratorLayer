"""
OpenAI-compatible API server for Agent Orchestrator.

Implements minimal endpoints required by Open WebUI:
- GET /v1/models
- POST /v1/chat/completions
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from main import build_pipeline, _normalize_intent_parameters


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

    history = conversation.get_history(session_id)
    intent = intent_adapter.extract(user_text, history, session_id=session_id)
    intent = _normalize_intent_parameters(intent, engine.orchestrator.domain_registry, entry_request=None)
    plan = planner.generate_plan(intent)
    output = await engine.execute_plan(plan, original_intent=intent)
    response_text = output.explanation if output.explanation else str(output.result)

    conversation.save(session_id, "user", user_text)
    conversation.save(session_id, "assistant", response_text)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "system_fingerprint": "agent-orchestrator-v1",
    }

