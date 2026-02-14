from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI

from app.models import DomainOutput, IntentInput
from app.telegram_service import TelegramService

DOMAIN_NAME = os.getenv("COMMUNICATION_DOMAIN_NAME", "communication")

app = FastAPI(
    title="Communication Domain Service",
    version="0.1.0",
    description="Domain service for outbound communication actions (Telegram).",
)

telegram_service = TelegramService()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/manifest")
def manifest() -> dict[str, Any]:
    return {
        "manifest_version": "1.0",
        "domain": DOMAIN_NAME,
        "domain_description": "Send notifications and operational messages via Telegram.",
        "capabilities": [
            {
                "name": "send_telegram_message",
                "description": "Send a Telegram message to a chat/user/group id.",
                "schema": {
                    "type": "object",
                    "properties": {
                        "chat_id": {"type": "string"},
                        "message": {"type": "string"},
                        "parse_mode": {"type": "string"},
                    },
                    "required": ["message"],
                },
                "metadata": {
                    "requires_auth": True,
                    "channel": "telegram",
                    "composition": {
                        "role": "notifier",
                        "priority": 100,
                        "param_map": {
                            "chat_id": {
                                "from_parameters": ["chat_id", "group_id"],
                                "default": "${ENV:TELEGRAM_DEFAULT_CHAT_ID}",
                            },
                            "message": {
                                "from_parameters": ["message", "notify_message"],
                                "default": "${1.explanation}",
                            },
                        },
                    },
                },
            },
            {
                "name": "send_telegram_group_message",
                "description": "Send a Telegram message to a group (group_id/chat_id).",
                "schema": {
                    "type": "object",
                    "properties": {
                        "group_id": {"type": "string"},
                        "chat_id": {"type": "string"},
                        "message": {"type": "string"},
                        "parse_mode": {"type": "string"},
                    },
                    "required": ["message"],
                },
                "metadata": {
                    "requires_auth": True,
                    "channel": "telegram",
                    "composition": {
                        "role": "notifier",
                        "priority": 90,
                        "param_map": {
                            "group_id": {
                                "from_parameters": ["group_id", "chat_id"],
                                "default": "${ENV:TELEGRAM_DEFAULT_CHAT_ID}",
                            },
                            "message": {
                                "from_parameters": ["message", "notify_message"],
                                "default": "${1.explanation}",
                            },
                        },
                    },
                },
            },
        ],
    }


@app.post("/execute", response_model=DomainOutput)
def execute(intent: IntentInput) -> DomainOutput:
    capability = intent.capability
    params = dict(intent.parameters)

    if capability not in ("send_telegram_message", "send_telegram_group_message"):
        return DomainOutput(
            status="failure",
            result={},
            explanation=f"Unsupported capability: {capability}",
            confidence=0.0,
            metadata={"error": "unsupported_capability"},
        )

    chat_id = (
        params.get("chat_id")
        or params.get("group_id")
    )
    message = str(params.get("message", ""))
    parse_mode = params.get("parse_mode")

    send_result = telegram_service.send_message(
        chat_id=str(chat_id) if chat_id is not None else None,
        message=message,
        parse_mode=str(parse_mode) if parse_mode else None,
    )
    if not send_result.get("success"):
        return DomainOutput(
            status="failure",
            result={},
            explanation="Failed to send Telegram message.",
            confidence=0.0,
            metadata={"error": send_result.get("error", "unknown_error")},
        )

    dry_run = bool(send_result.get("dry_run"))
    explanation = (
        f"Dry-run: Telegram message prepared for chat {send_result.get('chat_id')}."
        if dry_run
        else f"Telegram message sent to chat {send_result.get('chat_id')}."
    )
    return DomainOutput(
        status="success",
        result=send_result,
        explanation=explanation,
        confidence=1.0,
        metadata={"channel": "telegram", "dry_run": dry_run},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
