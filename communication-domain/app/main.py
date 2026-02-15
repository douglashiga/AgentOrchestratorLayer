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


def _classify_telegram_error(error_message: str) -> tuple[str, str, float, str]:
    message = (error_message or "").strip()
    lowered = message.lower()

    if "message cannot be empty" in lowered:
        return (
            "clarification",
            "A mensagem para envio está vazia. Informe o texto que deseja enviar.",
            0.7,
            "missing_message",
        )
    if "chat_id is required" in lowered:
        return (
            "clarification",
            "Informe um chat_id válido ou configure TELEGRAM_DEFAULT_CHAT_ID.",
            0.7,
            "missing_chat_id",
        )
    if "not in telegram_allowed_chat_ids" in lowered:
        return (
            "clarification",
            "Esse chat_id não está permitido. Atualize TELEGRAM_ALLOWED_CHAT_IDS ou use um chat autorizado.",
            0.7,
            "chat_not_allowed",
        )
    if "telegram_bot_token is not configured" in lowered or "telegram_unauthorized" in lowered:
        return (
            "clarification",
            "Token do Telegram ausente/inválido. Configure TELEGRAM_BOT_TOKEN corretamente.",
            0.6,
            "token_configuration",
        )
    if (
        "telegram_bad_request" in lowered
        or "chat not found" in lowered
        or "forbidden" in lowered
        or "bot was blocked" in lowered
    ):
        return (
            "clarification",
            "Não consegui enviar para esse chat. Confirme o chat_id e inicie o bot no Telegram antes do envio.",
            0.7,
            "invalid_chat_target",
        )

    return (
        "failure",
        "Failed to send Telegram message.",
        0.0,
        "telegram_runtime_error",
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/manifest")
def manifest() -> dict[str, Any]:
    return {
        "manifest_version": "1.0",
        "domain": DOMAIN_NAME,
        "domain_description": "Send notifications and operational messages via Telegram.",
        "domain_intent_hints": {
            "keywords": [
                "telegram",
                "envie",
                "manda",
                "notifique",
                "compartilhe",
                "grupo",
                "canal",
            ],
            "examples": [
                "envie no telegram",
                "manda essa resposta no grupo",
            ],
        },
        "capabilities": [
            {
                "name": "send_telegram_message",
                "description": (
                    "Send a Telegram message. Use when user asks to notify/send/share a previous result "
                    "(e.g., 'envie no telegram', 'manda no telegram')."
                ),
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
                    "parameter_specs": {
                        "chat_id": {
                            "type": "string",
                            "required": False,
                            "description": "Telegram target chat or user id.",
                            "examples": ["123456789", "-1001234567890"],
                            "default": "${ENV:TELEGRAM_DEFAULT_CHAT_ID}",
                        },
                        "message": {
                            "type": "string",
                            "required": True,
                            "description": "Text to send. If omitted in follow-up mode, use previous step explanation.",
                            "examples": ["Preço da PETR4: R$ 37,10"],
                        },
                        "parse_mode": {
                            "type": "string",
                            "required": False,
                            "enum": ["Markdown", "HTML"],
                        },
                    },
                    "parameter_sources": {
                        "chat_id": {
                            "aliases": ["group_id"],
                            "from_entry": "chat_id",
                            "from_env": "TELEGRAM_DEFAULT_CHAT_ID",
                            "validation": "numeric_id",
                        },
                    },
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
                "description": (
                    "Send a Telegram message to a group/channel. Use when user explicitly asks to send "
                    "output to Telegram groups/channels."
                ),
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
                    "parameter_specs": {
                        "group_id": {
                            "type": "string",
                            "required": False,
                            "description": "Telegram group/channel id.",
                            "examples": ["-1001234567890"],
                            "default": "${ENV:TELEGRAM_DEFAULT_CHAT_ID}",
                        },
                        "chat_id": {
                            "type": "string",
                            "required": False,
                            "description": "Alias of group_id for compatibility.",
                            "examples": ["-1001234567890"],
                        },
                        "message": {
                            "type": "string",
                            "required": True,
                            "description": "Text to send to group/channel.",
                            "examples": ["Resumo diário enviado."],
                        },
                        "parse_mode": {
                            "type": "string",
                            "required": False,
                            "enum": ["Markdown", "HTML"],
                        },
                    },
                    "parameter_sources": {
                        "group_id": {
                            "aliases": ["chat_id"],
                            "from_entry": "chat_id",
                            "from_env": "TELEGRAM_DEFAULT_CHAT_ID",
                            "validation": "numeric_id",
                        },
                    },
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
        error_message = str(send_result.get("error", "unknown_error"))
        status, explanation, confidence, error_category = _classify_telegram_error(error_message)
        return DomainOutput(
            status=status,
            result={},
            explanation=explanation,
            confidence=confidence,
            metadata={"error": error_message, "error_category": error_category},
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
