from __future__ import annotations

import os
from typing import Any

import httpx


class TelegramService:
    def __init__(self) -> None:
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.default_chat_id = os.getenv("TELEGRAM_DEFAULT_CHAT_ID", "").strip()
        self.timeout_seconds = float(os.getenv("TELEGRAM_TIMEOUT_SECONDS", "10"))
        self.dry_run = os.getenv("TELEGRAM_DRY_RUN", "false").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        allowlist_raw = os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", "").strip()
        self.allowlist = {
            item.strip()
            for item in allowlist_raw.split(",")
            if item.strip()
        }

    def send_message(self, chat_id: str | None, message: str, parse_mode: str | None = None) -> dict[str, Any]:
        target_chat_id = (chat_id or self.default_chat_id or "").strip()
        if not target_chat_id:
            return {
                "success": False,
                "error": "chat_id is required (or TELEGRAM_DEFAULT_CHAT_ID must be set).",
            }

        if self.allowlist and target_chat_id not in self.allowlist:
            return {
                "success": False,
                "error": f"chat_id '{target_chat_id}' is not in TELEGRAM_ALLOWED_CHAT_IDS.",
            }

        if not message.strip():
            return {"success": False, "error": "message cannot be empty."}

        if self.dry_run:
            return {
                "success": True,
                "dry_run": True,
                "chat_id": target_chat_id,
                "message": message,
            }

        if not self.bot_token:
            return {
                "success": False,
                "error": "TELEGRAM_BOT_TOKEN is not configured.",
            }

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": target_chat_id,
            "text": message,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
        except Exception as e:
            return {"success": False, "error": f"Telegram API error: {e}"}

        if not data.get("ok"):
            return {"success": False, "error": f"Telegram API rejected request: {data}"}

        result = data.get("result", {})
        return {
            "success": True,
            "dry_run": False,
            "chat_id": str(result.get("chat", {}).get("id", target_chat_id)),
            "message_id": result.get("message_id"),
            "text": result.get("text", message),
        }
