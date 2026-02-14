"""
Telegram Entry Adapter.

Responsibility:
- Poll Telegram updates (long polling)
- Normalize incoming messages to EntryRequest
- Send assistant replies back to Telegram
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from shared.models import EntryRequest

logger = logging.getLogger(__name__)


class TelegramEntryAdapter:
    """Telegram entry channel adapter (long polling)."""

    def __init__(
        self,
        bot_token: str,
        poll_timeout_seconds: int = 20,
        request_timeout_seconds: float = 30.0,
        allowed_chat_ids: set[str] | None = None,
    ):
        if not bot_token.strip():
            raise ValueError("Telegram bot token is required.")

        self._api_base = f"https://api.telegram.org/bot{bot_token.strip()}"
        self.poll_timeout_seconds = max(1, int(poll_timeout_seconds))
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.allowed_chat_ids = allowed_chat_ids or set()

    def _is_allowed(self, chat_id: str) -> bool:
        if not self.allowed_chat_ids:
            return True
        return chat_id in self.allowed_chat_ids

    async def poll_updates(self, offset: int | None = None) -> tuple[list[EntryRequest], int | None]:
        """
        Poll Telegram updates and convert them to EntryRequest.
        Returns (entries, next_offset).
        """
        params: dict[str, Any] = {
            "timeout": self.poll_timeout_seconds,
            "allowed_updates": '["message"]',
        }
        if offset is not None:
            params["offset"] = offset

        url = f"{self._api_base}/getUpdates"
        try:
            async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                payload = response.json()
        except Exception as e:
            logger.error("Telegram polling error: %s", e)
            return [], offset

        if not payload.get("ok", False):
            logger.warning("Telegram getUpdates returned non-ok payload: %s", payload)
            return [], offset

        updates = payload.get("result", [])
        entries: list[EntryRequest] = []
        max_update_id: int | None = None

        for update in updates:
            update_id = update.get("update_id")
            if isinstance(update_id, int):
                max_update_id = update_id if max_update_id is None else max(max_update_id, update_id)

            message = update.get("message") or {}
            text = str(message.get("text", "")).strip()
            if not text:
                continue

            chat = message.get("chat") or {}
            chat_id = str(chat.get("id", "")).strip()
            if not chat_id:
                continue
            if not self._is_allowed(chat_id):
                logger.info("Ignoring message from unauthorized chat_id=%s", chat_id)
                continue

            entry = EntryRequest(
                session_id=f"tg:{chat_id}",
                input_text=text,
                metadata={
                    "source": "telegram",
                    "chat_id": chat_id,
                    "update_id": update_id,
                    "username": (message.get("from") or {}).get("username", ""),
                },
            )
            entries.append(entry)

        next_offset = (max_update_id + 1) if max_update_id is not None else offset
        return entries, next_offset

    async def send_message(self, chat_id: str, text: str) -> bool:
        """Send a Telegram message to a specific chat."""
        if not chat_id:
            return False
        body = {
            "chat_id": chat_id,
            "text": text[:4000] if text else "(empty response)",
        }
        url = f"{self._api_base}/sendMessage"
        try:
            async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
                response = await client.post(url, json=body)
                response.raise_for_status()
                payload = response.json()
                return bool(payload.get("ok", False))
        except Exception as e:
            logger.error("Telegram sendMessage error: %s", e)
            return False
