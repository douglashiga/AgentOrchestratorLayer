"""
Simple integration check for communication domain.

Usage:
  TELEGRAM_DEFAULT_CHAT_ID=... python3 scripts/test_telegram_send_simple.py
"""

from __future__ import annotations

import os

from registry.http_handler import HttpDomainHandler
from shared.models import ExecutionIntent


def main() -> int:
    communication_url = os.getenv("COMMUNICATION_DOMAIN_URL", "http://localhost:8002")
    chat_id = os.getenv("TELEGRAM_DEFAULT_CHAT_ID", "").strip()

    handler = HttpDomainHandler(base_url=communication_url, timeout=20.0)
    intent = ExecutionIntent(
        domain="communication",
        capability="send_telegram_message",
        confidence=1.0,
        parameters={
            "chat_id": chat_id,
            "message": "Teste simples: AgentOrchestrator -> Telegram",
        },
        original_query="send telegram test",
    )

    output = handler.fetch_manifest()
    print("Manifest domain:", output.get("domain"))

    import asyncio

    result = asyncio.run(handler.execute(intent))
    print("Status:", result.status)
    print("Explanation:", result.explanation)
    print("Result:", result.result)
    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
