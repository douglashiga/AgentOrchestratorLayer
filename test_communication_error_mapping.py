from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "communication-domain"))

from app.main import _classify_telegram_error


def test_classify_missing_chat_id_as_clarification():
    status, explanation, confidence, category = _classify_telegram_error(
        "chat_id is required (or TELEGRAM_DEFAULT_CHAT_ID must be set)."
    )
    assert status == "clarification"
    assert confidence > 0.0
    assert category == "missing_chat_id"
    assert "chat_id" in explanation.lower()


def test_classify_bad_request_as_clarification():
    status, explanation, confidence, category = _classify_telegram_error(
        "telegram_bad_request: Bad Request: chat not found"
    )
    assert status == "clarification"
    assert confidence > 0.0
    assert category == "invalid_chat_target"
    assert "chat" in explanation.lower()


def test_classify_unknown_runtime_error_as_failure():
    status, _, confidence, category = _classify_telegram_error("telegram_network_error: timeout")
    assert status == "failure"
    assert confidence == 0.0
    assert category == "telegram_runtime_error"
