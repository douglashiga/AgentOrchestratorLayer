from __future__ import annotations

from models.selector import ModelSelector
from shared.models import ModelPolicy


class DummyResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class DummyClient:
    def __init__(self):
        self.calls: list[dict] = []

    def post(self, path, json=None, headers=None, timeout=None):
        self.calls.append(
            {
                "path": path,
                "json": json,
                "headers": headers or {},
                "timeout": timeout,
            }
        )
        return DummyResponse({"content": [{"type": "text", "text": "{\"ok\": true}"}]})

    def close(self) -> None:
        return None


def test_model_selector_auto_detects_anthropic_provider(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "auto")
    monkeypatch.setenv("MODEL_BASE_URL", "https://api.anthropic.com")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    selector = ModelSelector(ollama_url="http://localhost:11434")
    assert selector.provider == "anthropic"
    selector.close()


def test_model_selector_anthropic_json_mode_roundtrip(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("MODEL_BASE_URL", "https://api.anthropic.com")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    selector = ModelSelector(ollama_url="http://localhost:11434")
    fake_client = DummyClient()
    selector._client = fake_client

    out = selector.generate(
        messages=[
            {"role": "system", "content": "You are a JSON engine."},
            {"role": "user", "content": "return any json"},
        ],
        policy=ModelPolicy(
            model_name="claude-3-5-haiku-latest",
            json_mode=True,
            max_retries=1,
            timeout_seconds=5.0,
        ),
    )

    assert out == {"ok": True}
    assert len(fake_client.calls) == 1
    call = fake_client.calls[0]
    assert call["path"] == "/v1/messages"
    assert call["headers"].get("x-api-key") == "test-key"
    assert call["json"].get("model") == "claude-3-5-haiku-latest"
    assert "Return ONLY a valid JSON object." in str(call["json"].get("system", ""))
    selector.close()
