"""
Model Layer — LLM Abstraction & Policy Enforcement.

Responsibility:
- Abstract specific LLM client details (Ollama, etc.)
- Enforce timeouts and retries
- Fallback logic (switch models on failure)
- JSON validation helper

This is the ONLY place where LLMs are called.
"""

import json
import logging
from typing import Any

import httpx

from observability.logger import Observability
from shared.models import ModelPolicy

logger = logging.getLogger(__name__)


class ModelSelector:
    """Manages LLM calls with reliability policies."""

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url.rstrip("/")
        # Persistent client with connection pooling
        self._client = httpx.Client(
            base_url=self.ollama_url,
            timeout=60.0,  # default, overridden by policy
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
        self.observability = Observability()

    def generate(
        self,
        messages: list[dict],
        policy: ModelPolicy,
        session_id: str | None = None,
    ) -> dict[str, Any] | str:
        """
        Execute LLM generation with retry/timeout policy.
        Returns detailed dict if json_mode=True, else string.
        """
        obs = Observability(session_id)
        
        attempt = 0
        last_error = None
        active_policy = policy

        while attempt < policy.max_retries:
            attempt += 1
            try:
                with obs.measure("model_call", {"model": active_policy.model_name, "attempt": attempt}):
                    response_text = self._call_ollama(messages, active_policy)
                
                if active_policy.json_mode:
                    return self._parse_json(response_text)
                return response_text

            except Exception as e:
                last_error = e
                logger.warning(
                    "Model call failed (attempt %d/%d): %s",
                    attempt,
                    policy.max_retries,
                    e,
                )
                fallback_model = self._fallback_model_for_error(e, active_policy.model_name)
                if fallback_model:
                    logger.warning(
                        "Switching to fallback model '%s' after error on '%s'.",
                        fallback_model,
                        active_policy.model_name,
                    )
                    active_policy = active_policy.model_copy(update={"model_name": fallback_model})
                    continue
                if attempt >= policy.max_retries:
                    obs.log_event(
                        "model_failure",
                        {"error": str(e), "policy": active_policy.model_dump()},
                        level="ERROR",
                    )
                    raise

        raise last_error or RuntimeError("Unknown model failure")

    def _call_ollama(self, messages: list[dict], policy: ModelPolicy) -> str:
        """Low-level model API call with Ollama-first, OpenAI-compatible fallback."""
        try:
            return self._call_ollama_chat(messages, policy)
        except httpx.HTTPStatusError as e:
            # Some local providers expose only OpenAI-compatible APIs (/v1/chat/completions).
            if e.response is not None and e.response.status_code == 404:
                logger.info("Ollama endpoint not found; trying OpenAI-compatible chat endpoint.")
                return self._call_openai_chat(messages, policy)
            raise

    def _call_ollama_chat(self, messages: list[dict], policy: ModelPolicy) -> str:
        """Low-level Ollama /api/chat call."""
        payload = {
            "model": policy.model_name,
            "messages": messages,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": policy.temperature,
                # Default context and predict limits for safety
                "num_ctx": 4096, 
                "num_predict": 1024,
            },
        }

        if policy.json_mode:
            payload["format"] = "json"

        response = self._client.post(
            "/api/chat",
            json=payload,
            timeout=policy.timeout_seconds,
        )
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "")

    def _call_openai_chat(self, messages: list[dict], policy: ModelPolicy) -> str:
        """Fallback call for OpenAI-compatible /v1/chat/completions providers."""
        payload: dict[str, Any] = {
            "model": policy.model_name,
            "messages": messages,
            "temperature": policy.temperature,
            "stream": False,
        }
        if policy.json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = self._client.post(
            "/v1/chat/completions",
            json=payload,
            timeout=policy.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("OpenAI-compatible response missing choices")
        message = choices[0].get("message") or {}
        return str(message.get("content", ""))

    def _fallback_model_for_error(self, error: Exception, current_model: str) -> str | None:
        if not isinstance(error, httpx.HTTPStatusError):
            return None
        if error.response is None or error.response.status_code != 404:
            return None
        body = ""
        try:
            body = error.response.text or ""
        except Exception:
            body = ""
        if "model" not in body.lower() or "not found" not in body.lower():
            return None
        return self._first_available_model(exclude=current_model)

    def _first_available_model(self, exclude: str) -> str | None:
        # Prefer Ollama model listing.
        try:
            r = self._client.get("/api/tags", timeout=5.0)
            if r.status_code == 200:
                payload = r.json()
                models = payload.get("models") or []
                for item in models:
                    name = str(item.get("name", "")).strip()
                    if name and name != exclude:
                        return name
        except Exception:
            pass

        # OpenAI-compatible listing fallback.
        try:
            r = self._client.get("/v1/models", timeout=5.0)
            if r.status_code == 200:
                payload = r.json()
                models = payload.get("data") or []
                for item in models:
                    name = str(item.get("id", "")).strip()
                    if name and name != exclude:
                        return name
        except Exception:
            pass
        return None

    def _parse_json(self, text: str) -> dict[str, Any]:
        """Parse JSON response, handling common markdown issues."""
        clean_text = text.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.split("\n", 1)[1].rsplit("\n", 1)[0]
        
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from model: {e}") from e

    def close(self):
        """Close persistent connections."""
        self._client.close()
