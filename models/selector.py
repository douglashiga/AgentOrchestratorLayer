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

        while attempt < policy.max_retries:
            attempt += 1
            try:
                with obs.measure("model_call", {"model": policy.model_name, "attempt": attempt}):
                    response_text = self._call_ollama(messages, policy)
                
                if policy.json_mode:
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
                if attempt >= policy.max_retries:
                    obs.log_event(
                        "model_failure",
                        {"error": str(e), "policy": policy.model_dump()},
                        level="ERROR",
                    )
                    raise

        raise last_error or RuntimeError("Unknown model failure")

    def _call_ollama(self, messages: list[dict], policy: ModelPolicy) -> str:
        """Low-level Ollama API call."""
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
