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
import os
from typing import Any

import httpx

from observability.logger import Observability
from shared.models import ModelPolicy

logger = logging.getLogger(__name__)


class ModelSelector:
    """Manages LLM calls with reliability policies."""

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        configured_base_url = os.getenv("MODEL_BASE_URL", "").strip()
        self.ollama_url = (configured_base_url or ollama_url).rstrip("/")
        provider_raw = os.getenv("MODEL_PROVIDER", "auto").strip().lower()
        if provider_raw not in {"auto", "ollama", "openai_compatible", "anthropic"}:
            provider_raw = "auto"
        self.provider = self._resolve_provider(provider_raw, self.ollama_url)
        self.api_key = os.getenv("MODEL_API_KEY", "").strip()
        if not self.api_key:
            if self.provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
            elif self.provider == "openai_compatible":
                self.api_key = os.getenv("OPENAI_API_KEY", "").strip()

        base_headers: dict[str, str] = {}
        if self.provider == "openai_compatible" and self.api_key:
            base_headers["Authorization"] = f"Bearer {self.api_key}"

        # Persistent client with connection pooling
        self._client = httpx.Client(
            base_url=self.ollama_url,
            timeout=60.0,  # default, overridden by policy
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            headers=base_headers,
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
                with obs.measure(
                    "model_call",
                    {
                        "model": active_policy.model_name,
                        "attempt": attempt,
                        "provider": self.provider,
                    },
                ):
                    response_text = self._call_model(messages, active_policy)
                
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

    def _resolve_provider(self, provider_raw: str, base_url: str) -> str:
        if provider_raw != "auto":
            return provider_raw

        lowered = (base_url or "").strip().lower()
        if "anthropic.com" in lowered:
            return "anthropic"
        if "openai.com" in lowered or lowered.endswith("/v1"):
            return "openai_compatible"
        return "ollama"

    def _call_model(self, messages: list[dict], policy: ModelPolicy) -> str:
        """Low-level model API call dispatching by configured provider."""
        if self.provider == "anthropic":
            return self._call_anthropic_messages(messages, policy)
        if self.provider == "openai_compatible":
            return self._call_openai_chat(messages, policy)
        return self._call_ollama(messages, policy)

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

    def _call_anthropic_messages(self, messages: list[dict], policy: ModelPolicy) -> str:
        """Low-level Anthropic /v1/messages call."""
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY (or MODEL_API_KEY) is required when MODEL_PROVIDER=anthropic."
            )

        payload_messages: list[dict[str, str]] = []
        system_parts: list[str] = []
        for message in messages:
            role = str(message.get("role", "user")).strip().lower()
            content = message.get("content", "")
            if isinstance(content, list):
                text = " ".join(
                    str(item.get("text", "")).strip()
                    for item in content
                    if isinstance(item, dict) and str(item.get("type", "")).strip() == "text"
                ).strip()
            else:
                text = str(content).strip()
            if not text:
                continue
            if role == "system":
                system_parts.append(text)
                continue
            if role not in {"user", "assistant"}:
                role = "user"
            payload_messages.append({"role": role, "content": text})

        if not payload_messages:
            payload_messages = [{"role": "user", "content": "Hello"}]

        system_prompt = "\n\n".join(system_parts).strip()
        if policy.json_mode:
            json_guard = "Return ONLY a valid JSON object."
            system_prompt = f"{system_prompt}\n\n{json_guard}".strip() if system_prompt else json_guard

        payload: dict[str, Any] = {
            "model": policy.model_name,
            "messages": payload_messages,
            "temperature": policy.temperature,
            "max_tokens": 1024,
        }
        if system_prompt:
            payload["system"] = system_prompt

        response = self._client.post(
            "/v1/messages",
            json=payload,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
            },
            timeout=policy.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        content_blocks = data.get("content") or []
        text_parts: list[str] = []
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            if str(block.get("type", "")).strip() != "text":
                continue
            text_value = str(block.get("text", "")).strip()
            if text_value:
                text_parts.append(text_value)
        if not text_parts:
            raise ValueError("Anthropic response missing text content")
        return "\n".join(text_parts)

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
        body_l = body.lower()
        if "model" not in body_l:
            return None
        if "not found" not in body_l and "not_found" not in body_l:
            return None
        preferred_family = None
        current_l = str(current_model or "").strip().lower()
        for family in ("haiku", "sonnet", "opus"):
            if family in current_l:
                preferred_family = family
                break
        return self._first_available_model(exclude=current_model, preferred_family=preferred_family)

    def _first_available_model(self, exclude: str, preferred_family: str | None = None) -> str | None:
        def pick_from_names(names: list[str]) -> str | None:
            filtered = [name for name in names if name and name != exclude]
            if not filtered:
                return None
            if preferred_family:
                for name in filtered:
                    if preferred_family in name.lower():
                        return name
            return filtered[0]

        # Prefer Ollama model listing.
        try:
            r = self._client.get("/api/tags", timeout=5.0)
            if r.status_code == 200:
                payload = r.json()
                models = payload.get("models") or []
                names = [str(item.get("name", "")).strip() for item in models if isinstance(item, dict)]
                picked = pick_from_names(names)
                if picked:
                    return picked
        except Exception:
            pass

        # OpenAI-compatible listing fallback.
        try:
            r = self._client.get("/v1/models", timeout=5.0)
            if r.status_code == 200:
                payload = r.json()
                models = payload.get("data") or []
                names = [str(item.get("id", "")).strip() for item in models if isinstance(item, dict)]
                picked = pick_from_names(names)
                if picked:
                    return picked
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
