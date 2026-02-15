"""
HttpDomainHandler — Generic client for Remote Domains.

Responsibility:
- Implements Standard Domain Protocol (POST /execute)
- Serializes IntentOutput -> JSON
- Deserializes JSON -> DomainOutput
- Handles Network Errors
"""

import logging
import httpx
from typing import Any

from shared.models import IntentOutput, DomainOutput

logger = logging.getLogger(__name__)

class HttpDomainHandler:
    """Generic handler for remote HTTP domains."""

    def __init__(self, base_url: str, auth_token: str | None = None, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"

    async def execute(self, intent: IntentOutput) -> DomainOutput:
        """
        Execute intent via Remote Domain Protocol.
        POST /execute
        """
        url = f"{self.base_url}/execute"
        payload = intent.model_dump(mode="json")

        try:
            logger.info("Calling remote domain: %s capability=%s", url, intent.capability)
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload, headers=self.headers)
                
                if response.status_code != 200:
                    logger.error("Remote domain error %s: %s", response.status_code, response.text)
                    return DomainOutput(
                        status="failure",
                        explanation=f"Remote domain returned status {response.status_code}",
                        metadata={"status_code": response.status_code, "body": response.text}
                    )
                
                # Parse response
                data = response.json()
                return DomainOutput(**data)

        except httpx.RequestError as e:
            logger.error("Network error calling remote domain '%s': %r", url, e)
            return DomainOutput(
                status="failure",
                explanation=f"Network error connecting to domain: {e}",
                metadata={"error": str(e), "error_type": type(e).__name__, "url": url}
            )
        except Exception as e:
            logger.error("Unexpected error in HttpDomainHandler: %s", e)
            return DomainOutput(
                status="failure",
                explanation=f"Unexpected error: {e}",
                metadata={"error": str(e)}
            )

    def fetch_manifest(self) -> dict[str, Any]:
        """
        Fetch domain capabilities via GET /manifest.
        Returns dict with keys: 'domain', 'capabilities'.
        """
        url = f"{self.base_url}/manifest"
        try:
            logger.info("Fetching manifest from: %s", url)
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url, headers=self.headers)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error("Failed to fetch manifest from %s: %s", url, e)
            raise

    def fetch_openapi(self) -> dict[str, Any]:
        """
        Fetch OpenAPI schema via GET /openapi.json.
        """
        url = f"{self.base_url}/openapi.json"
        try:
            logger.info("Fetching OpenAPI from: %s", url)
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url, headers=self.headers)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("OpenAPI payload must be an object")
                return payload
        except Exception as e:
            logger.error("Failed to fetch OpenAPI from %s: %s", url, e)
            raise
