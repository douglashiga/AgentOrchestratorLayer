"""
Registry — Maps domains and capabilities to handlers.

Responsibility:
- Maintain mapping of domain_name -> Handler
- Maintain mapping of capability -> Handler
- Store metadata for capabilities (criticality, execution_mode)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class HandlerRegistry:
    """Registry for domain handlers and capabilities."""

    def __init__(self):
        self._domains: dict[str, Any] = {}
        self._capabilities: dict[str, Any] = {}
        self._metadata: dict[str, dict] = {}

    def register_domain(self, domain_name: str, handler: Any) -> None:
        """Register a handler for a high-level domain."""
        self._domains[domain_name] = handler
        logger.info("Registered domain: %s → %s", domain_name, handler.__class__.__name__)

    def register_capability(
        self, 
        capability: str, 
        handler: Any, 
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Register a specific capability to a handler."""
        self._capabilities[capability] = handler
        if metadata:
            self._metadata[capability] = metadata
        logger.info("Registered capability: %s → %s", capability, handler.__class__.__name__)

    def resolve_domain(self, domain_name: str) -> Any | None:
        """Resolve handler by domain name."""
        return self._domains.get(domain_name)

    def resolve_capability(self, capability: str) -> Any | None:
        """Resolve handler by capability."""
        return self._capabilities.get(capability)

    def get_metadata(self, capability: str) -> dict[str, Any]:
        """Get metadata for a capability."""
        return self._metadata.get(capability, {})

    @property
    def registered_domains(self) -> list[str]:
        return list(self._domains.keys())

    @property
    def registered_capabilities(self) -> list[str]:
        return list(self._capabilities.keys())
