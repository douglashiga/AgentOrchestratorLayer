"""
Domain Registry — Pure lookup, no logic.

Maps domain_name → DomainHandler.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from shared.models import Decision, Intent

logger = logging.getLogger(__name__)


class DomainHandler(Protocol):
    """Protocol that all domain handlers must implement."""

    def execute(self, intent: Intent) -> Decision:
        ...


class DomainRegistry:
    """Registry mapping domain names to their handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, DomainHandler] = {}

    def register(self, domain_name: str, handler: DomainHandler) -> None:
        """Register a domain handler."""
        logger.info("Registered domain: %s → %s", domain_name, type(handler).__name__)
        self._handlers[domain_name] = handler

    def resolve(self, domain_name: str) -> DomainHandler | None:
        """Resolve a domain name to its handler. Returns None if not found."""
        return self._handlers.get(domain_name)

    @property
    def registered_domains(self) -> list[str]:
        """List all registered domain names."""
        return list(self._handlers.keys())
