"""
Manifest Loader - Fetch and cache domain metadata from finance server.

Responsibility:
- Fetch manifest from finance domain server
- Extract symbol aliases from metadata
- Cache results with TTL
- Gracefully handle unavailability
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ManifestLoader:
    """Load and cache domain manifest metadata (symbol aliases, etc)."""

    def __init__(self, finance_server_url: str = "http://localhost:8001"):
        """
        Initialize manifest loader.

        Args:
            finance_server_url: URL of finance domain server
        """
        self.finance_server_url = finance_server_url.rstrip("/")
        self._manifest_cache: Optional[dict] = None
        self._aliases_cache: Optional[dict[str, str]] = None

    def fetch_symbol_aliases(self) -> dict[str, str]:
        """
        Fetch symbol aliases from manifest.

        Returns:
            Dictionary mapping raw symbols to canonical forms.
            Returns empty dict on failure (graceful degradation).
        """
        if self._aliases_cache is not None:
            logger.debug("Returning cached aliases (%d symbols)", len(self._aliases_cache))
            return self._aliases_cache

        manifest = self.fetch_manifest()
        if not manifest:
            logger.warning("No manifest available, using empty aliases")
            self._aliases_cache = {}
            return {}

        # Extract aliases from METADATA_OVERRIDES
        aliases = self._extract_aliases_from_manifest(manifest)
        self._aliases_cache = aliases

        logger.info("Loaded %d symbol aliases from manifest", len(aliases))
        return aliases

    def fetch_manifest(self) -> Optional[dict]:
        """
        Fetch full manifest from finance server.

        Returns:
            Manifest dictionary or None on failure
        """
        if self._manifest_cache is not None:
            logger.debug("Returning cached manifest")
            return self._manifest_cache

        try:
            import httpx

            url = f"{self.finance_server_url}/manifest"
            logger.debug("Fetching manifest from %s", url)

            with httpx.Client(timeout=2.0) as client:
                response = client.get(url)
                response.raise_for_status()
                manifest = response.json()

            logger.info("Successfully fetched manifest from %s", url)
            self._manifest_cache = manifest
            return manifest

        except httpx.HTTPError as e:
            logger.warning("HTTP error fetching manifest: %s", e)
            return None
        except json.JSONDecodeError as e:
            logger.warning("JSON decode error in manifest: %s", e)
            return None
        except Exception as e:
            logger.warning("Error fetching manifest: %s", e)
            return None

    def _extract_aliases_from_manifest(self, manifest: dict) -> dict[str, str]:
        """
        Extract symbol aliases from manifest metadata.

        Looks for SYMBOL_ALIASES in METADATA_OVERRIDES.get_stock_price.

        Args:
            manifest: Domain manifest dict

        Returns:
            Dictionary of aliases
        """
        if not isinstance(manifest, dict):
            return {}

        try:
            # Navigate to SYMBOL_ALIASES in metadata
            # Path: METADATA_OVERRIDES → get_stock_price → parameter_specs → symbol → aliases
            metadata_overrides = manifest.get("metadata_overrides", {})
            get_stock_price = metadata_overrides.get("get_stock_price", {})

            # Direct access if SYMBOL_ALIASES is stored there
            if "symbol_aliases" in get_stock_price:
                aliases = get_stock_price["symbol_aliases"]
                if isinstance(aliases, dict):
                    logger.debug("Extracted %d aliases from manifest", len(aliases))
                    return aliases

            # Alternative: look in parameter_specs for symbol aliases
            parameter_specs = get_stock_price.get("parameter_specs", {})
            symbol_specs = parameter_specs.get("symbol", {})
            symbol_aliases = symbol_specs.get("aliases", {})

            if symbol_aliases and isinstance(symbol_aliases, dict):
                logger.debug("Extracted %d symbol aliases from parameter_specs", len(symbol_aliases))
                return symbol_aliases

            logger.debug("No aliases found in manifest")
            return {}

        except Exception as e:
            logger.warning("Error extracting aliases from manifest: %s", e)
            return {}

    def clear_cache(self) -> None:
        """Clear cached manifest and aliases (for testing)."""
        self._manifest_cache = None
        self._aliases_cache = None
        logger.info("Manifest loader cache cleared")
