"""
Symbol Normalizer - Metadata-driven symbol alias resolution.

Responsibility:
- Convert raw symbol tokens to canonical forms (e.g., TELIA → TELIA.ST)
- Support multiple resolution strategies:
  1. Alias lookup (fast, from injected dictionary)
  2. Canonical pattern matching (deterministic)
  3. LLM-based fallback (for unknown symbols)
- Cache results for performance
- Log all resolution attempts for observability
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SymbolNormalizer:
    """Normalize raw symbol tokens to canonical forms using metadata-driven aliases."""

    def __init__(
        self,
        aliases: Optional[dict[str, str]] = None,
        llm_enabled: bool = False,
        llm_client=None,
    ):
        """
        Initialize symbol normalizer.

        Args:
            aliases: Dictionary mapping raw tokens to canonical symbols.
                    (e.g., {"TELIA": "TELIA.ST", "ITAU": "ITUB4.SA"})
            llm_enabled: Whether to use LLM for unknown symbols
            llm_client: Optional LLM client for fallback resolution
        """
        self.aliases = aliases or {}
        self.llm_enabled = llm_enabled
        self.llm_client = llm_client
        self._resolution_cache: dict[str, Optional[str]] = {}

        logger.info(
            "SymbolNormalizer initialized (aliases=%d, llm_enabled=%s)",
            len(self.aliases),
            llm_enabled,
        )

    def normalize(self, raw_symbol: str) -> Optional[str]:
        """
        Normalize a raw symbol token to canonical form.

        Resolution order:
        1. Try alias dictionary (fast)
        2. Try canonical pattern matching
        3. Try LLM (if enabled)
        4. Return original or None

        Args:
            raw_symbol: Raw token (e.g., "TELIA", "PETR4", "AAPL")

        Returns:
            Canonical symbol (e.g., "TELIA.ST") or None if not found
        """
        if not raw_symbol:
            return None

        # Check cache first
        normalized = raw_symbol.strip().upper()
        if normalized in self._resolution_cache:
            cached = self._resolution_cache[normalized]
            logger.debug("Symbol cache hit: %s → %s", raw_symbol, cached)
            return cached

        # Try resolution strategies
        result = (
            self._resolve_from_aliases(normalized)
            or self._normalize_canonical(normalized)
        )

        # Cache result (even if None)
        self._resolution_cache[normalized] = result

        if result:
            logger.debug("Symbol resolved: %s → %s (from cache)", raw_symbol, result)
        else:
            logger.debug("Symbol not resolved: %s (keeping as-is)", raw_symbol)

        return result

    def _resolve_from_aliases(self, symbol: str) -> Optional[str]:
        """Resolve symbol from alias dictionary (fast, deterministic)."""
        if not symbol:
            return None

        # Direct match
        if symbol in self.aliases:
            resolved = self.aliases[symbol]
            logger.debug("Alias match: %s → %s", symbol, resolved)
            return resolved

        # Try lowercase match
        symbol_lower = symbol.lower()
        for alias_key, alias_value in self.aliases.items():
            if alias_key.lower() == symbol_lower:
                logger.debug("Alias match (case-insensitive): %s → %s", symbol, alias_value)
                return alias_value

        return None

    def _normalize_canonical(self, symbol: str) -> Optional[str]:
        """
        Normalize canonical patterns without aliases.

        Patterns:
        - Explicit exchange suffix: PETR4.SA, VOLV-B.ST, AAPL
        - B3 base format: PETR4, VALE3, BOVA11 → add .SA
        - US base format: AAPL, TSLA
        """
        if not symbol:
            return None

        import re

        # Already has exchange suffix
        if bool(re.fullmatch(r"[A-Z0-9-]{1,12}\.[A-Z]{1,4}", symbol)):
            logger.debug("Canonical (explicit suffix): %s", symbol)
            return symbol

        # B3 base symbols: 4 letters + class (3/4/5/6/11)
        if bool(re.fullmatch(r"[A-Z]{4}(?:3|4|5|6|11)(?:F)?", symbol)):
            result = f"{symbol}.SA"
            logger.debug("Canonical (B3 format): %s → %s", symbol, result)
            return result

        # US tickers (letters only, short)
        if bool(re.fullmatch(r"[A-Z]{1,5}", symbol)):
            logger.debug("Canonical (US ticker): %s", symbol)
            return symbol

        return None

    def normalize_batch(self, symbols: list[str]) -> list[Optional[str]]:
        """Normalize multiple symbols (cache-aware for performance)."""
        return [self.normalize(symbol) for symbol in symbols]

    def get_resolution_stats(self) -> dict:
        """Get statistics about symbol resolution."""
        return {
            "total_symbols": len(self._resolution_cache),
            "aliases_loaded": len(self.aliases),
            "llm_enabled": self.llm_enabled,
        }

    def clear_cache(self) -> None:
        """Clear resolution cache (for testing)."""
        self._resolution_cache.clear()
        logger.info("Symbol resolution cache cleared")
