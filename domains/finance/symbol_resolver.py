"""
Symbol Resolver - Multi-strategy symbol resolution with LLM fallback.

Responsibility:
- Resolve symbol tokens to canonical forms with exchange info
- Strategy 1: Search in local database (search_symbol)
- Strategy 2: LLM-based resolution with structured output
- Strategy 3: Handle ambiguity by asking for clarification
- Return structured result: {symbol, exchange, confidence}
"""

import logging
from typing import Optional
from pydantic import BaseModel
from domains.finance.config import (
    get_focus_exchanges,
    is_exchange_in_focus,
    filter_by_focus_exchange,
    get_llm_exchange_guidance,
)

logger = logging.getLogger(__name__)


class ResolvedSymbol(BaseModel):
    """Structured symbol resolution result."""

    symbol: str  # Canonical symbol (e.g., PETR4.SA)
    exchange: str  # Exchange code (e.g., SA, ST, US)
    company_name: Optional[str] = None
    confidence: float = 1.0  # 0.0 to 1.0
    source: str = "unknown"  # database, llm, alias, pattern
    ambiguous: bool = False  # True if multiple matches found
    alternatives: list[dict] = []  # Other possible matches


class SymbolResolver:
    """
    Resolve raw symbol tokens to canonical forms with multi-strategy fallback.

    Strategy order:
    1. Alias mapping (TELIA → TELIA.ST)
    2. Local database search (search_symbol)
    3. LLM-based resolution (if enabled)
    4. Pattern matching (fallback)
    """

    def __init__(
        self,
        aliases: Optional[dict[str, str]] = None,
        skill_gateway=None,
        model_selector=None,
        enable_llm: bool = True,
    ):
        """
        Initialize symbol resolver.

        Args:
            aliases: Dictionary of known aliases
            skill_gateway: Gateway to call search_symbol skill
            model_selector: ModelSelector for LLM calls (from models.selector)
            enable_llm: Whether to use LLM fallback
        """
        self.aliases = aliases or {}
        self.skill_gateway = skill_gateway
        self.model_selector = model_selector
        self.enable_llm = enable_llm
        self._cache: dict[str, ResolvedSymbol] = {}

        logger.info(
            "SymbolResolver initialized (aliases=%d, llm_enabled=%s, db_enabled=%s)",
            len(self.aliases),
            enable_llm,
            bool(skill_gateway),
        )

    def resolve(self, raw_symbol: str) -> Optional[ResolvedSymbol]:
        """
        Resolve a raw symbol token to canonical form.

        Resolution order:
        1. Check cache
        2. Try alias mapping (fast)
        3. Try database search (accurate)
        4. Try LLM resolution (flexible)
        5. Try pattern matching (fallback)

        Args:
            raw_symbol: Raw token (e.g., "TELIA", "petro", "PETR4")

        Returns:
            ResolvedSymbol or None
        """
        if not raw_symbol:
            return None

        normalized = raw_symbol.strip().upper()

        # Check cache first
        if normalized in self._cache:
            logger.debug("Symbol cache hit: %s", raw_symbol)
            return self._cache[normalized]

        # Strategy 1: Try alias mapping (fast)
        result = self._resolve_via_alias(normalized)
        if result:
            logger.info("Symbol resolved via alias: %s → %s", raw_symbol, result.symbol)
            self._cache[normalized] = result
            return result

        # Strategy 2: Try database search (accurate)
        if self.skill_gateway:
            result = self._resolve_via_database(normalized)
            if result:
                logger.info("Symbol resolved via database: %s → %s", raw_symbol, result.symbol)
                self._cache[normalized] = result
                return result

        # Strategy 3: Try LLM resolution (flexible)
        if self.enable_llm and self.model_selector:
            result = self._resolve_via_llm(normalized)
            if result:
                logger.info("Symbol resolved via LLM: %s → %s", raw_symbol, result.symbol)
                self._cache[normalized] = result
                return result

        # Strategy 4: Try pattern matching (fallback)
        result = self._resolve_via_pattern(normalized)
        if result:
            logger.info("Symbol resolved via pattern: %s → %s", raw_symbol, result.symbol)
            self._cache[normalized] = result
            return result

        logger.debug("Symbol not resolved: %s", raw_symbol)
        return None

    def _resolve_via_alias(self, symbol: str) -> Optional[ResolvedSymbol]:
        """Resolve using alias mapping (fast, deterministic)."""
        if symbol not in self.aliases:
            return None

        canonical = self.aliases[symbol]
        exchange = self._extract_exchange(canonical)

        return ResolvedSymbol(
            symbol=canonical,
            exchange=exchange,
            confidence=1.0,
            source="alias",
        )

    def _resolve_via_database(self, symbol: str) -> Optional[ResolvedSymbol]:
        """
        Resolve using local database search (search_symbol skill).

        Calls skill_gateway to search in database.
        """
        if not self.skill_gateway:
            return None

        try:
            # Call search_symbol skill
            result = self.skill_gateway.execute(
                "mcp_finance",
                {"_action": "search_symbol", "query": symbol},
            )

            if not result.get("success"):
                logger.debug("Database search failed for: %s", symbol)
                return None

            # Extract candidates from result
            candidates = self._extract_candidates(result.get("data", {}))

            if not candidates:
                logger.debug("No candidates found in database for: %s", symbol)
                return None

            # Filter candidates by focus exchanges (prioritize focus exchanges)
            candidates = filter_by_focus_exchange(candidates)

            # If single match, return it
            if len(candidates) == 1:
                candidate = candidates[0]
                exchange = self._extract_exchange(candidate["symbol"])
                confidence = 1.0 if is_exchange_in_focus(exchange) else 0.6
                return ResolvedSymbol(
                    symbol=candidate["symbol"],
                    exchange=exchange,
                    company_name=candidate.get("name"),
                    confidence=confidence,
                    source="database",
                )

            # If multiple matches, return first with ambiguity flag
            if len(candidates) > 1:
                primary = candidates[0]
                exchange = self._extract_exchange(primary["symbol"])
                logger.warning("Ambiguous symbol search for: %s (%d matches)", symbol, len(candidates))
                return ResolvedSymbol(
                    symbol=primary["symbol"],
                    exchange=exchange,
                    company_name=primary.get("name"),
                    confidence=0.7 if is_exchange_in_focus(exchange) else 0.5,  # Lower for non-focus
                    source="database",
                    ambiguous=True,
                    alternatives=candidates[1:],
                )

        except Exception as e:
            logger.warning("Database search error for %s: %s", symbol, e)
            return None

    def _resolve_via_llm(self, symbol: str) -> Optional[ResolvedSymbol]:
        """
        Resolve using LLM with structured output via ModelSelector.

        Uses the application's LLM infrastructure (Claude Haiku) to resolve
        ambiguous symbols and return structured data: symbol, exchange, company_name
        """
        if not self.model_selector:
            return None

        try:
            prompt = self._build_llm_prompt(symbol)

            # Use ModelSelector to call Claude Haiku
            # This integrates with your domain's LLM configuration
            response = self.model_selector.call_llm(
                prompt=prompt,
                model="claude-haiku-4-5-20251001",  # Your configured model
                temperature=0.2,  # Low temperature for structured output
                max_tokens=200,
            )

            if not response:
                logger.debug("LLM returned empty response for: %s", symbol)
                return None

            # Parse LLM response
            parsed = self._parse_llm_response(response)
            if not parsed:
                logger.debug("LLM could not resolve: %s", symbol)
                return None

            return ResolvedSymbol(
                symbol=parsed["symbol"],
                exchange=parsed.get("exchange", self._extract_exchange(parsed["symbol"])),
                company_name=parsed.get("company_name"),
                confidence=parsed.get("confidence", 0.8),
                source="llm",
                ambiguous=parsed.get("ambiguous", False),
                alternatives=parsed.get("alternatives", []),
            )

        except Exception as e:
            logger.warning("LLM resolution error for %s: %s", symbol, e)
            return None

    def _resolve_via_pattern(self, symbol: str) -> Optional[ResolvedSymbol]:
        """
        Resolve using canonical pattern matching (fallback).

        Patterns:
        - B3: PETR4 → PETR4.SA
        - US: AAPL → AAPL
        - Explicit: PETR4.SA → PETR4.SA (already normalized)
        """
        import re

        # Already has exchange suffix
        if bool(re.fullmatch(r"[A-Z0-9-]{1,12}\.[A-Z]{1,4}", symbol)):
            exchange = self._extract_exchange(symbol)
            return ResolvedSymbol(
                symbol=symbol,
                exchange=exchange,
                confidence=0.9,
                source="pattern",
            )

        # B3 base symbols: 4 letters + class (3/4/5/6/11)
        if bool(re.fullmatch(r"[A-Z]{4}(?:3|4|5|6|11)(?:F)?", symbol)):
            canonical = f"{symbol}.SA"
            return ResolvedSymbol(
                symbol=canonical,
                exchange="SA",
                confidence=0.9,
                source="pattern",
            )

        # US tickers (letters only, short)
        if bool(re.fullmatch(r"[A-Z]{1,5}", symbol)):
            return ResolvedSymbol(
                symbol=symbol,
                exchange="US",
                confidence=0.8,
                source="pattern",
            )

        return None

    def _build_llm_prompt(self, symbol: str) -> str:
        """Build prompt for LLM symbol resolution with focus exchange guidance."""
        exchange_guidance = get_llm_exchange_guidance()
        return f"""You are a financial symbol resolver. Given a symbol or company name,
resolve it to a canonical trading symbol with exchange information.

{exchange_guidance}

Input: {symbol}

Return a JSON response with this structure (if multiple possibilities, list them):
{{
  "symbol": "PETR4.SA",
  "exchange": "SA",
  "company_name": "Petrobras",
  "confidence": 0.95,
  "ambiguous": false,
  "alternatives": []
}}

If the input is ambiguous (could be multiple symbols), set ambiguous=true and list alternatives.
When listing alternatives, prioritize symbols from focus exchanges (SA, ST, US).

Return ONLY valid JSON, no other text."""

    def _parse_llm_response(self, response: str) -> Optional[dict]:
        """Parse LLM response into structured data."""
        import json

        try:
            # Extract JSON from response
            response_clean = response.strip()
            if response_clean.startswith("```"):
                # Remove markdown code blocks
                response_clean = response_clean.replace("```json", "").replace("```", "").strip()

            data = json.loads(response_clean)

            # Validate required fields
            if not data.get("symbol"):
                return None

            return {
                "symbol": data["symbol"].upper(),
                "exchange": (data.get("exchange") or "").upper(),
                "company_name": data.get("company_name"),
                "confidence": float(data.get("confidence", 0.8)),
                "ambiguous": data.get("ambiguous", False),
                "alternatives": data.get("alternatives", []),
            }

        except Exception as e:
            logger.warning("Failed to parse LLM response: %s", e)
            return None

    def _extract_candidates(self, data: dict) -> list[dict]:
        """Extract symbol candidates from database search result."""
        candidates = []

        # Support multiple response formats
        for key in ("results", "symbols", "quotes", "matches", "items"):
            if isinstance(data.get(key), list):
                candidates = data[key]
                break

        if not candidates and isinstance(data, dict):
            candidates = [data]

        # Normalize candidate format
        normalized = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            symbol = (
                item.get("symbol")
                or item.get("ticker")
                or item.get("code")
            )
            if symbol:
                normalized.append(
                    {
                        "symbol": symbol.upper(),
                        "name": item.get("name") or item.get("shortname"),
                    }
                )

        return normalized

    def _extract_exchange(self, symbol: str) -> str:
        """Extract exchange code from symbol."""
        if "." in symbol:
            return symbol.split(".")[-1].upper()

        # Infer from pattern
        import re

        if bool(re.fullmatch(r"[A-Z]{4}(?:3|4|5|6|11)(?:F)?", symbol)):
            return "SA"  # B3

        if bool(re.fullmatch(r"[A-Z]{1,5}", symbol)):
            return "US"  # US ticker

        return "UNKNOWN"

    def clear_cache(self) -> None:
        """Clear resolution cache."""
        self._cache.clear()
        logger.info("Symbol resolution cache cleared")

    def get_stats(self) -> dict:
        """Get resolver statistics."""
        return {
            "aliases": len(self.aliases),
            "cached_symbols": len(self._cache),
            "llm_enabled": self.enable_llm,
            "db_enabled": bool(self.skill_gateway),
        }
