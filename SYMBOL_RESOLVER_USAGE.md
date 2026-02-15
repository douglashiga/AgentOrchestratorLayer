# Symbol Resolver - Usage Guide

## Overview

The `SymbolResolver` is a multi-strategy symbol resolution system that uses:
1. **Alias mapping** (fastest, pre-configured)
2. **Database search** (accurate, your `search_symbol` capability)
3. **LLM resolution** (flexible, handles ambiguity)
4. **Pattern matching** (fallback, regex-based)

## Architecture

```
Raw Symbol ("petro", "PETR4", "abc123")
    ↓
SymbolResolver.resolve()
    ├─ Try: Alias mapping → "PETR4.SA" ✓
    ├─ Try: Database search → results with company info
    ├─ Try: LLM → structured with alternatives
    └─ Try: Pattern matching → B3/US patterns
    ↓
ResolvedSymbol {
    symbol: "PETR4.SA",
    exchange: "SA",
    company_name: "Petrobras",
    confidence: 0.95,
    source: "database",
    ambiguous: false,
    alternatives: []
}
```

## Integration in Your Domain Handler

```python
from domains.finance.symbol_resolver import SymbolResolver
from models.selector import ModelSelector

class FinanceDomainHandler:
    def __init__(self, skill_gateway, registry=None):
        # Initialize your resolver
        self._symbol_resolver = SymbolResolver(
            aliases=SYMBOL_ALIASES,          # From your config
            skill_gateway=skill_gateway,      # For search_symbol calls
            model_selector=ModelSelector(),   # For LLM fallback
            enable_llm=True                   # Enable LLM fallback
        )

    async def get_stock_price(self, intent, params):
        """Get stock price with symbol resolution."""

        # If symbol is missing or ambiguous, resolve it
        if not params.get("symbol"):
            raw_symbol = params.get("_raw_symbol") or intent.original_query
            resolved = self._symbol_resolver.resolve(raw_symbol)

            if not resolved:
                return DomainOutput(
                    status="clarification",
                    explanation=f"Could not find symbol '{raw_symbol}'",
                )

            # Handle ambiguity
            if resolved.ambiguous:
                alternatives = ", ".join(
                    f"{alt['symbol']} ({alt.get('name', 'Unknown')})"
                    for alt in resolved.alternatives
                )
                return DomainOutput(
                    status="clarification",
                    explanation=f"Multiple symbols found: {resolved.symbol}, {alternatives}. Which one?",
                    metadata={"alternatives": resolved.alternatives}
                )

            params["symbol"] = resolved.symbol

        # Continue with get_stock_price...
        return await self._execute_stock_price(params)
```

## LLM Prompt (What Claude Sees)

When database search fails, the LLM gets this prompt:

```
You are a financial symbol resolver. Given a symbol or company name,
resolve it to a canonical trading symbol with exchange information.

Input: petrobras

Return a JSON response with this structure (if multiple possibilities, list them):
{
  "symbol": "PETR4.SA",
  "exchange": "SA",
  "company_name": "Petrobras",
  "confidence": 0.95,
  "ambiguous": false,
  "alternatives": []
}

If the input is ambiguous (could be multiple symbols), set ambiguous=true and list alternatives.

Return ONLY valid JSON, no other text.
```

## Example Workflows

### Scenario 1: Known Alias
```
Input: "telia"
  ↓
Alias lookup: "TELIA" → "TELIA.ST"
  ↓
Result: ResolvedSymbol(
  symbol="TELIA.ST",
  source="alias",
  confidence=1.0
)
```

### Scenario 2: Database Search (Your search_symbol)
```
Input: "petrobras"
  ↓
Alias: No match
  ↓
Database: search_symbol("PETROBRAS")
  → Returns: [{symbol: "PETR4.SA", name: "Petrobras PN"}, ...]
  ↓
Result: ResolvedSymbol(
  symbol="PETR4.SA",
  company_name="Petrobras PN",
  source="database",
  confidence=1.0
)
```

### Scenario 3: LLM Fallback (Ambiguous)
```
Input: "ABC"
  ↓
Alias: No match
  ↓
Database: No results
  ↓
LLM: "What is ABC in finance?"
  ← Claude responds: {
    "symbol": "ABC",
    "exchange": "US",
    "company_name": "ABC Inc",
    "ambiguous": true,
    "alternatives": [
      {"symbol": "ABC.US", "name": "ABC Inc"},
      {"symbol": "ABC.L", "name": "ABC Limited"}
    ]
  }
  ↓
Result: ResolvedSymbol(
  symbol="ABC",
  exchange="US",
  source="llm",
  confidence=0.8,
  ambiguous=true,
  alternatives=[...]
)
```

### Scenario 4: Pattern Matching (Fallback)
```
Input: "PETR4"  (raw B3 code)
  ↓
Alias: No match
  ↓
Database: (disabled or failed)
  ↓
LLM: (disabled)
  ↓
Pattern: Matches B3 pattern [A-Z]{4}[3-6]
  → PETR4 → PETR4.SA
  ↓
Result: ResolvedSymbol(
  symbol="PETR4.SA",
  exchange="SA",
  source="pattern",
  confidence=0.9
)
```

## Configuration

### In Your Domain Handler's `__init__`:

```python
# Minimal (no LLM fallback)
self._symbol_resolver = SymbolResolver(
    aliases=SYMBOL_ALIASES,
    skill_gateway=skill_gateway,
    enable_llm=False
)

# Full (with LLM)
from models.selector import ModelSelector

model_selector = ModelSelector()  # Your app's LLM provider
self._symbol_resolver = SymbolResolver(
    aliases=SYMBOL_ALIASES,
    skill_gateway=skill_gateway,
    model_selector=model_selector,
    enable_llm=True
)
```

## Handling Ambiguity

When a symbol is ambiguous (multiple matches), you have two options:

### Option 1: Ask User via Clarification
```python
if resolved.ambiguous:
    return DomainOutput(
        status="clarification",
        explanation="Which symbol did you mean?",
        metadata={
            "alternatives": [
                {"symbol": alt["symbol"], "name": alt.get("name")}
                for alt in resolved.alternatives
            ]
        }
    )
```

### Option 2: Use Primary + Log Warning
```python
if resolved.ambiguous:
    logger.warning(
        "Ambiguous symbol: %s (alternatives: %s)",
        resolved.symbol,
        [alt["symbol"] for alt in resolved.alternatives]
    )
    # Continue with primary match
    params["symbol"] = resolved.symbol
```

## Performance

- **Alias lookup**: <1ms
- **Database search**: 50-200ms (network + DB)
- **LLM resolution**: 200-500ms (API call)
- **Pattern matching**: <1ms

Cache keeps successful resolutions, so repeat queries are instant.

## Caching

The resolver caches all successful resolutions. To clear:

```python
# Clear entire cache
self._symbol_resolver.clear_cache()

# Or check stats
stats = self._symbol_resolver.get_stats()
print(f"Cached symbols: {stats['cached_symbols']}")
```

## Error Handling

All errors are logged and don't break the flow:

```
- Database error? → Falls back to LLM
- LLM error? → Falls back to pattern matching
- Pattern matching fails? → Returns None (ask user for clarification)
```

## Testing

```python
# Test with mock aliases
test_aliases = {
    "PETRO": "PETR4.SA",
    "VALE": "VALE3.SA",
}

resolver = SymbolResolver(
    aliases=test_aliases,
    skill_gateway=mock_gateway,
    model_selector=None,  # No LLM for tests
    enable_llm=False
)

resolved = resolver.resolve("PETRO")
assert resolved.symbol == "PETR4.SA"
assert resolved.source == "alias"
```
