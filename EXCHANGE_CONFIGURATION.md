# Exchange Configuration Guide

## Overview

The finance domain supports multiple stock exchanges and markets. A configuration system ensures the system focuses on specific exchanges (Sweden/ST, Brazil/SA, and USA) while gracefully degrading for other markets.

## Supported Exchanges

### Focus Exchanges (Prioritized)
These exchanges are prioritized in symbol resolution and LLM prompts:

1. **SA (B3 - Brazil Stock Exchange)**
   - Country: Brazil
   - Market Code: BR
   - Example: `PETR4.SA`, `VALE3.SA`, `ITUB4.SA`

2. **ST (Nasdaq Stockholm)**
   - Country: Sweden
   - Market Code: SE
   - Example: `TELIA.ST`, `VOLV-B.ST`, `HM-B.ST`

3. **US (US Stock Exchange)**
   - Countries: USA
   - Market Code: US
   - Example: `AAPL`, `TSLA`, `MSFT` (no suffix)

### Non-Focus Exchanges (For Reference)
Other exchanges like Hong Kong (HK), London (L), etc. are supported but not prioritized.

## Configuration System

### Configuration File
**Location:** `domains/finance/config.py`

#### Supported Exchanges Dictionary
```python
SUPPORTED_EXCHANGES = {
    "SA": {
        "name": "B3 (Brazil Stock Exchange)",
        "country": "Brazil",
        "country_code": "BR",
        "suffix_format": ".SA",
    },
    "ST": {
        "name": "Nasdaq Stockholm",
        "country": "Sweden",
        "country_code": "SE",
        "suffix_format": ".ST",
    },
    "US": {
        "name": "US Stock Exchange",
        "country": "United States",
        "country_code": "US",
        "suffix_format": "",
    },
}
```

#### Focus Exchanges
```python
FOCUS_EXCHANGES = ["SA", "ST", "US"]  # Exchanges to prioritize
FOCUS_COUNTRIES = ["BR", "SE", "US"]  # Corresponding countries
```

#### Market Aliases
User input is mapped to country/exchange codes:
```python
MARKET_ALIASES = {
    # Brazil
    "BOVESPA": "BR",
    "B3": "BR",
    "BRASIL": "BR",

    # USA
    "NYSE": "US",
    "NASDAQ": "US",
    "EUA": "US",

    # Sweden
    "SUECIA": "SE",
    "SWEDEN": "SE",
}
```

## Configuration in Action

### 1. Symbol Alias Resolution
**File:** `domains/finance/server.py` → `SYMBOL_ALIASES`

Contains 76 aliases organized by focus exchanges:
- **18 Brazilian companies** (B3/SA): Petrobras, Vale, Itau, Cosan, JBS, Ambev, Embraer, Natura
- **17 USA companies** (US): Apple, Tesla, Microsoft, Google, Amazon, Meta, Netflix
- **17 Swedish companies** (ST): Volvo, H&M, SKF, Electrolux, Ericsson, SEB, Swedbank

Example:
```python
SYMBOL_ALIASES = {
    # Brazilian
    "PETRO": "PETR4.SA",
    "VALE": "VALE3.SA",

    # USA
    "AAPL": "AAPL",
    "TSLA": "TSLA",

    # Swedish
    "TELIA": "TELIA.ST",
    "VOLVO": "VOLV-B.ST",
}
```

### 2. Database Search Filtering
**File:** `domains/finance/symbol_resolver.py` → `_resolve_via_database()`

When searching the database:
```python
# Filter candidates by focus exchanges (prioritize focus exchanges)
candidates = filter_by_focus_exchange(candidates)
```

**Behavior:**
- Reorders candidate list to put focus exchange symbols first
- Adjusts confidence scores:
  - Focus exchange: 1.0 (high confidence)
  - Non-focus exchange: 0.6 (lower confidence)

### 3. LLM Resolution Guidance
**File:** `domains/finance/symbol_resolver.py` → `_build_llm_prompt()`

The LLM receives explicit guidance:
```
IMPORTANT: Only resolve symbols to these exchanges: SA, ST, US
Corresponding countries: BR, SE, US
If a symbol exists on other exchanges, prefer the focus exchanges.
If only non-focus exchanges are available, return the best match but indicate low confidence.
```

**Example LLM Prompt:**
```
You are a financial symbol resolver. Given a symbol or company name,
resolve it to a canonical trading symbol with exchange information.

IMPORTANT: Only resolve symbols to these exchanges: SA, ST, US
Corresponding countries: BR, SE, US
If a symbol exists on other exchanges, prefer the focus exchanges.
If only non-focus exchanges are available, return the best match but indicate low confidence.

Input: apple

Return a JSON response with this structure...
```

## How It Works: Symbol Resolution Flow

### 1. Input: "qual o valor da PETR4, TELIA e AAPL?"
```
↓
Alias Mapping Strategy
├─ "PETR4" → "PETR4.SA" (recognized B3 format)
├─ "TELIA" → "TELIA.ST" (in SYMBOL_ALIASES)
├─ "AAPL" → "AAPL" (in SYMBOL_ALIASES)
↓
Output: ["PETR4.SA", "TELIA.ST", "AAPL"]
All focus exchanges ✓
```

### 2. Input: "qual o valor da Petrobras, Volvo e Apple?"
```
↓
Alias Mapping Strategy
├─ "PETROBRAS" → "PETR4.SA" (SYMBOL_ALIASES)
├─ "VOLVO" → "VOLV-B.ST" (SYMBOL_ALIASES)
├─ "APPLE" → "AAPL" (SYMBOL_ALIASES)
↓
Output: ["PETR4.SA", "VOLV-B.ST", "AAPL"]
All focus exchanges ✓
```

### 3. Input: "qual o valor da Sony?" (Non-focus Exchange)
```
↓
Alias Mapping Strategy
├─ "SONY" → Not found in SYMBOL_ALIASES
↓
Database Search Strategy
├─ search_symbol("SONY")
├─ Results: [6698.T (Tokyo), SNE (US)]
├─ filter_by_focus_exchange([6698.T, SNE])
├─ Reorder: [SNE (focus), 6698.T (non-focus)]
↓
Return: SNE with confidence 0.7 (focus exchange)
↓
If user wants Tokyo stock: Ambiguity flag + alternatives
```

## Usage in Code

### Using Configuration Functions

```python
from domains.finance.config import (
    get_focus_exchanges,
    is_exchange_in_focus,
    filter_by_focus_exchange,
    get_llm_exchange_guidance,
)

# Check if exchange is in focus
if is_exchange_in_focus("SA"):
    print("Brazil exchange is in focus")  # True

# Get list of focus exchanges
focus = get_focus_exchanges()
# ["SA", "ST", "US"]

# Filter and reorder symbols
symbols = [
    {"symbol": "6698.T", "name": "Sony"},
    {"symbol": "SNE", "name": "Sony"},
]
filtered = filter_by_focus_exchange(symbols)
# [{"symbol": "SNE", ...}, {"symbol": "6698.T", ...}]

# Get guidance for LLM prompts
guidance = get_llm_exchange_guidance()
# "IMPORTANT: Only resolve symbols to these exchanges: SA, ST, US..."
```

### Modifying Focus Exchanges

To change which exchanges are prioritized:

1. **Edit** `domains/finance/config.py`:
   ```python
   FOCUS_EXCHANGES = ["SA", "ST", "US", "L"]  # Add London
   FOCUS_COUNTRIES = ["BR", "SE", "US", "UK"]
   ```

2. **Update** `domains/finance/server.py` SYMBOL_ALIASES with new companies

3. **Update** test files to add new test cases

4. **Run tests** to verify configuration works

## Confidence Scores

Symbol resolution returns confidence scores:

| Scenario | Score | Reason |
|----------|-------|--------|
| Alias match + focus exchange | 1.0 | Direct match, prioritized market |
| Database single match + focus | 1.0 | Exact database match from focus market |
| Database single match + non-focus | 0.6 | Exact match but non-prioritized market |
| Database multiple matches (1st focus) | 0.7 | Ambiguous, but best choice is focus |
| Database multiple matches (1st non-focus) | 0.5 | Ambiguous, best choice is non-prioritized |
| LLM resolution + focus | 0.8-0.95 | LLM decides confidence |
| LLM resolution + non-focus | 0.3-0.6 | LLM found non-focus match |
| Pattern match + focus format | 0.9 | Recognized pattern from focus market |

## Market Aliases

User-friendly aliases for markets/exchanges:

```python
MARKET_ALIASES = {
    # Brazil
    "BOVESPA": "BR",
    "IBOVESPA": "BR",
    "IBOV": "BR",
    "B3": "BR",
    "BRASIL": "BR",
    "BRAZIL": "BR",

    # USA
    "NYSE": "US",
    "NASDAQ": "US",
    "EUA": "US",
    "USA": "US",

    # Sweden
    "SUECIA": "SE",
    "SWEDEN": "SE",
    "STOCKHOLM": "SE",
}
```

Users can say "qual o valor da AAPL na NASDAQ?" and the system understands it refers to the US market.

## Testing

All configuration behavior is tested:

```python
# Test focus exchange filtering
def test_filter_by_focus_exchange():
    symbols = [{"symbol": "6698.T", "exchange": "JP"}, {"symbol": "SNE", "exchange": "US"}]
    filtered = filter_by_focus_exchange(symbols)
    assert filtered[0]["symbol"] == "SNE"  # Focus first

# Test LLM guidance
def test_llm_exchange_guidance():
    guidance = get_llm_exchange_guidance()
    assert "SA, ST, US" in guidance
```

See `test_finance_symbol_resolution.py` for complete test coverage.

## Future Enhancements

1. **Per-User Configuration**: Allow users to select focus markets
2. **Temporal Focus**: Different focus exchanges for different times/markets
3. **API Endpoint**: Expose exchange configuration via REST API
4. **Dynamic Configuration**: Load focus exchanges from database or API
5. **Regional Settings**: Auto-detect user region and adjust focus exchanges

## Summary

The exchange configuration system:
- ✅ Centralizes market/exchange definitions in `config.py`
- ✅ Filters and prioritizes symbols from focus exchanges (SA, ST, US)
- ✅ Reduces LLM confusion by limiting scope
- ✅ Provides confidence scores based on exchange focus
- ✅ Maintains graceful degradation for non-focus exchanges
- ✅ Is testable, maintainable, and extensible
