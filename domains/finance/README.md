# Finance Domain

Financial data domain. Exposes quotes, history, screener, top movers, dividends, options, and fundamental data via MCP (finance server).

- **Internal port:** `8001`
- **Compose port:** `8003`
- **Type:** `remote_http`
- **Entry point:** `domains/finance/server.py`

---

## Endpoints

| Endpoint | Description |
|----------|-----------|
| `GET /health` | Health check |
| `GET /manifest` | Goals + capabilities + metadata |
| `POST /execute` | Executes an `ExecutionIntent` |

### `/execute` — Payload

```python
# Input (serialized ExecutionIntent)
{
    "domain": "finance",
    "capability": "get_stock_price",
    "confidence": 0.95,
    "parameters": {
        "symbol_text": "Nordea"        # or "symbol": "NDA-SE.ST" if already resolved
    },
    "original_query": "qual o preço da Nordea?"
}

# Output (DomainOutput)
{
    "status": "success",
    "result": {
        "symbol": "NDA-SE.ST",
        "price": 112.5,
        "currency": "SEK",
        "change": -0.8,
        "_market_context": {
            "market": "SE",
            "exchange": "OMX",
            "timezone": "Europe/Stockholm",
            "currency": "SEK"
        }
    },
    "explanation": "Nordea (NDA-SE.ST) está em 112.50 SEK",
    "confidence": 1.0,
    "metadata": {}
}
```

---

## Goals and Capabilities

### GET_QUOTE — Current quote

**Capabilities:** `get_stock_price`

**Entities schema:**
```python
{
    "symbol_text": {"type": "string", "required": True,
                    "description": "Symbol or company name as the user said it"},
    "symbols_text": {"type": "array", "description": "Multiple symbols for comparison"}
}
```

**Intent example:**
```python
IntentOutput(
    primary_domain="finance", goal="GET_QUOTE",
    entities={"symbol_text": "Nordea"},
    confidence=0.95
)
# → ExecutionIntent(capability="get_stock_price", parameters={"symbol_text": "Nordea"})
```

**Output:**
```json
{
    "symbol": "NDA-SE.ST",
    "price": 112.5,
    "currency": "SEK",
    "change": -0.8,
    "change_pct": -0.71
}
```

---

### TOP_MOVERS — Biggest gainers/losers

**Capabilities:** `get_top_gainers`, `get_top_losers`

**Entities schema (with capability_map):**
```python
{
    "direction": {
        "type": "enum",
        "values": ["GAINERS", "LOSERS", "BOTH"],
        "required": True,
        "default": "BOTH",
        "capability_map": {
            "GAINERS": "get_top_gainers",
            "LOSERS":  "get_top_losers",
            "BOTH":    ["get_top_gainers", "get_top_losers"]
        }
    },
    "market_text": {"type": "string"},
    "period_text":  {"type": "string"}
}
```

**Resolution by GoalResolver:**
```
direction=GAINERS → capability=get_top_gainers
direction=LOSERS  → capability=get_top_losers
direction=BOTH    → two parallel steps in ExecutionPlan
no direction      → fallback: get_top_gainers
```

**Output:**
```json
{
    "items": [
        {"symbol": "VALE3.SA", "change_pct": 4.2, "price": 67.80},
        {"symbol": "PETR4.SA", "change_pct": 3.1, "price": 38.50}
    ],
    "count": 10,
    "market": "BR"
}
```

---

### VIEW_HISTORY — OHLC History

**Capabilities:** `get_historical_data`

**Entities schema:**
```python
{
    "symbol_text": {"type": "string", "required": True},
    "period_text":   {"type": "string", "description": "e.g. 'last month', '1 year', 'ytd'"},
    "interval_text": {"type": "string", "description": "e.g. 'daily', 'weekly'"}
}
```

**Output:**
```json
{
    "symbol": "PETR4.SA",
    "period": "1mo",
    "interval": "1d",
    "data": [
        {"date": "2026-01-15", "open": 37.2, "high": 38.9, "low": 36.8, "close": 38.5, "volume": 12500000}
    ]
}
```

---

### SCREEN_STOCKS — Stock screener

**Capabilities:** `get_stock_screener`

**Entities schema:**
```python
{
    "market_text": {"type": "string"},
    "sector_text":  {"type": "string"}
}
```

**Output:**
```json
{
    "items": [
        {"symbol": "VALE3.SA", "pe_ratio": 6.2, "dividend_yield": 8.1, "market_cap": 280000000000}
    ],
    "count": 25
}
```

---

### DIVIDEND_ANALYSIS — Dividends

**Capabilities:** `get_top_dividend_payers`, `get_dividends`

**Entities schema (with capability_map):**
```python
{
    "focus": {
        "type": "enum",
        "values": ["RANKING", "HISTORY"],
        "capability_map": {
            "RANKING": "get_top_dividend_payers",
            "HISTORY": "get_dividends"
        }
    },
    "symbol_text":  {"type": "string"},
    "market_text":  {"type": "string"}
}
```

---

### OPTIONS_DATA — Options

**Capabilities:** `get_option_chain`, `get_option_greeks`

**Entities schema:**
```python
{
    "focus": {
        "type": "enum",
        "values": ["CHAIN", "GREEKS"],
        "capability_map": {
            "CHAIN":  "get_option_chain",
            "GREEKS": "get_option_greeks"
        }
    },
    "symbol_text": {"type": "string", "required": True}
}
```

---

### TECHNICAL_SCAN — Technical signals

**Capabilities:** `get_technical_signals`

**Output:**
```json
{
    "items": [
        {"symbol": "PETR4.SA", "rsi": 32.4, "macd_signal": "bullish_crossover"}
    ]
}
```

---

### COMPARE_STOCKS — Fundamental comparison

**Capabilities:** `compare_fundamentals`

**Entities schema:**
```python
{
    "symbols_text": {"type": "array", "required": True,
                     "description": "List of companies to compare"}
}
```

---

### Other Goals

| Goal | Capabilities | Description |
|------|-------------|-----------|
| `FUNDAMENTALS` | `get_fundamentals` | Fundamental data for a stock |
| `COMPANY_PROFILE` | `get_company_info` | Company profile and sector |
| `FINANCIAL_STATEMENTS` | `get_financial_statements` | Income statement, balance sheet |
| `ACCOUNT_OVERVIEW` | `get_account_summary` | Brokerage account summary |
| `SEARCH_SYMBOL` | `search_symbol`, `yahoo_search` | Search ticker by name |
| `PIPELINE_STATUS` | `list_jobs`, `get_job_status` | Data pipeline job status |

---

## Internal Architecture

```
ExecutionIntent (capability + parameters)
  │
  ├─ FinanceDomainHandler.execute()
  │    ├─ _resolve_parameters()        # applies manifest defaults
  │    ├─ get_type_hints() → typed dispatch (e.g. get_top_gainers)
  │    └─ _run_pipeline()
  │         ├─ _apply_pre_flow()       # resolve_symbol / resolve_symbol_list
  │         │    └─ SymbolResolver
  │         │         ├─ aliases dict  (metadata-driven, e.g. NORDEA → NDA-SE.ST)
  │         │         └─ search_symbol (MCP fallback)
  │         ├─ ContextResolver         # symbol → DomainContext (market, currency, exchange)
  │         ├─ SkillGateway            # → MCP finance server
  │         └─ StrategyCore           # skill data → Decision → DomainOutput
  │
  └─ DomainOutput
```

### Pre-flow: Symbol Resolution

The handler resolves human-friendly names → technical tickers **before** calling the skill:

```
parameters.symbol_text = "Nordea"
  → SymbolResolver.resolve("NORDEA")
  → check aliases dict → "NDA-SE.ST" ✓  (deterministic alias)
  → (fallback) search_symbol MCP call
  → (clarification) if ambiguous: returns candidates
```

### Context Resolution

After resolving the symbol, the `ContextResolver` determines the market context:

```python
DomainContext(
    market="SE",
    currency="SEK",
    exchange="OMX",
    exchange_suffix=".ST",
    exchange_timezone="Europe/Stockholm",
    settlement_days=2,
    tax_model="ISK",
    tax_rate_gains=0.30
)
```

---

## Environment Variables

| Variable | Default | Description |
|----------|--------|-----------|
| `MCP_URL` | `http://localhost:8000/sse` | MCP finance server URL |
| `MCP_ADAPTER_CALL_TIMEOUT_SECONDS` | `90` | MCP call timeout |
| `FINANCE_DOMAIN_PORT` | `8001` | Service port |
| `SYMBOL_ALIASES_JSON` | — | Alias override in JSON |

---

## Supported Flows (pre-flow metadata)

```json
{
    "flow": {
        "pre": [
            {
                "type": "resolve_symbol",
                "param": "symbol",
                "search_capability": "search_symbol",
                "search_fallback_capabilities": ["yahoo_search"],
                "required": true
            }
        ]
    }
}
```

When `flow.pre` is not declared, the handler infers automatically:
- `symbol` field in schema → `resolve_symbol`
- `symbols` field in schema → `resolve_symbol_list`

---

## Adding a New Capability

1. Declare in `DOMAIN_MANIFEST["capabilities"]` in `server.py`
2. Define `entities_schema` in the corresponding goal (or create a new goal)
3. Implement a typed method in `handler.py`:
   ```python
   async def nova_capability(self, intent: IntentOutput | ExecutionIntent, params: NovaInput) -> DomainOutput:
       return await self._run_pipeline(intent, params.model_dump())
   ```
4. Create Pydantic schema in `schemas.py`
5. GoalResolver and TaskDecomposer work automatically via metadata
