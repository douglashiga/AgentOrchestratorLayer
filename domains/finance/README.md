# Finance Domain

Domínio de dados financeiros. Expõe cotações, histórico, screener, top movers, dividendos, opções e dados fundamentalistas via MCP (finance server).

- **Porta interna:** `8001`
- **Porta no compose:** `8003`
- **Tipo:** `remote_http`
- **Entry point:** `domains/finance/server.py`

---

## Endpoints

| Endpoint | Descrição |
|----------|-----------|
| `GET /health` | Health check |
| `GET /manifest` | Goals + capabilities + metadata |
| `POST /execute` | Executa uma `ExecutionIntent` |

### `/execute` — Payload

```python
# Input (ExecutionIntent serializado)
{
    "domain": "finance",
    "capability": "get_stock_price",
    "confidence": 0.95,
    "parameters": {
        "symbol_text": "Nordea"        # ou "symbol": "NDA-SE.ST" se já resolvido
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

## Goals e Capabilities

### GET_QUOTE — Cotação atual

**Capabilities:** `get_stock_price`

**Entities schema:**
```python
{
    "symbol_text": {"type": "string", "required": True,
                    "description": "Símbolo ou nome da empresa como o usuário disse"},
    "symbols_text": {"type": "array", "description": "Múltiplos símbolos para comparação"}
}
```

**Exemplo de intent:**
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

### TOP_MOVERS — Maiores altas/baixas

**Capabilities:** `get_top_gainers`, `get_top_losers`

**Entities schema (com capability_map):**
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

**Resolução pelo GoalResolver:**
```
direction=GAINERS → capability=get_top_gainers
direction=LOSERS  → capability=get_top_losers
direction=BOTH    → dois steps paralelos no ExecutionPlan
sem direction     → fallback: get_top_gainers
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

### VIEW_HISTORY — Histórico OHLC

**Capabilities:** `get_historical_data`

**Entities schema:**
```python
{
    "symbol_text": {"type": "string", "required": True},
    "period_text":   {"type": "string", "description": "ex: 'ultimo mes', '1 ano', 'ytd'"},
    "interval_text": {"type": "string", "description": "ex: 'diário', 'semanal'"}
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

### SCREEN_STOCKS — Screener de ações

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

### DIVIDEND_ANALYSIS — Dividendos

**Capabilities:** `get_top_dividend_payers`, `get_dividends`

**Entities schema (com capability_map):**
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

### OPTIONS_DATA — Opções

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

### TECHNICAL_SCAN — Sinais técnicos

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

### COMPARE_STOCKS — Comparação fundamentalista

**Capabilities:** `compare_fundamentals`

**Entities schema:**
```python
{
    "symbols_text": {"type": "array", "required": True,
                     "description": "Lista de empresas para comparar"}
}
```

---

### Outros Goals

| Goal | Capabilities | Descrição |
|------|-------------|-----------|
| `FUNDAMENTALS` | `get_fundamentals` | Dados fundamentalistas de uma ação |
| `COMPANY_PROFILE` | `get_company_info` | Perfil e setor da empresa |
| `FINANCIAL_STATEMENTS` | `get_financial_statements` | DRE, balanço patrimonial |
| `ACCOUNT_OVERVIEW` | `get_account_summary` | Resumo da conta na corretora |
| `SEARCH_SYMBOL` | `search_symbol`, `yahoo_search` | Busca de ticker por nome |
| `PIPELINE_STATUS` | `list_jobs`, `get_job_status` | Status de jobs do pipeline de dados |

---

## Arquitetura Interna

```
ExecutionIntent (capability + parameters)
  │
  ├─ FinanceDomainHandler.execute()
  │    ├─ _resolve_parameters()        # aplica defaults do manifest
  │    ├─ get_type_hints() → typed dispatch (ex: get_top_gainers)
  │    └─ _run_pipeline()
  │         ├─ _apply_pre_flow()       # resolve_symbol / resolve_symbol_list
  │         │    └─ SymbolResolver
  │         │         ├─ aliases dict  (metadata-driven, ex: NORDEA → NDA-SE.ST)
  │         │         └─ search_symbol (MCP fallback)
  │         ├─ ContextResolver         # symbol → DomainContext (mercado, moeda, exchange)
  │         ├─ SkillGateway            # → MCP finance server
  │         └─ StrategyCore           # skill data → Decision → DomainOutput
  │
  └─ DomainOutput
```

### Pre-flow: Symbol Resolution

O handler resolve nomes human-friendly → tickers técnicos **antes** de chamar o skill:

```
parameters.symbol_text = "Nordea"
  → SymbolResolver.resolve("NORDEA")
  → check aliases dict → "NDA-SE.ST" ✓  (alias determinístico)
  → (fallback) search_symbol MCP call
  → (clarification) se ambíguo: retorna candidatos
```

### Context Resolution

Após resolver o símbolo, o `ContextResolver` determina o contexto de mercado:

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

## Variáveis de Ambiente

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `MCP_URL` | `http://localhost:8000/sse` | URL do MCP finance server |
| `MCP_ADAPTER_CALL_TIMEOUT_SECONDS` | `90` | Timeout de chamadas MCP |
| `FINANCE_DOMAIN_PORT` | `8001` | Porta do serviço |
| `SYMBOL_ALIASES_JSON` | — | Override de aliases em JSON |

---

## Flows Suportados (pre-flow metadata)

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

Quando `flow.pre` não está declarado, o handler infere automaticamente:
- campo `symbol` no schema → `resolve_symbol`
- campo `symbols` no schema → `resolve_symbol_list`

---

## Adicionando uma Nova Capability

1. Declarar em `DOMAIN_MANIFEST["capabilities"]` em `server.py`
2. Definir `entities_schema` no goal correspondente (ou criar novo goal)
3. Implementar método tipado em `handler.py`:
   ```python
   async def nova_capability(self, intent: IntentOutput | ExecutionIntent, params: NovaInput) -> DomainOutput:
       return await self._run_pipeline(intent, params.model_dump())
   ```
4. Criar schema Pydantic em `schemas.py`
5. GoalResolver e TaskDecomposer funcionam automaticamente via metadata
