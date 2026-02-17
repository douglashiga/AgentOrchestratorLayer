# Agent Orchestrator Layer — Architecture

## Overview

Sistema multi-domínio orientado a goals (objetivos de negócio). A camada de orquestração é **genérica e metadata-driven**: toda lógica de domínio vive nos **Domain Manifests**, nunca no core do orquestrador.

**Fluxo principal:**

```
User Input
  → Entry Layer         (normaliza input)
  → Intent Adapter      (LLM → IntentOutput  com goal + entities)
  → Goal Resolver       (determinístico: goal → capability)
  → Planner Service     (injeta memória, decompõe em plano)
  → Execution Engine    (executa DAG de steps)
  → Orchestrator        (roteia capability → domain handler)
  → Domain Handler      (executa, resolve símbolos, retorna DomainOutput)
```

---

## Camadas e Payloads

### 1. Entry Layer

**Arquivo:** `main.py`, `api/openai_server.py`

**Responsabilidade:** Normalizar qualquer canal (CLI, Telegram, HTTP) em `EntryRequest`.

**Payload de saída:**

```python
EntryRequest(
    session_id="user-abc123",
    input_text="qual o preço da Nordea?",
    metadata={}
)
```

---

### 2. Intent Layer — `intent/adapter.py`

**Responsabilidade:** LLM extrai **goal + entities human-friendly** do texto do usuário. Nunca infere tickers, IDs técnicos, ou job IDs — isso é responsabilidade do domínio.

**O que o LLM vê:** catálogo de domínios/goals + `entities_schema` por goal (quais entidades extrair e em qual formato).

**Payload de saída — `IntentOutput`:**

```python
IntentOutput(
    primary_domain="finance",
    goal="GET_QUOTE",
    entities={
        "symbol_text": "Nordea"      # nome como o usuário disse — nunca "NDA-SE.ST"
    },
    confidence=0.95,
    original_query="qual o preço da Nordea?"
)
```

**Exemplo com enum (TOP_MOVERS):**

```python
IntentOutput(
    primary_domain="finance",
    goal="TOP_MOVERS",
    entities={
        "direction": "GAINERS",      # valor enum do entities_schema
        "market_text": "Brasil"      # contexto human-friendly
    },
    confidence=0.92,
    original_query="maiores altas do Brasil"
)
```

**Regras:**
- Chaves `*_text` → strings exatamente como o usuário disse (nome, mercado, período)
- Chaves enum → valor exato do `entities_schema.values[]`
- ❌ Nunca: `ticker="NDA-SE.ST"`, `job_id="abc"`, `symbol="PETR4.SA"`

---

### 3. Goal Resolver — `planner/goal_resolver.py`

**Responsabilidade:** Mapeamento **determinístico** `IntentOutput → ExecutionIntent`.
Usa `entities_schema.capability_map` do manifest para resolver qual capability executar.

**Casos:**

| Goal | Entities | Capability resolvida |
|------|----------|----------------------|
| `GET_QUOTE` | qualquer | `get_stock_price` (único) |
| `TOP_MOVERS` | `direction=GAINERS` | `get_top_gainers` |
| `TOP_MOVERS` | `direction=LOSERS` | `get_top_losers` |
| `TOP_MOVERS` | sem direction | `get_top_gainers` (fallback = first) |

**Payload de saída — `ExecutionIntent`:**

```python
ExecutionIntent(
    domain="finance",
    capability="get_stock_price",      # capability resolvida deterministicamente
    parameters={
        "symbol_text": "Nordea"        # entities passadas como parameters
    },
    confidence=0.95,
    original_query="qual o preço da Nordea?"
)
```

---

### 4. Planner Service — `planner/service.py`

**Responsabilidade:** Injetar memória, decompor o `ExecutionIntent` em `ExecutionPlan`.
Pode usar `TaskDecomposer` (determinístico) e opcionalmente `FunctionCallingPlanner` (LLM).

**Payload de saída — `ExecutionPlan`:**

```python
ExecutionPlan(
    execution_mode="sequential",   # "sequential" | "parallel" | "dag"
    combine_mode="last",           # "last" | "report" | "merge"
    max_concurrency=4,
    steps=[
        ExecutionStep(
            id=1,
            domain="finance",
            capability="get_stock_price",
            params={"symbol_text": "Nordea"},
            depends_on=[],
            required=True,
            output_key="primary"
        )
    ]
)
```

**Exemplo multi-step (compose/notify):**

```python
ExecutionPlan(
    execution_mode="dag",
    combine_mode="report",
    steps=[
        ExecutionStep(id=1, capability="get_stock_price",      params={"symbol": "PETR4.SA"}, depends_on=[]),
        ExecutionStep(id=2, capability="send_telegram_message", params={"message": "${1.explanation}"}, depends_on=[1], required=False)
    ]
)
```

---

### 5. Execution Engine — `execution/engine.py`

**Responsabilidade:** Executar o `ExecutionPlan` em DAG. Para cada step, cria um `ExecutionIntent` local e chama o `Orchestrator`.

**Step intent criado internamente:**

```python
ExecutionIntent(
    domain="finance",
    capability="get_stock_price",
    confidence=0.95,
    parameters={"symbol_text": "Nordea"},
    original_query="qual o preço da Nordea?"
)
```

**Payload de saída — `DomainOutput` (combinado):**

```python
DomainOutput(
    status="success",              # "success" | "failure" | "clarification"
    result={
        "price": 112.5,
        "currency": "SEK",
        "_market_context": {
            "market": "SE",
            "exchange": "OMX",
            "timezone": "Europe/Stockholm"
        }
    },
    explanation="Nordea (NDA-SE.ST) está em 112.50 SEK",
    confidence=1.0,
    metadata={}
)
```

---

### 6. Orchestrator — `orchestrator/orchestrator.py`

**Responsabilidade:** Rotear `ExecutionIntent` → handler correto via registry. Aplicar confidence gate.

**Confidence gate:**
- Se `confidence < threshold (0.94)` e domínio não é `general` → retorna `clarification`
- Se confidence ok → chama `handler.execute(intent)`

**Resolve por:**
1. `capability` → handler direto (preferencial)
2. `domain` → handler do domínio (fallback)

---

### 7. Domain Handler — `domains/finance/handler.py`

**Responsabilidade:** Recebe `ExecutionIntent`, resolve entidades human-friendly → técnicas (symbol), executa skill, retorna `DomainOutput`.

**Pre-flow (determinístico):**

```
ExecutionIntent.parameters.symbol_text = "Nordea"
  → SymbolResolver.resolve("NORDEA")
  → "NDA-SE.ST"                          # alias ou search_symbol
  → ExecutionIntent.parameters.symbol = "NDA-SE.ST"
```

**Payload de saída — `DomainOutput`:**

```python
DomainOutput(
    status="success",
    result={
        "symbol": "NDA-SE.ST",
        "price": 112.5,
        "currency": "SEK",
        "_market_context": {"market": "SE", "exchange": "OMX", ...}
    },
    explanation="Nordea (NDA-SE.ST) está em 112.50 SEK",
    confidence=1.0,
    metadata={"risk_metrics": {}}
)
```

**Clarification (símbolo ambíguo):**

```python
DomainOutput(
    status="clarification",
    result={"candidates": [{"symbol": "NDA-SE.ST"}, {"symbol": "NDA-FI.HE"}]},
    explanation="Encontrei mais de um ticker para 'Nordea'. Qual deles?",
    confidence=1.0,
    metadata={"resolution": "ambiguous_symbol"}
)
```

---

## Diagrama de Fluxo Completo

```
┌──────────────────────────────────────────────────────────────────────┐
│  USER INPUT: "qual o preço da Nordea?"                               │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
            ┌──────────────▼──────────────┐
            │      INTENT ADAPTER (LLM)   │
            │  Vê: goal catalog +         │
            │       entities_schema       │
            └──────────────┬──────────────┘
                           │ IntentOutput
                           │ { primary_domain: "finance"
                           │   goal: "GET_QUOTE"
                           │   entities: { symbol_text: "Nordea" }
                           │   confidence: 0.95 }
            ┌──────────────▼──────────────┐
            │      GOAL RESOLVER          │
            │  (determinístico)           │
            │  GET_QUOTE → get_stock_price│
            └──────────────┬──────────────┘
                           │ ExecutionIntent
                           │ { domain: "finance"
                           │   capability: "get_stock_price"
                           │   parameters: { symbol_text: "Nordea" }
                           │   confidence: 0.95 }
            ┌──────────────▼──────────────┐
            │      PLANNER SERVICE        │
            │  + memory injection         │
            │  + TaskDecomposer           │
            └──────────────┬──────────────┘
                           │ ExecutionPlan
                           │ { steps: [ { id:1, capability: "get_stock_price"
                           │              params: { symbol_text: "Nordea" } } ] }
            ┌──────────────▼──────────────┐
            │      EXECUTION ENGINE       │
            │  (DAG executor)             │
            └──────────────┬──────────────┘
                           │ ExecutionIntent (per step)
            ┌──────────────▼──────────────┐
            │      ORCHESTRATOR           │
            │  capability → handler lookup│
            └──────────────┬──────────────┘
                           │
            ┌──────────────▼──────────────┐
            │    FINANCE DOMAIN HANDLER   │
            │  symbol_text → "NDA-SE.ST"  │
            │  → SkillGateway → MCP       │
            │  → StrategyCore             │
            └──────────────┬──────────────┘
                           │ DomainOutput
                           │ { status: "success"
                           │   result: { price: 112.5, currency: "SEK" }
                           │   explanation: "Nordea está em 112.50 SEK" }
└──────────────────────────────────────────────────────────────────────┘
```

---

## Contracts por Camada

### Goal Manifest (entities_schema)

Cada goal no manifest do domínio declara quais entidades o LLM deve extrair:

```python
# Exemplo: TOP_MOVERS em domains/finance/server.py
"entities_schema": {
    "direction": {
        "type": "enum",
        "values": ["GAINERS", "LOSERS", "BOTH"],
        "required": True,
        "default": "BOTH",
        "capability_map": {
            "GAINERS": "get_top_gainers",
            "LOSERS":  "get_top_losers",
            "BOTH":    ["get_top_gainers", "get_top_losers"],
        }
    },
    "market_text": {
        "type": "string",
        "description": "Mercado como o usuário mencionou"
    }
}
```

```python
# Exemplo: GET_QUOTE
"entities_schema": {
    "symbol_text": {
        "type": "string",
        "required": True,
        "description": "Nome/ticker como o usuário disse"
    }
}
```

### Capability Metadata Contract (manifest)

```json
{
  "capability": "get_stock_price",
  "metadata": {
    "intent_hints": {
      "keywords": ["preço", "valor", "cotação"],
      "examples": ["qual o preço da PETR4?"]
    },
    "decomposition": {
      "array_params": [
        {
          "param_name": "symbols",
          "single_param_name": "symbol",
          "max_concurrency": 4
        }
      ]
    },
    "composition": {
      "followup_roles": ["notifier"],
      "enabled_if": {"path": "parameters.notify", "equals": true},
      "param_map": {
        "message": {"from_parameters": ["_step_explanation"]}
      }
    },
    "flow": {
      "pre": [
        {"type": "resolve_symbol", "param": "symbol", "search_capability": "search_symbol"}
      ]
    }
  }
}
```

---

## Camadas — Responsabilidades e Limites

| Camada | Extrai/Decide | Nunca faz |
|--------|---------------|-----------|
| **Intent Adapter** | goal, entities human-friendly | inferir ticker, resolver símbolo |
| **Goal Resolver** | capability via capability_map | chamar LLM, acessar domínio |
| **Planner** | ExecutionPlan, injetar memória | lógica de domínio |
| **Execution Engine** | executa DAG, resolve `${ref}` | regra de negócio |
| **Orchestrator** | roteia capability → handler | lógica de domínio |
| **Domain Handler** | resolve entidades, executa skill | chamar outro domínio diretamente |

---

## Anti-Patterns

### ❌ LLM inferindo ticker técnico

```python
# ERRADO — adapter.py
entities = {"ticker": "NDA-SE.ST"}  # LLM não deve inferir isso

# CORRETO
entities = {"symbol_text": "Nordea"}  # human-friendly, domínio resolve
```

### ❌ Lógica de domínio no decomposer

```python
# ERRADO
if intent.capability == "get_stock_price":  # hardcoded
    ...

# CORRETO — ler metadata.decomposition
for rule in capability_metadata["decomposition"]["array_params"]:
    ...
```

### ❌ Routing hardcoded no orchestrator

```python
# ERRADO
if intent.domain == "finance":
    apply_finance_rules()

# CORRETO — registry resolve via metadata
handler = self.domain_registry.resolve_capability(intent.capability)
```

---

## Adicionando um Novo Domínio

1. Criar `domains/new_domain/` com endpoints `/health`, `/manifest`, `/execute`
2. Manifest declara: `capabilities[]` + `goals[]` com `entities_schema`
3. Registrar em `domains.bootstrap.json`
4. **Zero mudanças** em `intent/`, `planner/`, `execution/`, `orchestrator/`

## Adicionando uma Nova Capability

1. Declarar no manifest do domínio
2. Definir `entities_schema` se o goal mapeia múltiplas capabilities
3. Implementar handler no domínio
4. TaskDecomposer/GoalResolver resolvem automaticamente via metadata

---

**Arquivos de referência:**
- Modelos: `shared/models.py`
- Intent: `intent/adapter.py`
- Resolver: `planner/goal_resolver.py`
- Planner: `planner/service.py`, `planner/task_decomposer.py`
- Engine: `execution/engine.py`
- Orquestrador: `orchestrator/orchestrator.py`
- Finance manifest: `domains/finance/server.py`
- Workflow contracts: `shared/workflow_contracts.py`

**Última atualização:** 2026-02-17
**Versão:** 2.0 (Goal-based Intent + GoalResolver)
