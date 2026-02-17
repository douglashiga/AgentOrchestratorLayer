# Agent Orchestrator Layer — Architecture

## Overview

Multi-domain system oriented around goals (business objectives). The orchestration layer is **generic and metadata-driven**: all domain logic lives in the **Domain Manifests**, never in the orchestrator core.

**Main flow:**

```
User Input
  → Entry Layer         (normalizes input)
  → Intent Adapter      (LLM → IntentOutput  with goal + entities)
  → Goal Resolver       (deterministic: goal → capability)
  → Planner Service     (injects memory, decomposes into plan)
  → Execution Engine    (executes DAG of steps)
  → Orchestrator        (routes capability → domain handler)
  → Domain Handler      (executes, resolves symbols, returns DomainOutput)
```

---

## Layers and Payloads

### 1. Entry Layer

**File:** `main.py`, `api/openai_server.py`

**Responsibility:** Normalize any channel (CLI, Telegram, HTTP) into an `EntryRequest`.

**Output payload:**

```python
EntryRequest(
    session_id="user-abc123",
    input_text="qual o preço da Nordea?",
    metadata={}
)
```

---

### 2. Intent Layer — `intent/adapter.py`

**Responsibility:** The LLM extracts **goal + human-friendly entities** from the user's text. It never infers tickers, technical IDs, or job IDs — that is the domain's responsibility.

**What the LLM sees:** domain/goal catalog + `entities_schema` per goal (which entities to extract and in what format).

**Output payload — `IntentOutput`:**

```python
IntentOutput(
    primary_domain="finance",
    goal="GET_QUOTE",
    entities={
        "symbol_text": "Nordea"      # name as the user said it — never "NDA-SE.ST"
    },
    confidence=0.95,
    original_query="qual o preço da Nordea?"
)
```

**Example with enum (TOP_MOVERS):**

```python
IntentOutput(
    primary_domain="finance",
    goal="TOP_MOVERS",
    entities={
        "direction": "GAINERS",      # enum value from entities_schema
        "market_text": "Brasil"      # human-friendly context
    },
    confidence=0.92,
    original_query="maiores altas do Brasil"
)
```

**Rules:**
- `*_text` keys → strings exactly as the user said (name, market, period)
- enum keys → exact value from `entities_schema.values[]`
- ❌ Never: `ticker="NDA-SE.ST"`, `job_id="abc"`, `symbol="PETR4.SA"`

---

### 3. Goal Resolver — `planner/goal_resolver.py`

**Responsibility:** **Deterministic** mapping of `IntentOutput → ExecutionIntent`.
Uses `entities_schema.capability_map` from the manifest to resolve which capability to execute.

**Cases:**

| Goal | Entities | Resolved capability |
|------|----------|----------------------|
| `GET_QUOTE` | any | `get_stock_price` (single) |
| `TOP_MOVERS` | `direction=GAINERS` | `get_top_gainers` |
| `TOP_MOVERS` | `direction=LOSERS` | `get_top_losers` |
| `TOP_MOVERS` | no direction | `get_top_gainers` (fallback = first) |

**Output payload — `ExecutionIntent`:**

```python
ExecutionIntent(
    domain="finance",
    capability="get_stock_price",      # capability resolved deterministically
    parameters={
        "symbol_text": "Nordea"        # entities passed as parameters
    },
    confidence=0.95,
    original_query="qual o preço da Nordea?"
)
```

---

### 4. Planner Service — `planner/service.py`

**Responsibility:** Inject memory, decompose the `ExecutionIntent` into an `ExecutionPlan`.
Can use `TaskDecomposer` (deterministic) and optionally `FunctionCallingPlanner` (LLM).

**Output payload — `ExecutionPlan`:**

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

**Multi-step example (compose/notify):**

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

**Responsibility:** Execute the `ExecutionPlan` as a DAG. For each step, creates a local `ExecutionIntent` and calls the `Orchestrator`.

**Step intent created internally:**

```python
ExecutionIntent(
    domain="finance",
    capability="get_stock_price",
    confidence=0.95,
    parameters={"symbol_text": "Nordea"},
    original_query="qual o preço da Nordea?"
)
```

**Output payload — `DomainOutput` (combined):**

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

**Responsibility:** Route `ExecutionIntent` → correct handler via registry. Apply confidence gate.

**Confidence gate:**
- If `confidence < threshold (0.94)` and domain is not `general` → returns `clarification`
- If confidence is OK → calls `handler.execute(intent)`

**Resolves by:**
1. `capability` → direct handler (preferred)
2. `domain` → domain handler (fallback)

---

### 7. Domain Handler — `domains/finance/handler.py`

**Responsibility:** Receives `ExecutionIntent`, resolves human-friendly entities → technical ones (symbol), executes skill, returns `DomainOutput`.

**Pre-flow (deterministic):**

```
ExecutionIntent.parameters.symbol_text = "Nordea"
  → SymbolResolver.resolve("NORDEA")
  → "NDA-SE.ST"                          # alias or search_symbol
  → ExecutionIntent.parameters.symbol = "NDA-SE.ST"
```

**Output payload — `DomainOutput`:**

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

**Clarification (ambiguous symbol):**

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

## Complete Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│  USER INPUT: "qual o preço da Nordea?"                               │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
            ┌──────────────▼──────────────┐
            │      INTENT ADAPTER (LLM)   │
            │  Sees: goal catalog +       │
            │        entities_schema      │
            └──────────────┬──────────────┘
                           │ IntentOutput
                           │ { primary_domain: "finance"
                           │   goal: "GET_QUOTE"
                           │   entities: { symbol_text: "Nordea" }
                           │   confidence: 0.95 }
            ┌──────────────▼──────────────┐
            │      GOAL RESOLVER          │
            │  (deterministic)            │
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

## Contracts per Layer

### Goal Manifest (entities_schema)

Each goal in the domain manifest declares which entities the LLM should extract:

```python
# Example: TOP_MOVERS in domains/finance/server.py
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
        "description": "Market as the user mentioned it"
    }
}
```

```python
# Example: GET_QUOTE
"entities_schema": {
    "symbol_text": {
        "type": "string",
        "required": True,
        "description": "Name/ticker as the user said it"
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

## Layers — Responsibilities and Boundaries

| Layer | Extracts/Decides | Never does |
|--------|---------------|-----------|
| **Intent Adapter** | goal, human-friendly entities | infer ticker, resolve symbol |
| **Goal Resolver** | capability via capability_map | call LLM, access domain |
| **Planner** | ExecutionPlan, inject memory | domain logic |
| **Execution Engine** | execute DAG, resolve `${ref}` | business rules |
| **Orchestrator** | route capability → handler | domain logic |
| **Domain Handler** | resolve entities, execute skill | call another domain directly |

---

## Anti-Patterns

### ❌ LLM inferring a technical ticker

```python
# WRONG — adapter.py
entities = {"ticker": "NDA-SE.ST"}  # LLM should not infer this

# CORRECT
entities = {"symbol_text": "Nordea"}  # human-friendly, domain resolves
```

### ❌ Domain logic in the decomposer

```python
# WRONG
if intent.capability == "get_stock_price":  # hardcoded
    ...

# CORRECT — read metadata.decomposition
for rule in capability_metadata["decomposition"]["array_params"]:
    ...
```

### ❌ Hardcoded routing in the orchestrator

```python
# WRONG
if intent.domain == "finance":
    apply_finance_rules()

# CORRECT — registry resolves via metadata
handler = self.domain_registry.resolve_capability(intent.capability)
```

---

## Adding a New Domain

1. Create `domains/new_domain/` with endpoints `/health`, `/manifest`, `/execute`
2. Manifest declares: `capabilities[]` + `goals[]` with `entities_schema`
3. Register in `domains.bootstrap.json`
4. **Zero changes** in `intent/`, `planner/`, `execution/`, `orchestrator/`

## Adding a New Capability

1. Declare it in the domain manifest
2. Define `entities_schema` if the goal maps to multiple capabilities
3. Implement the handler in the domain
4. TaskDecomposer/GoalResolver resolve automatically via metadata

---

**Reference files:**
- Models: `shared/models.py`
- Intent: `intent/adapter.py`
- Resolver: `planner/goal_resolver.py`
- Planner: `planner/service.py`, `planner/task_decomposer.py`
- Engine: `execution/engine.py`
- Orchestrator: `orchestrator/orchestrator.py`
- Finance manifest: `domains/finance/server.py`
- Workflow contracts: `shared/workflow_contracts.py`

**Last updated:** 2026-02-17
**Version:** 2.0 (Goal-based Intent + GoalResolver)
