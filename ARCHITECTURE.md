# Agent Orchestrator Layer - Architecture Instructions

## Overview

This document defines the architectural principles and guidelines for the Agent Orchestrator Layer. **All development MUST follow these rules** to maintain a clean, generic, and extensible system.

**Core Principle**: Keep orchestrator layers **generic and metadata-driven**. Domain-specific logic belongs in **Domain Manifests**, NOT in the orchestrator core.

---

## Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER ENTRY POINTS                        │
│              (CLI, Telegram, Open WebUI, etc)                │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              APPLICATION LAYER (Generic)                    │
├─────────────────────────────────────────────────────────────┤
│  - Intent Adapter      → Extract user intent (generic)      │
│  - Planner Service     → Plan composition (generic)          │
│  - Task Decomposer     → Decompose to steps (generic)        │
│  - Execution Engine    → Run steps (generic)                 │
│  - Orchestrator        → Route to domains (generic)          │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│            DOMAIN LAYER (Domain-Specific)                   │
├─────────────────────────────────────────────────────────────┤
│  - Finance Domain      → Manifest + Business Logic          │
│  - Communication       → Manifest + Business Logic          │
│  - Any New Domain      → Manifest + Business Logic          │
└─────────────────────────────────────────────────────────────┘
```

---

## Layer Responsibilities

### 1. Intent Adapter Layer (`intent/adapter.py`)

**Purpose**: Extract structured intent from user input using LLM.

**Rules**:
- ❌ **DO NOT** include domain-specific logic (e.g., multi-symbol extraction)
- ❌ **DO NOT** hardcode parameter transformations
- ✅ **DO** use manifest as reference for valid actions/domains
- ✅ **DO** return generic `IntentOutput` structure
- ✅ **DO** keep system prompt focused on structure, not domain rules

**What it OWNS**:
- Converting free text → `IntentOutput` (domain, capability, parameters, confidence)
- Validating against manifest

**What it DELEGATES TO**:
- Domain manifests (for action names, parameter specs)
- TaskDecomposer (for multi-step planning)

---

### 2. Task Decomposer Layer (`planner/task_decomposer.py`)

**Purpose**: Convert single intent into multi-step execution plan.

**Rules**:
- ❌ **DO NOT** hardcode capability-specific logic (e.g., "if get_stock_price, do X")
- ❌ **DO NOT** make assumptions about parameter names
- ✅ **DO** read decomposition rules from capability metadata
- ✅ **DO** apply rules generically for ALL domains/capabilities
- ✅ **DO** support both single-step and multi-step plans

**What it OWNS**:
- Reading `decomposition` metadata from manifests
- Creating parallel/sequential step plans
- Dependency management between steps

**What it DELEGATES TO**:
- Domain manifests (for decomposition rules)
- Execution Engine (for step execution)

### 2.1 Decomposition Metadata Contract

Every capability CAN define decomposition rules in its manifest:

```json
{
  "capability": "get_stock_price",
  "metadata": {
    "decomposition": {
      "array_params": [
        {
          "param_name": "symbols",           // Which parameter is an array
          "single_param_name": "symbol",     // What to call it per step
          "max_concurrency": 4,              // Parallel execution limit
          "create_parallel_steps": true      // Optional: always true if present
        }
      ]
    }
  }
}
```

**Algorithm** (in TaskDecomposer):
```python
for rule in capability_metadata.decomposition.array_params:
    if intent.parameters[rule.param_name] is list AND len > 1:
        for item in array:
            create_step(capability, {rule.single_param_name: item})
        return parallel_plan()
```

**Examples**:
- `get_stock_price`: decompose `symbols` → `symbol` (get price of each)
- `get_option_chain`: decompose `symbols` → `symbol` (get options for each)
- `build_wheel_plan`: decompose `symbols` → `symbol` (build plan per stock)
- Future capability: decompose `report_ids` → `report_id` (fetch each report)

---

### 3. Execution Engine Layer (`execution/engine.py`)

**Purpose**: Execute step plans deterministically.

**Rules**:
- ❌ **DO NOT** add business logic
- ❌ **DO NOT** make domain-specific decisions
- ✅ **DO** execute steps in DAG order (parallel where possible)
- ✅ **DO** collect and combine results
- ✅ **DO** handle errors consistently

**What it OWNS**:
- Dependency resolution (DAG execution)
- Parallel/sequential execution
- Result combination
- Error handling

**What it DELEGATES TO**:
- Orchestrator (for step capability routing)
- Domain handlers (for step execution)

---

### 4. Orchestrator (`orchestrator/orchestrator.py`)

**Purpose**: Route capabilities to domains and apply confidence gates.

**Rules**:
- ❌ **DO NOT** hardcode domain routing
- ❌ **DO NOT** make domain-specific decisions
- ✅ **DO** use registry to find domains/capabilities
- ✅ **DO** apply confidence thresholds consistently
- ✅ **DO** delegate execution to domain handlers

**What it OWNS**:
- Capability → Domain mapping
- Confidence gates
- Handler routing

**What it DELEGATES TO**:
- Registry (for capability lookup)
- Domain handlers (for execution)

---

### 5. Domain Layer (Domain-Specific)

**Purpose**: Implement business logic for a domain.

**Examples**: Finance, Communication, Reporting, etc.

**What it OWNS**:
- Business logic implementation
- Manifest definition (capabilities, metadata, rules)
- Domain-specific handlers

**What it MUST PROVIDE**:
1. `/manifest` endpoint returning:
   ```json
   {
     "capabilities": [
       {
         "name": "get_stock_price",
         "description": "...",
         "schema": {...},
         "metadata": {
           "intent_hints": {...},
           "parameter_specs": {...},
           "decomposition": {...},
           "composition": {...},
           "flow": {...}
         }
       }
     ]
   }
   ```

2. `/execute` endpoint implementing capabilities

3. Decomposition rules for multi-item parameters (in metadata)

---

## Metadata Contract

### Capability Metadata Structure

```json
{
  "capability": "get_stock_price",
  "metadata": {
    // Intent Extraction Hints
    "intent_hints": {
      "keywords": ["preço", "valor", "cotação"],
      "examples": ["qual o preço da PETR4?"]
    },

    // Parameter Specifications (for validation & hints)
    "parameter_specs": {
      "symbol": {
        "type": "string",
        "required": true,
        "examples": ["PETR4.SA", "VALE3.SA"],
        "aliases": {"PETRO": "PETR4.SA"}
      },
      "symbols": {
        "type": "array",
        "items": {"type": "string"},
        "examples": [["PETR4.SA", "VALE3.SA"]]
      }
    },

    // Decomposition Rules (TaskDecomposer reads this)
    "decomposition": {
      "array_params": [
        {
          "param_name": "symbols",
          "single_param_name": "symbol",
          "max_concurrency": 4
        }
      ]
    },

    // Composition Rules (for multi-step workflows)
    "composition": {
      "followup_roles": ["notifier"],
      "enabled_if": {"path": "parameters.notify", "equals": true},
      "followup_required": false,
      "param_map": {...}
    },

    // Pre-execution Flow (deterministic schema-driven flows)
    "flow": {
      "pre": [
        {
          "type": "resolve_symbol",
          "param": "symbol",
          "search_capability": "search_symbol"
        }
      ]
    }
  }
}
```

---

## Implementation Checklist

### When Adding a New Capability

- [ ] Define in domain manifest `/capabilities` endpoint
- [ ] Add `intent_hints` (keywords, examples for LLM)
- [ ] Add `parameter_specs` (types, required fields, examples, aliases)
- [ ] Add `decomposition` rules IF it handles array parameters
- [ ] Add `composition` rules IF it can chain to other capabilities
- [ ] Add `flow` rules IF it needs pre/post-execution deterministic steps
- [ ] Implement `/execute` handler in domain
- [ ] NO domain-specific code in `adapter.py`, `decomposer.py`, `engine.py`, `orchestrator.py`

### When Adding a New Domain

1. Create `/domains/your_domain/` directory
2. Implement:
   - `/health` endpoint
   - `/manifest` endpoint (with all capabilities & metadata)
   - `/execute` endpoint (for capability execution)
3. Register domain in `domains.bootstrap.json` or via admin commands
4. NO changes needed to orchestrator layers

---

## Anti-Patterns (DO NOT DO THIS)

### ❌ Anti-Pattern 1: Domain Logic in Adapter

```python
# WRONG: adapter.py
def extract(self, text):
    if "petro" in text and "vale" in text:  # ❌ Domain-specific
        intent.parameters["symbols"] = ["PETR4", "VALE3"]
```

**Solution**: Let LLM extract from manifest, TaskDecomposer handles array decomposition.

---

### ❌ Anti-Pattern 2: Hardcoded Capability Logic in Decomposer

```python
# WRONG: decomposer.py
def decompose(self, intent):
    if intent.capability == "get_stock_price":  # ❌ Hardcoded
        # multi-symbol logic here
```

**Solution**: Read `decomposition` metadata, apply generically.

---

### ❌ Anti-Pattern 3: Domain Rules in Orchestrator

```python
# WRONG: orchestrator.py
def route(self, intent):
    if intent.domain == "finance":  # ❌ Domain-specific
        apply_finance_rules()
```

**Solution**: Apply rules via metadata, orchestrator stays generic.

---

## Token Optimization Guidelines

Following this architecture reduces tokens because:

1. **LLM prompts are stable** - Manifest-driven, not hardcoded domain logic
2. **Reusable patterns** - Same decomposition logic works for all domains
3. **Less debugging** - Generic layers are tested once, not per-domain
4. **Future AI developers don't need context** - Architecture rules are explicit

### For New AI Developers

When you inherit this codebase:
1. Read this `ARCHITECTURE.md` first
2. Check the domain manifest structure
3. Implement ONLY in the domain layer
4. Never touch `intent/`, `planner/`, `execution/`, `orchestrator/` unless fixing bugs

---

## Example: Adding Multi-Currency Stock Price Support

### ✅ RIGHT WAY (Metadata-Driven)

1. Update Finance manifest `get_stock_price` capability:
   ```json
   {
     "parameter_specs": {
       "currencies": {
         "type": "array",
         "items": {"type": "string"},
         "examples": [["USD", "BRL"]]
       }
     },
     "decomposition": {
       "array_params": [
         {
           "param_name": "currencies",
           "single_param_name": "currency",
           "max_concurrency": 4
         }
       ]
     }
   }
   ```

2. Update handler to support `currency` parameter
3. Done! TaskDecomposer automatically handles parallel execution

### ❌ WRONG WAY (Hardcoded)

```python
# In decomposer.py or adapter.py
if "currencies" in intent.parameters:
    for currency in intent.parameters["currencies"]:
        # ❌ Don't do this
```

---

## Validation Questions for Code Reviews

When reviewing changes:

1. **Is domain-specific logic in domain layer only?** ✓ Yes
2. **Does TaskDecomposer use generic metadata rules?** ✓ Yes
3. **Can this logic apply to other domains?** ✓ Yes
4. **Did we read the manifest before hardcoding?** ✓ Yes
5. **Will future AI developers understand this without context?** ✓ Yes

---

## References

- **Metadata Contracts**: `shared/workflow_contracts.py`
- **Intent Extraction**: `intent/adapter.py`
- **Decomposition**: `planner/task_decomposer.py`
- **Execution**: `execution/engine.py`
- **Orchestration**: `orchestrator/orchestrator.py`
- **Example Domain**: `domains/finance/`

---

**Last Updated**: 2026-02-15
**Architecture Version**: 1.0 (Generic Metadata-Driven)
