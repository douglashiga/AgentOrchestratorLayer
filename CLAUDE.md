# CLAUDE.md

## Project Overview

**Agent Orchestrator Layer** — a multi-domain orchestrator for LLM agents that separates the LLM layer (intent extraction, planning) from a deterministic core (execution engine, domain logic) and remote integrations (HTTP-based domain services, MCP skills).

- **Language**: Python 3.11+
- **Architecture**: Domain-Driven Design (DDD) layered architecture
- **Key frameworks**: Pydantic, httpx, FastAPI, Rich
- **Database**: SQLite (embedded, no external DB required)

## Repository Structure

```
AgentOrchestratorLayer/
├── main.py                    # CLI entrypoint (run, run-telegram, admin commands)
├── api/openai_server.py       # OpenAI-compatible API server (FastAPI)
├── entry/                     # Channel adapters (CLI, Telegram)
├── intent/adapter.py          # LLM-powered intent extraction
├── planner/                   # Plan generation (TaskDecomposer, FunctionCallingPlanner)
├── execution/                 # DAG execution engine, result combiner, task state
├── orchestrator/              # Stateless domain router (confidence gating)
├── registry/                  # Domain/capability registry (SQLite-backed)
├── memory/store.py            # Structured memory store (SQLite, namespaced)
├── models/selector.py         # LLM abstraction (Ollama, OpenAI-compatible, Anthropic)
├── domains/
│   ├── finance/               # Finance domain (handler, context, core, server, schemas)
│   └── general/               # General chat fallback handler
├── skills/                    # Skill gateway + MCP adapter
├── shared/
│   ├── models.py              # Core contracts (IntentOutput, ExecutionStep, DomainOutput)
│   ├── workflow_contracts.py  # Generic workflow specs (MethodSpec, WorkflowNodeSpec)
│   ├── safe_eval.py           # Safe expression evaluator (no raw eval)
│   ├── response_formatter.py  # Output formatting
│   └── delivery_layer.py      # Streaming, suggestions, history persistence
├── observability/logger.py    # Structured logging
├── communication-domain/      # Standalone Telegram notification service
├── scripts/                   # Evaluation and probe scripts
├── test_*.py                  # 36+ pytest test files (root level)
├── Dockerfile                 # Multi-purpose container image
├── docker-compose.yml         # 5 services (finance, communication, agent, API, webui)
├── pyproject.toml             # Package config + pytest settings
├── requirements.txt           # Full dependency list
└── domains.bootstrap.json     # Remote domain bootstrap config
```

## Common Commands

### Running the application

```bash
python3 main.py run                # Interactive CLI mode
python3 main.py run-telegram       # Telegram bot entry
uvicorn api.openai_server:app --host 0.0.0.0 --port 8010  # OpenAI-compatible API
```

### Running tests

```bash
python3 -m pytest -q                              # Run all tests
python3 -m pytest test_specific_file.py -q         # Run specific test file
python3 -m pytest -k "test_name" -q                # Run tests matching pattern
PYTHONPATH=. python3 scripts/evaluate_capabilities.py  # Live capability evaluation
```

- pytest is configured in `pyproject.toml` with `norecursedirs = ["scripts", ".venv", "__pycache__"]`
- Test files are at the repository root (`test_*.py`)
- Live evaluation tests require `RUN_LIVE_CAPABILITY_TESTS=1`

### Docker

```bash
docker compose up -d                  # Start all services
docker compose up finance-server -d   # Finance domain on port 8003
docker compose up agent-api -d        # OpenAI API on port 8010
```

### Admin commands

```bash
python3 main.py domain-list
python3 main.py domain-add <name> <type> <config_json>
python3 main.py domain-sync <domain>
python3 main.py memory-set <key> <value> [--namespace <ns>]
python3 main.py memory-get <key>
python3 main.py memory-search <query>
```

## Architecture & Execution Flow

```
User Input → Entry Layer (CLI/Telegram/API)
  → ConversationManager (load session history)
  → IntentAdapter (LLM → IntentOutput)
  → Orchestrator (confidence gating, resolve handler)
  → PlannerService (memory injection, task decomposition)
  → ExecutionEngine (DAG/parallel batch execution)
  → DomainHandler (deterministic business logic)
  → ResponseFormatter → DeliveryLayer → User
```

### Key architectural rules

1. **Orchestrator is stateless** — no business logic, no skills, no state, no MCP. It only routes requests to domain handlers with confidence gating.
2. **ModelSelector is the only LLM callsite** — all LLM interactions go through `models/selector.py`. Never call LLM APIs directly from other modules.
3. **Business logic is deterministic** — domain handlers, context resolution, and strategy logic operate outside LLM context. The LLM is used only for intent extraction and planning.
4. **Pydantic contracts throughout** — use frozen Pydantic models for all data contracts (`shared/models.py`, `shared/workflow_contracts.py`).
5. **Registry-based discovery** — domains and capabilities are registered in `RegistryDB` (SQLite) and synced from remote HTTP manifests at startup.

## Code Conventions

### Data contracts

- Core types are in `shared/models.py`: `IntentOutput`, `ExecutionStep`, `ExecutionPlan`, `DomainOutput`
- Workflow types are in `shared/workflow_contracts.py`: `MethodSpec`, `WorkflowSpec`, `WorkflowNodeSpec`, `TaskInstance`
- All models use Pydantic v2 with `model_config = ConfigDict(frozen=True)` where applicable

### Domain handlers

- Each domain lives under `domains/<name>/`
- Handlers implement a standard interface: receive intent/params, return `DomainOutput`
- Remote domains expose: `GET /health`, `GET /manifest`, `GET /openapi.json`, `POST /execute`
- Capability metadata includes `intent_hints`, `parameter_specs`, `flow` (pre-steps), and `composition` (multi-step orchestration)

### Testing patterns

- Tests use pytest + pytest-asyncio
- Test files are at the root level, named `test_*.py`
- Mock LLM responses and HTTP calls; avoid live LLM calls in unit tests
- Integration scripts live under `scripts/`

### Error handling

- `DomainOutput.status` is one of: `success`, `failure`, `clarification`
- Confidence gating blocks low-confidence intents (threshold configurable via `ORCHESTRATOR_CONFIDENCE_THRESHOLD`)
- Execution engine supports retry/backoff/timeout policies via `MethodPolicy`

## Environment Configuration

Key environment variables (see README.md for full reference):

| Variable | Purpose | Default |
|---|---|---|
| `OLLAMA_URL` | Ollama server URL | `http://localhost:11434` |
| `MODEL_PROVIDER` | LLM provider (`ollama`, `openai_compatible`, `anthropic`, `auto`) | `auto` |
| `INTENT_MODEL_NAME` | Model for intent extraction | `llama3.1:8b` |
| `DB_PATH` | Conversation SQLite path | `agent.db` |
| `REGISTRY_DB_PATH` | Registry SQLite path | `registry.db` |
| `MEMORY_DB_PATH` | Memory SQLite path | `memory.db` |
| `ORCHESTRATOR_CONFIDENCE_THRESHOLD` | Minimum confidence to proceed | `0.94` |
| `AUTO_SYNC_REMOTE_CAPABILITIES` | Sync remote manifests at startup | `true` |
| `PLANNER_FUNCTION_CALLING_ENABLED` | Enable function-calling planner loop | `true` |
| `BOOTSTRAP_DOMAINS_FILE` | Path to domain bootstrap JSON | `domains.bootstrap.json` |

Environment is loaded from `.env` file via python-dotenv (priority: `.env` > system env vars).

## Important Notes for AI Assistants

- **Read before modifying**: Always read a file before proposing changes. Understand existing patterns.
- **Respect the layering**: Don't mix concerns across layers (e.g., no LLM calls in domain handlers, no business logic in the orchestrator).
- **Test coverage**: Run `python3 -m pytest -q` after changes to ensure nothing breaks.
- **No linting/formatting tools configured**: The project does not have black, flake8, or mypy configured. Follow existing code style (type hints, standard Python conventions).
- **No CI/CD pipeline**: There is no automated CI. Always run tests locally before finalizing changes.
- **Large main.py**: `main.py` (~65KB) is the CLI entrypoint and contains significant initialization logic. Be aware of its size when making changes.
