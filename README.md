# Agent Orchestrator Layer

Multi-domain orchestrator with:
- LLM intent extraction
- Metadata-driven planning (TaskDecomposer)
- Optional function-calling planner loop (Semantic-Kernel inspired)
- Deterministic execution engine (DAG/parallel)
- Deterministic structured memory store (outside the LLM)
- Remote domain integration via HTTP (`/manifest`, `/execute`, `/openapi.json`)

## Architecture

```mermaid
graph TD
    User((User)) --> Entry[Entry Layer]
    Entry --> Intent[Intent Adapter]
    Intent --> Planner[Planner Service]
    Mem[(Memory Store)] --> Planner
    Planner --> Decomposer[TaskDecomposer\n(metadata-driven)]
    Planner --> FCPlanner[FunctionCallingPlanner\n(optional)]
    Decomposer --> Exec[Execution Engine\n(DAG/parallel)]
    FCPlanner --> Exec
    Exec --> Orch[Orchestrator]
    Orch --> Reg[Registry]
    Reg --> Fin[Finance Domain\n(remote_http)]
    Reg --> Com[Communication Domain\n(remote_http)]
```

## Key Features

- Dynamic domains from DB/bootstrap (no hardcoded finance/communication in orchestrator runtime).
- Capability discovery from remote manifests.
- Multi-step composition driven by capability metadata (`metadata.composition`).
- Function-calling planning loop with strict validation/fallback.
- Deterministic guardrail: notifier steps are allowed only with explicit `notify=true`.
- Capability pre-flow engine in finance service (metadata/schema-driven, no method hardcode).
- Soft confirmation flow for low-confidence intents.
- Telegram entry channel with per-step JSON debug trace.
- OpenAI-compatible API for Open WebUI.
- Streaming support (`stream=true`) with incremental status updates for better UX.

## Memory Store (Semantic-Kernel style)

Interface:

```python
class MemoryStore:
    def save(key, value)
    def get(key)
    def search(query)
```

Implementation:
- `memory/store.py` (`SQLiteMemoryStore`)

Planner integration:
- Memory is read before decomposition.
- Deterministic fields are injected into intent params when allowed by schema.
- Memory context is included in planner debug trace.

Structured memory example:

```json
{
  "preferred_market": "sweden",
  "risk_mode": "moderate",
  "wheel_active": true,
  "last_wheel_operation": "2026-02-14: sell_put VALE3.SA",
  "capital_available": 25000
}
```

## Capability Flow Engine (Service-side)

The finance service now applies deterministic pre-flows before skill execution, driven by capability metadata (manifest) and schema inference.

Supported pre-flow steps:
- `resolve_symbol`
- `resolve_symbol_list`

Example metadata (manifest):

```json
{
  "flow": {
    "pre": [
      { "type": "resolve_symbol", "param": "symbol", "search_capability": "search_symbol" }
    ]
  }
}
```

Fallback behavior when `flow.pre` is not declared:
- if schema has `symbol` -> auto `resolve_symbol`
- if schema has `symbols` -> auto `resolve_symbol_list`

This keeps flows dynamic and reusable across methods.

## Project Structure

```text
AgentOrchestratorLayer/
├── main.py
├── docker-compose.yml
├── domains.bootstrap.json
├── api/
│   └── openai_server.py
├── memory/
│   ├── __init__.py
│   └── store.py
├── planner/
│   ├── service.py
│   ├── task_decomposer.py
│   └── function_calling_planner.py
├── execution/
│   ├── engine.py
│   └── result_combiner.py
├── orchestrator/
│   └── orchestrator.py
├── registry/
│   ├── db.py
│   ├── loader.py
│   ├── domain_registry.py
│   └── http_handler.py
├── domains/
│   ├── finance/
│   └── general/
├── communication-domain/
│   └── app/
└── scripts/
```

## Domains

### Finance Domain (remote HTTP)
- Internal port: `8001`
- Host port in compose: `8003`
- Endpoints: `/health`, `/manifest`, `/execute`

### Communication Domain (remote HTTP)
- Port: `8002`
- Endpoints: `/health`, `/manifest`, `/execute`
- Capabilities:
  - `send_telegram_message`
  - `send_telegram_group_message`

## Configuration

### Core Agent env vars

- `OLLAMA_URL` (default `http://localhost:11434`)
- `MCP_URL` (default `http://localhost:8000/sse`)
- `DB_PATH` (default `agent.db`)
- `REGISTRY_DB_PATH` (default `registry.db`)
- `MEMORY_DB_PATH` (default `memory.db`)
- `MEMORY_BOOTSTRAP_JSON` (optional)
- `BOOTSTRAP_DOMAINS_JSON` (optional)
- `BOOTSTRAP_DOMAINS_FILE` (optional)
- `AUTO_SYNC_REMOTE_CAPABILITIES` (default `true`)
- `SEED_CORE_DEFAULTS` (default `true`)

### Confidence / Clarification

- `SOFT_CONFIRMATION_ENABLED` (default `true`)
- `SOFT_CONFIRM_THRESHOLD` (default `0.94`)
- `ORCHESTRATOR_CONFIDENCE_THRESHOLD` (default from `SOFT_CONFIRM_THRESHOLD`, fallback `0.94`)

### Function-Calling Planner

- `PLANNER_FUNCTION_CALLING_ENABLED` (default `true`)
- `PLANNER_FUNCTION_CHOICE_MODE` (`auto|required|none`, default `auto`)
- `PLANNER_MAX_ITERATIONS` (default `2`)
- `PLANNER_MODEL` (default `llama3.1:8b`)
- `PLANNER_TIMEOUT_SECONDS` (default `8`)
- `PLANNER_INCLUDED_DOMAINS` / `PLANNER_EXCLUDED_DOMAINS`
- `PLANNER_INCLUDED_CAPABILITIES` / `PLANNER_EXCLUDED_CAPABILITIES`

### Planner + Memory integration

- `PLANNER_MEMORY_ENABLED` (default `true`)
- `PLANNER_MEMORY_PARAM_MAP_JSON` (optional JSON map)
- `PLANNER_MEMORY_ALLOW_WITHOUT_SCHEMA` (default `false`)

Example `PLANNER_MEMORY_PARAM_MAP_JSON`:

```json
{
  "preferred_market": "market",
  "risk_mode": "risk_mode",
  "wheel_active": "wheel_active",
  "capital_available": "capital"
}
```

### Telegram Entry Channel (agent)

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_DEFAULT_CHAT_ID`
- `TELEGRAM_ENTRY_POLL_TIMEOUT_SECONDS` (default `20`)
- `TELEGRAM_ENTRY_REQUEST_TIMEOUT_SECONDS` (default `35`)
- `TELEGRAM_ENTRY_ALLOWED_CHAT_IDS`
- `TELEGRAM_ENTRY_DEBUG` (default `true`)
- `TELEGRAM_ENTRY_DEBUG_MAX_CHARS` (default `3200`)

### Communication Domain (telegram sender)

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_DEFAULT_CHAT_ID`
- `TELEGRAM_DRY_RUN` (service default `false`; compose default sets `true`)
- `TELEGRAM_ALLOWED_CHAT_IDS`
- `TELEGRAM_TIMEOUT_SECONDS`

### OpenAI-Compatible API / Open WebUI

- `OPENAI_API_DEBUG_TRACE` (default `true`)
- `OPENAI_API_INCLUDE_SUGGESTIONS` (default `true`)
- `OPENAI_API_STREAM_STATUS_UPDATES` (default `true`)
- `OPENAI_API_STREAM_CHUNK_SIZE` (default `160`)
- `OPENAI_API_BASE_URL` (for Open WebUI)
- `OPENAI_API_KEY` (for Open WebUI)

## Bootstrap Domains

Domains are injected from:
1. `BOOTSTRAP_DOMAINS_JSON`
2. `BOOTSTRAP_DOMAINS_FILE`

Example (`domains.bootstrap.json`):

```json
[
  {
    "name": "finance",
    "type": "remote_http",
    "config": {
      "url": "http://finance-server:8001",
      "timeout": 30.0
    },
    "sync_capabilities": true
  },
  {
    "name": "communication",
    "type": "remote_http",
    "config": {
      "url": "http://communication-domain:8002",
      "timeout": 15.0
    },
    "sync_capabilities": true
  }
]
```

## Run

### Docker Compose

```bash
docker compose up --build
```

Services:
- `finance-server` -> host `:8003` (container `:8001`)
- `communication-domain` -> host `:8002`
- `agent` (CLI)
- `agent-api` -> host `:8010`
- `open-webui` -> host `:3000`

### CLI

```bash
python3 main.py run
```

### Telegram Entry

```bash
python3 main.py run-telegram
```

### OpenAI-compatible API

```bash
uvicorn api.openai_server:app --host 0.0.0.0 --port 8010
```

Endpoints:
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

Streaming:
- Send `"stream": true` to receive SSE chunks (`chat.completion.chunk`).
- The API emits early status messages before final content to improve perceived latency.

## Admin Commands

### Domains

```bash
python3 main.py domain-list
python3 main.py domain-add finance remote_http '{"url":"http://localhost:8003"}'
python3 main.py domain-sync finance
```

### Memory

```bash
python3 main.py memory-set preferred_market '"SE"'
python3 main.py memory-set risk_mode '"moderate"' --namespace session:abc123
python3 main.py memory-get preferred_market
python3 main.py memory-search market
```

## Tests / Validation

### Simple integration scripts

Use `PYTHONPATH=.` to avoid import issues when running from `scripts/`.

```bash
PYTHONPATH=. python3 scripts/test_telegram_send_simple.py
PYTHONPATH=. python3 scripts/test_stock_price_notify_simple.py
```

### Unit tests

```bash
python3 -m pytest -q
```

Important:
- If `pytest` is not installed in your environment, install dependencies first.

## Troubleshooting

- `Name or service not known` for `finance-server`:
  - If running outside compose, use `http://localhost:8003` (host port), not `http://finance-server:8001`.
- Telegram returns no messages and `getUpdates` is empty:
  - Send at least one message to the bot, then call `getUpdates` again.
- Many clarification prompts:
  - Check `SOFT_CONFIRM_THRESHOLD` and `ORCHESTRATOR_CONFIDENCE_THRESHOLD`.
- Wrong ticker / ambiguous ticker on `get_stock_price`:
  - The finance handler now runs deterministic `search_symbol` resolution first.
  - If multiple matches are found, it returns a clarification asking which ticker you mean.
- Port conflict on `8001`:
  - Finance host port is already moved to `8003` in compose.

## Notes

- Communication domain is standalone and can be moved to another repository/container.
- Planner decomposition is metadata-driven; improve domain manifests/capability metadata for better planning quality.

## License

MIT
