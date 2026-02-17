# Agent Orchestrator Layer

Sistema multi-domínio orientado a goals com intent extraction via LLM, planejamento metadata-driven e execução determinística em DAG.

## Arquitetura Resumida

```
User Input
  ↓ EntryRequest
Intent Adapter (LLM)
  ↓ IntentOutput { primary_domain, goal, entities{*_text / enum} }
Goal Resolver (determinístico)
  ↓ ExecutionIntent { domain, capability, parameters, confidence }
Planner Service + Memory
  ↓ ExecutionPlan { steps[], execution_mode, combine_mode }
Execution Engine (DAG)
  ↓ ExecutionIntent (por step)
Orchestrator (registry lookup)
  ↓
Domain Handler
  ↓ DomainOutput { status, result, explanation }
```

> Detalhes completos de cada camada e payloads: **[ARCHITECTURE.md](./ARCHITECTURE.md)**

---

## Diagrama

```mermaid
graph TD
    User((User)) --> Entry[Entry Layer\nCLI / Telegram / HTTP]
    Entry --> Intent[Intent Adapter\nLLM → IntentOutput]
    Intent --> Resolver[Goal Resolver\ngoal → capability]
    Resolver --> Planner[Planner Service\n+ Memory injection]
    Mem[(Memory Store)] --> Planner
    Planner --> Decomposer[TaskDecomposer\nmetadata-driven]
    Planner --> FCPlanner[FunctionCallingPlanner\noptional LLM loop]
    Decomposer --> Exec[Execution Engine\nDAG / parallel]
    FCPlanner --> Exec
    Exec --> Orch[Orchestrator\nconfidence gate + routing]
    Orch --> Reg[Registry]
    Reg --> Fin[Finance Domain\nremote_http :8001]
    Reg --> Com[Communication Domain\nremote_http :8002]
```

---

## Payloads por Camada

### EntryRequest

```python
EntryRequest(
    session_id="user-abc123",
    input_text="qual o preço da Nordea?",
    metadata={}
)
```

### IntentOutput (saída do Intent Adapter)

O LLM extrai **goal** e **entities human-friendly** — nunca tickers técnicos ou IDs.

```python
IntentOutput(
    primary_domain="finance",
    goal="GET_QUOTE",
    entities={"symbol_text": "Nordea"},   # nome como o usuário disse
    confidence=0.95,
    original_query="qual o preço da Nordea?"
)
```

```python
# Goal com enum (TOP_MOVERS)
IntentOutput(
    primary_domain="finance",
    goal="TOP_MOVERS",
    entities={"direction": "GAINERS", "market_text": "Brasil"},
    confidence=0.92,
    original_query="maiores altas do Brasil"
)
```

### ExecutionIntent (saída do Goal Resolver)

Mapeamento determinístico `goal + entities → capability`. Sem LLM.

```python
ExecutionIntent(
    domain="finance",
    capability="get_stock_price",          # resolvido pelo GoalResolver
    parameters={"symbol_text": "Nordea"},  # entities viram parameters
    confidence=0.95,
    original_query="qual o preço da Nordea?"
)
```

### ExecutionPlan (saída do Planner)

```python
ExecutionPlan(
    execution_mode="dag",
    combine_mode="report",
    steps=[
        ExecutionStep(id=1, capability="get_stock_price",
                      params={"symbol_text": "Nordea"}, depends_on=[]),
        ExecutionStep(id=2, capability="send_telegram_message",
                      params={"message": "${1.explanation}"}, depends_on=[1], required=False)
    ]
)
```

### DomainOutput (saída do Domain Handler)

```python
DomainOutput(
    status="success",        # "success" | "failure" | "clarification"
    result={
        "symbol": "NDA-SE.ST",
        "price": 112.5,
        "currency": "SEK",
        "_market_context": {"market": "SE", "exchange": "OMX"}
    },
    explanation="Nordea está em 112.50 SEK",
    confidence=1.0,
    metadata={}
)
```

---

## Estrutura do Projeto

```text
AgentOrchestratorLayer/
├── main.py                          # Entry CLI + Telegram
├── api/openai_server.py             # OpenAI-compatible API (Open WebUI)
│
├── intent/adapter.py                # LLM → IntentOutput
│
├── planner/
│   ├── goal_resolver.py             # IntentOutput → ExecutionIntent (determinístico)
│   ├── service.py                   # orquestra planner + memória
│   ├── task_decomposer.py           # metadata-driven step decomposition
│   └── function_calling_planner.py  # LLM loop opcional
│
├── execution/
│   ├── engine.py                    # DAG executor + workflow runtime
│   ├── result_combiner.py           # combina outputs dos steps
│   └── task_state_store.py          # persiste TaskInstance + WorkflowEvent
│
├── orchestrator/orchestrator.py     # confidence gate + capability routing
│
├── registry/
│   ├── db.py                        # SQLite: domains, capabilities, goals
│   ├── loader.py                    # carrega manifests → registry
│   ├── domain_registry.py           # HandlerRegistry em memória
│   └── http_handler.py              # handler para domínios remote_http
│
├── shared/
│   ├── models.py                    # todos os Pydantic models
│   └── workflow_contracts.py        # MethodSpec, WorkflowSpec, TaskInstance
│
├── memory/store.py                  # SQLiteMemoryStore
├── models/selector.py               # ModelSelector (Ollama/OpenAI-compat)
├── skills/                          # SkillGateway + MCP adapter
│
├── domains/
│   ├── finance/                     # Finance domain (ver domains/finance/README.md)
│   └── general/handler.py           # General domain (chat)
│
├── communication-domain/            # Communication domain (ver communication-domain/README.md)
│
├── scripts/                         # scripts de teste e avaliação
├── domains.bootstrap.json           # bootstrap de domínios
└── docker-compose.yml
```

---

## Domínios

Cada domínio tem seu próprio README com manifest, capabilities e exemplos:

- **[Finance Domain](./domains/finance/README.md)** — cotações, top movers, screener, histórico
- **[Communication Domain](./communication-domain/README.md)** — envio via Telegram

---

## Features Principais

- **Goal-based intent:** LLM extrai goal + entities human-friendly; GoalResolver mapeia para capability sem LLM
- **Metadata-driven decomposition:** decomposição em steps paralelos configurada no manifest, não no código
- **DAG execution:** steps com dependências explícitas, execução paralela com `max_concurrency`
- **Workflow declarativo:** `MethodSpec` + `WorkflowSpec` para fluxos com `human_gate`, `decision`, `validate`, `call`, `return`
- **Pause/resume:** `TaskInstance` persiste estado; `resume_task(ClarificationAnswer)` retoma de onde parou
- **Memory injection:** memória estruturada (SQLite) injetada no planner antes da decomposição
- **Symbol resolver:** Finance handler resolve nomes → tickers via alias metadata + `search_symbol` como fallback
- **Soft confirmation:** intents com `confidence < 0.94` retornam clarification antes de executar
- **Streaming:** SSE com status updates incrementais; fast-path com token streaming real para chat geral
- **OpenAI-compatible API:** integração direta com Open WebUI

---

## Configuração

### Variáveis principais

```bash
# LLM / Modelos
OLLAMA_URL=http://localhost:11434

# Domínios remotos
BOOTSTRAP_DOMAINS_FILE=domains.bootstrap.json

# Bancos
DB_PATH=agent.db
REGISTRY_DB_PATH=registry.db
MEMORY_DB_PATH=memory.db

# Confidence
SOFT_CONFIRM_THRESHOLD=0.94

# Telegram entry
TELEGRAM_BOT_TOKEN=...
TELEGRAM_DEFAULT_CHAT_ID=...

# OpenAI API
OPENAI_API_DEBUG_TRACE=false
```

### Bootstrap de domínios (`domains.bootstrap.json`)

```json
[
  {
    "name": "finance",
    "type": "remote_http",
    "config": {"url": "http://finance-server:8001", "timeout": 90.0},
    "sync_capabilities": true
  },
  {
    "name": "communication",
    "type": "remote_http",
    "config": {"url": "http://communication-domain:8002", "timeout": 15.0},
    "sync_capabilities": true
  }
]
```

---

## Como Rodar

### Docker Compose

```bash
docker compose up --build
```

Serviços:
- `finance-server` → host `:8003`
- `communication-domain` → host `:8002`
- `agent-api` → host `:8010`
- `open-webui` → host `:3000`

### CLI

```bash
python3 main.py run
python3 main.py run-telegram
```

### API

```bash
uvicorn api.openai_server:app --host 0.0.0.0 --port 8010
```

Endpoints:
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions` (com `stream: true` para SSE)

### Admin

```bash
python3 main.py domain-list
python3 main.py domain-add finance remote_http '{"url":"http://localhost:8003"}'
python3 main.py domain-sync finance
python3 main.py memory-set preferred_market '"SE"'
python3 main.py memory-get preferred_market
```

---

## Testes

```bash
# unit tests
python3 -m pytest -q

# scripts de integração
PYTHONPATH=. python3 scripts/test_stock_price_notify_simple.py
PYTHONPATH=. python3 scripts/test_telegram_send_simple.py

# avaliação de capabilities (requer domínios rodando)
FINANCE_DOMAIN_URL=http://localhost:8003 python3 scripts/evaluate_capabilities.py
```

---

## Troubleshooting

| Sintoma | Causa provável | Solução |
|---------|----------------|---------|
| Muitas clarifications | `SOFT_CONFIRM_THRESHOLD` alto | Reduzir para `0.85` |
| Ticker errado | LLM inferiu ticker direto | Verificar `entities_schema` do goal |
| `Name or service not known` para `finance-server` | Fora do compose | Usar `http://localhost:8003` |
| Telegram não recebe mensagens | Bot sem mensagem inicial | Enviar uma mensagem ao bot primeiro |
| Port conflict 8001 | Finance usa 8001 interno | Host port é 8003 no compose |

## License

MIT
