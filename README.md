# 🤖 Agent Orchestrator Layer

Multi-layer, multi-domain agent orchestrator with **deterministic core** and **isolated LLM**. Built in Python, powered by [Ollama](https://ollama.ai) and integrated with [MCP Finance Server](https://github.com/douglashiga/MCP-Finance-Server).

## 📐 Architecture

```
[Telegram]      [HTTP API]      [CLI]
        ↓             ↓             ↓
            Entry Adapters
                   ↓
         Conversation State Manager
                   ↓
              Intent Adapter (LLM)
                   ↓
            Orchestrator (Stateless)
                   ↓
              Domain Registry
                   ↓
           Domain Handler
                   ↓
           Context Resolver
                   ↓
              Skill Gateway
                   ↓
              Skill Registry
                   ↓
          Skill Implementations
                   ↓
            Execution Context
                   ↓
             Strategy Core
                   ↓
                Decision
                   ↓
         Conversation Persist
                   ↓
                Response
```

### Layers

| Layer | File | Responsibility |
|-------|------|---------------|
| **Entry** | `entry/cli.py` | Receives input, normalizes to `EntryRequest`. No logic. |
| **Conversation** | `conversation/manager.py` | SQLite history per `session_id`. No business logic. |
| **Intent Adapter** | `intent/adapter.py` | Ollama LLM → structured `Intent` (JSON, temp=0). No calculations. |
| **Orchestrator** | `orchestrator/orchestrator.py` | Stateless router. Resolves domain, delegates. No business logic. |
| **Domain Registry** | `registry/domain_registry.py` | Maps `domain_name → Handler`. Pure lookup. |
| **Finance Domain** | `domains/finance/handler.py` | Orchestrates Context → Skills → Strategy internally. |
| **Context Resolver** | `domains/finance/context.py` | Deterministic market detection from ticker suffix. |
| **Strategy Core** | `domains/finance/core.py` | Deterministic calculations. No LLM, no HTTP. 100% testable. |
| **General Domain** | `domains/general/handler.py` | Conversational responses for non-finance queries. |
| **Skill Gateway** | `skills/gateway.py` | Controlled access to skills via registry. |
| **Skill Registry** | `skills/registry.py` | Maps `skill_name → Implementation`. Pure lookup. |
| **MCP Adapter** | `skills/implementations/mcp_adapter.py` | Calls MCP Finance Server via SSE protocol. |
| **Shared Models** | `shared/models.py` | Pydantic models (all frozen/immutable). |

### Execution Flow

```
1. Entry recebe requisição
2. Conversation Manager recupera histórico
3. Intent Adapter gera Intent estruturado (LLM)
4. Intent é validado (Pydantic schema)
5. Orchestrator resolve domínio
6. DomainHandler executa:
   • Context Resolver (determinístico)
   • Skill Gateway (busca dados via MCP)
   • Strategy Core (cálculo determinístico)
7. Decision é gerada
8. Conversation Manager salva interação
9. Response retorna ao Entry Adapter
10. Entry responde ao usuário
```

### Anti-Hallucination Rules

| # | Rule |
|---|------|
| 1 | LLM **never** participates in calculations |
| 2 | History **never** replaces real data |
| 3 | Strategy Core **never** accesses HTTP |
| 4 | Skills **never** decide when they are called |
| 5 | Orchestrator **never** knows business logic |
| 6 | No layer skipping — each layer calls only the next |
| 7 | All external inputs validated (Pydantic) |
| 8 | All external responses validated |
| 9 | Contexts are **immutable** after creation |
| 10 | No layer alters a previous context |

---

## 📦 Project Structure

```
AgentsOrchstratorLayer/
├── main.py                          # CLI entrypoint, wires all layers
├── pyproject.toml                   # Dependencies
├── shared/
│   └── models.py                    # Pydantic models (frozen)
├── entry/
│   └── cli.py                       # CLI adapter
├── conversation/
│   └── manager.py                   # SQLite state manager
├── intent/
│   └── adapter.py                   # Ollama LLM intent extraction
├── orchestrator/
│   └── orchestrator.py              # Stateless router
├── registry/
│   └── domain_registry.py           # Domain lookup
├── domains/
│   ├── general/
│   │   └── handler.py               # Conversational responses
│   └── finance/
│       ├── handler.py               # Finance domain orchestration
│       ├── context.py               # Market/currency resolver
│       └── core.py                  # Deterministic strategy engine
└── skills/
    ├── gateway.py                   # Controlled skill access
    ├── registry.py                  # Skill lookup
    └── implementations/
        └── mcp_adapter.py           # MCP Finance Server (SSE)
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.11+**
- **Ollama** with `qwen2.5-coder:32b` model
- **MCP Finance Server** (optional, for real market data)

### 1. Clone & Install

```bash
git clone https://github.com/douglashiga/AgentsOrchstratorLayer.git
cd AgentsOrchstratorLayer
pip install -e .
```

### 2. Setup Ollama

```bash
# Install Ollama (macOS)
brew install ollama

# Pull the model
ollama pull qwen2.5-coder:32b

# Start Ollama (runs on port 11434)
ollama serve
```

### 3. Setup MCP Finance Server (optional)

```bash
# Clone and run the MCP Finance Server
git clone https://github.com/douglashiga/MCP-Finance-Server.git
cd MCP-Finance-Server
docker-compose up -d
# Server runs on http://localhost:8000/sse
```

### 4. Run

```bash
python main.py
```

---

## 💬 Usage

```
╔══════════════ 🤖 ══════════════╗
║ Agent Orchestrator              ║
║ Multi-layer finance agent       ║
║ Type your question or 'exit'    ║
╚═════════════════════════════════╝

You → oi
🤖 Assistant: Oi! Como posso ajudar? Posso buscar preços, fundamentos, dividendos...

You → qual o preço da AAPL?
🧠 Intent: domain=finance, action=get_stock_price, symbol=AAPL, confidence=100%
✅ Result: AAPL is currently trading at 150.25 USD

You → me mostra os fundamentos da Tesla
🧠 Intent: domain=finance, action=get_fundamentals, symbol=TSLA, confidence=95%
✅ Result: Fundamentals data for TSLA (US market)

You → exit
Goodbye! 👋
```

### Supported Finance Actions

| Action | Description | Example |
|--------|-------------|---------|
| `get_stock_price` | Real-time price | "preço da AAPL" |
| `get_fundamentals` | Company fundamentals | "fundamentos da Tesla" |
| `get_dividends` | Dividend history | "dividendos da KO" |
| `get_company_info` | Company information | "info da Microsoft" |
| `get_historical_data` | OHLCV history | "histórico da PETR4" |
| `get_option_chain` | Option chains | "opções da AAPL" |
| `get_option_greeks` | Option Greeks | "greeks da AAPL call 150" |
| `get_financial_statements` | Financial statements | "balanço da MSFT" |
| `get_exchange_info` | Exchange info | "exchange da VOW3.DE" |
| `get_account_summary` | Account summary | "resumo da conta" |
| `search_symbol` | Search symbols | "buscar Apple" |
| `yahoo_search` | General search | "bancos brasileiros" |

### Market Detection

The system automatically detects markets from ticker suffixes:

| Suffix | Market | Currency | Tax Model |
|--------|--------|----------|-----------|
| *(none)* | US | USD | standard |
| `.ST` | Sweden | SEK | ISK |
| `.SA` | Brazil | BRL | standard |
| `.DE` | Germany | EUR | standard |
| `.L` | UK | GBP | ISA |
| `.TO` | Canada | CAD | TFSA |
| `.HK` | Hong Kong | HKD | standard |

---

## ⚙️ Configuration

Environment variables in `main.py`:

```python
OLLAMA_URL = "http://localhost:11434"    # Ollama API
OLLAMA_MODEL = "qwen2.5-coder:32b"     # LLM model
MCP_URL = "http://localhost:8000/sse"   # MCP Finance Server (SSE)
DB_PATH = "conversations.db"            # SQLite database
```

---

## 🔌 Extending

### Adding a New Domain

1. Create `domains/your_domain/handler.py` implementing `execute(intent) → Decision`
2. Register in `main.py`:
   ```python
   domain_registry.register("your_domain", YourHandler())
   ```
3. Update the Intent Adapter system prompt in `intent/adapter.py`

### Adding a New Skill

1. Create `skills/implementations/your_skill.py` implementing `execute(params) → dict`
2. Register in `main.py`:
   ```python
   skill_registry.register("your_skill", YourSkill())
   ```

### Adding a New Entry Adapter

1. Create `entry/your_adapter.py` that produces `EntryRequest`
2. Wire it in your own entrypoint (e.g., FastAPI, Telegram bot)

---

## 📄 License

MIT
