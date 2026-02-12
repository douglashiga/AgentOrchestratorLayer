# 🤖 Agent Orchestrator Layer

Multi-layer, multi-domain agent orchestrator with **deterministic core** and **isolated LLM**. Built in Python, powered by [Ollama](https://ollama.ai) and integrated with [MCP Finance Server](https://github.com/douglashiga/MCP-Finance-Server).

## 📐 Architecture

## 📐 Architecture

### 8-Layer Reliability Architecture

```
[User]
  ↓
[1. Entry Layer]      (CLI/API)
  ↓
[2. Intent Layer]     (Strict JSON + Confidence)
  ↓
[3. Planner Layer]    (Decomposition & Plan Generation)
  ↓
[4. Execution Engine] (Timeout, Retry, Dependency Management)
  ↓
[5. Orchestrator]     (Registry Lookup & Schema Validation)
  ↓
[6. Domain Layer]     (Pure Logic & Isolation)
  ↓      ↖ (Retry/Fallback)
[7. Model Layer]      (LLM Abstraction & Policy)
  ↓
[8. Observability]    (Structured Logging & Metrics)
```

### Layers Responsibilities

| Layer | Responsibility | Key Feature |
|-------|---------------|-------------|
| **1. Entry** | I/O Normalization | Protocol Agnostic |
| **2. Intent** | Classification & Confidence | Strict Schema (Pydantic) |
| **3. Planner** | Task Decomposition | Structured Execution Plan |
| **4. Execution** | Run Management | Timeouts & Dependencies |
| **5. Orchestrator** | Routing & Validation | "Dumb" Registry Lookup |
| **6. Domain** | Business Logic | Isolated & Deterministic |
| **7. Model** | LLM Handling | Retry, Fallback & Policy |
| **8. Observability** | Insight & Debugging | Structured Events |

### Anti-Hallucination Rules

1. **Strict Schemas**: All LLM outputs must be valid JSON matching Pydantic models.
2. **Confidence Gating**: Intents with confidence < 0.5 are rejected or escalated.
3. **No Logic in LLM**: Models only classify or formats; they never execute business rules.
4. **Timeouts**: Mandatory timeouts at Execution and Model layers.
5. **Retries**: Max 3 schema validation retries before failure.

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
