# Communication Domain

Outbound communication domain. Sends messages via Telegram to users and groups/channels.

- **Port:** `8002`
- **Type:** `remote_http`
- **Entry point:** `communication-domain/app/main.py`
- **Standalone:** can be moved to another repository/container without changes to the orchestrator

---

## Endpoints

| Endpoint | Description |
|----------|-----------|
| `GET /health` | Health check |
| `GET /manifest` | Goals + capabilities + metadata |
| `GET /openapi.json` | OpenAPI spec |
| `POST /execute` | Executes an `ExecutionIntent` |

### `/execute` — Payload

```python
# Input (serialized ExecutionIntent)
{
    "domain": "communication",
    "capability": "send_telegram_message",
    "confidence": 1.0,
    "parameters": {
        "message": "Nordea está em 112.50 SEK",
        "chat_id": "123456789"           # optional if TELEGRAM_DEFAULT_CHAT_ID is configured
    },
    "original_query": "manda o preço no telegram"
}

# Output (DomainOutput)
{
    "status": "success",
    "result": {
        "sent": true,
        "chat_id": "123456789",
        "message_id": 42
    },
    "explanation": "Mensagem enviada para o chat 123456789.",
    "confidence": 1.0,
    "metadata": {}
}
```

---

## Goals and Capabilities

### SEND_NOTIFICATION — Direct message

**Goal:** `SEND_NOTIFICATION`
**Capabilities:** `send_telegram_message`

**Execution parameters:**

| Parameter | Type | Required | Description |
|-----------|------|-------------|-----------|
| `message` | string | Yes | Message text |
| `chat_id` | string | No | Chat ID. Default: `TELEGRAM_DEFAULT_CHAT_ID` |

**Example:**
```python
ExecutionIntent(
    domain="communication",
    capability="send_telegram_message",
    parameters={
        "message": "Nordea (NDA-SE.ST) está em 112.50 SEK (-0.71%)",
        "chat_id": "123456789"
    }
)
```

---

### SEND_GROUP_MESSAGE — Message to group/channel

**Goal:** `SEND_GROUP_MESSAGE`
**Capabilities:** `send_telegram_group_message`

**Execution parameters:**

| Parameter | Type | Required | Description |
|-----------|------|-------------|-----------|
| `message` | string | Yes | Message text |
| `group_id` | string | No | Group/channel ID. Default: `TELEGRAM_DEFAULT_CHAT_ID` |
| `chat_id` | string | No | Alias for `group_id` |

---

## Usage as Notifier (Composition)

The communication domain is typically the **final notification step** in multi-step plans, added by the `TaskDecomposer` or `FunctionCallingPlanner` when the intent contains `notify=true`.

**Multi-step ExecutionPlan example:**

```python
ExecutionPlan(
    execution_mode="dag",
    combine_mode="report",
    steps=[
        ExecutionStep(
            id=1,
            domain="finance",
            capability="get_stock_price",
            params={"symbol": "NDA-SE.ST"},
            depends_on=[]
        ),
        ExecutionStep(
            id=2,
            domain="communication",
            capability="send_telegram_message",
            params={
                "message": "${1.explanation}",          # reference to step 1 output
                "chat_id": "${ENV:TELEGRAM_DEFAULT_CHAT_ID}"
            },
            depends_on=[1],
            required=False,
            output_key="notification"
        )
    ]
)
```

---

## Error Handling

| Situation | Status | Explanation |
|----------|--------|------------|
| Message sent | `success` | `"Mensagem enviada para o chat X."` |
| `TELEGRAM_DRY_RUN=true` | `success` | `"Dry-run: Telegram message prepared for chat X."` |
| `chat_id` missing/invalid | `clarification` | `"Informe um chat_id válido..."` |
| Unauthorized chat | `clarification` | `"Esse chat_id não está permitido..."` |
| Token missing | `clarification` | `"Token do Telegram ausente/inválido..."` |
| Bot not started in chat | `clarification` | `"Não consegui enviar... inicie o bot no Telegram antes"` |
| Unknown capability | `failure` | `"Unsupported capability: X"` |

---

## Internal Architecture

```
ExecutionIntent (capability + parameters)
  │
  ├─ execute(intent)
  │    ├─ extract capability, params
  │    ├─ resolve chat_id (params → ENV fallback)
  │    └─ TelegramService.send_message(chat_id, message)
  │         └─ Telegram Bot API
  │
  └─ DomainOutput { status, result, explanation }
```

---

## Environment Variables

| Variable | Required | Description |
|----------|-------------|-----------|
| `TELEGRAM_BOT_TOKEN` | Yes | Bot token (BotFather) |
| `TELEGRAM_DEFAULT_CHAT_ID` | No | Default chat ID when not specified |
| `TELEGRAM_ALLOWED_CHAT_IDS` | No | List of allowed chat IDs (CSV). Empty = all |
| `TELEGRAM_TIMEOUT_SECONDS` | No | API call timeout (default: `10`) |
| `TELEGRAM_DRY_RUN` | No | `true` to simulate without sending (default: `false`; compose: `true`) |

---

## How to Run

```bash
pip install -r requirements.txt
python -m app.main
```

Server on port `:8002`.

---

## Adding a New Communication Capability

1. Declare in `DOMAIN_MANIFEST["capabilities"]` and `DOMAIN_MANIFEST["goals"]` in `main.py`
2. Implement in `execute()`:
   ```python
   elif capability == "nova_capability":
       # logic here
   ```
3. No changes required in the orchestrator
