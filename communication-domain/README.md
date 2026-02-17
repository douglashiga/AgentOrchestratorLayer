# Communication Domain

Domínio de comunicação outbound. Envia mensagens via Telegram para usuários e grupos/canais.

- **Porta:** `8002`
- **Tipo:** `remote_http`
- **Entry point:** `communication-domain/app/main.py`
- **Standalone:** pode ser movido para outro repositório/container sem mudanças no orquestrador

---

## Endpoints

| Endpoint | Descrição |
|----------|-----------|
| `GET /health` | Health check |
| `GET /manifest` | Goals + capabilities + metadata |
| `GET /openapi.json` | OpenAPI spec |
| `POST /execute` | Executa uma `ExecutionIntent` |

### `/execute` — Payload

```python
# Input (ExecutionIntent serializado)
{
    "domain": "communication",
    "capability": "send_telegram_message",
    "confidence": 1.0,
    "parameters": {
        "message": "Nordea está em 112.50 SEK",
        "chat_id": "123456789"           # opcional se TELEGRAM_DEFAULT_CHAT_ID configurado
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

## Goals e Capabilities

### SEND_NOTIFICATION — Mensagem direta

**Goal:** `SEND_NOTIFICATION`
**Capabilities:** `send_telegram_message`

**Parâmetros de execução:**

| Parâmetro | Tipo | Obrigatório | Descrição |
|-----------|------|-------------|-----------|
| `message` | string | ✅ | Texto da mensagem |
| `chat_id` | string | ❌ | ID do chat. Default: `TELEGRAM_DEFAULT_CHAT_ID` |

**Exemplo:**
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

### SEND_GROUP_MESSAGE — Mensagem para grupo/canal

**Goal:** `SEND_GROUP_MESSAGE`
**Capabilities:** `send_telegram_group_message`

**Parâmetros de execução:**

| Parâmetro | Tipo | Obrigatório | Descrição |
|-----------|------|-------------|-----------|
| `message` | string | ✅ | Texto da mensagem |
| `group_id` | string | ❌ | ID do grupo/canal. Default: `TELEGRAM_DEFAULT_CHAT_ID` |
| `chat_id` | string | ❌ | Alias de `group_id` |

---

## Uso como Notifier (Composition)

O domínio de comunicação é tipicamente o **step final de notificação** em planos multi-step, adicionado pelo `TaskDecomposer` ou `FunctionCallingPlanner` quando o intent contém `notify=true`.

**Exemplo de ExecutionPlan multi-step:**

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
                "message": "${1.explanation}",          # referência ao output do step 1
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

## Tratamento de Erros

| Situação | Status | Explicação |
|----------|--------|------------|
| Mensagem enviada | `success` | `"Mensagem enviada para o chat X."` |
| `TELEGRAM_DRY_RUN=true` | `success` | `"Dry-run: Telegram message prepared for chat X."` |
| `chat_id` ausente/inválido | `clarification` | `"Informe um chat_id válido..."` |
| Chat não autorizado | `clarification` | `"Esse chat_id não está permitido..."` |
| Token ausente | `clarification` | `"Token do Telegram ausente/inválido..."` |
| Bot não iniciado no chat | `clarification` | `"Não consegui enviar... inicie o bot no Telegram antes"` |
| Capability desconhecida | `failure` | `"Unsupported capability: X"` |

---

## Arquitetura Interna

```
ExecutionIntent (capability + parameters)
  │
  ├─ execute(intent)
  │    ├─ extrair capability, params
  │    ├─ resolver chat_id (params → ENV fallback)
  │    └─ TelegramService.send_message(chat_id, message)
  │         └─ Telegram Bot API
  │
  └─ DomainOutput { status, result, explanation }
```

---

## Variáveis de Ambiente

| Variável | Obrigatória | Descrição |
|----------|-------------|-----------|
| `TELEGRAM_BOT_TOKEN` | ✅ | Token do bot (BotFather) |
| `TELEGRAM_DEFAULT_CHAT_ID` | ❌ | Chat ID padrão quando não especificado |
| `TELEGRAM_ALLOWED_CHAT_IDS` | ❌ | Lista de chat IDs permitidos (CSV). Vazio = todos |
| `TELEGRAM_TIMEOUT_SECONDS` | ❌ | Timeout das chamadas à API (default: `10`) |
| `TELEGRAM_DRY_RUN` | ❌ | `true` para simular sem enviar (default: `false`; compose: `true`) |

---

## Como Rodar

```bash
pip install -r requirements.txt
python -m app.main
```

Servidor na porta `:8002`.

---

## Adicionando uma Nova Capability de Comunicação

1. Declarar em `DOMAIN_MANIFEST["capabilities"]` e `DOMAIN_MANIFEST["goals"]` em `main.py`
2. Implementar no `execute()`:
   ```python
   elif capability == "nova_capability":
       # lógica aqui
   ```
3. Nenhuma mudança necessária no orquestrador
