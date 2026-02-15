# Setup do Open WebUI com Agent Orchestrator

## Pré-requisitos

1. Docker e Docker Compose instalados
2. Agent Orchestrator rodando (o `agent-api` serviço)
3. Variáveis de ambiente configuradas

## Configuração Rápida

### 1. Copiar arquivo .env
```bash
cp .env.example .env
```

### 2. Editar .env (opcional, os defaults funcionam)
```bash
# Verificar OPENAI_API_BASE_URL
OPENAI_API_BASE_URL=http://agent-api:8010/v1
OPENAI_API_KEY=sk-local-dev

# Ollama (se estiver usando)
OLLAMA_URL=http://host.docker.internal:11434
```

### 3. Iniciar Docker Compose
```bash
docker compose up --build
```

Ou apenas o Open WebUI:
```bash
docker compose up open-webui -d
```

### 4. Acessar Open WebUI
Abrir navegador em: **http://localhost:3000**

## Configurar o Modelo no Open WebUI

### Via Web Interface:

1. Ao entrar, ir para **Admin Panel** (ícone de engrenagem)
2. Clique em **Settings** → **Connections**
3. Verificar:
   - **OpenAI API Base URL**: `http://agent-api:8010/v1`
   - **API Key**: `sk-local-dev`

### Ou via .env (mais simples):
As variáveis já estão no `docker-compose.yml`:
```yaml
environment:
  - OPENAI_API_BASE_URL=http://agent-api:8010/v1
  - OPENAI_API_KEY=sk-local-dev
```

## Usar o Chat

### 1. Selecionar Modelo
Dropdown no topo → **agent-orchestrator** (full pipeline)
ou **agent-orchestrator-fastpath** (fast-path mode)

### 2. Testar com um comando financeiro:
```
Qual é o preço da ação VALE3?
```

### 3. Observar progresso
O chat mostrará:
- ✅ Status updates (se `OPENAI_API_STREAM_STATUS_UPDATES=true`)
- 🧠 Progress events (intent extracted, plan gerado, etc)
- 📊 Resultados
- 💡 Sugestões (se `OPENAI_API_INCLUDE_SUGGESTIONS=true`)

## Troubleshooting

### Erro: "Connection refused"
- Verificar se `agent-api` está rodando: `docker compose ps`
- Se fora do Docker, usar: `OPENAI_API_BASE_URL=http://localhost:8010/v1`

### Erro: "Model not found"
- Certifique-se que o endpoint `/v1/models` está retornando models
- Teste: `curl http://agent-api:8010/v1/models`

### Chat não está recebendo respostas
1. Verificar logs: `docker compose logs agent-api`
2. Verificar se domains (finance, communication) estão conectadas
3. Habilitar debug: `OPENAI_API_DEBUG_TRACE=true`

### Open WebUI não conecta ao API
1. Dentro do container, verificar conectividade:
   ```bash
   docker compose exec open-webui curl http://agent-api:8010/health
   ```
2. Se falhar, pode ser issue de networking entre containers

## Variáveis de Ambiente Úteis

```bash
# Debug completo
OPENAI_API_DEBUG_TRACE=true

# Desabilitar fast-path para sempre usar pipeline completo
GENERAL_FASTPATH_ENABLED=false

# Sugestões inline no content (não recomendado)
OPENAI_API_SUGGESTIONS_IN_CONTENT=true

# Log level
LOG_LEVEL=DEBUG
```

## Dicas de Performance

1. **Fast-path habilitado** (padrão):
   - Detecta chats gerais e usa streaming direto
   - Mais rápido para conversas simples

2. **Desabilitar preload de modelos**:
   ```bash
   OPENAI_API_PRELOAD_MODELS=false
   ```

3. **Aumentar timeout para operações lentas**:
   ```bash
   OPENAI_API_PRELOAD_TIMEOUT_SECONDS=60
   PLANNER_TIMEOUT_SECONDS=30
   ```

## Verificar Status dos Serviços

```bash
# Health check
curl http://localhost:8010/health

# Listar modelos disponíveis
curl http://localhost:8010/v1/models

# Testar chat (via CLI)
curl -X POST http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "agent-orchestrator",
    "messages": [{"role": "user", "content": "Oi"}],
    "stream": false
  }'
```

## Customizações

### Adicionar novos tipos de sugestões:
Ver `api/openai_server.py` linhas ~529-560

### Modificar fonte de dados (domains):
Editar `domains.bootstrap.json`

### Mudar modelo de intent extraction:
`INTENT_MODEL_NAME=` (defaults to `OLLAMA_INTENT_MODEL`)
