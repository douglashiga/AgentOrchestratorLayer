# Troubleshooting Open WebUI + Agent Orchestrator

## Problemas Comuns

### 1. Open WebUI não conecta ao Agent Orchestrator API

**Sintomas:**
- Erro "Connection refused" ao enviar mensagem
- Chat fica carregando infinitamente

**Soluções:**

```bash
# Verificar se agent-api está rodando
docker compose ps

# Se não estiver, iniciar:
docker compose up agent-api -d

# Testar conectividade direto
curl http://localhost:8010/health

# Dentro do container do Open WebUI
docker compose exec open-webui curl http://agent-api:8010/health
```

**Se ainda não funcionar:**
- Verificar se a porta 8010 não está bloqueada
- Checar logs: `docker compose logs agent-api`
- Reiniciar: `docker compose restart agent-api`

---

### 2. Open WebUI não mostra nenhum modelo

**Sintomas:**
- Dropdown de modelos vazio
- Chat desabilitado

**Testar:**

```bash
# Endpoint deve retornar 2 modelos
curl http://localhost:8010/v1/models | jq .

# Resposta esperada:
# {
#   "data": [
#     {
#       "id": "agent-orchestrator",
#       "object": "model",
#       "owned_by": "agent"
#     },
#     {
#       "id": "agent-orchestrator-fastpath",
#       "object": "model",
#       "owned_by": "agent"
#     }
#   ]
# }
```

**Se estiver vazio:**
- Recarregar página: F5 ou Cmd+R
- Limpar cache do navegador (Ctrl+Shift+Delete)
- Reiniciar container: `docker compose restart open-webui`

---

### 3. Chat responde muito lentamente

**Verificar modelo:**
```bash
# Se estiver usando fast-path desnecessariamente
GENERAL_FASTPATH_ENABLED=false docker compose up agent-api -d

# Ou aumentar timeout
PLANNER_TIMEOUT_SECONDS=30 docker compose up agent-api -d
```

**Checklist:**
- [ ] Ollama está rodando? `curl http://localhost:11434/api/tags`
- [ ] Modelo está carregado? `ollama list`
- [ ] CPU/memória disponível? `docker stats`

---

### 4. Erro 401/403 ao conectar

**Possível causa:** API Key está errada

```bash
# Verificar no docker-compose.yml
grep OPENAI_API_KEY docker-compose.yml

# Deve ser:
# OPENAI_API_KEY=sk-local-dev

# Se estiver diferente, editar ou adicionar:
# OPENAI_API_KEY=sk-local-dev
```

---

### 5. Open WebUI mostra erro 502 Bad Gateway

**Significa:** Open WebUI não consegue contactar a API

```bash
# Verificar se agent-api está saudável
curl -v http://agent-api:8010/health

# Dentro do container
docker compose exec open-webui bash
curl -v http://agent-api:8010/health
exit

# Se falhar, problema é de networking
docker network ls | grep agent
docker network inspect agent-orchestrator
```

**Solução:**
```bash
# Recrear rede
docker compose down
docker network rm agent-orchestrator 2>/dev/null || true
docker compose up -d
```

---

### 6. Chat funciona mas não mostra progresso/status

**Se não está vendo `[progress]` ou `[step]` updates:**

```bash
# Verificar variáveis
echo "Progress Events: $OPENAI_API_STREAM_PROGRESS_EVENTS"
echo "Stream Status: $OPENAI_API_STREAM_STATUS_UPDATES"

# Habilitar (no .env)
OPENAI_API_STREAM_PROGRESS_EVENTS=true
OPENAI_API_STREAM_STATUS_UPDATES=true

# Reiniciar
docker compose restart agent-api
```

---

### 7. Responses aparecem sem formatação

**Verificar streaming format:**

```bash
# No .env, configurar:
OPENAI_API_STREAM_PROGRESS_FORMAT=human  # ou json_pretty

# Reiniciar
docker compose restart agent-api
```

---

### 8. Chat não integra com Finance Domain

**Testar se finance domain está respondendo:**

```bash
# Health
curl http://localhost:8003/health

# Manifest
curl http://localhost:8003/manifest | jq .

# Se falhar, finance-server não iniciou
docker compose logs finance-server | tail -20
```

**Resolver:**
```bash
docker compose restart finance-server
```

---

### 9. Memory/Database issues

**Se ver erros sobre SQLite:**

```bash
# Checar arquivos
ls -lah *.db

# Limpar se corrompido
rm agent.db registry.db memory.db
docker compose restart agent-api
```

---

### 10. Open WebUI perde histórico do chat

**Isso é esperado se:**
- Containerizer removeu volume
- Reiniciou sem `-d` flag

**Para persistir:**
```bash
# Usar volume nomeado
docker volume ls | grep open-webui

# Usar container com volume
docker compose up -d open-webui
# (já está configurado no docker-compose.yml)
```

---

## Comandos Úteis

### Verificar Status Global
```bash
# Status de todos containers
docker compose ps

# Logs em tempo real
docker compose logs -f agent-api

# Stats de uso
docker stats
```

### Debug Detalhado
```bash
# Habilitar debug trace
OPENAI_API_DEBUG_TRACE=true docker compose up agent-api -d

# Revisar response com debug
curl -X POST http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "agent-orchestrator",
    "messages": [{"role": "user", "content": "Oi"}],
    "stream": false
  }' | jq .x_openwebui.debug
```

### Reiniciar Tudo
```bash
docker compose down
docker compose up --build -d
```

### Limpar Volumes (⚠️ Deleta dados)
```bash
docker compose down -v
docker compose up -d
```

---

## Network Topology

```
┌─────────────────────────────────────────────────┐
│         Docker Network: agent-orchestrator       │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌──────────────┐      ┌──────────────┐         │
│  │  Open WebUI  │      │  agent-api   │         │
│  │  :3000       │─────→│  :8010       │         │
│  └──────────────┘      │  (FastAPI)   │         │
│                        └──────────────┘         │
│                               ↓                 │
│                        ┌──────────────┐         │
│                        │ Orchestrator │         │
│                        │   Registry   │         │
│                        └──────────────┘         │
│                               ↓                 │
│                  ┌────────────┴────────────┐    │
│                  ↓                         ↓    │
│          ┌─────────────┐         ┌──────────────┐
│          │   finance   │         │ communication│
│          │   :8001     │         │    :8002     │
│          └─────────────┘         └──────────────┘
│
└─────────────────────────────────────────────────┘
         ↓
   ┌─────────────┐
   │   Ollama    │
   │  :11434     │
   │ (local/ext) │
   └─────────────┘
```

---

## Logs Úteis para Análise

### Coletar logs para debug
```bash
# Todos os containers
docker compose logs > orchestrator.logs 2>&1

# Específico
docker compose logs agent-api > agent-api.logs
docker compose logs open-webui > open-webui.logs

# Com timestamps
docker compose logs --timestamps --follow agent-api
```

### Buscar erros
```bash
docker compose logs | grep -i error
docker compose logs | grep -i exception
docker compose logs | grep -i 502
docker compose logs | grep -i timeout
```

---

## Performance Tuning

### Se está lento:

1. **Verificar CPU/Memory:**
   ```bash
   docker stats --no-stream
   ```

2. **Aumentar recursos (docker-compose.yml):**
   ```yaml
   agent-api:
     deploy:
       resources:
         limits:
           cpus: '2'
           memory: 4G
   ```

3. **Cache de modelos (Ollama):**
   ```bash
   # Pré-carregar modelo
   ollama pull llama3.1:8b
   ```

4. **Reduzir chunk size (streaming rápido):**
   ```bash
   OPENAI_API_STREAM_CHUNK_SIZE=80
   ```

---

## Getting Help

1. **Verificar logs completos:**
   ```bash
   docker compose logs --tail=100
   ```

2. **Validar configuração:**
   ```bash
   docker compose config | grep -A5 agent-api
   ```

3. **Testar API manualmente:**
   ```bash
   curl -v http://localhost:8010/health
   curl http://localhost:8010/v1/models | jq .
   ```

4. **Checar volumes:**
   ```bash
   docker volume ls
   docker inspect open-webui-data
   ```
