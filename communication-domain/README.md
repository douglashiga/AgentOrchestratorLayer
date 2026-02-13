# Communication Domain Service

Standalone domain service to send outbound notifications via Telegram.

## Endpoints

- `GET /health`
- `GET /manifest`
- `GET /openapi.json`
- `POST /execute`

## Environment Variables

- `TELEGRAM_BOT_TOKEN`: Telegram bot token.
- `TELEGRAM_DEFAULT_CHAT_ID`: Default chat/group id when request does not provide one.
- `TELEGRAM_TIMEOUT_SECONDS`: HTTP timeout for Telegram API calls. Default `10`.
- `TELEGRAM_DRY_RUN`: If true, do not call Telegram API, just simulate success.
- `TELEGRAM_ALLOWED_CHAT_IDS`: Optional comma-separated allowlist.

## Local Run

```bash
pip install -r requirements.txt
python -m app.main
```

Server listens on `:8002`.
