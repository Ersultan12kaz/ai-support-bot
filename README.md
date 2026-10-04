# AI Support Bot for a Beauty Salon

A Telegram assistant that answers customer questions 24/7 using the business's own information
(services, prices, hours, policies) and **hands the conversation to a human** when it should:
bookings, complaints, "can I talk to a person?", or anything the knowledge base doesn't cover.

Powered by **Claude (Anthropic API)**, Python 3.13 and aiogram 3.

## How it behaves

- Answers only from `knowledge.md` — it is told never to invent prices, times or services
- Replies in the customer's language (Kazakh, Russian or English), short and plain
- Remembers the conversation per customer (bounded history, `/reset` to clear)
- **Human handoff via tool use:** Claude calls a `handoff_to_human` tool (strict JSON schema with a
  reason and a summary); the bot posts it to the staff chat with a link to the customer
- Safe failure: rate limits, network errors and declined requests return a friendly message and
  roll back the turn, so the conversation history never ends up half-written

## Engineering notes

- The system prompt (instructions + knowledge base) is identical for every customer and uses
  **prompt caching**, so repeated requests reuse it at a fraction of the cost
- Server-side refusal fallback is enabled (`fallbacks="default"`): if a safety classifier declines,
  the API retries on a suitable model instead of failing
- `effort` is configurable (`CLAUDE_EFFORT`, default `medium`) to trade depth for cost and speed
- Business logic (`support_bot/assistant.py`) is separate from Telegram glue (`support_bot/bot.py`)
- 8 tests run against a fake Claude client — no API key or spend needed

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # from console.anthropic.com
export BOT_TOKEN=...                # from @BotFather
export ADMIN_CHAT_ID=...            # staff chat for handoffs
export BUSINESS_NAME="Aru Beauty Studio"
.venv/bin/python -m support_bot.bot
```

Edit `knowledge.md` to adapt the bot to any business — a café, clinic, school or online shop.

## Tests

```bash
.venv/bin/pip install pytest pytest-asyncio
.venv/bin/python -m pytest
```
