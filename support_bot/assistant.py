"""AI support assistant: answers from a knowledge base and hands off to a human when needed."""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import anthropic

log = logging.getLogger(__name__)

MODEL = "claude-opus-5"
MAX_HISTORY_MESSAGES = 30  # per user; older turns are dropped from the front in whole exchanges
MAX_TOOL_ROUNDS = 3

SYSTEM_TEMPLATE = """You are the friendly online assistant of {business}. You chat with customers in Telegram.

Answer only from the business information below. If the answer is not there, say you are not sure \
and offer to connect the customer with a staff member. Never invent prices, times or services.

Reply in the customer's language (Kazakh, Russian or English). Keep replies short: 1–4 sentences, \
plain text, no Markdown tables.

Call the handoff_to_human tool when the customer asks for a person, wants to book or change an \
appointment, has a complaint, or asks something the business information does not cover. After \
the tool returns, tell the customer a staff member will message them soon.

<business_information>
{knowledge}
</business_information>"""

HANDOFF_TOOL = {
    "name": "handoff_to_human",
    "description": (
        "Forward the conversation to a staff member. Use for bookings, complaints, requests to talk "
        "to a person, or questions the business information does not answer."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "reason": {"type": "string", "enum": ["booking", "complaint", "asked_for_human", "unknown_answer"]},
            "summary": {"type": "string", "description": "One or two sentences: what the customer needs."},
        },
        "required": ["reason", "summary"],
        "additionalProperties": False,
    },
}

FALLBACK_REPLY = "Sorry, I can't help with that here. A staff member will get back to you."


@dataclass(frozen=True)
class Handoff:
    user_id: int
    user_name: str
    reason: str
    summary: str


HandoffCallback = Callable[[Handoff], Awaitable[None]]


class SupportAssistant:
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        business: str,
        knowledge: str,
        on_handoff: HandoffCallback,
        effort: str = "medium",
    ) -> None:
        self.client = client
        self.system = [
            {
                "type": "text",
                "text": SYSTEM_TEMPLATE.format(business=business, knowledge=knowledge),
                # The system prompt is identical for every customer, so it is cached across all chats.
                "cache_control": {"type": "ephemeral"},
            }
        ]
        self.on_handoff = on_handoff
        self.effort = effort
        self.histories: dict[int, list[dict]] = {}

    def reset(self, user_id: int) -> None:
        self.histories.pop(user_id, None)

    async def reply(self, user_id: int, user_name: str, text: str) -> str:
        history = self.histories.setdefault(user_id, [])
        turn_start = len(history)  # on failure, roll back to here so history never holds half a turn
        history.append({"role": "user", "content": text})

        for _ in range(MAX_TOOL_ROUNDS):
            try:
                response = await self.client.beta.messages.create(
                    model=MODEL,
                    max_tokens=4096,
                    system=self.system,
                    tools=[HANDOFF_TOOL],
                    messages=history,
                    output_config={"effort": self.effort},
                    # If a safety classifier declines, the API retries on a suitable fallback model.
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                )
            except anthropic.RateLimitError:
                del history[turn_start:]
                return "We're getting a lot of messages right now. Please try again in a minute."
            except anthropic.APIConnectionError:
                del history[turn_start:]
                return "Connection problem on our side. Please try again shortly."
            except anthropic.APIStatusError as exc:
                log.error("Claude API error %s: %s", exc.status_code, exc.message)
                del history[turn_start:]
                return "Something went wrong on our side. Please try again later."

            if response.stop_reason == "refusal":
                del history[turn_start:]
                return FALLBACK_REPLY

            # Keep the full content (including thinking and tool_use blocks) so later turns stay valid.
            history.append({"role": "assistant", "content": response.content})

            tool_calls = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_calls:
                self._trim(user_id)
                return _text_of(response.content) or FALLBACK_REPLY

            results = []
            for call in tool_calls:
                results.append(await self._run_tool(call, user_id, user_name))
            history.append({"role": "user", "content": results})

        self._trim(user_id)
        return "A staff member will get back to you soon."

    async def _run_tool(self, call, user_id: int, user_name: str) -> dict:
        if call.name != "handoff_to_human":
            return {"type": "tool_result", "tool_use_id": call.id, "content": "Unknown tool", "is_error": True}
        args = call.input if isinstance(call.input, dict) else {}
        handoff = Handoff(user_id, user_name, str(args.get("reason", "")), str(args.get("summary", "")))
        try:
            await self.on_handoff(handoff)
            content = "Staff notified. They will reply to the customer in this chat."
        except Exception:
            log.exception("Handoff failed")
            return {"type": "tool_result", "tool_use_id": call.id,
                    "content": "Could not reach staff right now.", "is_error": True}
        return {"type": "tool_result", "tool_use_id": call.id, "content": content}

    def _trim(self, user_id: int) -> None:
        """Drop the oldest exchanges so history stays bounded, always starting on a plain user message."""
        history = self.histories.get(user_id, [])
        while len(history) > MAX_HISTORY_MESSAGES:
            history.pop(0)
            while history and not (history[0]["role"] == "user" and isinstance(history[0]["content"], str)):
                history.pop(0)


def _text_of(content) -> str:
    return "\n".join(b.text for b in content if b.type == "text").strip()
