"""Tests run against a fake Claude client, so they need no API key and cost nothing."""

from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from support_bot.assistant import FALLBACK_REPLY, MAX_HISTORY_MESSAGES, SupportAssistant


def text(t):
    return SimpleNamespace(type="text", text=t)


def tool_use(reason="booking", summary="Wants a manicure on Friday"):
    return SimpleNamespace(type="tool_use", id="toolu_1", name="handoff_to_human",
                           input={"reason": reason, "summary": summary})


def resp(content, stop="end_turn"):
    return SimpleNamespace(content=content, stop_reason=stop)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make(responses):
    handoffs = []

    async def on_handoff(h):
        handoffs.append(h)

    client = FakeClient(responses)
    return SupportAssistant(client, "Aru Beauty", "Manicure: 7 000 ₸", on_handoff), client, handoffs


async def test_plain_answer_and_request_shape():
    bot, client, _ = make([resp([text("Manicure costs 7 000 ₸.")])])
    assert await bot.reply(1, "Aru", "How much is a manicure?") == "Manicure costs 7 000 ₸."
    call = client.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["fallbacks"] == "default"
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "Manicure: 7 000 ₸" in call["system"][0]["text"]


async def test_history_is_kept_between_turns():
    bot, client, _ = make([resp([text("Hi!")]), resp([text("You asked about hours.")])])
    await bot.reply(1, "Aru", "Hello")
    await bot.reply(1, "Aru", "What did I ask?")
    sent = client.calls[1]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert sent[0]["content"] == "Hello"


async def test_handoff_tool_round_trip():
    bot, client, handoffs = make([
        resp([text("Let me connect you."), tool_use()], stop="tool_use"),
        resp([text("A staff member will message you soon.")]),
    ])
    answer = await bot.reply(7, "Dana", "Book me for Friday")
    assert answer == "A staff member will message you soon."
    assert handoffs[0].user_id == 7 and handoffs[0].reason == "booking"
    tool_result = client.calls[1]["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "toolu_1"
    assert "is_error" not in tool_result


async def test_failed_handoff_reported_as_tool_error():
    async def broken(_):
        raise RuntimeError("telegram down")

    client = FakeClient([resp([tool_use()], stop="tool_use"), resp([text("Sorry, please call us.")])])
    bot = SupportAssistant(client, "Aru Beauty", "kb", broken)
    assert await bot.reply(1, "A", "complaint") == "Sorry, please call us."
    assert client.calls[1]["messages"][-1]["content"][0]["is_error"] is True


async def test_refusal_rolls_back_the_turn():
    bot, _, _ = make([resp([], stop="refusal")])
    assert await bot.reply(1, "A", "something odd") == FALLBACK_REPLY
    assert bot.histories[1] == []


async def test_api_error_rolls_back_the_turn():
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.RateLimitError("slow down", response=httpx2.Response(429, request=req), body=None)
    bot, _, _ = make([resp([text("ok")]), err])
    await bot.reply(1, "A", "first")
    answer = await bot.reply(1, "A", "second")
    assert "try again" in answer
    assert len(bot.histories[1]) == 2  # only the successful first exchange remains


async def test_history_is_bounded_and_starts_with_user_text():
    replies = [resp([text(f"answer {i}")]) for i in range(40)]
    bot, _, _ = make(replies)
    for i in range(40):
        await bot.reply(1, "A", f"q{i}")
    h = bot.histories[1]
    assert len(h) <= MAX_HISTORY_MESSAGES
    assert h[0]["role"] == "user" and isinstance(h[0]["content"], str)


async def test_reset():
    bot, _, _ = make([resp([text("hi")])])
    await bot.reply(1, "A", "hello")
    bot.reset(1)
    assert 1 not in bot.histories
