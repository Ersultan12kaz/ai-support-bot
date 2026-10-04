"""Telegram glue around SupportAssistant."""

import asyncio
import logging
import os
from html import escape
from pathlib import Path

import anthropic
from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction, ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from support_bot.assistant import Handoff, SupportAssistant

REASONS = {
    "booking": "📅 Booking",
    "complaint": "⚠️ Complaint",
    "asked_for_human": "🙋 Wants a person",
    "unknown_answer": "❓ Question the bot couldn't answer",
}


def build_router(assistant: SupportAssistant, business: str) -> Router:
    router = Router()

    @router.message(CommandStart())
    async def start(message: Message) -> None:
        assistant.reset(message.from_user.id)
        await message.answer(
            f"Hi! I'm the assistant of <b>{escape(business)}</b>. Ask me about services, prices or opening "
            "hours — or ask for a staff member any time."
        )

    @router.message(Command("reset"))
    async def reset(message: Message) -> None:
        assistant.reset(message.from_user.id)
        await message.answer("Conversation cleared.")

    @router.message(F.text)
    async def chat(message: Message, bot: Bot) -> None:
        await bot.send_chat_action(message.chat.id, ChatAction.TYPING)
        answer = await assistant.reply(message.from_user.id, message.from_user.full_name, message.text[:2000])
        await message.answer(escape(answer))

    @router.message()
    async def other(message: Message) -> None:
        await message.answer("I can only read text messages for now.")

    return router


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    token = os.environ["BOT_TOKEN"]
    admin_chat = int(os.getenv("ADMIN_CHAT_ID", "0"))
    business = os.getenv("BUSINESS_NAME", "Aru Beauty Studio")
    knowledge = Path(os.getenv("KNOWLEDGE_FILE", "knowledge.md")).read_text(encoding="utf-8")

    bot = Bot(token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    async def notify_staff(h: Handoff) -> None:
        text = (
            f"{REASONS.get(h.reason, h.reason)}\n"
            f"Customer: {escape(h.user_name)} (<a href=\"tg://user?id={h.user_id}\">open chat</a>)\n"
            f"{escape(h.summary)}"
        )
        if admin_chat:
            await bot.send_message(admin_chat, text)
        else:
            logging.warning("ADMIN_CHAT_ID not set; handoff: %s", h)

    assistant = SupportAssistant(
        anthropic.AsyncAnthropic(), business, knowledge, notify_staff, effort=os.getenv("CLAUDE_EFFORT", "medium")
    )
    dp = Dispatcher()
    dp.include_router(build_router(assistant, business))
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
