"""Stream editors for Telegram messages (append mode and edit mode).

Append mode: each flushed chunk is sent as a NEW message.
Edit mode: a single message is continuously edited in-place.

Use :func:`create_stream_editor` to obtain the appropriate implementation
based on configuration.
"""

from __future__ import annotations

import html
import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest

from ductor_bot.bot.buttons import extract_buttons
from ductor_bot.bot.formatting import (
    TELEGRAM_MSG_LIMIT,
    markdown_to_telegram_html,
    split_html_message,
)

if TYPE_CHECKING:
    from aiogram import Bot
    from aiogram.types import Message

    from ductor_bot.config import StreamingConfig

logger = logging.getLogger(__name__)


@runtime_checkable
class StreamEditorProtocol(Protocol):
    """Interface shared by append-mode and edit-mode stream editors."""

    @property
    def has_content(self) -> bool: ...
    async def append_text(self, text: str) -> None: ...
    async def append_tool(self, tool_name: str) -> None:
        """Send a tool indicator as a new message."""
        name = tool_name
        desc = None
        if "|" in tool_name:
            name, desc = tool_name.split("|", 1)

        if desc:
            indicator = f"<b>[TOOL: {html.escape(name)}]</b> <i>{html.escape(desc)}</i>"
        else:
            indicator = f"<b>[TOOL: {html.escape(name)}]</b>"
        await self._send(indicator)

    async def append_system(self, text: str) -> None:
        """Send a system status indicator as a new message."""
        indicator = f"<i>[{html.escape(text)}]</i>"
        await self._send(indicator)

    async def finalize(self, full_text: str) -> None:
        """Attach button keyboard to the last sent message, if any."""
        if self._last_msg is None:
            return
        _, markup = extract_buttons(full_text)
        if markup is None:
            return
        try:
            await self._bot.edit_message_reply_markup(
                chat_id=self._chat_id,
                message_id=self._last_msg.message_id,
                reply_markup=markup,
            )
        except TelegramBadRequest:
            logger.warning("Failed to attach button keyboard")

    async def _send(
        self,
        text: str,
        *,
        raw_fallback: str = "",
        parse_mode: ParseMode | None = ParseMode.HTML,
    ) -> None:
        """Send a single message, using reply_to for the first one."""
        display = text[:TELEGRAM_MSG_LIMIT]
        if not display.strip():
            return

        try:
            if self._messages_sent == 0 and self._reply_to:
                msg = await self._reply_to.answer(display, parse_mode=parse_mode)
            else:
                msg = await self._bot.send_message(
                    chat_id=self._chat_id, text=display, parse_mode=parse_mode
                )
            self._last_msg = msg
            self._messages_sent += 1
        except TelegramBadRequest:
            if parse_mode is not None:
                logger.warning("HTML send failed, falling back to plain text")
                fallback = (raw_fallback or text)[:TELEGRAM_MSG_LIMIT]
                await self._send(fallback, parse_mode=None)
            else:
                logger.exception("Failed to send stream chunk even as plain text")


def create_stream_editor(
    bot: Bot,
    chat_id: int,
    *,
    reply_to: Message | None = None,
    cfg: StreamingConfig | None = None,
) -> StreamEditorProtocol:
    """Create the appropriate stream editor based on config."""
    from ductor_bot.config import StreamingConfig

    c = cfg or StreamingConfig()
    if c.append_mode:
        return StreamEditor(bot, chat_id, reply_to=reply_to)
    from ductor_bot.bot.edit_streaming import EditStreamEditor

    return EditStreamEditor(
        bot,
        chat_id,
        reply_to=reply_to,
        edit_interval_seconds=c.edit_interval_seconds,
        max_edit_failures=c.max_edit_failures,
    )
