"""
Stub LLM for end-to-end pipeline testing without the Mac.

Returns a canned response that echoes the user's last message.
Drops in as a 1:1 replacement for MacLLMService — same frame contract,
no network. Swap it out when you're ready to test against the Mac.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService

log = logging.getLogger(__name__)


_CANNED_EN = "I heard you. The Mac LLM is offline for testing — this is a stub response."
_CANNED_AR = "سمعتك. الخادم متوقف للتجربة. هذا رد تجريبي."


class StubLLMService(LLMService):
    """Returns a fixed sentence, token-by-token, simulating a stream."""

    def __init__(self, *, token_delay_s: float = 0.03, **kwargs) -> None:
        super().__init__(**kwargs)
        self._token_delay = token_delay_s

    async def _fake_stream(self, messages: list[dict]) -> AsyncGenerator[str, None]:
        # Detect Arabic by inspecting last user message for Arabic chars.
        last_user = next(
            (m for m in reversed(messages) if m.get("role") == "user"),
            {},
        )
        content = str(last_user.get("content", ""))
        is_ar = any("؀" <= c <= "ۿ" for c in content)
        reply = _CANNED_AR if is_ar else _CANNED_EN

        log.info("stub_llm.replying", extra={"lang": "ar" if is_ar else "en"})

        # Stream word-by-word so downstream TTS sees realistic token cadence.
        for word in reply.split(" "):
            yield word + " "
            await asyncio.sleep(self._token_delay)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            messages = frame.context.get_messages()
            await self.push_frame(LLMFullResponseStartFrame(), direction)
            try:
                async for tok in self._fake_stream(messages):
                    await self.push_frame(LLMTextFrame(tok), direction)
            finally:
                await self.push_frame(LLMFullResponseEndFrame(), direction)
        else:
            await self.push_frame(frame, direction)
