"""
BotSpeakingGate — drops mic audio while the bot is speaking.

Prevents SILMA's playback from looping back through the iPhone mic,
causing false VAD triggers and "Thank you" hallucinations.

Place between transport.input() and VADProcessor:

    Pipeline([
        transport.input(),
        BotSpeakingGate(cooldown_secs=0.4),
        vad_processor,
        ...
    ])

How it works:
- BotStartedSpeakingFrame flows UPSTREAM from the output transport.
  The gate catches it and sets _muted = True.
- BotStoppedSpeakingFrame flows UPSTREAM; gate clears mute + starts
  a short cooldown to absorb any remaining echo tail.
- UserAudioRawFrame (mic audio) flows DOWNSTREAM; gate drops it
  while muted so VAD never fires on the bot's own voice.
"""
from __future__ import annotations

import asyncio
import time

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    UserAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class BotSpeakingGate(FrameProcessor):
    """Suppress mic input while the bot is speaking + a short cooldown after."""

    def __init__(self, cooldown_secs: float = 0.4):
        super().__init__()
        self._muted = False
        self._cooldown_secs = cooldown_secs
        self._unmute_time: float = 0.0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        # BotStartedSpeakingFrame travels UPSTREAM — mute mic
        if isinstance(frame, BotStartedSpeakingFrame):
            self._muted = True
            self._unmute_time = 0.0

        # BotStoppedSpeakingFrame travels UPSTREAM — start cooldown
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._muted = False
            self._unmute_time = time.monotonic() + self._cooldown_secs

        # Drop mic audio while muted or in cooldown
        elif (
            isinstance(frame, UserAudioRawFrame)
            and direction == FrameDirection.DOWNSTREAM
            and (self._muted or time.monotonic() < self._unmute_time)
        ):
            return  # drop — do NOT push

        await self.push_frame(frame, direction)
