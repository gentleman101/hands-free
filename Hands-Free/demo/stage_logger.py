"""
StageLogger — timestamped pipeline diagnostics.

Place TWO instances in the pipeline:
  - early (after transport.input) → catches VAD / speech events
  - late  (end of pipeline)       → catches LLM / TTS events

Logs enough to answer:
  1. TTFA  (end-of-speech → first TTSStartedFrame)
  2. Why flicker / interruptions happen mid-sentence
  3. Why sentences stop early
  4. Whether quality issues are STT, LLM, or TTS
"""
from __future__ import annotations

import time
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    InterruptionTaskFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


def _ts() -> str:
    return time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"


class StageLogger(FrameProcessor):
    """Timestamped stage diagnostics."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._speech_end_t: float | None = None
        self._llm_start_t: float | None = None
        self._tts_first_t: float | None = None
        self._llm_buf: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        now = time.monotonic()

        # ── VAD / speech ────────────────────────────────────────────────────
        if isinstance(frame, (UserStartedSpeakingFrame, VADUserStartedSpeakingFrame)):
            print(f"\n[{_ts()}] 🗣️  speech start", flush=True)

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # Stamp from VAD stop — this is the true end of speech
            self._speech_end_t = now
            self._tts_first_t = None
            self._llm_buf = []
            print(f"[{_ts()}] ↳ VAD speech end  ← latency clock starts", flush=True)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            # Don't overwrite _speech_end_t — just log the turn-strategy firing time
            gap = int((now - self._speech_end_t) * 1000) if self._speech_end_t else "?"
            print(f"[{_ts()}] ↳ turn end (+{gap}ms after VAD)", flush=True)

        # ── STT ─────────────────────────────────────────────────────────────
        elif isinstance(frame, TranscriptionFrame):
            lang = getattr(frame.language, "name", str(frame.language)) if frame.language else "?"
            stt_ms = int((now - self._speech_end_t) * 1000) if self._speech_end_t else "?"
            print(f"[{_ts()}] 📝 STT +{stt_ms}ms [{lang.lower()[:2]}] \"{frame.text}\"", flush=True)

        # ── LLM ─────────────────────────────────────────────────────────────
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._llm_start_t = now
            llm_ms = int((now - self._speech_end_t) * 1000) if self._speech_end_t else "?"
            print(f"[{_ts()}] 🤖 LLM start +{llm_ms}ms from speech end", flush=True)

        elif isinstance(frame, TextFrame) and frame.text:
            self._llm_buf.append(frame.text)

        elif isinstance(frame, LLMFullResponseEndFrame):
            response = "".join(self._llm_buf).strip()
            llm_dur = int((now - self._llm_start_t) * 1000) if self._llm_start_t else "?"
            print(f"[{_ts()}] ↳ LLM done ({llm_dur}ms) → \"{response[:120]}\"", flush=True)

        # ── TTS ─────────────────────────────────────────────────────────────
        elif isinstance(frame, TTSStartedFrame):
            if self._tts_first_t is None:
                self._tts_first_t = now
                ttfa = int((now - self._speech_end_t) * 1000) if self._speech_end_t else "?"
                print(f"[{_ts()}] 🔊 TTS first chunk  ← TTFA: {ttfa}ms", flush=True)
            else:
                print(f"[{_ts()}] 🔊 TTS next chunk", flush=True)

        elif isinstance(frame, TTSStoppedFrame):
            print(f"[{_ts()}] ↳ TTS chunk done", flush=True)

        elif isinstance(frame, (BotStartedSpeakingFrame,)):
            real_ttfa = int((now - self._speech_end_t) * 1000) if self._speech_end_t else "?"
            synth_gap = int((now - self._tts_first_t) * 1000) if self._tts_first_t else "?"
            print(f"[{_ts()}] 📢 bot speaking  ← REAL TTFA: {real_ttfa}ms  (WebRTC buffer gap: {synth_gap}ms)", flush=True)

        elif isinstance(frame, (BotStoppedSpeakingFrame,)):
            print(f"[{_ts()}] ↳ bot stopped speaking", flush=True)

        # ── Interruptions (issues 2 & 3) ────────────────────────────────────
        elif isinstance(frame, InterruptionFrame):
            print(f"[{_ts()}] ⚡ INTERRUPTION ← VAD fired while bot speaking (echo/barge-in)", flush=True)

        elif isinstance(frame, InterruptionTaskFrame):
            print(f"[{_ts()}] ⚡ INTERRUPTION TASK ← pipeline task interrupted", flush=True)

        # ── Errors ──────────────────────────────────────────────────────────
        elif isinstance(frame, ErrorFrame):
            print(f"[{_ts()}] ❌ ERROR: {frame.error}", flush=True)

        await self.push_frame(frame, direction)
