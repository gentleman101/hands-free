"""
RammasSTTService — Pipecat WhisperSTTService with EN/AR auto-detect.

Pipecat's stack handles the heavy lifting:
- RNNoise filter (in the transport) → noise suppression before VAD
- Silero VAD (in VADProcessor)      → speech detection
- Smart Turn v3                     → semantic endpointing
- WhisperSTTService.no_speech_prob  → built-in hallucination filter

The one thing Pipecat doesn't do is *language* auto-detection per utterance —
it transcribes in the fixed language from settings. We override run_stt
just to pass language=None to faster-whisper and surface info.language
on the emitted TranscriptionFrame for LanguageRouter to consume.

That's it. No custom RMS / logprob / compression / hallucination / script gates —
those duplicated work Pipecat (or RNNoise + VAD + Whisper internals) already does.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator

import numpy as np
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

log = logging.getLogger(__name__)


class RammasSTTService(WhisperSTTService):
    """WhisperSTTService with EN/AR auto-detection. Everything else is Pipecat default."""

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        print(f"[STT] run_stt audio={len(audio)} bytes", flush=True)
        if not self._model:
            yield ErrorFrame("Whisper model not available")
            return

        await self.start_processing_metrics()

        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        # language=None → faster-whisper auto-detects. info.language exposed
        # so we can route per-utterance.
        try:
            segments_iter, info = await asyncio.to_thread(
                self._model.transcribe,
                audio_float,
                language=None,
                beam_size=1,                       # EXP-009: greedy, 80-200ms faster
                vad_filter=False,                  # EXP-008: Pipecat already ran VAD
                condition_on_previous_text=False,  # prevents hypothesis drift
            )
            segments = list(segments_iter)
        except Exception as e:
            await self.stop_processing_metrics()
            yield ErrorFrame(f"Whisper transcribe failed: {e}")
            return

        await self.stop_processing_metrics()

        # Reject anything that's not EN or AR (rest of pipeline assumes one of these)
        if info.language not in ("en", "ar"):
            log.debug("stt: filtered — non-EN/AR", extra={"lang": info.language})
            return

        # Trust Pipecat's built-in no_speech_prob gate
        no_speech_threshold = 0.4  # WhisperSTTService default
        text = ""
        for segment in segments:
            if getattr(segment, "no_speech_prob", 0.0) < no_speech_threshold:
                text += segment.text + " "
        text = text.strip()

        if not text:
            return

        lang_enum = Language.AR if info.language == "ar" else Language.EN
        logger.debug(f"Transcription: [{text}] ({info.language})")
        yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), lang_enum)
