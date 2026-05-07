"""
Pipeline orchestrator: STT → LLM (remote) → SentenceBuffer → TTS, with filler audio.

Apple-branch carry-overs:
- EXP-005: SentenceBuffer with _MIN_SENTENCE_CHARS=8 + comma boundary →
  ships first clause to TTS earlier (saves 200-500ms TTFA)
- pipeline_ttfa_ms is the PRIMARY METRIC

New for distributed setup:
- LLM is called via llm_client.py (HTTP RPC to Mac)
- Filler audio plays during LLM wait, cancelled on first TTS chunk
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import AsyncIterator

from backend.database import fire_and_forget, insert_event
from backend.services import llm_client, prompt_builder
from backend.services import tts as tts_svc
from backend.services.filler import FillerPlayer

log = logging.getLogger(__name__)


# EXP-005: comma included so first clause ships to TTS earlier.
# Min chars 8 (was 15) — Fish Speech first-chunk latency dominates pipeline TTFA,
# so shorter first emit-to-TTS = faster pipeline_ttfa.
_BOUNDARIES = frozenset(".?!\n،؟！,")
_MIN_SENTENCE_CHARS = 8


@dataclass
class PipelineLatency:
    stt_ms: int = 0
    llm_ttft_ms: int = 0
    llm_total_ms: int = 0
    tts_ttfa_ms: int = 0
    tts_total_ms: int = 0
    pipeline_ttfa_ms: int = 0   # PRIMARY METRIC
    total_ms: int = 0


@dataclass
class PipelineResult:
    transcription: str = ""
    language: str = ""
    response_text: str = ""
    latency: PipelineLatency = field(default_factory=PipelineLatency)


# ─── SentenceBuffer ─────────────────────────────────────────────────────────
class SentenceBuffer:
    """
    Accumulate LLM token stream and emit complete sentences.
    Emits on a boundary char after _MIN_SENTENCE_CHARS.
    Call flush() to drain remaining buffered text at end of stream.
    """

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, token: str) -> str | None:
        """Add a token. Return a complete sentence if a boundary is detected, else None."""
        self._buf += token
        if len(self._buf) < _MIN_SENTENCE_CHARS:
            return None
        for i, ch in enumerate(self._buf):
            if ch in _BOUNDARIES:
                sentence = self._buf[: i + 1].strip()
                self._buf = self._buf[i + 1 :]
                if sentence:
                    return sentence
        return None

    def flush(self) -> str | None:
        """Return any remaining buffered text (end-of-stream)."""
        remaining = self._buf.strip()
        self._buf = ""
        return remaining if remaining else None


# ─── Realtime pipeline ──────────────────────────────────────────────────────
async def run_realtime(
    utterance_bytes: bytes,
    *,
    request_id: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    filler: FillerPlayer | None = None,
    out: PipelineResult | None = None,
) -> AsyncIterator[bytes]:
    """
    Run a full realtime pipeline turn:
      audio bytes → STT → LLM stream (remote) → SentenceBuffer → TTS stream → audio bytes.

    Yields TTS audio chunks as Fish Speech generates them. First audio chunk
    arrives as soon as the first sentence is synthesized.

    If `filler` is provided, plays a filler clip during the LLM wait and
    cancels it the moment the first real TTS audio chunk is available.

    Populates `out` (if given) with transcription, language, and final response.
    """
    from backend.services import stt as stt_svc  # lazy: avoids soundfile at import time

    rid = request_id or str(uuid.uuid4())
    t_pipeline_start = time.perf_counter()

    # 1. STT
    transcript = await stt_svc.transcribe(utterance_bytes, request_id=rid)
    if out is not None:
        out.transcription = transcript.text
        out.language = transcript.language
        out.latency.stt_ms = transcript.duration_ms

    if not transcript.text.strip():
        log.debug("empty transcription — skipping pipeline", extra={"request_id": rid})
        return

    # 2. Start filler audio (covers LLM latency)
    if filler is not None and filler.has_fillers(transcript.language):
        await filler.start(transcript.language)

    # 3. Build prompt + stream LLM tokens (remote on Mac)
    messages = prompt_builder.build_messages(
        transcript.text,
        language=transcript.language,
        conversation_history=conversation_history,
    )

    buf = SentenceBuffer()
    response_chunks: list[str] = []
    pipeline_ttfa_recorded = False
    first_tts_started = False

    t_llm = time.perf_counter()
    llm_first = True
    llm_ttft_ms = 0

    async for token in llm_client.generate_stream(
        messages, language=transcript.language, request_id=rid
    ):
        if llm_first:
            llm_ttft_ms = int((time.perf_counter() - t_llm) * 1000)
            llm_first = False

        response_chunks.append(token)
        sentence = buf.feed(token)
        if sentence is None:
            continue

        # First complete sentence → start TTS streaming
        async for audio_chunk in tts_svc.synthesize_stream(
            sentence, transcript.language, request_id=rid
        ):
            # On the very first audio chunk, stop the filler and record pipeline_ttfa
            if not first_tts_started:
                first_tts_started = True
                if filler is not None:
                    await filler.stop()

            if not pipeline_ttfa_recorded:
                pipeline_ttfa = int((time.perf_counter() - t_pipeline_start) * 1000)
                pipeline_ttfa_recorded = True
                if out is not None:
                    out.latency.pipeline_ttfa_ms = pipeline_ttfa
                    out.latency.llm_ttft_ms = llm_ttft_ms
                log.info(
                    "pipeline ttfa",
                    extra={
                        "pipeline_ttfa_ms": pipeline_ttfa,
                        "request_id": rid,
                    },
                )
                fire_and_forget(insert_event(
                    request_id=rid,
                    stage="pipeline_ttfa",
                    duration_ms=pipeline_ttfa,
                    ttfa_ms=pipeline_ttfa,
                    timestamp=datetime.now(UTC).isoformat(),
                    language=transcript.language,
                    pipeline_phase=os.environ.get("HF_PIPELINE_PHASE", "raw"),
                    success=True,
                ))

            yield audio_chunk

    # LLM stream done → flush buffer + synthesize tail
    llm_total_ms = int((time.perf_counter() - t_llm) * 1000)
    remaining = buf.flush()
    if remaining:
        response_chunks.append(remaining)
        async for audio_chunk in tts_svc.synthesize_stream(
            remaining, transcript.language, request_id=rid
        ):
            if not first_tts_started:
                first_tts_started = True
                if filler is not None:
                    await filler.stop()
            yield audio_chunk

    # If we never produced any TTS at all (empty LLM response), at least stop the filler
    if not first_tts_started and filler is not None:
        await filler.stop()

    if out is not None:
        out.response_text = "".join(response_chunks).strip()
        out.latency.llm_total_ms = llm_total_ms
        out.latency.total_ms = int((time.perf_counter() - t_pipeline_start) * 1000)
