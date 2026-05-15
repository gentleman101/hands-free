"""
Pipeline orchestrator: STT → intent → filler → LLM → SentenceBuffer → TTS.

Flow per turn:
  1. STT transcribes utterance
  2. Intent classifier picks filler variant
  3. Filler WAV bytes yielded immediately (from memory, ~instant) — user hears
     something natural while LLM + TTS work in parallel
  4. LLM streams tokens → SentenceBuffer emits sentences → TTS synthesizes
  5. Real TTS audio chunks yielded

Apple-branch carry-overs:
- EXP-005: SentenceBuffer min_chars=8 + comma boundary → 200-500ms TTFA gain
- pipeline_ttfa_ms is the PRIMARY METRIC (filler hides it perceptually)
"""
from __future__ import annotations

import asyncio
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
from backend.services.filler import pick_filler, pick_filler_semantic, stream_filler
from backend.services.intent_classifier import classify
from backend.services.tts_orchestrator import TTSOrchestrator

log = logging.getLogger(__name__)


# EXP-005: comma boundary + min_chars=8 ships first clause to TTS earlier.
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
    intent: str = ""
    response_text: str = ""
    latency: PipelineLatency = field(default_factory=PipelineLatency)


# ─── SentenceBuffer ─────────────────────────────────────────────────────────
class SentenceBuffer:
    """
    Accumulate LLM token stream and emit complete sentences.
    Emits on a boundary char after _MIN_SENTENCE_CHARS accumulated.
    Call flush() to drain remaining text at end of stream.
    """

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, token: str) -> str | None:
        self._buf += token
        if len(self._buf) < _MIN_SENTENCE_CHARS:
            return None
        for i, ch in enumerate(self._buf):
            if ch in _BOUNDARIES:
                sentence = self._buf[: i + 1].strip()
                self._buf = self._buf[i + 1 :]
                return sentence if sentence else None
        return None

    def flush(self) -> str | None:
        remaining = self._buf.strip()
        self._buf = ""
        return remaining if remaining else None


# ─── Realtime pipeline ──────────────────────────────────────────────────────
async def run_realtime(
    utterance_bytes: bytes,
    *,
    request_id: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    out: PipelineResult | None = None,
) -> AsyncIterator[bytes]:
    """
    Run a full realtime pipeline turn. Yields audio bytes.

    First yields filler WAV bytes (from memory, near-instant) so the caller
    can begin playback immediately. Then yields real TTS audio as it streams.
    """
    from backend.services import stt as stt_svc  # lazy: avoids soundfile at import time

    rid = request_id or str(uuid.uuid4())
    t_pipeline_start = time.perf_counter()

    # ── 1. STT ───────────────────────────────────────────────────────────────
    transcript = await stt_svc.transcribe(utterance_bytes, request_id=rid)
    if out is not None:
        out.transcription = transcript.text
        out.language = transcript.language
        out.latency.stt_ms = transcript.duration_ms

    if not transcript.text.strip():
        log.debug("empty transcription — skipping pipeline", extra={"request_id": rid})
        return

    # ── 2. Intent classification ──────────────────────────────────────────────
    intent = classify(transcript.text)
    if out is not None:
        out.intent = intent
    log.debug("intent", extra={"intent": intent, "request_id": rid})

    # ── 3. Filler — semantic match to transcription (not just intent bucket) ──
    # pick_filler_semantic embeds the transcription (~3-5 ms CPU) and cosine-sims
    # against all pre-embedded filler texts, picking from the top-3 closest.
    # Falls back to TF-IDF intent bucket if sentence-transformers unavailable.
    filler_data, filler_text = pick_filler_semantic(
        transcript.text, transcript.language, fallback_intent=intent
    )
    if filler_data:
        chunk_size = 4096
        for i in range(0, len(filler_data), chunk_size):
            yield filler_data[i : i + chunk_size]

    # ── 4. LLM stream — seed with filler text so reply stitches naturally ────
    messages = prompt_builder.build_messages(
        transcript.text,
        language=transcript.language,
        conversation_history=conversation_history,
        filler_text=filler_text or None,
    )

    buf = SentenceBuffer()
    response_chunks: list[str] = []
    pipeline_ttfa_recorded = False

    t_llm = time.perf_counter()
    llm_first = True
    llm_ttft_ms = 0

    async def _tts_and_yield(sentence: str):
        """Synthesize a sentence and yield chunks, recording TTFA on first chunk."""
        nonlocal pipeline_ttfa_recorded
        async for audio_chunk in tts_svc.synthesize_stream(
            sentence, transcript.language, request_id=rid
        ):
            if not pipeline_ttfa_recorded:
                pipeline_ttfa = int((time.perf_counter() - t_pipeline_start) * 1000)
                pipeline_ttfa_recorded = True
                if out is not None:
                    out.latency.pipeline_ttfa_ms = pipeline_ttfa
                    out.latency.llm_ttft_ms = llm_ttft_ms
                log.info(
                    "pipeline ttfa",
                    extra={"pipeline_ttfa_ms": pipeline_ttfa, "request_id": rid},
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

    # Sequential per-sentence TTS — wait for each sentence to finish before next
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
        async for audio_chunk in _tts_and_yield(sentence):
            yield audio_chunk

    # Flush tail
    llm_total_ms = int((time.perf_counter() - t_llm) * 1000)
    remaining = buf.flush()
    if remaining:
        response_chunks.append(remaining)
        async for audio_chunk in _tts_and_yield(remaining):
            yield audio_chunk

    if out is not None:
        out.response_text = "".join(response_chunks).strip()
        out.latency.llm_total_ms = llm_total_ms
        out.latency.total_ms = int((time.perf_counter() - t_pipeline_start) * 1000)
