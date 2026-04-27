"""
Pipeline orchestrator: STT → LLM → TTS with sentence-level streaming.

SentenceBuffer accumulates LLM tokens and emits complete sentences to TTS
immediately — this is the key to "live person" feel.
First audio chunk plays while LLM is still generating the rest.

Target pipeline_ttfa < 800ms after utterance end.
"""
import logging
import time
import uuid
from dataclasses import dataclass
from typing import AsyncIterator

from backend.monitoring.latency_tracker import track_latency
from backend.services import llm as llm_svc
from backend.services import prompt_builder
from backend.services import stt as stt_svc
from backend.services import tts as tts_svc
log = logging.getLogger(__name__)

# Arabic + English sentence boundaries
_BOUNDARIES = frozenset(".?!\n،؟！")
# Minimum chars before considering a boundary (avoids splitting "e.g.")
_MIN_SENTENCE_CHARS = 15


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
    transcription: str
    language: str
    response_text: str
    audio_bytes: bytes
    latency: PipelineLatency


class SentenceBuffer:
    """
    Accumulates LLM token stream and emits complete sentences.
    Emits on sentence boundary chars after _MIN_SENTENCE_CHARS.
    Call flush() at end to emit any remaining text.
    """

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, token: str) -> str | None:
        """
        Add a token. Returns a complete sentence if boundary detected, else None.
        """
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
        """Return any remaining buffered text (end of LLM response)."""
        remaining = self._buf.strip()
        self._buf = ""
        return remaining if remaining else None


async def run_realtime(
    utterance_bytes: bytes,
    *,
    request_id: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> AsyncIterator[bytes]:
    """
    Full pipelined run: STT → LLM stream → SentenceBuffer → TTS stream.
    Yields audio bytes as Fish Speech generates them.
    First audio chunk arrives as soon as the first sentence is synthesised.
    """
    rid = request_id or str(uuid.uuid4())
    t_start = time.perf_counter()
    ttfa_recorded = False

    # Layer 3: STT
    transcript = await stt_svc.transcribe(utterance_bytes, request_id=rid)
    stt_ms = int((time.perf_counter() - t_start) * 1000)
    log.info(
        "pipeline stt done",
        extra={"stt_ms": stt_ms, "language": transcript.language, "request_id": rid},
    )

    # Layer 4: LLM streaming + SentenceBuffer
    messages = prompt_builder.build_messages(
        transcript.text, language=transcript.language, conversation_history=conversation_history
    )

    buf = SentenceBuffer()

    async for token in llm_svc.generate_stream(messages, request_id=rid):
        sentence = buf.feed(token)
        if sentence:
            log.debug("sentence emitted to tts", extra={"chars": len(sentence), "request_id": rid})
            # Layer 5: TTS streaming — start immediately on each sentence
            async for audio_chunk in tts_svc.synthesize_stream(
                sentence, transcript.language, request_id=rid
            ):
                if not ttfa_recorded:
                    pipeline_ttfa = int((time.perf_counter() - t_start) * 1000)
                    ttfa_recorded = True
                    from backend.database import insert_event
                    from datetime import UTC, datetime
                    import os

                    await insert_event(
                        request_id=rid,
                        stage="pipeline_ttfa",
                        duration_ms=pipeline_ttfa,
                        ttfa_ms=pipeline_ttfa,
                        timestamp=datetime.now(UTC).isoformat(),
                        language=transcript.language,
                        pipeline_phase=os.environ.get("HF_PIPELINE_PHASE", "raw"),
                        success=True,
                    )
                    log.info(
                        "pipeline ttfa",
                        extra={"pipeline_ttfa_ms": pipeline_ttfa, "request_id": rid},
                    )
                yield audio_chunk

    # Flush remaining buffer
    remaining = buf.flush()
    if remaining:
        async for audio_chunk in tts_svc.synthesize_stream(
            remaining, transcript.language, request_id=rid
        ):
            yield audio_chunk


@track_latency("pipeline_total")
async def run(
    utterance_bytes: bytes,
    *,
    language: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> PipelineResult:
    """
    Non-streaming pipeline run. Collects full audio + all latency stats.
    Use run_realtime for actual pipeline; use this for testing/benchmarking.
    """
    rid = str(uuid.uuid4())
    t0 = time.perf_counter()

    # STT
    t_stt = time.perf_counter()
    transcript = await stt_svc.transcribe(utterance_bytes, request_id=rid)
    stt_ms = int((time.perf_counter() - t_stt) * 1000)

    # Skip pipeline if STT returned nothing (noise/silence)
    if not transcript.text.strip():
        log.warning("empty transcription — skipping pipeline", extra={"request_id": rid})
        return PipelineResult(
            transcription="",
            language=transcript.language,
            response_text="",
            audio_bytes=b"",
            latency=PipelineLatency(stt_ms=stt_ms),
        )

    # LLM (non-streaming for simplicity in non-realtime path)
    messages = prompt_builder.build_messages(
        transcript.text,
        language=transcript.language,
        conversation_history=conversation_history,
    )
    t_llm = time.perf_counter()
    llm_result = await llm_svc.generate(messages, language=transcript.language)
    llm_total_ms = int((time.perf_counter() - t_llm) * 1000)

    # TTS
    t_tts = time.perf_counter()
    tts_result = await tts_svc.synthesize(llm_result.text, transcript.language)
    tts_total_ms = int((time.perf_counter() - t_tts) * 1000)

    total_ms = int((time.perf_counter() - t0) * 1000)

    latency = PipelineLatency(
        stt_ms=stt_ms,
        llm_ttft_ms=llm_result.ttft_ms,
        llm_total_ms=llm_total_ms,
        tts_ttfa_ms=tts_result.ttfa_ms,
        tts_total_ms=tts_total_ms,
        pipeline_ttfa_ms=stt_ms + llm_result.ttft_ms + tts_result.ttfa_ms,
        total_ms=total_ms,
    )

    log.info(
        "pipeline complete",
        extra={
            "stt_ms": stt_ms,
            "llm_ttft_ms": llm_result.ttft_ms,
            "llm_total_ms": llm_total_ms,
            "tts_ttfa_ms": tts_result.ttfa_ms,
            "pipeline_ttfa_ms": latency.pipeline_ttfa_ms,
            "total_ms": total_ms,
            "language": transcript.language,
            "request_id": rid,
        },
    )

    return PipelineResult(
        transcription=transcript.text,
        language=transcript.language,
        response_text=llm_result.text,
        audio_bytes=tts_result.audio_bytes,
        latency=latency,
    )
