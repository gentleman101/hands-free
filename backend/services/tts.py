"""
TTS service using Fish Speech 1.5 with HTTP chunked streaming.

Fish Speech runs as a sidecar server (fish-speech serve --port 8001).
This client calls it via HTTP, streaming audio chunks as they arrive.

For TTS without voice cloning: only the LM + DAC decoder runs.
The voice cloning codec encoder is NOT used here (Phase 6+).

Prosody tags ([laugh], [emphasis], etc.) are passed through to Fish Speech
which reads them natively. Tags are stripped only if falling back to a
non-Fish TTS in future.
"""
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import AsyncIterator

import httpx

from backend.monitoring.latency_tracker import track_latency

log = logging.getLogger(__name__)


# Fish Speech S2 inline tag pattern — strip before non-Fish fallback
_TAG_RE = re.compile(r"\[([a-z_]+)\]", re.IGNORECASE)

_DEFAULT_SAMPLE_RATE = 44100  # Fish Speech firefly_gan_vq outputs at 44100Hz


@dataclass
class SynthesizeResult:
    audio_bytes: bytes
    sample_rate: int
    ttfa_ms: int      # time-to-first-audio-chunk
    duration_ms: int  # total synthesis time


def _fish_url() -> str:
    return os.environ.get("HF_FISH_SPEECH_URL", "http://localhost:8001")


def strip_prosody_tags(text: str) -> str:
    """Remove Fish Speech inline tags. Use only when falling back to non-Fish TTS."""
    return _TAG_RE.sub("", text).strip()


async def synthesize_stream(
    text: str,
    language: str = "en",
    *,
    request_id: str | None = None,
) -> AsyncIterator[bytes]:
    """
    Stream audio chunks from Fish Speech as they are generated.
    First chunk arrives as soon as Fish Speech begins decoding.
    Yields raw PCM bytes at _DEFAULT_SAMPLE_RATE Hz.
    TTFA is logged on first chunk.
    """
    rid = request_id or str(uuid.uuid4())
    url = f"{_fish_url()}/v1/tts"
    t0 = time.perf_counter()
    ttfa_logged = False

    payload = {
        "text": text,
        "streaming": True,
        "format": "wav",
        "chunk_length": 200,
        "normalize": True,
    }

    log.debug("tts stream start", extra={"request_id": rid, "language": language, "chars": len(text)})

    async with httpx.AsyncClient(timeout=60.0).stream("POST", url, json=payload) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes(chunk_size=4096):
            if chunk:
                if not ttfa_logged:
                    ttfa_ms = int((time.perf_counter() - t0) * 1000)
                    ttfa_logged = True
                    from backend.database import insert_event
                    from datetime import UTC, datetime

                    await insert_event(
                        request_id=rid,
                        stage="tts_ttfa",
                        duration_ms=ttfa_ms,
                        ttfa_ms=ttfa_ms,
                        timestamp=datetime.now(UTC).isoformat(),
                        language=language,
                        pipeline_phase=os.environ.get("HF_PIPELINE_PHASE", "raw"),
                        success=True,
                    )
                    log.info("tts ttfa", extra={"ttfa_ms": ttfa_ms, "request_id": rid})
                yield chunk


@track_latency("tts_total")
async def synthesize(
    text: str,
    language: str = "en",
) -> SynthesizeResult:
    """
    Non-streaming synthesis. Collects full audio + timing.
    Use synthesize_stream for pipeline; use this for testing/benchmarking.
    """
    t0 = time.perf_counter()
    chunks: list[bytes] = []
    ttfa_ms = 0
    first = True

    async for chunk in synthesize_stream(text, language):
        if first:
            ttfa_ms = int((time.perf_counter() - t0) * 1000)
            first = False
        chunks.append(chunk)

    audio_bytes = b"".join(chunks)
    duration_ms = int((time.perf_counter() - t0) * 1000)

    log.info(
        "tts complete",
        extra={
            "bytes": len(audio_bytes),
            "ttfa_ms": ttfa_ms,
            "duration_ms": duration_ms,
            "language": language,
        },
    )
    return SynthesizeResult(
        audio_bytes=audio_bytes,
        sample_rate=_DEFAULT_SAMPLE_RATE,
        ttfa_ms=ttfa_ms,
        duration_ms=duration_ms,
    )
