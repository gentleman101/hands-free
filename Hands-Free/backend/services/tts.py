"""
TTS service: Fish Speech 1.5 via HTTP chunked streaming.

Apple-branch optimizations carried over:
- EXP-010: module-level singleton httpx.AsyncClient (saves 5-15ms per call by
  reusing the underlying connection pool)
- chunk_length=200 → small chunks → low TTFA
- Prosody tags ([laugh], [emphasis], etc.) passed through to Fish Speech
"""
from __future__ import annotations

import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import AsyncIterator

import httpx

from backend.database import fire_and_forget, insert_event
from backend.monitoring.latency_tracker import track_latency

log = logging.getLogger(__name__)


# Fish Speech S2 inline tag pattern — strip when falling back to non-Fish TTS
_TAG_RE = re.compile(r"\[([a-z_]+)\]", re.IGNORECASE)
_DEFAULT_SAMPLE_RATE = 44100  # Fish Speech firefly_gan_vq output rate


@dataclass
class SynthesizeResult:
    audio_bytes: bytes
    sample_rate: int
    ttfa_ms: int
    duration_ms: int


def _fish_url() -> str:
    return os.environ.get("HF_FISH_SPEECH_URL", "http://localhost:8001")


# ─── Singleton httpx client (EXP-010) ───────────────────────────────────────
_http_client: httpx.AsyncClient | None = None


def _tts_client() -> httpx.AsyncClient:
    """Return the module-level httpx.AsyncClient, lazy-init on first call."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=60.0,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
    return _http_client


async def aclose() -> None:
    """Close the httpx client (call on shutdown)."""
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


def strip_prosody_tags(text: str) -> str:
    """Remove Fish Speech inline tags. Use only when falling back to non-Fish TTS."""
    return _TAG_RE.sub("", text).strip()


# ─── Streaming synth ────────────────────────────────────────────────────────
async def synthesize_stream(
    text: str,
    language: str = "en",
    *,
    request_id: str | None = None,
) -> AsyncIterator[bytes]:
    """
    Stream audio chunks from Fish Speech as they are generated.

    First chunk arrives as soon as Fish Speech begins decoding.
    TTFA event is recorded on first chunk.
    """
    rid = request_id or str(uuid.uuid4())
    url = f"{_fish_url()}/v1/tts"
    t0 = time.perf_counter()
    ttfa_logged = False

    payload = {
        "text": text,
        "streaming": True,
        "format": "wav",
        "chunk_length": 200,           # smaller chunks = lower TTFA
        "normalize": True,
    }

    log.debug(
        "tts stream start",
        extra={"request_id": rid, "language": language, "chars": len(text)},
    )

    async with _tts_client().stream("POST", url, json=payload) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes(chunk_size=4096):
            if not chunk:
                continue
            if not ttfa_logged:
                ttfa_ms = int((time.perf_counter() - t0) * 1000)
                ttfa_logged = True
                log.info("tts ttfa", extra={"ttfa_ms": ttfa_ms, "request_id": rid})
                fire_and_forget(insert_event(
                    request_id=rid,
                    stage="tts_ttfa",
                    duration_ms=ttfa_ms,
                    ttfa_ms=ttfa_ms,
                    timestamp=datetime.now(UTC).isoformat(),
                    language=language,
                    pipeline_phase=os.environ.get("HF_PIPELINE_PHASE", "raw"),
                    success=True,
                ))
            yield chunk


# ─── Non-streaming convenience ──────────────────────────────────────────────
@track_latency("tts_total")
async def synthesize(
    text: str,
    language: str = "en",
    *,
    request_id: str | None = None,
) -> SynthesizeResult:
    """Collect full audio + timing. Use synthesize_stream in pipeline; this for tests."""
    rid = request_id or str(uuid.uuid4())
    t0 = time.perf_counter()
    chunks: list[bytes] = []
    ttfa_ms = 0
    first = True

    async for chunk in synthesize_stream(text, language, request_id=rid):
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


async def health_check(timeout_s: float = 2.0) -> bool:
    """Return True if Fish Speech sidecar is reachable."""
    try:
        resp = await _tts_client().get(f"{_fish_url()}/v1/health", timeout=timeout_s)
        return resp.status_code == 200
    except Exception:
        return False
