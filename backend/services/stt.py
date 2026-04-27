"""
Speech-to-Text service using faster-whisper with Silero VAD.

Designed for voice assistant use: utterance-triggered (not live captioning).
VAD detects end-of-utterance, then transcribes the complete buffer.
"""
import asyncio
import logging
from dataclasses import dataclass
from functools import lru_cache

from faster_whisper import WhisperModel  # type: ignore[import-untyped]

from backend.monitoring.latency_tracker import track_latency

log = logging.getLogger(__name__)

# Confidence below this → treat language detection as uncertain
_LANG_CONFIDENCE_THRESHOLD = 0.70


@dataclass
class TranscribeResult:
    text: str
    language: str        # "en" | "ar" | other ISO 639-1
    confidence: float    # language detection confidence
    duration_ms: int


@lru_cache(maxsize=1)
def _load_model(model_size: str = "small") -> WhisperModel:
    """Load once, reuse forever. Called lazily on first transcription."""
    log.info("loading whisper model", extra={"model_size": model_size})
    model = WhisperModel(
        model_size,
        device="auto",           # uses Metal on Apple Silicon
        compute_type="int8",     # fastest on CPU/Metal
    )
    log.info("whisper model ready", extra={"model_size": model_size})
    return model


def preload(model_size: str | None = None) -> None:
    """Eagerly load Whisper at startup — eliminates cold-start on first utterance."""
    import os
    size = model_size or os.environ.get("HF_WHISPER_MODEL_SIZE", "small")
    _load_model(size)


def _transcribe_sync(
    audio_bytes: bytes,
    model_size: str,
) -> TranscribeResult:
    """Blocking transcription — run in thread pool."""
    import io
    import time

    import soundfile as sf  # type: ignore[import-untyped]

    t0 = time.perf_counter()
    model = _load_model(model_size)

    # Decode audio bytes to float32 numpy array
    audio_io = io.BytesIO(audio_bytes)
    audio_np, _ = sf.read(audio_io, dtype="float32")

    # HF_WHISPER_LANGUAGE=ar forces Arabic (faster + more accurate for AR-only demos)
    # Leave unset for bilingual auto-detect
    import os as _os
    lang_hint = _os.environ.get("HF_WHISPER_LANGUAGE") or None

    segments, info = model.transcribe(
        audio_np,
        beam_size=2,             # lower latency; minimal accuracy impact for short utterances
        language=lang_hint,      # None = auto-detect, "ar" = force Arabic, "en" = force English
        vad_filter=True,         # Silero VAD to strip silence
        vad_parameters={
            "min_silence_duration_ms": 300,
            "speech_pad_ms": 100,
        },
    )

    text = " ".join(seg.text.strip() for seg in segments).strip()
    duration_ms = int((time.perf_counter() - t0) * 1000)

    return TranscribeResult(
        text=text,
        language=info.language,
        confidence=info.language_probability,
        duration_ms=duration_ms,
    )


@track_latency("stt")
async def transcribe(
    audio_bytes: bytes,
    *,
    model_size: str = "small",
    language: str | None = None,
) -> TranscribeResult:
    """
    Transcribe a complete utterance.
    audio_bytes: raw PCM WAV bytes at 16kHz mono.
    Returns TranscribeResult with text, detected language, and confidence.
    """
    import os

    size = os.environ.get("HF_WHISPER_MODEL_SIZE", model_size)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, _transcribe_sync, audio_bytes, size
    )

    if result.confidence < _LANG_CONFIDENCE_THRESHOLD:
        log.warning(
            "low language confidence",
            extra={
                "language": result.language,
                "confidence": result.confidence,
                "text_preview": result.text[:60],
            },
        )

    log.info(
        "transcribed",
        extra={
            "language": result.language,
            "confidence": round(result.confidence, 3),
            "chars": len(result.text),
            "duration_ms": result.duration_ms,
        },
    )
    return result
