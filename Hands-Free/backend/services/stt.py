"""
STT service using faster-whisper large-v3-turbo on CUDA (Windows).

Apple-branch innovations carried over:
- EXP-002: EN/AR language constraint via probability gates (no French/Portuguese leakage)
- Confidence gates: no_speech_prob, avg_logprob, compression_ratio
  → catches hallucinations + noise + low-confidence transcriptions
- _script_matches_language(): validates Arabic chars vs Latin chars match the
  detected language → catches third-language misclassifications
- EXP-008: NO Whisper internal vad_filter — audio_capture already does VAD
- EXP-009: beam_size=1 (greedy) — 80-200ms faster, negligible accuracy loss
  on short single-speaker utterances
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from backend.monitoring.latency_tracker import track_latency

log = logging.getLogger(__name__)


# ─── Tuning constants (from Apple stt.py) ───────────────────────────────────
_NO_SPEECH_THRESHOLD = 0.3            # discard if Whisper >30% says not speech
_MIN_AVG_LOGPROB = -1.0               # discard low-confidence transcriptions
_MAX_COMPRESSION_RATIO = 2.4          # hallucinations have repetitive tokens

# Whisper hallucinates these common YouTube-trained phrases on noise/silence.
# Drop them outright. Exact match after lower+strip-punctuation.
_HALLUCINATION_PHRASES = frozenset({
    "thank you", "thanks", "thanks for watching", "thank you for watching",
    "bye", "goodbye", "see you next time", "please subscribe",
    "you", ".", "...", "okay", "ok", "yeah", "uh", "um", "hmm",
    "i'm here to help you", "how can i help you",
    "شكراً", "شكرا", "مع السلامة",
})


# ─── Result type ────────────────────────────────────────────────────────────
@dataclass
class TranscribeResult:
    text: str
    language: str          # "en" | "ar" | "unknown"
    confidence: float      # 0.0 – 1.0
    duration_ms: int = 0


# ─── Script validation ──────────────────────────────────────────────────────
def _script_matches_language(text: str, language: str) -> bool:
    """
    Validate that the transcribed text's script matches the detected language.

    Catches false-positive language detections where a third language is
    misclassified as EN or AR and transcribed as garbage.

    - "ar": at least 30% of alphabetic characters must be Arabic script (U+0600–U+06FF)
    - "en": at least 50% of alphabetic characters must be ASCII Latin
    """
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return False

    if language == "ar":
        arabic = sum(1 for c in alpha_chars if "؀" <= c <= "ۿ")
        return (arabic / len(alpha_chars)) >= 0.30

    latin = sum(1 for c in alpha_chars if ord(c) < 128)
    return (latin / len(alpha_chars)) >= 0.50


# ─── Model loading (cached) ─────────────────────────────────────────────────
@lru_cache(maxsize=4)
def _get_model(model_size: str, device: str, compute_type: str):
    """Lazy-load and cache the faster-whisper model."""
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]
    except ImportError as e:
        raise RuntimeError(
            "faster-whisper not installed. Run: pip install faster-whisper"
        ) from e

    log.info(
        "loading whisper model",
        extra={"model": model_size, "device": device, "compute_type": compute_type},
    )
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def preload(model_size: str | None = None) -> None:
    """Eagerly warm up the Whisper model + run a 0.5s silence inference."""
    size = model_size or os.environ.get("HF_WHISPER_MODEL", "large-v3-turbo")
    device = os.environ.get("HF_WHISPER_DEVICE", "cuda")
    compute = os.environ.get("HF_WHISPER_COMPUTE_TYPE", "int8")
    model = _get_model(size, device, compute)
    # Warm with 0.5s silence
    silence = np.zeros(8000, dtype=np.float32)
    list(model.transcribe(silence, beam_size=1)[0])
    log.info("whisper ready", extra={"model": size})


# ─── Sync transcribe (runs in executor) ─────────────────────────────────────
def _transcribe_sync(audio_bytes: bytes, model_size: str) -> TranscribeResult:
    """Blocking transcription using faster-whisper. Returns confidence-filtered result."""
    import time

    device = os.environ.get("HF_WHISPER_DEVICE", "cuda")
    compute = os.environ.get("HF_WHISPER_COMPUTE_TYPE", "int8")
    model = _get_model(model_size, device, compute)

    t0 = time.perf_counter()

    import soundfile as sf  # type: ignore[import-untyped]  # lazy: not installed in test env

    # Read WAV bytes → float32 mono numpy
    audio_np, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    if sr != 16000:
        log.warning("audio not at 16kHz", extra={"sr": sr})

    # Energy gate: distinguish close-mic user speech from distant ambient
    # conversation that VAD misfires on. Real near-field speech RMS is typically
    # 0.05-0.3; office ambient murmur is 0.005-0.02.
    rms = float(np.sqrt(np.mean(audio_np ** 2)))
    _MIN_RMS = float(os.environ.get("HF_STT_MIN_RMS", "0.025"))
    if rms < _MIN_RMS:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        log.info(
            "stt: filtered — low energy (ambient noise)",
            extra={"rms": round(rms, 4), "min_rms": _MIN_RMS},
        )
        return TranscribeResult(text="", language="unknown", confidence=0.0, duration_ms=duration_ms)

    # Single-pass transcribe — Whisper detects language as part of decoding.
    # EXP-009: beam_size=1 (greedy)
    # vad_filter=True: trims silent regions before decoding → prevents "Thank you"
    # style hallucinations on noisy office audio. Worth the ~30-50ms cost.
    segments_iter, info = model.transcribe(
        audio_np,
        beam_size=1,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
    )
    segments = list(segments_iter)

    # Trust Whisper's detected language. Reject anything that's not en/ar.
    duration_ms = int((time.perf_counter() - t0) * 1000)
    if info.language not in ("en", "ar"):
        log.info("stt: filtered — non-EN/AR language", extra={"detected": info.language})
        return TranscribeResult(text="", language="unknown", confidence=0.0, duration_ms=duration_ms)
    detected = info.language

    duration_ms = int((time.perf_counter() - t0) * 1000)

    # Step 3: Confidence gates
    if not segments:
        return TranscribeResult(text="", language=detected, confidence=0.0, duration_ms=duration_ms)

    avg_no_speech = sum(getattr(s, "no_speech_prob", 0.0) for s in segments) / len(segments)
    avg_logprob = sum(getattr(s, "avg_logprob", 0.0) for s in segments) / len(segments)
    avg_compression = sum(getattr(s, "compression_ratio", 0.0) for s in segments) / len(segments)

    text_preview = " ".join(s.text.strip() for s in segments).strip()

    log.info(
        "stt scores",
        extra={
            "lang": detected,
            "rms": round(rms, 4),
            "no_speech_prob": round(avg_no_speech, 3),
            "avg_logprob": round(avg_logprob, 3),
            "compression_ratio": round(avg_compression, 3),
            "text": text_preview[:80],
        },
    )

    if avg_no_speech > _NO_SPEECH_THRESHOLD:
        log.info(
            "stt: filtered — no_speech",
            extra={"no_speech_prob": round(avg_no_speech, 3), "text": text_preview[:80]},
        )
        return TranscribeResult(text="", language=detected, confidence=0.0, duration_ms=duration_ms)

    if avg_compression > _MAX_COMPRESSION_RATIO:
        log.info(
            "stt: filtered — hallucination (compression)",
            extra={"compression": round(avg_compression, 3), "text": text_preview[:80]},
        )
        return TranscribeResult(text="", language=detected, confidence=0.0, duration_ms=duration_ms)

    if avg_logprob < _MIN_AVG_LOGPROB:
        log.info(
            "stt: filtered — low logprob",
            extra={"avg_logprob": round(avg_logprob, 3), "text": text_preview[:80]},
        )
        return TranscribeResult(text="", language=detected, confidence=0.0, duration_ms=duration_ms)

    text = text_preview

    # Step 4: Hallucination denylist (common YouTube-trained Whisper hallucinations)
    text_normalized = text.lower().strip(".,!?¿¡،؟ \t\n")
    if text_normalized in _HALLUCINATION_PHRASES:
        log.info("stt: filtered — hallucination phrase", extra={"text": text})
        return TranscribeResult(text="", language=detected, confidence=0.0, duration_ms=duration_ms)

    # Step 5: Script validation (catches third-language misclassification)
    if not _script_matches_language(text, detected):
        log.debug(
            "stt: skipping — script mismatch",
            extra={"detected": detected, "text_preview": text[:40]},
        )
        return TranscribeResult(text="", language=detected, confidence=0.0, duration_ms=duration_ms)

    # Confidence: rough 0–1 score from logprob (-1.0 → 0, 0.0 → 1)
    confidence = max(0.0, min(1.0, 1.0 + avg_logprob))

    return TranscribeResult(
        text=text,
        language=detected,
        confidence=round(confidence, 3),
        duration_ms=duration_ms,
    )


# ─── Async wrapper ──────────────────────────────────────────────────────────
@track_latency("stt")
async def transcribe(
    audio_bytes: bytes,
    *,
    model_size: str | None = None,
    request_id: str | None = None,
) -> TranscribeResult:
    """Transcribe WAV bytes. Runs blocking model in thread pool."""
    size = model_size or os.environ.get("HF_WHISPER_MODEL", "large-v3-turbo")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _transcribe_sync, audio_bytes, size)

    log.info(
        "transcribed",
        extra={
            "language": result.language,
            "confidence": result.confidence,
            "chars": len(result.text),
            "duration_ms": result.duration_ms,
        },
    )
    return result
