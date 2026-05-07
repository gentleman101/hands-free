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
import soundfile as sf  # type: ignore[import-untyped]

from backend.monitoring.latency_tracker import track_latency

log = logging.getLogger(__name__)


# ─── Tuning constants (from Apple stt.py) ───────────────────────────────────
_NO_SPEECH_THRESHOLD = 0.3            # discard if Whisper >30% says not speech
_MIN_AVG_LOGPROB = -1.0               # discard low-confidence transcriptions
_MAX_COMPRESSION_RATIO = 2.4          # hallucinations have repetitive tokens
_LANG_PROB_THRESHOLD = 0.30           # min EN-or-AR probability to keep result


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

    # Read WAV bytes → float32 mono numpy
    audio_np, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    if sr != 16000:
        log.warning("audio not at 16kHz", extra={"sr": sr})

    # Step 1: Detect language probabilities; force EN or AR (EXP-002)
    # detect_language returns (lang_code, lang_prob, all_probs)
    try:
        lang, lang_prob, lang_probs = model.detect_language(audio_np)
    except Exception:
        lang_probs = {}
        lang_prob = 0.0

    en_prob = float(lang_probs.get("en", 0.0)) if isinstance(lang_probs, dict) else 0.0
    ar_prob = float(lang_probs.get("ar", 0.0)) if isinstance(lang_probs, dict) else 0.0

    duration_ms = int((time.perf_counter() - t0) * 1000)

    if max(en_prob, ar_prob) < _LANG_PROB_THRESHOLD:
        log.debug(
            "stt: lang prob below threshold",
            extra={"en_prob": en_prob, "ar_prob": ar_prob},
        )
        return TranscribeResult(text="", language="unknown", confidence=0.0, duration_ms=duration_ms)

    detected = "en" if en_prob > ar_prob else "ar"

    # Step 2: Transcribe with forced language
    # EXP-008: NO vad_filter (audio_capture already does it)
    # EXP-009: beam_size=1 (greedy)
    segments_iter, info = model.transcribe(
        audio_np,
        language=detected,
        beam_size=1,
    )
    segments = list(segments_iter)

    duration_ms = int((time.perf_counter() - t0) * 1000)

    # Step 3: Confidence gates
    if not segments:
        return TranscribeResult(text="", language=detected, confidence=0.0, duration_ms=duration_ms)

    avg_no_speech = sum(getattr(s, "no_speech_prob", 0.0) for s in segments) / len(segments)
    avg_logprob = sum(getattr(s, "avg_logprob", 0.0) for s in segments) / len(segments)
    avg_compression = sum(getattr(s, "compression_ratio", 0.0) for s in segments) / len(segments)

    log.debug(
        "stt confidence",
        extra={
            "lang": detected,
            "no_speech_prob": round(avg_no_speech, 3),
            "avg_logprob": round(avg_logprob, 3),
            "compression_ratio": round(avg_compression, 3),
            "duration_ms": duration_ms,
        },
    )

    if avg_no_speech > _NO_SPEECH_THRESHOLD:
        log.debug("stt: skipping — likely not speech", extra={"no_speech_prob": avg_no_speech})
        return TranscribeResult(text="", language=detected, confidence=0.0, duration_ms=duration_ms)

    if avg_compression > _MAX_COMPRESSION_RATIO:
        log.debug("stt: skipping — hallucination", extra={"compression": avg_compression})
        return TranscribeResult(text="", language=detected, confidence=0.0, duration_ms=duration_ms)

    if avg_logprob < _MIN_AVG_LOGPROB:
        log.debug("stt: skipping — low confidence", extra={"avg_logprob": avg_logprob})
        return TranscribeResult(text="", language=detected, confidence=0.0, duration_ms=duration_ms)

    text = " ".join(s.text.strip() for s in segments).strip()

    # Step 4: Script validation (catches third-language misclassification)
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
