"""
Speech-to-Text service using mlx-whisper on Apple Silicon GPU.

mlx-whisper runs Whisper natively via Apple MLX (Metal/Neural Engine) —
3-5× faster than faster-whisper on CPU (2000ms → 150-300ms on 'small').

Language is constrained to English and Arabic:
- Single-pass: auto-detect + transcribe in one call
- Discard result if detected language is neither EN nor AR
- EXP-015: replaces faster-whisper (CTranslate2 has no MPS support)
"""
import asyncio
import logging
from dataclasses import dataclass
from functools import lru_cache

from backend.monitoring.latency_tracker import track_latency

log = logging.getLogger(__name__)

# Confidence gates — applied to every transcription result
# no_speech_prob: Whisper's estimate that the audio contains no speech (0–1).
#   > 0.6 means Whisper itself thinks it's ambient noise → discard.
_NO_SPEECH_THRESHOLD = 0.3  # discard if Whisper thinks >30% chance not speech
# avg_logprob: average log-probability of transcription tokens.
#   Values close to 0 = high confidence. Very negative = model was guessing
#   (typical when wrong language is forced or audio is noisy) → discard.
_MIN_AVG_LOGPROB = -1.0
_MAX_COMPRESSION_RATIO = 2.4  # hallucinations repeat tokens and compress extremely well; normal speech is 1.5–2.4


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
        return False  # no alphabetic content at all — discard

    if language == "ar":
        arabic = sum(1 for c in alpha_chars if "؀" <= c <= "ۿ")
        return (arabic / len(alpha_chars)) >= 0.30

    # "en" — expect mostly ASCII Latin
    latin = sum(1 for c in alpha_chars if ord(c) < 128)
    return (latin / len(alpha_chars)) >= 0.50

# Map HF_WHISPER_MODEL_SIZE → MLX quantised HuggingFace repo (q4 = 4-bit, fastest on Apple Silicon)
_MLX_REPOS: dict[str, str] = {
    "tiny":   "mlx-community/whisper-tiny-mlx",
    "base":   "mlx-community/whisper-base-mlx",
    "small":  "mlx-community/whisper-small-mlx",    # full precision — better Arabic accuracy than q4
    "medium": "mlx-community/whisper-medium-mlx",
    "large":  "mlx-community/whisper-large-v3-mlx",
}
_DEFAULT_MODEL = "small"


@dataclass
class TranscribeResult:
    text: str
    language: str    # "en" | "ar"
    confidence: float
    duration_ms: int


def preload(model_size: str | None = None) -> None:
    """Eagerly warm up mlx-whisper — eliminates cold-start on first utterance."""
    import io
    import os
    import wave

    import numpy as np

    size = model_size or os.environ.get("HF_WHISPER_MODEL_SIZE", _DEFAULT_MODEL)

    # 0.5s of silence — just enough to trigger model download + MLX compile
    sr = 16000
    silence = np.zeros(sr // 2, dtype=np.float32)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((silence * 32767).astype(np.int16).tobytes())

    log.info("preloading mlx-whisper", extra={"repo": _MLX_REPOS.get(size, size)})
    _transcribe_sync(buf.getvalue(), size)
    log.info("mlx-whisper ready", extra={"model_size": size})


def _transcribe_sync(audio_bytes: bytes, model_size: str) -> TranscribeResult:
    """
    Blocking transcription on Apple Silicon GPU via mlx-whisper.
    Single pass: language auto-detection + transcription in one call.
    Returns TranscribeResult with text="" if audio is not EN or AR.
    """
    import io
    import time
    import warnings

    import soundfile as sf  # type: ignore[import-untyped]
    import mlx_whisper      # type: ignore[import-untyped]

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    t0 = time.perf_counter()
    repo = _MLX_REPOS.get(model_size, _MLX_REPOS[_DEFAULT_MODEL])

    audio_io = io.BytesIO(audio_bytes)
    audio_np, _ = sf.read(audio_io, dtype="float32")

    result = mlx_whisper.transcribe(
        audio_np,
        path_or_hf_repo=repo,
        language=None,          # auto-detect; we constrain to EN/AR below
        word_timestamps=False,
        verbose=False,
        # beam_size not supported by mlx_whisper — greedy decoding is the default
    )

    duration_ms = int((time.perf_counter() - t0) * 1000)
    detected = (result.get("language") or "").strip().lower()

    # ── confidence filtering via segment stats ────────────────────────────────
    segments = result.get("segments") or []
    if segments:
        avg_no_speech = sum(s.get("no_speech_prob", 0.0) for s in segments) / len(segments)
        avg_logprob   = sum(s.get("avg_logprob", 0.0)    for s in segments) / len(segments)
    else:
        avg_no_speech = 1.0   # no segments → treat as silence
        avg_logprob   = -999.0

    log.debug(
        "stt confidence",
        extra={
            "lang": detected,
            "no_speech_prob": round(avg_no_speech, 3),
            "avg_logprob": round(avg_logprob, 3),
            "duration_ms": duration_ms,
        },
    )

    avg_compression = sum(s.get("compression_ratio", 0.0) for s in segments) / len(segments) if segments else 0.0

    if avg_no_speech > _NO_SPEECH_THRESHOLD:
        log.debug("skipping — likely not speech", extra={"no_speech_prob": round(avg_no_speech, 3)})
        return TranscribeResult(text="", language="en", confidence=0.0, duration_ms=duration_ms)

    if avg_compression > _MAX_COMPRESSION_RATIO:
        log.debug("skipping — hallucination detected", extra={"compression_ratio": round(avg_compression, 3)})
        return TranscribeResult(text="", language="en", confidence=0.0, duration_ms=duration_ms)

    if avg_logprob < _MIN_AVG_LOGPROB:
        log.debug("skipping — low confidence", extra={"avg_logprob": round(avg_logprob, 3)})
        return TranscribeResult(text="", language="en", confidence=0.0, duration_ms=duration_ms)

    if detected not in ("en", "ar"):
        log.debug("skipping — not EN or AR", extra={"detected": detected})
        return TranscribeResult(text="", language="en", confidence=0.0, duration_ms=duration_ms)

    text = (result.get("text") or "").strip()

    if not _script_matches_language(text, detected):
        log.debug(
            "skipping — script mismatch (third language misdetected as %s)", detected,
            extra={"detected": detected, "text_preview": text[:40]},
        )
        return TranscribeResult(text="", language="en", confidence=0.0, duration_ms=duration_ms)
    # ─────────────────────────────────────────────────────────────────────────
    confidence = max(0.0, min(1.0, 1.0 + avg_logprob))  # rough 0–1 score from logprob

    return TranscribeResult(
        text=text,
        language=detected,
        confidence=round(confidence, 3),
        duration_ms=duration_ms,
    )


@track_latency("stt")
async def transcribe(
    audio_bytes: bytes,
    *,
    model_size: str = _DEFAULT_MODEL,
    language: str | None = None,
    request_id: str | None = None,
) -> TranscribeResult:
    """
    Transcribe a complete utterance, constrained to English and Arabic.
    Runs mlx-whisper in a thread pool (non-blocking).
    Returns TranscribeResult with text="" if audio is neither EN nor AR.
    """
    import os

    size = os.environ.get("HF_WHISPER_MODEL_SIZE", model_size)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _transcribe_sync, audio_bytes, size)

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
