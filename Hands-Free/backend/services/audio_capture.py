"""
Microphone capture with VAD and device priority routing.

VAD backend:
  - "ten"    → TEN VAD (preferred; faster transitions, higher accuracy than Silero)
  - "silero" → Silero VAD (fallback if TEN VAD unavailable)
  - "energy" → RMS energy gate (last-resort fallback if neither installed)

Apple-branch tuning carried over:
- EXP-011: speech threshold 0.80 (was 0.70) — cuts HVAC/breath false fires
- 600ms silence to end utterance — LiveKit/Pipecat-style turn-taking
- 128ms pre-roll prevents first-word clipping
- 30s hard cap prevents runaway buffers
- Device priority: External (BT/USB) > built-in > first available
"""
from __future__ import annotations

import asyncio
import io
import logging
import queue
import threading
import wave
from collections import deque
from typing import AsyncIterator

import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]

log = logging.getLogger(__name__)


# ─── Constants ──────────────────────────────────────────────────────────────
_SAMPLE_RATE = 16000
_CHANNELS = 1
_CHUNK_SAMPLES = 512                  # 32ms at 16kHz; matches Silero VAD frame size
_CHUNK_MS = int(_CHUNK_SAMPLES / _SAMPLE_RATE * 1000)

# VAD tuning (from Apple EXP-011 + audio_capture.py)
# Override at runtime with HF_VAD_THRESHOLD env var (e.g. 0.5 for quieter mics)
import os as _os
_SPEECH_THRESHOLD = float(_os.environ.get("HF_VAD_THRESHOLD", "0.80"))
_SILENCE_DURATION_MS = 600            # ms of silence to end utterance
_SILENCE_CHUNKS = _SILENCE_DURATION_MS // _CHUNK_MS
_MIN_SPEECH_CHUNKS = 8                # ~256ms minimum speech (Silero default)
_PREFIX_CHUNKS = 4                    # ~128ms pre-roll (prevents first-word clipping)
_MAX_UTTERANCE_S = 30                 # hard cap to prevent runaway buffers
_ENERGY_FALLBACK_THRESHOLD = 0.02     # RMS gate when no neural VAD available

# ─── Device selection ───────────────────────────────────────────────────────
def _resolve_device(env_value: str) -> int | str | None:
    """Return device to use for input.

    Priority:
    1. HF_AUDIO_INPUT_DEVICE env var (numeric id or name substring)
    2. Windows default input device (if one is set)
    3. First available input device
    """
    if env_value:
        if env_value.isdigit():
            return int(env_value)
        return env_value

    # Use Windows default if set
    default_input = sd.default.device[0]
    if default_input >= 0:
        log.info("using system default input device", extra={"device_id": default_input})
        return default_input

    # Fall back to first available input
    inputs = [i for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
    if inputs:
        log.info("no default input set, using first available", extra={"device_id": inputs[0]})
        return inputs[0]

    return None


# ─── VAD backends ───────────────────────────────────────────────────────────
class _VAD:
    """Abstract VAD interface."""

    def is_speech(self, chunk: np.ndarray) -> bool:
        raise NotImplementedError


class _TenVAD(_VAD):
    """TEN VAD wrapper — frame-level, low-latency speech detection."""

    def __init__(self, threshold: float = _SPEECH_THRESHOLD) -> None:
        from ten_vad import TenVad  # type: ignore[import-untyped]
        self._vad = TenVad()
        self._threshold = threshold
        log.info("VAD backend: TEN VAD", extra={"threshold": threshold})

    def is_speech(self, chunk: np.ndarray) -> bool:
        # ten-vad expects int16 PCM at 16kHz, 256-sample frames
        # We feed 512 samples in two halves
        pcm16 = (chunk * 32767).astype(np.int16)
        max_prob = 0.0
        for i in (0, 256):
            prob, _ = self._vad.process(pcm16[i:i + 256])
            if prob > max_prob:
                max_prob = prob
        # Log every chunk at DEBUG so we can see noise vs speech distribution
        log.debug("vad", extra={"prob": round(float(max_prob), 3)})
        return max_prob >= self._threshold


class _SileroVAD(_VAD):
    """Silero VAD wrapper — used as fallback if TEN VAD unavailable."""

    def __init__(self, threshold: float = _SPEECH_THRESHOLD) -> None:
        import torch  # type: ignore[import-untyped]
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self._model = model
        self._torch = torch
        self._threshold = threshold
        log.info("VAD backend: Silero", extra={"threshold": threshold})

    def is_speech(self, chunk: np.ndarray) -> bool:
        with self._torch.no_grad():
            tensor = self._torch.from_numpy(chunk).float()
            prob = float(self._model(tensor, _SAMPLE_RATE).item())
        return prob >= self._threshold


class _EnergyVAD(_VAD):
    """Pure-RMS fallback when neural VAD is unavailable."""

    def __init__(self, threshold: float = _ENERGY_FALLBACK_THRESHOLD) -> None:
        self._threshold = threshold
        log.warning("VAD backend: energy-only (install ten-vad or silero for accuracy)")

    def is_speech(self, chunk: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        return rms >= self._threshold


def _make_vad(backend: str) -> _VAD:
    """Construct the requested VAD, falling back if unavailable."""
    backend = backend.lower()
    if backend == "ten":
        try:
            return _TenVAD()
        except Exception as e:
            log.warning("ten-vad unavailable, falling back to Silero", extra={"err": str(e)})
            backend = "silero"
    if backend == "silero":
        try:
            return _SileroVAD()
        except Exception as e:
            log.warning("silero unavailable, falling back to energy", extra={"err": str(e)})
    return _EnergyVAD()


# ─── PCM → WAV conversion ───────────────────────────────────────────────────
def _pcm_to_wav(pcm: np.ndarray, sample_rate: int = _SAMPLE_RATE) -> bytes:
    """Convert float32 numpy array to WAV bytes."""
    pcm_int16 = (pcm * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(_CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


# ─── AudioCapture ───────────────────────────────────────────────────────────
class AudioCapture:
    """
    Microphone capture with VAD-driven utterance segmentation.

    Yields complete utterances as PCM WAV bytes after VAD detects end-of-speech.
    """

    def __init__(
        self,
        *,
        vad_backend: str = "ten",
        device_env: str = "",
        sample_rate: int = _SAMPLE_RATE,
    ) -> None:
        self._sample_rate = sample_rate
        self._device = _resolve_device(device_env)
        self._vad = _make_vad(vad_backend)
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._running = False
        self._paused = False  # When True, callback drops audio (barge-in suppression)
        self._max_chunks = (_MAX_UTTERANCE_S * 1000) // _CHUNK_MS

    def pause(self) -> None:
        """Stop accepting audio (legacy — prefer set_vad_threshold for barge-in)."""
        self._paused = True
        # Drain any audio already in the queue
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    def resume(self) -> None:
        """Re-enable audio capture after playback completes."""
        self._paused = False

    def set_vad_threshold(self, threshold: float) -> None:
        """
        Dynamically adjust VAD speech threshold.

        Use 0.95 during AI's turn (processing/playback) — only barge-in
        (loud user speech) triggers; quiet AI audio leakage is ignored.

        Use 0.70 during user's turn (listening) — more sensitive to soft speech.
        """
        if hasattr(self._vad, "_threshold"):
            self._vad._threshold = threshold
            log.info("vad threshold updated", extra={"threshold": threshold})

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            log.warning("sounddevice status", extra={"status": str(status)})
        if self._paused:
            return  # drop audio while paused (barge-in suppression during playback)
        # mono float32
        self._queue.put(indata[:, 0].copy())

    def start(self) -> None:
        if self._running:
            return
        if self._device is None:
            raise RuntimeError("No input device found. Connect a microphone.")
        self._stream = sd.InputStream(
            device=self._device,
            channels=_CHANNELS,
            samplerate=self._sample_rate,
            blocksize=_CHUNK_SAMPLES,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        self._running = True
        log.info("mic stream started", extra={"device": self._device, "sr": self._sample_rate})

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._running = False
        log.info("mic stream stopped")

    async def utterances(
        self,
        on_speech_start: callable | None = None,  # type: ignore[type-arg]
    ) -> AsyncIterator[bytes]:
        """
        Yield complete utterances as WAV bytes.

        Algorithm:
        1. Buffer all audio in a sliding pre-roll deque (4 chunks = 128ms)
        2. On speech detected, flush pre-roll into utterance buffer
        3. Keep accumulating until SILENCE_CHUNKS in a row are non-speech
        4. Emit utterance if it has >= MIN_SPEECH_CHUNKS of speech

        Args:
            on_speech_start: optional sync callback fired the moment speech is detected
                (use this to update UI state to "capturing").
        """
        if not self._running:
            self.start()

        prefix: deque[np.ndarray] = deque(maxlen=_PREFIX_CHUNKS)
        utterance: list[np.ndarray] = []
        speech_count = 0
        silence_count = 0
        in_speech = False

        loop = asyncio.get_event_loop()

        while self._running:
            chunk = await loop.run_in_executor(None, self._queue.get)
            if chunk.size == 0:
                continue

            is_speech = self._vad.is_speech(chunk)

            if not in_speech:
                # Idle: keep rolling pre-roll
                prefix.append(chunk)
                if is_speech:
                    in_speech = True
                    silence_count = 0
                    speech_count = 1
                    utterance = list(prefix)  # flush pre-roll
                    utterance.append(chunk)
                    if on_speech_start is not None:
                        try:
                            on_speech_start()
                        except Exception:
                            log.exception("on_speech_start callback failed")
            else:
                # Inside an utterance
                utterance.append(chunk)
                if is_speech:
                    speech_count += 1
                    silence_count = 0
                else:
                    silence_count += 1
                    if silence_count >= _SILENCE_CHUNKS:
                        # End of utterance — emit if substantial
                        if speech_count >= _MIN_SPEECH_CHUNKS:
                            pcm = np.concatenate(utterance)
                            wav = _pcm_to_wav(pcm, self._sample_rate)
                            log.info(
                                "utterance",
                                extra={
                                    "duration_ms": int(len(pcm) / self._sample_rate * 1000),
                                    "speech_chunks": speech_count,
                                },
                            )
                            yield wav
                        else:
                            log.debug(
                                "skipped short utterance",
                                extra={"speech_chunks": speech_count},
                            )
                        # Reset
                        utterance = []
                        speech_count = 0
                        silence_count = 0
                        in_speech = False
                        prefix.clear()

                # Hard cap
                if len(utterance) >= self._max_chunks:
                    pcm = np.concatenate(utterance)
                    wav = _pcm_to_wav(pcm, self._sample_rate)
                    log.warning(
                        "utterance hit max length",
                        extra={"duration_s": _MAX_UTTERANCE_S},
                    )
                    yield wav
                    utterance = []
                    speech_count = 0
                    silence_count = 0
                    in_speech = False
                    prefix.clear()
