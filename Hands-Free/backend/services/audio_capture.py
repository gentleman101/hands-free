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
_SPEECH_THRESHOLD = 0.80              # EXP-011: raised from 0.70 to cut false fires
_SILENCE_DURATION_MS = 600            # ms of silence to end utterance
_SILENCE_CHUNKS = _SILENCE_DURATION_MS // _CHUNK_MS
_MIN_SPEECH_CHUNKS = 8                # ~256ms minimum speech (Silero default)
_PREFIX_CHUNKS = 4                    # ~128ms pre-roll (prevents first-word clipping)
_MAX_UTTERANCE_S = 30                 # hard cap to prevent runaway buffers
_ENERGY_FALLBACK_THRESHOLD = 0.02     # RMS gate when no neural VAD available

# Device priority (Apple)
_SKIP_KEYWORDS = ("nomachine", "virtual", "aggregate", "displayport", "hdmi")
_EXTERNAL_KEYWORDS = (
    "bluetooth", "airpods", "headset", "headphone", "earpod", "buds",
    "bose", "sony", "jabra", "beats", "usb", "external",
)
_BUILTIN_KEYWORDS = ("built-in", "macbook", "mac mini", "internal", "microphone array")


# ─── Device selection ───────────────────────────────────────────────────────
def list_input_devices() -> list[tuple[int, str]]:
    """Return [(index, name)] for every real input device (skips virtual/HDMI)."""
    return [
        (i, str(dev["name"]))  # type: ignore[index]
        for i, dev in enumerate(sd.query_devices())
        if dev["max_input_channels"] > 0  # type: ignore[index]
        and not any(kw in str(dev["name"]).lower() for kw in _SKIP_KEYWORDS)  # type: ignore[index]
    ]


def _pick_input_device(devices: list[tuple[int, str]]) -> int | None:
    """Priority: External (BT/USB) > built-in > first available."""
    if not devices:
        log.warning("no input devices found — connect a mic or set HF_AUDIO_INPUT_DEVICE")
        return None

    name_lower = [(i, n, n.lower()) for i, n in devices]

    for i, name, nl in name_lower:
        if any(kw in nl for kw in _EXTERNAL_KEYWORDS):
            log.info("selected external input", extra={"device_id": i, "name": name})
            return i

    for i, name, nl in name_lower:
        if any(kw in nl for kw in _BUILTIN_KEYWORDS):
            log.info("selected built-in input", extra={"device_id": i, "name": name})
            return i

    i, name, _ = name_lower[0]
    log.info("selected first available input", extra={"device_id": i, "name": name})
    return i


def _resolve_device(env_value: str) -> int | None:
    """Resolve HF_AUDIO_INPUT_DEVICE env value (id, name, or empty for auto)."""
    devices = list_input_devices()
    log.debug("available input devices", extra={"devices": devices})

    if env_value:
        # Numeric id?
        if env_value.isdigit():
            idx = int(env_value)
            if any(i == idx for i, _ in devices):
                return idx
        # Name match (case-insensitive substring)
        env_lower = env_value.lower()
        for i, n in devices:
            if env_lower in n.lower():
                return i
        log.warning("HF_AUDIO_INPUT_DEVICE not matched", extra={"value": env_value})

    return _pick_input_device(devices)


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
        for i in (0, 256):
            prob, _ = self._vad.process(pcm16[i:i + 256])
            if prob >= self._threshold:
                return True
        return False


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
        self._max_chunks = (_MAX_UTTERANCE_S * 1000) // _CHUNK_MS

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            log.warning("sounddevice status", extra={"status": str(status)})
        # mono float32
        self._queue.put(indata[:, 0].copy())

    def start(self) -> None:
        if self._running:
            return
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

    async def utterances(self) -> AsyncIterator[bytes]:
        """
        Yield complete utterances as WAV bytes.

        Algorithm:
        1. Buffer all audio in a sliding pre-roll deque (4 chunks = 128ms)
        2. On speech detected, flush pre-roll into utterance buffer
        3. Keep accumulating until SILENCE_CHUNKS in a row are non-speech
        4. Emit utterance if it has >= MIN_SPEECH_CHUNKS of speech
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
