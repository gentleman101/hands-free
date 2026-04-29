"""
Microphone capture with Silero VAD via faster-whisper.

Yields complete utterances (PCM WAV bytes) after VAD detects end-of-speech.
No mid-sentence streaming — utterance-triggered is the correct pattern
for a voice assistant.
"""
import asyncio
import io
import logging
import queue
import threading
import wave
from typing import AsyncIterator

import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]

log = logging.getLogger(__name__)

_SAMPLE_RATE = 16000


# Devices to always skip — virtual/remote adapters, display audio
_SKIP_KEYWORDS = ("nomachine", "virtual", "aggregate", "displayport", "hdmi")

# External devices get priority over built-in
_EXTERNAL_KEYWORDS = ("bluetooth", "airpods", "headset", "headphone", "earpod",
                      "buds", "bose", "sony", "jabra", "beats", "usb", "external")
_BUILTIN_KEYWORDS = ("built-in", "macbook", "mac mini", "internal")


def list_input_devices() -> list[tuple[int, str]]:
    """Return [(index, name)] for every real input device (skips virtual/NoMachine)."""
    return [
        (i, str(dev["name"]))  # type: ignore[index]
        for i, dev in enumerate(sd.query_devices())
        if dev["max_input_channels"] > 0  # type: ignore[index]
        and not any(kw in str(dev["name"]).lower() for kw in _SKIP_KEYWORDS)  # type: ignore[index]
    ]


def _pick_input_device(devices: list[tuple[int, str]]) -> int | None:
    """
    Priority order:
      1. External device (BT headset, USB mic, etc.)
      2. Built-in mic
      3. First available
    """
    if not devices:
        log.warning("no input devices found — connect a mic or set HF_AUDIO_INPUT_DEVICE")
        return None

    name_lower = [(i, n, n.lower()) for i, n in devices]

    # 1. External (BT / USB / named external device)
    for i, name, nl in name_lower:
        if any(kw in nl for kw in _EXTERNAL_KEYWORDS):
            log.info("selected external input", extra={"device_id": i, "name": name})
            return i

    # 2. Built-in mic
    for i, name, nl in name_lower:
        if any(kw in nl for kw in _BUILTIN_KEYWORDS):
            log.info("selected built-in input", extra={"device_id": i, "name": name})
            return i

    # 3. First available
    i, name, _ = name_lower[0]
    log.info("selected first available input", extra={"device_id": i, "name": name})
    return i


def _default_input_device() -> int | None:
    """Select best available input device using priority routing."""
    devices = list_input_devices()
    log.debug("available input devices", extra={"devices": devices})
    return _pick_input_device(devices)
_CHANNELS = 1
_CHUNK_MS = 32          # sounddevice callback chunk size (ms) — 512 samples, Silero VAD minimum
_CHUNK_SAMPLES = 512    # must be exactly 512 at 16kHz for Silero VAD

# VAD state
_SPEECH_THRESHOLD = 0.80        # Silero VAD probability — raised from 0.7 to cut HVAC/breath false fires (EXP-011)
_SILENCE_DURATION_MS = 600      # ms of silence to end utterance — LiveKit default is 550ms, Pipecat 500ms
_SILENCE_CHUNKS = _SILENCE_DURATION_MS // _CHUNK_MS
_MIN_SPEECH_CHUNKS = 8          # ~256ms minimum speech — Silero upstream default is 250ms
_PREFIX_CHUNKS = 4              # ~128ms pre-roll prepended to each utterance (prevents first-word clipping)
_MAX_UTTERANCE_S = 30           # hard cap to prevent runaway buffers
_ENERGY_FALLBACK_THRESHOLD = 0.02  # RMS threshold when Silero VAD unavailable (0.01 was too sensitive)


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


class AudioCapture:
    """
    Captures mic audio and yields complete utterances via async generator.

    Usage:
        capture = AudioCapture()
        await capture.start()
        async for utterance_wav in capture.utterances():
            result = await transcribe(utterance_wav)
        await capture.stop()
    """

    def __init__(self) -> None:
        self._running = False
        self._muted = False          # when True, VAD runs but utterances are not queued
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._utterance_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._stream: sd.InputStream | None = None
        self._vad_thread: threading.Thread | None = None
        self._silence_count = 0
        self._speech_buffer: list[np.ndarray] = []
        self._in_speech = False

    def mute(self) -> None:
        """Block utterances from reaching the queue (call before TTS playback)."""
        self._muted = True
        log.debug("audio capture muted")

    def unmute(self) -> None:
        """Re-open the utterance queue (call after TTS playback finishes)."""
        self._muted = False
        log.debug("audio capture unmuted")

    def _sd_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """sounddevice callback — runs in audio thread."""
        if status:
            log.warning("audio callback status", extra={"status": str(status)})
        self._audio_queue.put(indata[:, 0].copy())

    def _load_vad(self):  # type: ignore[no-untyped-def]
        """Load Silero VAD model (lazy, cached by torch hub). Uses MPS if available."""
        import ssl
        import torch  # type: ignore[import-untyped]

        # macOS Python 3.13 ships without SSL certs; bypass for torch.hub GitHub validation
        _orig_ctx = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context
        try:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
                verbose=False,
                trust_repo=True,
            )
        finally:
            ssl._create_default_https_context = _orig_ctx
        get_speech_prob = utils[0]
        # VAD stays on CPU — model is tiny, MPS dropout ops cause instability
        return model, get_speech_prob

    def _vad_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Blocking VAD loop running in its own thread."""
        try:
            model, get_speech_prob = self._load_vad()
        except Exception:
            log.exception("failed to load Silero VAD — using energy-based fallback")
            model = None

        silence_count = 0
        speech_buffer: list[np.ndarray] = []
        pre_roll: list[np.ndarray] = []   # rolling window of recent non-speech chunks
        in_speech = False
        max_chunks = int(_MAX_UTTERANCE_S * 1000 / _CHUNK_MS)

        while self._running:
            try:
                chunk = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # VAD inference
            if model is not None:
                import torch  # type: ignore[import-untyped]

                tensor = torch.from_numpy(chunk).unsqueeze(0)
                prob = float(model(tensor, _SAMPLE_RATE).item())
                is_speech = prob > _SPEECH_THRESHOLD
            else:
                # Energy fallback
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                is_speech = rms > _ENERGY_FALLBACK_THRESHOLD

            if is_speech:
                if not in_speech:
                    # Speech just started — prepend pre-roll so first word isn't clipped
                    speech_buffer = list(pre_roll)
                    pre_roll = []
                in_speech = True
                silence_count = 0
                speech_buffer.append(chunk)
            else:
                if not in_speech:
                    # Keep a rolling pre-roll window of recent non-speech chunks
                    pre_roll.append(chunk)
                    if len(pre_roll) > _PREFIX_CHUNKS:
                        pre_roll.pop(0)
                if in_speech:
                    silence_count += 1
                    speech_buffer.append(chunk)  # include trailing silence
                    if silence_count >= _SILENCE_CHUNKS or len(speech_buffer) >= max_chunks:
                        # Discard utterances shorter than minimum speech duration —
                        # filters clicks, breath sounds, brief noise bursts
                        speech_chunk_count = len(speech_buffer) - silence_count
                        if speech_chunk_count < _MIN_SPEECH_CHUNKS:
                            log.debug(
                                "utterance too short — discarded",
                                extra={"speech_chunks": speech_chunk_count, "min": _MIN_SPEECH_CHUNKS},
                            )
                        elif self._muted:
                            log.debug("utterance suppressed (muted during TTS playback)")
                        else:
                            audio = np.concatenate(speech_buffer)
                            wav_bytes = _pcm_to_wav(audio)
                            asyncio.run_coroutine_threadsafe(
                                self._utterance_queue.put(wav_bytes), loop
                            )
                            log.debug(
                                "utterance ready",
                                extra={"chunks": len(speech_buffer), "bytes": len(wav_bytes)},
                            )
                        speech_buffer = []
                        in_speech = False
                        silence_count = 0

    async def start(self) -> None:
        """Start mic capture and VAD thread."""
        import os

        # Allow explicit device override via env var (e.g. HF_AUDIO_INPUT_DEVICE=3)
        device_env = os.environ.get("HF_AUDIO_INPUT_DEVICE")
        device: int | None = int(device_env) if device_env is not None else _default_input_device()

        self._running = True
        loop = asyncio.get_event_loop()
        self._vad_thread = threading.Thread(
            target=self._vad_loop, args=(loop,), daemon=True
        )
        self._vad_thread.start()
        self._stream = sd.InputStream(
            device=device,
            samplerate=_SAMPLE_RATE,
            channels=_CHANNELS,
            dtype="float32",
            blocksize=_CHUNK_SAMPLES,
            callback=self._sd_callback,
        )
        self._stream.start()
        log.info("audio capture started", extra={"sample_rate": _SAMPLE_RATE, "device": device})

    async def stop(self) -> None:
        """Stop capture and VAD thread."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
        if self._vad_thread:
            self._vad_thread.join(timeout=2.0)
        log.info("audio capture stopped")

    async def utterances(self) -> AsyncIterator[bytes]:
        """Yield complete utterance WAV bytes when VAD detects end-of-speech."""
        while self._running:
            try:
                wav = await asyncio.wait_for(
                    self._utterance_queue.get(), timeout=1.0
                )
                yield wav
            except asyncio.TimeoutError:
                continue
