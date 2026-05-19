"""
Windows entry point: starts API server + mic loop.

Usage:
  set HF_MACHINE_ROLE=client
  set HF_LLM_SERVICE_URL=http://mac-mini.local:11434
  python scripts/start.py

This:
  1. Verifies role is "client"
  2. Health-checks Mac LLM server (fail-fast on office network)
  3. Starts FastAPI in a background task
  4. Runs the mic loop: capture → pipeline → speaker
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import struct
import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]
import soundfile as sf    # type: ignore[import-untyped]
import io


def _decode_wav(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Decode WAV bytes to float32 numpy array + sample rate.
    Handles streaming WAVs where the data chunk size is 0 or 0xFFFFFFFF
    (libsndfile rejects these; we parse manually in that case).
    """
    try:
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        return data, sr
    except Exception:
        pass

    # Manual parse: walk chunks to find fmt + data
    if wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        # Fish Speech streaming mode returns raw int16 PCM @ 44100 Hz mono
        samples = np.frombuffer(wav_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, 44100

    pos = 12
    sr, channels, bits = 44100, 1, 16
    while pos < len(wav_bytes) - 8:
        chunk_id = wav_bytes[pos:pos+4]
        chunk_size = struct.unpack_from("<I", wav_bytes, pos+4)[0]
        if chunk_id == b"fmt ":
            channels = struct.unpack_from("<H", wav_bytes, pos+8)[0]
            sr       = struct.unpack_from("<I", wav_bytes, pos+10)[0]
            bits     = struct.unpack_from("<H", wav_bytes, pos+22)[0]
        elif chunk_id == b"data":
            pcm_start = pos + 8
            pcm = wav_bytes[pcm_start:] if chunk_size in (0, 0xFFFFFFFF) else wav_bytes[pcm_start:pcm_start+chunk_size]
            dtype = np.int16 if bits == 16 else np.int32
            samples = np.frombuffer(pcm, dtype=dtype).astype(np.float32) / (2 ** (bits - 1))
            data = samples.reshape(-1, channels) if channels > 1 else samples
            return data, sr
        pos += 8 + (chunk_size if chunk_size != 0xFFFFFFFF else 0)

    raise ValueError("WAV data chunk not found")


async def _play_wav(wav_bytes: bytes) -> None:
    """Play WAV bytes through default output device."""
    data, sr = _decode_wav(wav_bytes)
    sd.play(data, sr)
    try:
        await asyncio.get_event_loop().run_in_executor(None, sd.wait)
    except asyncio.CancelledError:
        sd.stop()
        raise


async def _mic_loop(log: logging.Logger) -> None:
    from backend.services.audio_capture import AudioCapture
    from backend.services.pipeline import PipelineResult, run_realtime
    from backend import config as cfg
    from backend.cli_ui import get as get_ui

    config = cfg.load()
    capture = AudioCapture(
        vad_backend=config.vad_backend,
        device_env=config.audio_input_device,
    )

    ui = get_ui()
    ui.start()
    turn = 0
    import time as _time
    # Dynamic VAD thresholds
    SPEAK_THRESHOLD = 0.60       # sensitive to soft speech (RMS gate filters noise downstream)
    PROCESSING_THRESHOLD = 0.95  # strict during AI's turn — only barge-in catches it
    capture.set_vad_threshold(SPEAK_THRESHOLD)

    try:
        ui.info(f"\n{'-' * 60}\nRammas — ready. Speak into the mic. Ctrl+C to exit.\n{'-' * 60}")
        ui.stage("SPEAK", "your turn")

        async for utterance in capture.utterances():
            turn += 1
            ui.banner(f"Turn {turn}")
            # Switch to PROCESSING — raise VAD threshold so AI's own audio
            # (echo/leakage) doesn't trigger; only loud user speech (barge-in) does.
            capture.set_vad_threshold(PROCESSING_THRESHOLD)
            ui.stage("PROCESSING", "stt → llm → tts")

            out = PipelineResult()
            filler_chunks: list[bytes] = []
            tts_chunks: list[bytes] = []
            filler_done = asyncio.Event()
            transcription_shown = False

            async for chunk in run_realtime(utterance, out=out, filler_done=filler_done):
                if not transcription_shown and out.transcription:
                    ui.info(f"  [{out.language}] You: {out.transcription}")
                    transcription_shown = True
                if filler_done.is_set():
                    tts_chunks.append(chunk)
                else:
                    filler_chunks.append(chunk)

            if not transcription_shown:
                ui.info("  (no response — transcription empty)")
                capture.set_vad_threshold(SPEAK_THRESHOLD)
                ui.stage("SPEAK", "your turn")
                continue

            if filler_chunks:
                await _play_wav(b"".join(filler_chunks))

            ui.info(f"  Rammas: {out.response_text}")

            if tts_chunks:
                await _play_wav(b"".join(tts_chunks))

            # Per-stage timings from pipeline (each measured at its own start)
            stt_ms     = out.latency.stt_ms or 0
            llm_ttft   = out.latency.llm_ttft_ms or 0
            llm_total  = out.latency.llm_total_ms or 0
            pipe_ttfa  = out.latency.pipeline_ttfa_ms or 0
            tts_ttfa   = max(0, pipe_ttfa - stt_ms - llm_ttft)
            total_ms   = out.latency.total_ms or pipe_ttfa
            ui.info(
                f"  stt={stt_ms}ms  llm_ttft={llm_ttft}ms  "
                f"tts_ttfa={tts_ttfa}ms  llm_total={llm_total}ms  total={total_ms}ms"
            )

            await asyncio.sleep(0.3)
            capture.set_vad_threshold(SPEAK_THRESHOLD)
            ui.stage("SPEAK", "your turn")
    finally:
        ui.stop()
        capture.stop()


async def _serve_api() -> None:
    """Run FastAPI in the same event loop."""
    import uvicorn
    from backend import config as cfg

    config = cfg.load()
    server_config = uvicorn.Config(
        "backend.app:app",
        host="127.0.0.1",
        port=int(__import__("os").environ.get("HF_API_PORT", "8000")),
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(server_config)
    await server.serve()


async def _main() -> int:
    import os as _os
    from backend.logging_config import setup_logging
    from backend import config as cfg
    from backend.services import llm_client

    # CLI UI takes over stdout; route logs to a file so the screen stays clean.
    # Pass HF_LOG_TO_STDERR=1 to keep old behavior (useful for debugging).
    if not _os.environ.get("HF_LOG_TO_STDERR"):
        log_path = Path(__file__).resolve().parent.parent / "data" / "start.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _os.environ["HF_LOG_FILE"] = str(log_path)
    setup_logging()

    log = logging.getLogger("scripts.start")
    config = cfg.load()

    if config.machine_role != "client":
        log.error(
            "scripts/start.py is for the client. Mac uses scripts/start_llm_server.py.",
            extra={"role": config.machine_role},
        )
        return 1

    # Pre-warm Whisper so first utterance doesn't pay model load cost
    log.info("warming up Whisper model")
    from backend.services import stt as _stt, tts as _tts
    await asyncio.get_event_loop().run_in_executor(None, _stt.preload)
    log.info("Whisper ready")

    # Pre-encode TTS reference audio — first call costs ~12s, cached ~20ms
    log.info("warming up Fish Speech (voice cloning cache)")
    await _tts.warmup()
    log.info("Fish Speech ready")

    # Strict startup: fail fast if Mac is not reachable (office network)
    log.info("checking Mac LLM server", extra={"url": config.llm_service_url})
    if not await llm_client.health_check():
        log.error(
            "Mac LLM server is not reachable. Start it on the Mac:",
            extra={"url": config.llm_service_url},
        )
        log.error("  ssh mac-mini")
        log.error("  cd hands-free && python scripts/start_llm_server.py")
        return 2
    log.info("Mac LLM server is reachable")

    api_task = asyncio.create_task(_serve_api())
    mic_task = asyncio.create_task(_mic_loop(log))

    try:
        await asyncio.gather(api_task, mic_task)
    except KeyboardInterrupt:
        log.info("interrupted by user")
        api_task.cancel()
        mic_task.cancel()
    return 0


def main() -> int:
    try:
        return asyncio.run(_main())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    # Ensure project root is on sys.path when running directly
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.exit(main())
