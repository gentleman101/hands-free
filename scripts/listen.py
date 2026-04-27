#!/usr/bin/env python3
"""
Hands-Free live mic loop.

Listens on the microphone, detects end-of-utterance via VAD,
runs the full STT → LLM → TTS pipeline, and plays audio back.

Usage:
    python scripts/listen.py
    python scripts/listen.py --phase prompted
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))


async def _play(audio_bytes: bytes, sample_rate: int = 22050) -> None:
    import io
    import wave

    import numpy as np
    import sounddevice as sd  # type: ignore[import-untyped]

    buf = io.BytesIO(audio_bytes)
    with wave.open(buf, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    sd.play(pcm, samplerate=sr, blocking=True)


async def listen_loop() -> None:
    from backend.database import init_db
    from backend.logging_config import configure_logging
    from backend.services.audio_capture import AudioCapture
    from backend.services.pipeline import run

    configure_logging("INFO")
    await init_db()

    capture = AudioCapture()
    await capture.start()

    print("\n  Hands-Free is listening. Speak — pause 300ms to send.")
    print("  Ctrl+C to quit.\n")

    try:
        async for utterance_wav in capture.utterances():
            print("  [you stopped speaking — processing...]")
            try:
                result = await run(utterance_wav)
                lat = result.latency
                print(
                    f"  [{result.language.upper()}] \"{result.transcription}\"\n"
                    f"  → \"{result.response_text}\"\n"
                    f"  STT={lat.stt_ms}ms  TTFT={lat.llm_ttft_ms}ms  "
                    f"TTFA={lat.tts_ttfa_ms}ms  Pipeline TTFA={lat.pipeline_ttfa_ms}ms\n"
                )
                await _play(result.audio_bytes)
            except Exception as exc:
                print(f"  [pipeline error] {exc}")
    except KeyboardInterrupt:
        print("\n  Stopping.")
    finally:
        await capture.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hands-Free live mic loop")
    parser.add_argument(
        "--phase",
        default=None,
        help="Pipeline phase label (raw/prompted/lora_v1 etc.) — written to latency log",
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_OLLAMA_BASE_URL", "http://localhost:11434/v1")
    os.environ.setdefault("HF_OLLAMA_MODEL", "gemma4:e2b")
    os.environ.setdefault("HF_WHISPER_MODEL_SIZE", "small")
    os.environ.setdefault("HF_FISH_SPEECH_URL", "http://localhost:8001")
    os.environ.setdefault("HF_PIPELINE_PHASE", args.phase or "raw")

    asyncio.run(listen_loop())


if __name__ == "__main__":
    main()
