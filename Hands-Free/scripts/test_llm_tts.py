"""
End-to-end test: text → LLM (Mac Mini) → TTS (Fish Speech WSL2) → play audio.

Benchmarks:
  - LLM TTFT: time to first token from Mac
  - TTS TTFA: time to first audio chunk (not file write)
  - Total:    wall clock from send to first audio byte

Usage:
  venv\Scripts\python scripts/test_llm_tts.py
  venv\Scripts\python scripts/test_llm_tts.py --text "ما هو رصيد فاتورتي؟" --lang ar
"""
from __future__ import annotations

import argparse
import asyncio
import io
import sys
import time
import wave
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from backend.database import init_db
from backend.services import llm_client, tts
from backend.services.prompt_builder import build_messages


async def run(text: str, language: str) -> None:
    await init_db()

    print(f"\nInput  : {text.encode('ascii', 'replace').decode()}")
    print(f"Language: {language}")
    print(f"LLM    : {llm_client._http_client().base_url}")
    print(f"TTS    : {tts._fish_url()}")
    print()

    # ── 1. LLM ──────────────────────────────────────────────────────────────
    print("Checking LLM health...", end=" ", flush=True)
    if not await llm_client.health_check():
        print("UNREACHABLE — is the Mac Mini server running?")
        return
    print("OK")

    print("Checking TTS health...", end=" ", flush=True)
    if not await tts.health_check():
        print("UNREACHABLE — is Fish Speech running in WSL2?")
        return
    print("OK\n")

    messages = build_messages(text, language=language)

    print("Streaming LLM response...")
    t_start = time.perf_counter()
    llm_ttft_ms: int | None = None
    full_text = ""

    async for token in llm_client.generate_stream(messages, language=language):
        if llm_ttft_ms is None:
            llm_ttft_ms = int((time.perf_counter() - t_start) * 1000)
            print(f"  LLM TTFT      : {llm_ttft_ms} ms")
            print(f"  LLM response  : ", end="", flush=True)
        print(token, end="", flush=True)
        full_text += token

    llm_total_ms = int((time.perf_counter() - t_start) * 1000)
    print(f"\n  LLM total     : {llm_total_ms} ms\n")

    if not full_text.strip():
        print("Empty LLM response — nothing to synthesize.")
        return

    # ── 2. TTS ──────────────────────────────────────────────────────────────
    print("Streaming TTS...")
    t_tts = time.perf_counter()
    tts_ttfa_ms: int | None = None
    audio_chunks: list[bytes] = []

    async for chunk in tts.synthesize_stream(full_text, language):
        if tts_ttfa_ms is None:
            tts_ttfa_ms = int((time.perf_counter() - t_tts) * 1000)
            print(f"  TTS TTFA      : {tts_ttfa_ms} ms  <-- first audio chunk")
        audio_chunks.append(chunk)

    tts_total_ms = int((time.perf_counter() - t_tts) * 1000)
    total_ms = int((time.perf_counter() - t_start) * 1000)
    audio_bytes = b"".join(audio_chunks)

    print(f"  TTS total     : {tts_total_ms} ms")
    print(f"  Audio size    : {len(audio_bytes) / 1024:.1f} KB")
    print()
    print("-- Benchmark ---------------------------------------")
    print(f"  LLM TTFT      : {llm_ttft_ms} ms")
    print(f"  TTS TTFA      : {tts_ttfa_ms} ms  (first audio chunk, not file write)")
    print(f"  Pipeline total: {total_ms} ms")
    print("----------------------------------------------------\n")

    # ── 3. Play audio ────────────────────────────────────────────────────────
    try:
        import sounddevice as sd
        import numpy as np

        # Fish Speech streaming sends raw PCM (no WAV header per chunk).
        # Try WAV parse first; fall back to raw PCM at Fish Speech's known rate.
        sample_rate = 44100
        n_channels = 1
        audio_np = None

        if audio_bytes[:4] == b"RIFF":
            with wave.open(io.BytesIO(audio_bytes)) as wf:
                sample_rate = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())
            dtype = np.int16 if sampwidth == 2 else np.int32
            audio_np = np.frombuffer(frames, dtype=dtype)
        else:
            # Raw PCM: Fish Speech outputs 16-bit signed, 44100Hz, mono
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

        if n_channels > 1:
            audio_np = audio_np.reshape(-1, n_channels)

        print("Playing audio...")
        sd.play(audio_np, samplerate=sample_rate)
        sd.wait()
        print("Done.")

    except ImportError:
        print("sounddevice not installed — saving to output.wav instead")
        Path("output.wav").write_bytes(audio_bytes)
        print("Saved to output.wav")
    except Exception as e:
        print(f"Playback error: {e} — saving to output.wav")
        Path("output.wav").write_bytes(audio_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test LLM → TTS end-to-end")
    parser.add_argument("--text", default="What is my electricity bill balance?",
                        help="Input text to send to LLM")
    parser.add_argument("--lang", default="en", choices=["en", "ar"],
                        help="Language (en or ar)")
    args = parser.parse_args()
    asyncio.run(run(args.text, args.lang))


if __name__ == "__main__":
    main()
