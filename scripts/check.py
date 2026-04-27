#!/usr/bin/env python3
"""
Module-by-module health check.

Usage:
    python scripts/check.py           # all modules
    python scripts/check.py --llm     # LLM only
    python scripts/check.py --stt     # STT only
    python scripts/check.py --tts     # TTS only
    python scripts/check.py --pipeline  # full pipeline (fixture wav)
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_OLLAMA_BASE_URL", "http://localhost:11434/v1")
os.environ.setdefault("HF_OLLAMA_MODEL", "gemma4:e2b")
os.environ.setdefault("HF_WHISPER_MODEL_SIZE", "small")
os.environ.setdefault("HF_FISH_SPEECH_URL", "http://localhost:8001")
os.environ.setdefault("HF_PIPELINE_PHASE", "raw")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

SEP = "─" * 50


# ── LLM ───────────────────────────────────────────────────────────────────────

async def check_llm() -> bool:
    import time
    from openai import AsyncOpenAI

    print(f"\n{SEP}")
    print("  LLM — Ollama / gemma4:e2b")
    print(SEP)
    try:
        client = AsyncOpenAI(
            base_url=os.environ.get("HF_OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",
        )
        messages = [{"role": "user", "content": "Reply with exactly: Hello from Gemma."}]
        print("  Sending prompt...", end="", flush=True)

        t0 = time.perf_counter()
        result = await client.chat.completions.create(
            model=os.environ.get("HF_OLLAMA_MODEL", "gemma4:e2b"),
            messages=messages,  # type: ignore[arg-type]
            max_tokens=2048,
            stream=False,
        )
        total_ms = int((time.perf_counter() - t0) * 1000)
        response = result.choices[0].message.content or ""
        print(f" done ({total_ms}ms)")
        print(f"  Response: \"{response.strip()}\"")
        return True
    except Exception as exc:
        print(f"\n  FAIL: {exc}")
        return False


# ── STT ───────────────────────────────────────────────────────────────────────

async def check_stt() -> bool:
    print(f"\n{SEP}")
    print("  STT — faster-whisper")
    print(SEP)
    try:
        from backend.services.stt import transcribe
        from backend.database import init_db
        await init_db()

        fixture = ROOT / "fixtures" / "audio" / "en_greeting.wav"
        audio = fixture.read_bytes()
        print(f"  Transcribing {fixture.name} ({len(audio)} bytes)...", end="", flush=True)
        result = await transcribe(audio)
        print(f" done ({result.duration_ms}ms)")
        print(f"  Language : {result.language} (confidence {result.confidence:.2f})")
        print(f"  Text     : \"{result.text}\"")
        return True
    except Exception as exc:
        print(f"\n  FAIL: {exc}")
        return False


# ── TTS ───────────────────────────────────────────────────────────────────────

async def check_tts() -> bool:
    print(f"\n{SEP}")
    print("  TTS — Fish Speech")
    print(SEP)
    try:
        from backend.services.tts import synthesize
        from backend.database import init_db
        await init_db()

        text = "Hello, this is a test of Fish Speech synthesis."
        print(f"  Synthesising: \"{text}\"")
        print("  Waiting...", end="", flush=True)
        result = await synthesize(text, language="en")
        print(" done")
        print(f"  Audio    : {len(result.audio_bytes)} bytes")
        print(f"  TTFA     : {result.ttfa_ms}ms")
        print(f"  Total    : {result.duration_ms}ms")

        # Save to file so you can listen
        out = ROOT / "data" / "tts_check.wav"
        out.write_bytes(result.audio_bytes)
        print(f"  Saved to : {out}")
        print(f"  Play with: afplay {out}")
        return True
    except Exception as exc:
        print(f"\n  FAIL: {exc}")
        return False


# ── Pipeline ──────────────────────────────────────────────────────────────────

async def check_text(text: str) -> bool:
    """Send text to LLM via direct API call, print response. No TTS."""
    import time
    from openai import AsyncOpenAI
    from backend.services.prompt_builder import build_messages

    print(f"\n{SEP}")
    print("  LLM only")
    print(SEP)
    try:
        client = AsyncOpenAI(
            base_url=os.environ.get("HF_OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",
        )
        messages = build_messages(text, language="en")
        print(f"  Input : \"{text}\"")
        print("  Waiting...", end="", flush=True)

        t0 = time.perf_counter()
        result = await client.chat.completions.create(
            model=os.environ.get("HF_OLLAMA_MODEL", "gemma4:e2b"),
            messages=messages,  # type: ignore[arg-type]
            max_tokens=2048,
            stream=False,
        )
        total_ms = int((time.perf_counter() - t0) * 1000)
        msg = result.choices[0].message
        response = msg.content or ""
        print(f" {total_ms}ms  (finish: {result.choices[0].finish_reason})")
        print()
        print("  ── Gemma response ──────────────────────────────")
        print(f"  {response.strip()}")
        print("  ────────────────────────────────────────────────")
        return True
    except Exception as exc:
        print(f"\n  FAIL: {exc}")
        import traceback
        traceback.print_exc()
        return False


async def check_mic() -> bool:
    """Record 4 seconds from mic and transcribe live."""
    import io
    import wave
    import sounddevice as sd  # type: ignore[import-untyped]
    from backend.services.stt import transcribe
    from backend.database import init_db
    await init_db()

    print(f"\n{SEP}")
    print("  STT — live mic recording")
    print(SEP)
    try:
        sample_rate = 16000
        seconds = 4
        print(f"  Recording {seconds}s — speak now...")
        audio_np = sd.rec(
            int(seconds * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
        )
        sd.wait()
        print("  Transcribing...", end="", flush=True)

        # Pack to WAV bytes
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_np.tobytes())
        wav_bytes = buf.getvalue()

        result = await transcribe(wav_bytes)
        print(f" done ({result.duration_ms}ms)")
        print(f"  Language : {result.language} (confidence {result.confidence:.2f})")
        print(f"  Text     : \"{result.text}\"")
        return True
    except Exception as exc:
        print(f"\n  FAIL: {exc}")
        import traceback
        traceback.print_exc()
        return False


async def check_vad() -> bool:
    """Test VAD + mic: speak, pause, see utterance detected and transcribed."""
    from backend.services.audio_capture import AudioCapture
    from backend.services.stt import transcribe
    from backend.database import init_db
    await init_db()

    print(f"\n{SEP}")
    print("  VAD — speak then pause 300ms to trigger utterance")
    print(SEP)
    print("  Listening... (Ctrl+C to stop after 1 utterance)")

    capture = AudioCapture()
    await capture.start()
    try:
        async for utterance_wav in capture.utterances():
            print(f"  Utterance detected ({len(utterance_wav)} bytes) — transcribing...", end="", flush=True)
            result = await transcribe(utterance_wav)
            print(f" done ({result.duration_ms}ms)")
            print(f"  Language : {result.language} (confidence {result.confidence:.2f})")
            print(f"  Text     : \"{result.text}\"")
            break  # stop after first utterance
    except KeyboardInterrupt:
        print("\n  Stopped.")
    finally:
        await capture.stop()
    return True


async def check_pipeline() -> bool:
    print(f"\n{SEP}")
    print("  Pipeline — STT → LLM → TTS (fixture: en_greeting.wav)")
    print(SEP)
    try:
        from backend.services.pipeline import run
        from backend.database import init_db
        await init_db()

        fixture = ROOT / "fixtures" / "audio" / "en_greeting.wav"
        audio = fixture.read_bytes()
        print("  Running pipeline...", end="", flush=True)
        result = await run(audio)
        lat = result.latency
        print(" done")
        print(f"  Heard    : \"{result.transcription}\"")
        print(f"  Lang     : {result.language}")
        print(f"  Gemma    : \"{result.response_text}\"")
        print(f"  STT      : {lat.stt_ms}ms")
        print(f"  LLM ttft : {lat.llm_ttft_ms}ms  total: {lat.llm_total_ms}ms")
        print(f"  TTS ttfa : {lat.tts_ttfa_ms}ms  total: {lat.tts_total_ms}ms")
        print(f"  TTFA     : {lat.pipeline_ttfa_ms}ms  |  total: {lat.total_ms}ms")

        if result.audio_bytes:
            out = ROOT / "data" / "pipeline_check.wav"
            out.write_bytes(result.audio_bytes)
            print(f"  Audio    : {out}")
            print(f"  Play with: afplay {out}")
        return True
    except Exception as exc:
        print(f"\n  FAIL: {exc}")
        import traceback
        traceback.print_exc()
        return False


# ── main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(description="Module health check")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--stt", action="store_true")
    parser.add_argument("--tts", action="store_true")
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--mic", action="store_true", help="Record 4s from mic and transcribe")
    parser.add_argument("--vad", action="store_true", help="Test VAD: speak then pause to trigger")
    parser.add_argument("--text", type=str, default=None, help="Send text to LLM→TTS directly")
    args = parser.parse_args()

    if args.text:
        await check_text(args.text)
        return

    if args.mic:
        await check_mic()
        return

    if args.vad:
        await check_vad()
        return

    run_all = not any([args.llm, args.stt, args.tts, args.pipeline])

    results = {}
    if args.llm or run_all:
        results["LLM"] = await check_llm()
    if args.stt or run_all:
        results["STT"] = await check_stt()
    if args.tts or run_all:
        results["TTS"] = await check_tts()
    if args.pipeline or run_all:
        results["Pipeline"] = await check_pipeline()

    print(f"\n{SEP}")
    print("  Summary")
    print(SEP)
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {name}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
