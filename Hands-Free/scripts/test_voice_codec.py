# -*- coding: utf-8 -*-
"""
Test Arabic voice codec:
  1. VQGAN-encode the reference WAV (measures encode time)
  2. Run TTS with that reference voice
  3. Save output WAV and open with Windows Media Player

Usage:
    venv\Scripts\python scripts/test_voice_codec.py
"""
from __future__ import annotations

import asyncio
import base64
import io
import subprocess
import sys
import time
import wave
from pathlib import Path

import httpx
import ormsgpack

FISH_URL = "http://172.29.211.12:8001"
VOICES_DIR = Path(__file__).parent.parent / "assets" / "voices"

REFERENCE_WAV = VOICES_DIR / "hamdan_ar.wav"
REFERENCE_TEXT = "مرحباً، أنا رمّاس، مساعدك الذكي من هيئة كهرباء ومياه دبي. كيف يمكنني مساعدتك اليوم؟"

TTS_TEXT = "فاتورتك الشهرية ٤٥٠ درهم. يمكنك دفعها عبر تطبيق ديوا أو الموقع الإلكتروني."


async def encode_reference(client: httpx.AsyncClient, wav_bytes: bytes) -> list:
    """Call /v1/vqgan/encode and return VQ tokens."""
    payload = ormsgpack.packb(
        {"audios": [wav_bytes]},
        option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
    )
    t0 = time.perf_counter()
    resp = await client.post(
        f"{FISH_URL}/v1/vqgan/encode",
        data=payload,
        headers={"Content-Type": "application/msgpack"},
        timeout=30.0,
    )
    encode_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    result = ormsgpack.unpackb(resp.content)
    tokens = result["tokens"][0]  # shape: [codebook_size, time_steps]
    print(f"  VQGAN encode : {encode_ms:.0f} ms  ({len(tokens)} codebooks, {len(tokens[0])} frames)")
    return tokens


async def tts_with_reference(client: httpx.AsyncClient, wav_bytes: bytes) -> bytes:
    """Run TTS with reference audio, return raw WAV bytes (streaming)."""
    payload = {
        "text": TTS_TEXT,
        "streaming": True,
        "format": "wav",
        "chunk_length": 200,
        "normalize": True,
        "use_memory_cache": "on",
        "references": [
            {
                "audio": base64.b64encode(wav_bytes).decode(),
                "text": REFERENCE_TEXT,
            }
        ],
    }

    chunks: list[bytes] = []
    ttfa_ms: float | None = None
    t0 = time.perf_counter()

    async with client.stream("POST", f"{FISH_URL}/v1/tts", json=payload, timeout=60.0) as resp:
        if resp.status_code != 200:
            print(f"  TTS error: HTTP {resp.status_code}")
            return b""
        async for chunk in resp.aiter_bytes(chunk_size=4096):
            if chunk:
                if ttfa_ms is None:
                    ttfa_ms = (time.perf_counter() - t0) * 1000
                chunks.append(chunk)

    total_ms = (time.perf_counter() - t0) * 1000
    audio = b"".join(chunks)
    print(f"  TTFA         : {ttfa_ms:.0f} ms")
    print(f"  Total        : {total_ms:.0f} ms")
    print(f"  Audio        : {len(audio) / 1024:.1f} KB")
    return audio


def save_and_play(raw_audio: bytes, out_path: Path) -> None:
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(raw_audio)
    print(f"  Saved        : {out_path}")
    subprocess.Popen(["cmd", "/c", "start", "", str(out_path.resolve())])


async def main() -> None:
    if not REFERENCE_WAV.exists():
        print(f"ERROR: Reference WAV not found: {REFERENCE_WAV}")
        sys.exit(1)

    wav_bytes = REFERENCE_WAV.read_bytes()
    print(f"Reference WAV : {REFERENCE_WAV.name}  ({len(wav_bytes) // 1024} KB)")
    print()

    async with httpx.AsyncClient() as client:
        # Step 1: Pre-encode the reference audio
        print("[1] Encoding reference voice (VQGAN)...")
        try:
            tokens = await encode_reference(client, wav_bytes)
        except Exception as e:
            print(f"  Encode failed: {e}")
            tokens = None

        # Step 2: TTS with cached reference
        print("\n[2] Running TTS with Arabic text...")
        print(f"  Text: {TTS_TEXT[:60].encode('ascii', 'replace').decode()}...")
        raw = await tts_with_reference(client, wav_bytes)

    if raw:
        out = Path("test_voice_codec_ar.wav")
        save_and_play(raw, out)
        print("\nDone. Playing on default audio device...")
    else:
        print("\nNo audio returned.")


if __name__ == "__main__":
    asyncio.run(main())
