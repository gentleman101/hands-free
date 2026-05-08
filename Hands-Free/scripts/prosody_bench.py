# -*- coding: utf-8 -*-
"""
Quick prosody latency test.

Sends a short EN + AR snippet with Fish Speech prosody tags,
measures time-to-first-chunk (TTFA) and total synthesis time,
then plays the audio so you can hear the prosody.

Usage:
    # 1. Start Fish Speech in another terminal:
    #    venv\Scripts\fish-speech serve --port 8001
    #
    # 2. Run this script:
    #    venv\Scripts\python scripts/prosody_bench.py
"""
from __future__ import annotations

import asyncio
import sys
import time

import httpx

FISH_URL = "http://172.29.211.12:8001/v1/tts"

SNIPPETS = [
    {
        "label": "EN with prosody",
        "text": "Your bill is [emphasis]450 AED[/emphasis]. [laugh] Don't worry, that's normal.",
        "language": "en",
    },
    {
        "label": "AR with prosody",
        "text": "فاتورتك [emphasis]٤٥٠ درهم[/emphasis]. لا تقلق، هذا طبيعي.",
        "language": "ar",
    },
    {
        "label": "EN plain (baseline)",
        "text": "Your bill is 450 AED. Don't worry, that's normal.",
        "language": "en",
    },
]


async def bench_one(client: httpx.AsyncClient, snippet: dict) -> None:
    label = snippet["label"]
    payload = {
        "text": snippet["text"],
        "streaming": True,
        "format": "wav",
        "chunk_length": 200,
        "normalize": True,
    }

    chunks: list[bytes] = []
    ttfa_ms: float | None = None
    t0 = time.perf_counter()

    try:
        async with client.stream("POST", FISH_URL, json=payload, timeout=30.0) as resp:
            if resp.status_code != 200:
                print(f"  [{label}] HTTP {resp.status_code}")
                return
            async for chunk in resp.aiter_bytes(chunk_size=4096):
                if chunk:
                    if ttfa_ms is None:
                        ttfa_ms = (time.perf_counter() - t0) * 1000
                    chunks.append(chunk)
    except httpx.ConnectError:
        print(f"  [{label}] Fish Speech not reachable at {FISH_URL}")
        print("  Start it with: venv\\Scripts\\fish-speech serve --port 8001")
        return

    total_ms = (time.perf_counter() - t0) * 1000
    audio_bytes = b"".join(chunks)

    print(f"\n[{label}]")
    print(f"  Text     : {snippet['text'][:60].encode('ascii', 'replace').decode()}...")
    print(f"  TTFA     : {ttfa_ms:.0f} ms  (time to first audio chunk)")
    print(f"  Total    : {total_ms:.0f} ms")
    print(f"  Audio    : {len(audio_bytes) / 1024:.1f} KB")

    # Save as WAV and open with Windows default player (AirPods / default device)
    import wave, os as _os, subprocess
    out = f"prosody_bench_{label.replace(' ', '_')}.wav"
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(audio_bytes)
    abs_out = _os.path.abspath(out)
    print(f"  Saved to : {abs_out}")
    subprocess.Popen(["cmd", "/c", "start", "", abs_out])


async def main() -> None:
    print("Prosody latency benchmark")
    print("=" * 40)
    async with httpx.AsyncClient() as client:
        for snippet in SNIPPETS:
            await bench_one(client, snippet)
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
