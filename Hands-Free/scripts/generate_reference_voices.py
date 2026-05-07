"""
Generate reference voice clips for Fish Speech voice cloning.
Uses Edge TTS to create clean reference audio for Hamdan (AR) and Brian (EN).

Run once:
    venv\Scripts\python scripts/generate_reference_voices.py
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import edge_tts

REFERENCES_DIR = Path("assets/voices")

VOICES = [
    {
        "file": "hamdan_ar.wav",
        "voice": "ar-AE-HamdanNeural",
        "text": "مرحباً، أنا رمّاس، مساعدك الذكي من هيئة كهرباء ومياه دبي. كيف يمكنني مساعدتك اليوم؟",
    },
    {
        "file": "brian_en.wav",
        "voice": "en-US-BrianNeural",
        "text": "Hello, I'm Rammas, your smart assistant from Dubai Electricity and Water Authority. How can I help you today?",
    },
]


async def generate(voice: dict) -> None:
    out = REFERENCES_DIR / voice["file"]
    if out.exists():
        print(f"  Skipping {voice['file']} (already exists)")
        return

    print(f"  Generating {voice['file']} with {voice['voice']}...")
    tts = edge_tts.Communicate(voice["text"], voice["voice"])
    await tts.save(str(out))
    print(f"  Saved {out} ({out.stat().st_size // 1024}KB)")


async def main() -> None:
    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating reference voice clips...")
    for v in VOICES:
        await generate(v)
    print("\nDone. Files saved to assets/voices/")
    print("  hamdan_ar.wav  -> AR reference (Fish Speech will clone this voice)")
    print("  brian_en.wav   -> EN reference (Fish Speech will clone this voice)")


if __name__ == "__main__":
    asyncio.run(main())
