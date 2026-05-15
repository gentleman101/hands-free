# -*- coding: utf-8 -*-
"""
Generate filler audio WAVs for all intents x languages via Fish Speech.

Fillers are played immediately after VAD end while STT + LLM + TTS process,
so the user hears something natural instead of silence.

Run once (Fish Speech must be running):
    venv\Scripts\python scripts/generate_fillers.py

Output: assets/fillers/{en,ar}/{intent}_{n}.wav
"""
from __future__ import annotations

import asyncio
import base64
import sys
import time
import wave
from pathlib import Path

import httpx

FISH_URL = "http://172.29.211.12:8001"
VOICES_DIR = Path("assets/voices")
OUT_DIR = Path("assets/fillers")

REFS = {
    "en": {
        "file": "brian_en.wav",
        "text": "Hello, I'm Rammas, your smart assistant from Dubai Electricity and Water Authority. How can I help you today?",
    },
    "ar": {
        "file": "hamdan_ar.wav",
        "text": "مرحباً، أنا رمّاس، مساعدك الذكي من هيئة كهرباء ومياه دبي. كيف يمكنني مساعدتك اليوم؟",
    },
}

# 7 intents x 2 languages x 6 variants = 84 WAVs
# Variant structure per intent:
#   1-2: clean, short, generic
#   3-4: light prosody (ohh, okay)
#   5-6: heavy prosody (umm..., aaah..., ohhh kaaay) — sounds most human
FILLERS: dict[str, dict[str, list[str]]] = {
    "billing": {
        "en": [
            "Let me pull that up.",
            "Checking your account now.",
            "Ohh okay, one moment.",
            "Sure, let me grab those details.",
            "Umm... yeah sure, getting it right away.",
            "Aaah... yes... got it, one sec.",
        ],
        "ar": [
            "دعني أجلب ذلك.",
            "أتحقق من حسابك الآن.",
            "أوه حسناً، لحظة.",
            "نعم، دعني آخذ تلك التفاصيل.",
            "مم... نعم بالتأكيد، على الفور.",
            "آه... نعم... تمام، ثانية.",
        ],
    },
    "location": {
        "en": [
            "Let me find that for you.",
            "Checking what's nearest to you.",
            "Ohh okay, looking that up.",
            "Sure, finding the closest one!",
            "Umm... yeah, let me check what's near you.",
            "Aaah... yes, on it.",
        ],
        "ar": [
            "دعني أجد ذلك لك.",
            "أتحقق مما هو أقرب إليك.",
            "أوه حسناً، أبحث عن ذلك.",
            "نعم، أجد الأقرب!",
            "مم... نعم، دعني أتحقق مما هو قريب منك.",
            "آه... نعم، على الفور.",
        ],
    },
    "outage": {
        "en": [
            "Let me check on that.",
            "Checking your area now.",
            "Ohh okay, looking into that.",
            "Sure, on it!",
            "Umm... yeah, let me see what's happening.",
            "Aaah... yes, checking right now.",
        ],
        "ar": [
            "دعني أتحقق من ذلك.",
            "أتحقق من منطقتك الآن.",
            "أوه حسناً، أنظر في ذلك.",
            "نعم، حاضر!",
            "مم... نعم، دعني أرى ما يحدث.",
            "آه... نعم، أتحقق الآن.",
        ],
    },
    "account": {
        "en": [
            "Let me pull that up.",
            "Checking your details.",
            "Ohh okay, one moment.",
            "Sure, got it!",
            "Umm... yeah, let me get that for you.",
            "Aaah... yes... got it.",
        ],
        "ar": [
            "دعني أجلب ذلك.",
            "أتحقق من بياناتك.",
            "أوه حسناً، لحظة.",
            "نعم، تمام!",
            "مم... نعم، دعني أجلب ذلك لك.",
            "آه... نعم... تمام.",
        ],
    },
    "service_request": {
        "en": [
            "Let me look into that.",
            "Checking what we can do.",
            "Ohh okay, let me see.",
            "Yes, sure, on it!",
            "Umm... yeah sure, getting it right away.",
            "Aaah... yes... got it, one moment.",
        ],
        "ar": [
            "دعني أنظر في ذلك.",
            "أتحقق مما يمكن فعله.",
            "أوه حسناً، دعني أرى.",
            "نعم، بالتأكيد، حاضر!",
            "مم... نعم بالتأكيد، جاري الحصول عليه.",
            "آه... نعم... تمام، لحظة.",
        ],
    },
    "sustainability": {
        "en": [
            "Let me find that for you.",
            "Checking on that now.",
            "Ohhh kaaay, let me look that up.",
            "Sure, on it!",
            "Umm... yeah, let me get that info.",
            "Aaah... yes, looking that up.",
        ],
        "ar": [
            "دعني أجد ذلك لك.",
            "أتحقق من ذلك الآن.",
            "أووه حسناً، دعني أبحث عن ذلك.",
            "نعم، حاضر!",
            "مم... نعم، دعني أجلب تلك المعلومات.",
            "آه... نعم، أبحث عن ذلك.",
        ],
    },
    "general": {
        "en": [
            "One moment.",
            "Let me check.",
            "Ohh okay, one sec.",
            "Ohhh kaaay, let me fetch that quickly.",
            "Umm... yeah sure, getting it right away.",
            "Aaah... yes... got it.",
        ],
        "ar": [
            "لحظة.",
            "دعني أتحقق.",
            "أوه حسناً، ثانية.",
            "أووه حسناً، دعني أجلب ذلك بسرعة.",
            "مم... نعم بالتأكيد، على الفور.",
            "آه... نعم... تمام.",
        ],
    },
}


async def synthesize(client: httpx.AsyncClient, text: str, language: str) -> bytes:
    ref = REFS[language]
    ref_path = VOICES_DIR / ref["file"]
    ref_audio = base64.b64encode(ref_path.read_bytes()).decode()

    payload = {
        "text": text,
        "streaming": True,
        "format": "wav",
        "chunk_length": 200,
        "normalize": True,
        "use_memory_cache": "on",
        "references": [{"audio": ref_audio, "text": ref["text"]}],
    }

    chunks: list[bytes] = []
    async with client.stream("POST", f"{FISH_URL}/v1/tts", json=payload, timeout=60.0) as resp:
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}")
        async for chunk in resp.aiter_bytes(4096):
            if chunk:
                chunks.append(chunk)
    return b"".join(chunks)


def save_wav(raw: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(raw)


async def main(force: bool = False) -> None:
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{FISH_URL}/v1/health", timeout=3.0)
            if r.status_code != 200:
                raise RuntimeError
        except Exception:
            print(f"ERROR: Fish Speech not reachable at {FISH_URL}")
            sys.exit(1)

    total = sum(len(v[lang]) for v in FILLERS.values() for lang in ("en", "ar"))
    done = 0
    print(f"Generating {total} filler WAVs{' (force-overwrite)' if force else ''}...\n")

    async with httpx.AsyncClient() as client:
        for intent, langs in FILLERS.items():
            for lang, phrases in langs.items():
                for i, phrase in enumerate(phrases, start=1):
                    out = OUT_DIR / lang / f"{intent}_{i}.wav"
                    if out.exists() and not force:
                        done += 1
                        print(f"  [skip] {lang}/{intent}_{i}.wav")
                        continue
                    t0 = time.perf_counter()
                    try:
                        raw = await synthesize(client, phrase, lang)
                        save_wav(raw, out)
                        ms = int((time.perf_counter() - t0) * 1000)
                        display = phrase[:60].encode("ascii", "replace").decode()
                        print(f"  [{done+1:02d}/{total}] {lang}/{intent}_{i}.wav  ({ms}ms)  \"{display}\"")
                    except Exception as exc:
                        print(f"  [ERR] {lang}/{intent}_{i}: {exc}")
                    done += 1

    print(f"\nDone. {OUT_DIR}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="overwrite existing WAVs")
    args = parser.parse_args()
    asyncio.run(main(force=args.force))
