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

# 7 intents x 2 languages x 3 variants = 42 WAVs
FILLERS: dict[str, dict[str, list[str]]] = {
    "billing": {
        "en": [
            "Let me pull up your bill.",
            "Checking your account now.",
            "One moment, looking at your balance.",
            "Ooh okay, let me check your bill real quick.",
            "Absolutely, pulling up your account.",
            "Umm, let me grab those details for you.",
        ],
        "ar": [
            "دعني أتحقق من فاتورتك.",
            "لحظة وأجلب لك التفاصيل.",
            "سأتحقق من حسابك الآن.",
            "آه، دعني أتحقق من فاتورتك بسرعة.",
            "بالتأكيد، سأجلب تفاصيل حسابك.",
            "حسناً، لحظة وأرى ذلك.",
        ],
    },
    "location": {
        "en": [
            "Let me find the nearest center for you.",
            "Checking locations near you.",
            "One moment, searching nearby.",
            "Ooh sure, let me find that for you.",
            "Absolutely, looking up centers near you.",
            "Umm, let me search that real quick.",
        ],
        "ar": [
            "دعني أجد أقرب مركز لك.",
            "لحظة وأبحث في المواقع القريبة.",
            "سأتحقق من المواقع المتاحة.",
            "آه، دعني أجد أقرب مكان لك.",
            "بالتأكيد، سأبحث في المواقع القريبة.",
            "حسناً، لحظة وأبحث.",
        ],
    },
    "outage": {
        "en": [
            "Let me check the outage status.",
            "Checking your area right now.",
            "One moment, looking into that.",
            "Ohh okay, let me check what's happening.",
            "Sure, looking into the outage now.",
            "Umm, let me see what's going on in your area.",
        ],
        "ar": [
            "دعني أتحقق من حالة الانقطاع.",
            "لحظة وأتحقق من منطقتك.",
            "سأفحص الوضع الآن.",
            "آه، دعني أرى ما يحدث.",
            "بالتأكيد، سأتحقق من الانقطاع الآن.",
            "حسناً، لحظة وأفحص منطقتك.",
        ],
    },
    "account": {
        "en": [
            "Let me access your account.",
            "Checking your details now.",
            "One moment with your account.",
            "Ohh okay, pulling up your account now.",
            "Absolutely, let me grab your details.",
            "Sure thing, one sec.",
        ],
        "ar": [
            "دعني أصل إلى حسابك.",
            "لحظة وأتحقق من بياناتك.",
            "سأراجع حسابك الآن.",
            "آه، دعني أفتح حسابك الآن.",
            "بالتأكيد، سأجلب بياناتك.",
            "حسناً، لحظة واحدة.",
        ],
    },
    "service_request": {
        "en": [
            "Let me look into that service.",
            "Checking what's available for you.",
            "One moment, processing that.",
            "Ohh okay, let me see what we can do.",
            "Absolutely, looking into that right now.",
            "Sure thing, give me just a second.",
        ],
        "ar": [
            "دعني أتحقق من الخدمة المطلوبة.",
            "لحظة وأرى ما يمكن تقديمه.",
            "سأتحقق من الخيارات المتاحة.",
            "آه، دعني أرى ما يمكن فعله.",
            "بالتأكيد، سأتحقق من ذلك الآن.",
            "حسناً، لحظة وأرى الخيارات.",
        ],
    },
    "sustainability": {
        "en": [
            "Let me find that for you.",
            "Checking our sustainability info.",
            "One moment, looking that up.",
            "Ohh sure, looking that up now.",
            "Absolutely, let me find that info.",
            "Umm, give me just a moment on that.",
        ],
        "ar": [
            "دعني أجلب تلك المعلومات.",
            "لحظة وأبحث في ذلك.",
            "سأتحقق من معلومات الاستدامة.",
            "آه، دعني أجد تلك المعلومات.",
            "بالتأكيد، سأبحث في ذلك الآن.",
            "حسناً، لحظة وأتحقق.",
        ],
    },
    "general": {
        "en": [
            "One moment please.",
            "Let me check that for you.",
            "Sure, just a moment.",
            "Umm, one sec.",
            "Ohh okay, let me check.",
            "Absolutely, just a moment.",
        ],
        "ar": [
            "لحظة من فضلك.",
            "دعني أتحقق من ذلك.",
            "حسناً، لحظة واحدة.",
            "آه، ثانية واحدة.",
            "حسناً، دعني أتحقق.",
            "بالتأكيد، لحظة.",
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


async def main() -> None:
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
    print(f"Generating {total} filler WAVs...\n")

    async with httpx.AsyncClient() as client:
        for intent, langs in FILLERS.items():
            for lang, phrases in langs.items():
                for i, phrase in enumerate(phrases, start=1):
                    out = OUT_DIR / lang / f"{intent}_{i}.wav"
                    if out.exists():
                        done += 1
                        print(f"  [skip] {lang}/{intent}_{i}.wav")
                        continue
                    t0 = time.perf_counter()
                    try:
                        raw = await synthesize(client, phrase, lang)
                        save_wav(raw, out)
                        ms = int((time.perf_counter() - t0) * 1000)
                        display = phrase[:50].encode("ascii", "replace").decode()
                        print(f"  [{done+1:02d}/{total}] {lang}/{intent}_{i}.wav  ({ms}ms)  \"{display}\"")
                    except Exception as exc:
                        print(f"  [ERR] {lang}/{intent}_{i}: {exc}")
                    done += 1

    print(f"\nDone. {OUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
