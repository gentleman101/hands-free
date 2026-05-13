r"""
Filler audio player.

Plays a pre-recorded context-aware phrase while STT + LLM + TTS process,
so the user hears something natural instead of silence.

Intent → random filler variant, per language.
Generate WAVs first: venv\Scripts\python scripts/generate_fillers.py
"""
from __future__ import annotations

import asyncio
import logging
import random
from pathlib import Path
from typing import AsyncIterator

log = logging.getLogger(__name__)

_FILLERS_DIR = Path(__file__).parent.parent.parent / "assets" / "fillers"

_INTENTS = (
    "billing", "location", "outage", "account",
    "service_request", "sustainability", "general",
)
_VARIANTS = 6  # billing_1.wav … billing_6.wav

# Cache: {language: {intent: [wav_bytes, ...]}}
_cache: dict[str, dict[str, list[bytes]]] = {}
# Parallel text cache so pipeline can pass the played phrase to the LLM
_text_cache: dict[str, dict[str, list[str]]] = {}

# Filler texts mirror generate_fillers.py FILLERS — order must match WAV numbering
_FILLER_TEXTS: dict[str, dict[str, list[str]]] = {
    "billing": {
        "en": ["Let me pull up your bill.", "Checking your account now.", "One moment, looking at your balance.", "Ooh okay, let me check your bill real quick.", "Absolutely, pulling up your account.", "Umm, let me grab those details for you."],
        "ar": ["دعني أتحقق من فاتورتك.", "لحظة وأجلب لك التفاصيل.", "سأتحقق من حسابك الآن.", "آه، دعني أتحقق من فاتورتك بسرعة.", "بالتأكيد، سأجلب تفاصيل حسابك.", "حسناً، لحظة وأرى ذلك."],
    },
    "location": {
        "en": ["Let me find the nearest center for you.", "Checking locations near you.", "One moment, searching nearby.", "Ooh sure, let me find that for you.", "Absolutely, looking up centers near you.", "Umm, let me search for that."],
        "ar": ["دعني أجد أقرب مركز لك.", "لحظة وأبحث في المواقع القريبة.", "سأتحقق من المواقع المتاحة.", "آه، دعني أجد أقرب مكان لك.", "بالتأكيد، سأبحث في المواقع القريبة.", "حسناً، لحظة وأبحث."],
    },
    "outage": {
        "en": ["Let me check the outage status.", "Checking your area right now.", "One moment, looking into that.", "Ohh okay, let me check what's happening.", "Sure, looking into the outage now.", "Umm, let me check on that."],
        "ar": ["دعني أتحقق من حالة الانقطاع.", "لحظة وأتحقق من منطقتك.", "سأفحص الوضع الآن.", "آه، دعني أرى ما يحدث.", "بالتأكيد، سأتحقق من الانقطاع الآن.", "حسناً، لحظة وأتحقق."],
    },
    "account": {
        "en": ["Let me access your account.", "Checking your details now.", "One moment with your account.", "Ohh okay, pulling up your account now.", "Absolutely, let me grab your details.", "Umm, checking your account."],
        "ar": ["دعني أصل إلى حسابك.", "لحظة وأتحقق من بياناتك.", "سأراجع حسابك الآن.", "آه، دعني أفتح حسابك الآن.", "بالتأكيد، سأجلب بياناتك.", "حسناً، لحظة وأراجع."],
    },
    "service_request": {
        "en": ["Let me look into that service.", "Checking what's available for you.", "One moment, processing that.", "Ohh okay, let me see what we can do.", "Absolutely, looking into that right now.", "Umm, let me check on that service."],
        "ar": ["دعني أتحقق من الخدمة المطلوبة.", "لحظة وأرى ما يمكن تقديمه.", "سأتحقق من الخيارات المتاحة.", "آه، دعني أرى ما يمكن فعله.", "بالتأكيد، سأتحقق من ذلك الآن.", "حسناً، لحظة وأرى."],
    },
    "sustainability": {
        "en": ["Let me find that for you.", "Checking our sustainability info.", "One moment, looking that up.", "Ohh sure, looking that up now.", "Absolutely, let me find that info.", "Umm, let me look that up."],
        "ar": ["دعني أجلب تلك المعلومات.", "لحظة وأبحث في ذلك.", "سأتحقق من معلومات الاستدامة.", "آه، دعني أجد تلك المعلومات.", "بالتأكيد، سأجلب المعلومات.", "حسناً، لحظة وأبحث."],
    },
    "general": {
        "en": ["Let me check on that.", "One moment please.", "Looking into that now.", "Ohh okay, let me check.", "Sure, one moment.", "Umm, let me see."],
        "ar": ["دعني أتحقق من ذلك.", "لحظة من فضلك.", "سأتحقق الآن.", "آه، حسناً، لحظة.", "بالتأكيد، لحظة.", "حسناً، دعني أرى."],
    },
}


def _load() -> None:
    for lang in ("en", "ar"):
        _cache[lang] = {}
        _text_cache[lang] = {}
        for intent in _INTENTS:
            clips: list[bytes] = []
            for i in range(1, _VARIANTS + 1):
                path = _FILLERS_DIR / lang / f"{intent}_{i}.wav"
                if path.exists():
                    clips.append(path.read_bytes())
                else:
                    log.debug("filler missing", extra={"path": str(path)})
            _cache[lang][intent] = clips
            _text_cache[lang][intent] = _FILLER_TEXTS.get(intent, {}).get(lang, [])
    loaded = sum(len(v) for d in _cache.values() for v in d.values())
    log.info("fillers loaded", extra={"total": loaded})


def _ensure_loaded() -> None:
    if not _cache:
        _load()


def get_filler(intent: str, language: str) -> tuple[bytes, str] | tuple[None, None]:
    """Return (wav_bytes, text) for a random filler matching intent + language."""
    _ensure_loaded()
    lang = language if language in _cache else "en"
    audio_bucket = _cache[lang].get(intent) or _cache[lang].get("general") or []
    text_bucket = _text_cache[lang].get(intent) or _text_cache[lang].get("general") or []
    if not audio_bucket:
        return None, None
    idx = random.randrange(len(audio_bucket))
    text = text_bucket[idx] if idx < len(text_bucket) else ""
    return audio_bucket[idx], text


def get_filler_bytes(intent: str, language: str) -> bytes | None:
    """Return raw WAV bytes for a random filler (text discarded)."""
    data, _ = get_filler(intent, language)
    return data


async def stream_filler(
    intent: str, language: str, chunk_size: int = 4096
) -> AsyncIterator[bytes]:
    """
    Yield filler WAV in chunks — same interface as tts.synthesize_stream.
    Use pick_filler() first to get the text for LLM context.
    """
    data = get_filler_bytes(intent, language)
    if not data:
        return
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]
        await asyncio.sleep(0)  # allow other coroutines to run between chunks


def pick_filler(intent: str, language: str) -> tuple[bytes | None, str]:
    """
    Select a filler clip once and return (wav_bytes, text) together.
    Use this in the pipeline so the same variant is streamed AND passed to the LLM.
    """
    data, text = get_filler(intent, language)
    return data, text or ""
