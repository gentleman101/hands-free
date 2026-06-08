"""
Minimal filler player for the Pipecat demo.

Loads pre-recorded WAVs from the main repo's assets/fillers directory.
No sentence-transformers dependency — uses keyword-based intent classification.
Returns (pcm_bytes, sample_rate, filler_text) for playback + LLM continuation.
"""
from __future__ import annotations

import io
import logging
import random
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_MAIN_REPO = Path(r"C:\Users\dewac\Documents\Naman\Hands-Free")
_FILLERS_DIR = _MAIN_REPO / "assets" / "fillers"

_INTENTS = ("billing", "location", "outage", "account", "service_request", "sustainability", "general")

# Keyword → intent mapping for fast classification (no ML needed)
_KEYWORDS: dict[str, list[str]] = {
    "billing":          ["bill", "invoice", "payment", "balance", "charges", "فاتورة", "مبلغ", "دفع", "رصيد"],
    "location":         ["office", "center", "branch", "location", "address", "nearest", "مركز", "مكتب", "فرع", "أقرب"],
    "outage":           ["outage", "power", "blackout", "electricity", "انقطاع", "كهرباء", "تيار", "ضوء"],
    "account":          ["account", "meter", "profile", "login", "حساب", "عداد", "ملف", "تسجيل"],
    "service_request":  ["connect", "register", "apply", "new connection", "transfer", "توصيل", "تسجيل", "طلب", "جديد"],
    "sustainability":   ["solar", "green", "ev", "charger", "energy saving", "شمسي", "طاقة", "شاحن", "استدامة"],
}

# Spoken filler texts — must match WAV file content exactly (same order as generate_fillers.py)
_FILLER_TEXTS: dict[str, dict[str, list[str]]] = {
    "billing": {
        "en": ["Let me pull that up.", "Checking your account now.", "Ohh okay, one moment.",
               "Sure, let me grab those details.", "Umm... yeah sure, getting it right away.", "Aaah... yes... got it, one sec."],
        "ar": ["دعني أجلب ذلك.", "أتحقق من حسابك الآن.", "أوه حسناً، لحظة.",
               "نعم، دعني آخذ تلك التفاصيل.", "مم... نعم بالتأكيد، على الفور.", "آه... نعم... تمام، ثانية."],
    },
    "location": {
        "en": ["Let me find that for you.", "Checking what's nearest to you.", "Ohh okay, looking that up.",
               "Sure, finding the closest one!", "Umm... yeah, let me check what's near you.", "Aaah... yes, on it."],
        "ar": ["دعني أجد ذلك لك.", "أتحقق مما هو أقرب إليك.", "أوه حسناً، أبحث عن ذلك.",
               "نعم، أجد الأقرب!", "مم... نعم، دعني أتحقق مما هو قريب منك.", "آه... نعم، على الفور."],
    },
    "outage": {
        "en": ["Let me check on that.", "Checking your area now.", "Ohh okay, looking into that.",
               "Sure, on it!", "Umm... yeah, let me see what's happening.", "Aaah... yes, checking right now."],
        "ar": ["دعني أتحقق من ذلك.", "أتحقق من منطقتك الآن.", "أوه حسناً، أنظر في ذلك.",
               "نعم، حاضر!", "مم... نعم، دعني أرى ما يحدث.", "آه... نعم، أتحقق الآن."],
    },
    "account": {
        "en": ["Let me pull that up.", "Checking your details.", "Ohh okay, one moment.",
               "Sure, got it!", "Umm... yeah, let me get that for you.", "Aaah... yes... got it."],
        "ar": ["دعني أجلب ذلك.", "أتحقق من بياناتك.", "أوه حسناً، لحظة.",
               "نعم، تمام!", "مم... نعم، دعني أجلب ذلك لك.", "آه... نعم... تمام."],
    },
    "service_request": {
        "en": ["Let me look into that.", "Checking what we can do.", "Ohh okay, let me see.",
               "Yes, sure, on it!", "Umm... yeah sure, getting it right away.", "Aaah... yes... got it, one moment."],
        "ar": ["دعني أنظر في ذلك.", "أتحقق مما يمكن فعله.", "أوه حسناً، دعني أرى.",
               "نعم، بالتأكيد، حاضر!", "مم... نعم بالتأكيد، جاري الحصول عليه.", "آه... نعم... تمام، لحظة."],
    },
    "sustainability": {
        "en": ["Let me find that for you.", "Checking on that now.", "Ohhh kaaay, let me look that up.",
               "Sure, on it!", "Umm... yeah, let me get that info.", "Aaah... yes, looking that up."],
        "ar": ["دعني أجد ذلك لك.", "أتحقق من ذلك الآن.", "أووه حسناً، دعني أبحث عن ذلك.",
               "نعم، حاضر!", "مم... نعم، دعني أجلب تلك المعلومات.", "آه... نعم، أبحث عن ذلك."],
    },
    "general": {
        "en": ["One moment.", "Let me check.", "Ohh okay, one sec.",
               "Ohhh kaaay, let me fetch that quickly.", "Umm... yeah sure, getting it right away.", "Aaah... yes... got it."],
        "ar": ["لحظة.", "دعني أتحقق.", "أوه حسناً، ثانية.",
               "أووه حسناً، دعني أجلب ذلك بسرعة.", "مم... نعم بالتأكيد، على الفور.", "آه... نعم... تمام."],
    },
}

# WAV cache: {lang: {intent: [raw_wav_bytes, ...]}}
_cache: dict[str, dict[str, list[bytes]]] = {}
_loaded = False


def _classify_intent(text: str) -> str:
    text_lower = text.lower()
    for intent, keywords in _KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return intent
    return "general"


def _load_cache() -> None:
    global _loaded
    if _loaded:
        return
    for lang in ("en", "ar"):
        _cache[lang] = {}
        for intent in _INTENTS:
            clips: list[bytes] = []
            for i in range(1, 7):
                path = _FILLERS_DIR / lang / f"{intent}_{i}.wav"
                if path.exists():
                    clips.append(path.read_bytes())
            _cache[lang][intent] = clips
    total = sum(len(v) for d in _cache.values() for v in d.values())
    log.info(f"filler cache: {total} clips loaded from {_FILLERS_DIR}")
    _loaded = True


def pick_filler(text: str, language: str) -> tuple[bytes | None, str]:
    """Return (wav_bytes, filler_text) matching the utterance intent + language."""
    _load_cache()
    lang = language if language in _cache else "en"
    intent = _classify_intent(text)
    clips = _cache[lang].get(intent) or _cache[lang].get("general") or []
    texts = _FILLER_TEXTS.get(intent, {}).get(lang) or _FILLER_TEXTS["general"].get(lang, [])
    if not clips:
        return None, ""
    idx = random.randrange(len(clips))
    filler_text = texts[idx] if idx < len(texts) else ""
    return clips[idx], filler_text


def wav_to_pcm(wav_bytes: bytes) -> tuple[bytes, int]:
    """Decode WAV bytes → (int16 PCM bytes, sample_rate)."""
    import soundfile as sf
    audio_np, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    pcm = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    return pcm, int(sr)
