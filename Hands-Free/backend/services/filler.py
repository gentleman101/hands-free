r"""
Filler audio player.

Plays a pre-recorded context-aware phrase while STT + LLM + TTS process,
so the user hears something natural instead of silence.

Two selection modes:
  1. Semantic (preferred): embed transcription → cosine-sim against all pre-embedded
     filler texts → pick randomly from top-3 matches.  Uses
     paraphrase-multilingual-MiniLM-L12-v2 (~118 MB, CPU).  Added latency: ~3-5 ms.
  2. TF-IDF intent bucket (fallback): classic classify() → random variant.

Generate WAVs first: venv\Scripts\python scripts/generate_fillers.py
"""
from __future__ import annotations

import asyncio
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator

import numpy as np

log = logging.getLogger(__name__)

# ─── Sentence-transformers (optional) ───────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    _HAS_SBERT = True
except ImportError:  # pragma: no cover
    _HAS_SBERT = False

_SBERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 118 MB, 50+ langs incl. AR
_sbert: Any = None  # loaded lazily on first semantic call

# Pre-computed per-language: matrix (n_fillers × 384) + ordered item list
_sem_embeddings: dict[str, np.ndarray] = {}      # lang → float32 (n, 384)
_sem_items: dict[str, list[tuple[bytes, str]]] = {}  # lang → [(wav, text), ...]
_TOP_K = 3  # pick randomly from this many top matches

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

# ─── Semantic descriptions ──────────────────────────────────────────────────
# Used ONLY for SBERT embedding (not spoken, not sent to LLM).
# Richer domain vocabulary → better cosine-sim matches than the short spoken phrases.
# One description per filler variant, same order as _FILLER_TEXTS.
_FILLER_DESCRIPTIONS: dict[str, dict[str, list[str]]] = {
    # Design rule: each intent's keywords must NOT appear in other intents' keywords.
    # billing: financial/payment vocabulary only — NO utility names (electricity/water)
    # outage: failure/interruption vocabulary — NO water (conflicts with service_request)
    # service_request: application/registration vocabulary — NO cut/disconnect alone
    "billing": {
        "en": [
            "bill invoice payment due amount owe charges tariff cost",
            "outstanding balance overdue payment statement monthly charges",
            "bill total current balance what is my bill tariff amount",
            "invoice details billing summary charges breakdown",
            "billing history previous statements account statement",
            "bill explanation calculated tariff usage cost high bill expensive",
        ],
        "ar": [
            "فاتورة مبلغ مستحق رسوم تعريفة دفع تكلفة",
            "رصيد مستحق متأخر كشف حساب رسوم شهرية",
            "إجمالي الفاتورة كم أدفع ما هي الرسوم",
            "تفاصيل الفاتورة ملخص الرسوم",
            "سجل الفواتير كشوفات سابقة كشف الحساب",
            "شرح الفاتورة حساب الرسوم تكلفة مرتفعة فاتورة غالية",
        ],
    },
    "location": {
        "en": [
            "happiness center office address nearest branch location directions",
            "service center nearby customer care walk-in visit",
            "branch address office hours DEWA branch nearest",
            "where is DEWA office find center directions",
            "closest happiness center near me visit in person",
            "DEWA office map address service center location",
        ],
        "ar": [
            "مركز السعادة مكتب عنوان أقرب فرع موقع اتجاهات",
            "مركز الخدمة قريب خدمة العملاء زيارة شخصية",
            "عنوان الفرع ساعات العمل أقرب فرع ديوا",
            "أين مكتب ديوا إيجاد مركز اتجاهات",
            "أقرب مركز سعادة بالقرب زيارة شخصية",
            "خريطة ديوا عنوان مركز الخدمة موقع",
        ],
    },
    "outage": {
        "en": [
            "power outage blackout no power lights off failure cut",
            "supply interrupted stopped not working disruption fault",
            "scheduled maintenance planned shutdown when will it resume",
            "power failure sudden blackout building dark no electricity",
            "report outage restoration time when will power return",
            "supply not working fault interruption no service",
        ],
        "ar": [
            "انقطاع التيار بلاك اوت لا طاقة ظلام انقطاع قطع",
            "انقطاع الخدمة توقف الإمداد عطل",
            "صيانة مجدولة إغلاق مخطط متى تعود",
            "انقطاع مفاجئ المبنى مظلم لا كهرباء",
            "الإبلاغ عن انقطاع وقت الاستعادة متى يعود",
            "الإمداد لا يعمل عطل انقطاع لا خدمة",
        ],
    },
    "account": {
        "en": [
            "account details profile number meter reading login",
            "smart meter update contact phone number account settings",
            "register link account customer profile DEWA app",
            "account management personal details settings",
            "meter number connection status account information",
            "account access login customer app DEWA profile",
        ],
        "ar": [
            "حساب تفاصيل ملف رقم عداد قراءة تسجيل دخول",
            "عداد ذكي تحديث بيانات تواصل رقم هاتف إعدادات",
            "تسجيل ربط حساب ملف عميل تطبيق ديوا",
            "إدارة الحساب بيانات شخصية إعدادات",
            "رقم عداد توصيل حالة معلومات الحساب",
            "الوصول للحساب تسجيل دخول تطبيق ديوا ملف",
        ],
    },
    "service_request": {
        "en": [
            "apply subscribe register new connection activate utility setup",
            "transfer relocate move address service application form",
            "reconnect reactivate suspended service request application",
            "new meter setup upgrade capacity new premises connection",
            "submit application new subscription utility activation",
            "new home move setup register DEWA service activation",
        ],
        "ar": [
            "تقديم اشتراك تسجيل توصيل جديد تفعيل إعداد",
            "نقل تحويل عنوان جديد طلب خدمة نموذج",
            "إعادة توصيل تفعيل خدمة موقوفة طلب",
            "عداد جديد ترقية سعة توصيل عقار جديد",
            "تقديم طلب اشتراك جديد تفعيل مرافق",
            "منزل جديد انتقال إعداد تسجيل خدمات ديوا",
        ],
    },
    "sustainability": {
        "en": [
            "install solar panels photovoltaic rooftop Shams Dubai net metering",
            "green energy renewable clean power carbon footprint",
            "EV charger electric vehicle DEWA green charger",
            "energy saving reduce consumption efficiency tips",
            "sustainability program clean energy zero carbon green building",
            "conservation recycling environmental eco-friendly",
        ],
        "ar": [
            "تركيب ألواح شمسية طاقة شمسية شمس دبي قياس صافي",
            "طاقة خضراء متجددة نظيفة بصمة كربونية",
            "شاحن سيارات كهربائية ديوا أخضر",
            "توفير طاقة تقليل استهلاك كفاءة نصائح",
            "برنامج استدامة طاقة نظيفة صفر كربون مبنى أخضر",
            "ترشيد إعادة تدوير بيئي صديق للبيئة",
        ],
    },
    "general": {
        "en": [
            "hello hi greeting general question help",
            "what can you do capabilities how to use assistant",
            "general inquiry information question",
            "good morning afternoon pleasantry small talk",
            "thank you okay understood got it",
            "help assist support please",
        ],
        "ar": [
            "مرحبا أهلا تحية سؤال عام مساعدة",
            "ماذا تستطيع قدرات كيف تستخدم مساعد",
            "استفسار عام معلومات سؤال",
            "صباح الخير مساء الخير تحية حديث",
            "شكراً حسناً مفهوم تمام",
            "مساعدة دعم من فضلك",
        ],
    },
}

# Filler texts — MUST match generate_fillers.py FILLERS exactly (same order = same WAV index).
# Variants 1-2: clean/short. 3-4: light prosody. 5-6: heavy prosody (most human-sounding).
_FILLER_TEXTS: dict[str, dict[str, list[str]]] = {
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

    # Build semantic index after WAVs are loaded
    _build_semantic_index()


def _build_semantic_index() -> None:
    """
    Pre-embed all filler descriptions so per-call latency is just one query embed +
    dot-product (~8 ms on CPU).

    Embeddings use _FILLER_DESCRIPTIONS (rich domain vocabulary) not the short spoken
    phrases — this gives much better cosine-sim matching for domain-specific queries
    like "solar panels" or "EV charger" vs the generic "Let me look that up".
    """
    global _sbert
    if not _HAS_SBERT:
        log.warning(
            "sentence-transformers not installed — semantic filler matching disabled "
            "(pip install sentence-transformers to enable)"
        )
        return

    import time
    t0 = time.perf_counter()
    _sbert = _SentenceTransformer(_SBERT_MODEL)
    load_ms = int((time.perf_counter() - t0) * 1000)

    for lang in ("en", "ar"):
        items: list[tuple[bytes, str]] = []
        desc_texts: list[str] = []
        for intent in _INTENTS:
            audio_list = _cache[lang].get(intent, [])
            spoken_list = _text_cache[lang].get(intent, [])
            desc_list = _FILLER_DESCRIPTIONS.get(intent, {}).get(lang, [])
            for idx, wav in enumerate(audio_list):
                spoken = spoken_list[idx] if idx < len(spoken_list) else ""
                # Use description for embedding; spoken text goes to LLM
                desc = desc_list[idx] if idx < len(desc_list) else spoken
                items.append((wav, spoken))
                desc_texts.append(desc)
        if desc_texts:
            embs = _sbert.encode(desc_texts, normalize_embeddings=True, show_progress_bar=False)
            _sem_embeddings[lang] = embs.astype(np.float32)
            _sem_items[lang] = items

    total = sum(len(v) for v in _sem_items.values())
    log.info(
        "semantic filler index built",
        extra={"model": _SBERT_MODEL, "load_ms": load_ms, "total_fillers": total},
    )


import threading as _threading

_load_lock = _threading.Lock()
_loaded = False


def _ensure_loaded() -> None:
    """Thread-safe loader — safe to call from the pre-warm thread and the pipeline."""
    global _loaded
    if _loaded:
        return
    with _load_lock:
        if _loaded:   # double-check after acquiring lock
            return
        _load()
        _loaded = True


# ─── Background pre-warm ─────────────────────────────────────────────────────
# Start loading WAVs + SBERT model at import time so the 5-10 s model load
# happens during app startup, not during the first utterance.
# Daemon thread — does not block import or the asyncio event loop.
import os as _os
if _os.environ.get("HF_FILLER_ENABLED", "1") != "0":
    _preload_thread = _threading.Thread(target=_ensure_loaded, daemon=True, name="filler-preload")
    _preload_thread.start()


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


def pick_filler_semantic(
    transcription: str,
    language: str,
    *,
    fallback_intent: str = "general",
) -> tuple[bytes | None, str]:
    """
    Select filler by semantic similarity to the transcription.

    Algorithm:
    1. Embed transcription with multilingual MiniLM (~3-5 ms, CPU).
    2. Dot-product against pre-embedded filler matrix (normalised → cosine sim).
    3. Pick randomly from the top-_TOP_K matches (adds variety while staying relevant).
    4. Falls back to TF-IDF intent bucket if sentence-transformers unavailable.

    Added latency vs intent-bucket approach: ~3-5 ms (negligible).
    """
    import time
    _ensure_loaded()

    lang = language if language in _sem_embeddings else "en"

    # ── Semantic path ─────────────────────────────────────────────────────────
    if _sbert is not None and lang in _sem_embeddings:
        t0 = time.perf_counter()
        query_emb = _sbert.encode(
            [transcription], normalize_embeddings=True, show_progress_bar=False
        )  # (1, 384)
        sims: np.ndarray = (query_emb @ _sem_embeddings[lang].T).flatten()
        embed_ms = int((time.perf_counter() - t0) * 1000)

        n = len(sims)
        top_k = min(_TOP_K, n)
        # argpartition is O(n) — fine for n≤50
        top_indices = np.argpartition(sims, -top_k)[-top_k:]
        chosen = int(random.choice(top_indices))
        wav, text = _sem_items[lang][chosen]

        log.debug(
            "semantic filler selected",
            extra={
                "lang": lang,
                "embed_ms": embed_ms,
                "top_score": float(sims[top_indices].max()),
                "chosen_score": float(sims[chosen]),
                "text": text,
            },
        )
        return wav, text

    # ── Fallback: TF-IDF intent bucket ────────────────────────────────────────
    log.debug("semantic filler unavailable, falling back to intent bucket")
    return pick_filler(fallback_intent, language)
