"""
Lightweight TF-IDF intent classifier for DEWA filler selection.

Trained on a small set of bilingual (EN/AR) example queries per intent.
Inference: <5ms. No GPU. Falls back to "general" below confidence threshold.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ─── Training data ───────────────────────────────────────────────────────────

_EXAMPLES: dict[str, list[str]] = {
    "billing": [
        "what is my bill", "how much do I owe", "show me my invoice",
        "my electricity bill", "payment due", "outstanding amount",
        "bill for this month", "how much is my bill", "charge on my account",
        "فاتورتي", "كم فاتورتي", "المبلغ المستحق", "فاتورة الكهرباء",
        "فاتورة المياه", "كم عليّ", "رسوم الحساب", "الفاتورة الشهرية",
    ],
    "location": [
        "where is the happiness center", "nearest DEWA office", "branch location",
        "where can I go", "find a service center", "DEWA office near me",
        "where are you located", "customer service location", "walk in center",
        "أين مركز السعادة", "أقرب مكتب ديوا", "موقع الفرع", "أين يمكنني الذهاب",
        "مركز خدمة العملاء", "أين ديوا", "الفروع القريبة",
    ],
    "outage": [
        "power is out", "no electricity", "water cut", "outage in my area",
        "when will power come back", "report power outage", "no water supply",
        "electricity is off", "planned maintenance", "blackout",
        "انقطاع الكهرباء", "لا كهرباء", "انقطاع المياه", "متى يعود التيار",
        "بلاغ انقطاع", "صيانة مجدولة", "انقطاع في منطقتي",
    ],
    "account": [
        "my account details", "update my information", "change my number",
        "account number", "meter reading", "smart meter", "my profile",
        "register new account", "link my account", "DEWA account login",
        "حسابي", "تفاصيل الحساب", "تحديث بياناتي", "رقم العداد",
        "قراءة العداد", "حساب ديوا", "تسجيل حساب جديد", "ربط الحساب",
    ],
    "service_request": [
        "new connection", "move my connection", "disconnect service",
        "service request", "apply for electricity", "new water connection",
        "activate service", "temporary disconnection", "reconnect",
        "توصيل جديد", "نقل الاشتراك", "قطع الخدمة", "طلب خدمة",
        "توصيل كهرباء", "إعادة التوصيل", "تفعيل الخدمة",
    ],
    "sustainability": [
        "solar panels", "green energy", "save electricity", "shams dubai",
        "EV charger", "carbon footprint", "renewable energy", "energy saving tips",
        "ألواح شمسية", "طاقة خضراء", "توفير الكهرباء", "شمس دبي",
        "شاحن سيارة كهربائية", "طاقة متجددة", "نصائح توفير الطاقة",
    ],
    "general": [
        "hello", "hi", "help me", "what can you do", "tell me something",
        "I have a question", "good morning", "thank you", "ok",
        "مرحبا", "السلام عليكم", "ساعدني", "ماذا تستطيع", "شكرا", "حسنا",
    ],
}

# ─── Model ───────────────────────────────────────────────────────────────────

_CONFIDENCE_THRESHOLD = 0.15  # below this → fall back to "general"

def _build_model():
    texts, labels = [], []
    for intent, examples in _EXAMPLES.items():
        for ex in examples:
            texts.append(ex)
            labels.append(intent)

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",   # char n-grams — works for both Arabic and English
        ngram_range=(2, 4),
        min_df=1,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix, labels

_vectorizer, _matrix, _labels = _build_model()


# ─── Public API ──────────────────────────────────────────────────────────────

def classify(text: str) -> str:
    """Return intent label for the given transcription. Always returns a string."""
    if not text.strip():
        return "general"

    vec = _vectorizer.transform([text.lower()])
    sims = cosine_similarity(vec, _matrix).flatten()
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    if best_score < _CONFIDENCE_THRESHOLD:
        return "general"

    return _labels[best_idx]
