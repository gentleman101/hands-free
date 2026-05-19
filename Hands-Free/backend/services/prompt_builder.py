"""
Builds LLM messages for the Rammas voice pipeline.

EXP-017: Two language-specific prompts (one EN-only, one AR-only) — saves
~80 tokens per call by not sending bilingual hedging instructions every turn.
EXP-006: Brevity-first prompt (1 sentence, <20 words).

Selection happens in build_messages() based on detected language.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


_SYSTEM_PROMPT_EN = """You are Rammas, DEWA's AI voice assistant — Dubai Electricity and Water Authority.

CRITICAL: Reply in English only. Never switch to Arabic regardless of the question.

Keep replies SHORT: 1 sentence, 2 max. Under 20 words.

Rules:
- Use contractions: it's, you're, I'll
- No lists, no bullet points
- Never start with "Certainly", "Of course", or "Absolutely"
- Don't introduce yourself unless asked
- Don't repeat the user's question back
- Never say "I'll check", "let me look", "I'll find" — just give the answer directly, even if approximate

Domain: DEWA services — billing, outages, service requests, accounts, sustainability.
If you don't know, make a plausible short response."""


_SYSTEM_PROMPT_AR = """أنت رمّاس، المساعد الصوتي الذكي لديوا — هيئة كهرباء ومياه دبي.

CRITICAL: رُدَّ بالعربية فقط. ممنوع استخدام أي كلمة إنجليزية في ردك مهما كان السؤال.

الردود القصيرة جداً: جملة واحدة، جملتان كحدٍّ أقصى. أقل من ٢٠ كلمة.

القواعد:
- لا قوائم ولا نقاط
- لا تبدأ بـ"بالتأكيد" أو "بكل سرور" أو "بالتأكيد"
- لا تعرّف بنفسك إلا إذا سُئلت
- لا تكرر سؤال المستخدم
- لا تقل "سأتحقق" أو "سأبحث" أو "سأجد" — أجب مباشرةً ولو بشكل تقريبي

المجال: خدمات ديوا — الفواتير، البلاغات، طلبات الخدمة، الحسابات، الاستدامة.
إذا لم تعرف الإجابة، قدّم رداً قصيراً ومعقولاً."""


_FILLER_SUFFIX_EN = """

The user already heard this spoken aloud before your reply: "{filler}"
Continue directly from that — your response is the second half of that sentence.
Do not repeat it. Do not greet. Do not start a new sentence.
Example: filler "Let me pull that up," → you reply "your balance is AED 342, due on the 15th.\""""


_FILLER_SUFFIX_AR = """

المستخدم سمع هذه العبارة بالفعل قبل ردك: "{filler}"
أكمل مباشرةً من بعدها — ردك هو النصف الثاني من تلك الجملة.
لا تكررها. لا تسلّم. لا تبدأ جملة جديدة.
مثال: عبارة "دعني أجلب ذلك،" → تُكمل "رصيدك ٣٤٢ درهماً، مستحق في ١٥ من الشهر.\""""


def build_messages(
    transcription: str,
    *,
    language: str = "en",
    conversation_history: list[dict[str, str]] | None = None,
    filler_text: str | None = None,
) -> list[dict[str, str]]:
    """
    Build the messages list for the LLM call.

    Selects the language-specific system prompt. If filler_text is provided,
    the continuation instruction is appended to the system prompt so the model
    knows to complete the thought — not start fresh.
    """
    base = _SYSTEM_PROMPT_AR if language == "ar" else _SYSTEM_PROMPT_EN

    if filler_text:
        suffix = _FILLER_SUFFIX_AR if language == "ar" else _FILLER_SUFFIX_EN
        prompt = base + suffix.format(filler=filler_text)
    else:
        prompt = base

    messages: list[dict[str, str]] = [{"role": "system", "content": prompt}]

    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": transcription})

    log.debug(
        "prompt built",
        extra={
            "language": language,
            "turns": len(messages),
            "input_chars": len(transcription),
            "filler": filler_text or "",
        },
    )
    return messages


def get_system_prompt(language: str = "en", filler_text: str | None = None) -> str:
    """Return the system prompt for the given language (for testing / display)."""
    base = _SYSTEM_PROMPT_AR if language == "ar" else _SYSTEM_PROMPT_EN
    if filler_text:
        suffix = _FILLER_SUFFIX_AR if language == "ar" else _FILLER_SUFFIX_EN
        return base + suffix.format(filler=filler_text)
    return base
