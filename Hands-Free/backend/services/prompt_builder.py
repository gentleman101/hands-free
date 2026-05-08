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

Keep replies SHORT: 1 sentence, 2 max. Under 20 words. Voice users hate long answers.

Rules:
- Use contractions: it's, you're, I'll
- No lists, no bullet points
- Never start with "Certainly", "Of course", or "Absolutely"
- Don't introduce yourself unless asked

Domain: DEWA services — billing, outages, service requests, accounts, sustainability.
If you don't know, say so briefly and offer to connect with DEWA support."""


_SYSTEM_PROMPT_AR = """أنت رمّاس، المساعد الصوتي الذكي لديوا — هيئة كهرباء ومياه دبي.

أجب دائماً بالعربية فقط. الردود القصيرة جداً: جملة واحدة، جملتان كحدٍّ أقصى. أقل من ٢٠ كلمة.

القواعد:
- لا قوائم ولا نقاط
- لا تبدأ بـ"بالتأكيد" أو "بكل سرور"
- لا تعرّف بنفسك إلا إذا سُئلت

المجال: خدمات ديوا — الفواتير، البلاغات، طلبات الخدمة، الحسابات، الاستدامة.
إذا لم تعرف الإجابة، قل ذلك بإيجاز واعرض التواصل مع دعم ديوا."""


def build_messages(
    transcription: str,
    *,
    language: str = "en",
    conversation_history: list[dict[str, str]] | None = None,
    filler_text: str | None = None,
) -> list[dict[str, str]]:
    """
    Build the messages list for the LLM call.

    Selects the language-specific system prompt so the model receives
    only the instructions relevant to the current language.

    If filler_text is provided it is injected as a partial assistant turn
    so the model completes the thought naturally — the filler audio and the
    LLM reply stitch together as one seamless utterance.

    Args:
        transcription: What the user said.
        language: Detected language code ("en" | "ar").
        conversation_history: Prior turns as [{"role": "user"|"assistant", "content": str}].
        filler_text: The filler phrase already played to the user (e.g. "Let me check your bill.").

    Returns:
        Messages list ready for the LLM client.
    """
    prompt = _SYSTEM_PROMPT_AR if language == "ar" else _SYSTEM_PROMPT_EN
    messages: list[dict[str, str]] = [{"role": "system", "content": prompt}]

    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": transcription})

    if filler_text:
        # Seed the assistant turn with the filler so the reply flows from it
        messages.append({"role": "assistant", "content": filler_text})

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


def get_system_prompt(language: str = "en") -> str:
    """Return the system prompt for the given language (for testing / display)."""
    return _SYSTEM_PROMPT_AR if language == "ar" else _SYSTEM_PROMPT_EN
