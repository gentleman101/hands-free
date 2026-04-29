"""
Builds LLM messages for the Rammas voice pipeline.

Two dedicated prompts — one per language — so each call only sends
what's needed. No bilingual hedging, no language directive overhead.
Prompt selection is done in build_messages() based on detected language.
"""
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
) -> list[dict[str, str]]:
    """
    Build the messages list for the LLM call.

    Selects the language-specific system prompt so the model receives
    only the instructions relevant to the current language — fewer tokens,
    faster TTFT, and no cross-language confusion.

    Args:
        transcription: What the user said.
        language: Detected language code ("en" | "ar").
        conversation_history: Prior turns as [{"role": "user"|"assistant", "content": str}].

    Returns:
        Messages list ready for the Ollama client.
    """
    prompt = _SYSTEM_PROMPT_AR if language == "ar" else _SYSTEM_PROMPT_EN
    messages: list[dict[str, str]] = [{"role": "system", "content": prompt}]

    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": transcription})

    log.debug(
        "prompt built",
        extra={"language": language, "turns": len(messages), "input_chars": len(transcription)},
    )
    return messages


def get_system_prompt(language: str = "en") -> str:
    """Return the system prompt for the given language (for testing / display)."""
    return _SYSTEM_PROMPT_AR if language == "ar" else _SYSTEM_PROMPT_EN
