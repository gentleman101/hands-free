"""
Builds LLM messages for the bilingual voice pipeline.

System prompt teaches Gemma to write text that Fish Speech renders naturally:
- Inline prosody tags: [laugh], [emphasis], [whisper], [sigh], [excited]
- Punctuation as rhythm: commas = pause, ... = thinking, — = cut
- 1-2 sentences max, contractions, no lists, respond in user's language
"""
import logging

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a helpful bilingual voice assistant.
Always respond in the exact same language the user spoke.

Format responses for natural spoken delivery via text-to-speech:
- Keep responses to 1-2 short sentences maximum
- Use contractions: it's, you're, that's, I'll, we'll
- Never write lists, bullet points, or numbered items
- Use commas for natural breath pauses
- Use ... for a thinking pause (e.g. "Let me think... yes, that works")
- Use — for an abrupt mid-thought pause
- Add [laugh] where a real speaker would naturally laugh
- Add [emphasis] before a word you want stressed
- Add [sigh] before a sentence that warrants it
- Never start with "Certainly", "Of course", "Great question", or "Absolutely"

Domain: HR workflows, task management, and general office queries.
When you don't know, say so briefly and offer to help another way."""


def build_messages(
    transcription: str,
    *,
    language: str = "en",
    conversation_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """
    Build the messages list for the LLM call.

    Args:
        transcription: What the user said.
        language: Detected language code ("en" | "ar").
        conversation_history: Prior turns as [{"role": "user"|"assistant", "content": str}].

    Returns:
        Messages list ready for openai client.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": transcription})

    log.debug(
        "prompt built",
        extra={"language": language, "turns": len(messages), "input_chars": len(transcription)},
    )
    return messages


def get_system_prompt() -> str:
    """Return the system prompt string (for testing / display)."""
    return _SYSTEM_PROMPT
