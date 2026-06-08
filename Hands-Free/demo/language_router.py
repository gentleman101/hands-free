"""
LanguageRouter — sits between STT and context_aggregator.user() in the pipeline.

Per TranscriptionFrame it:
  1. Detects EN vs AR from the frame's language field.
  2. Optionally picks + plays a filler WAV immediately (before LLM starts).
  3. Swaps LLMContext.messages[0] to the language-specific system prompt.
     If a filler was played, appends the continuation hint so the LLM
     completes the filler sentence rather than starting fresh.
  4. Updates MacLLMService and SilmaTTSService language.
  5. Forwards the TranscriptionFrame unchanged for normal pipeline processing.

All other frame types are passed through untouched.
"""
from __future__ import annotations

import logging

from pipecat.frames.frames import Frame, TranscriptionFrame, TTSAudioRawFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transcriptions.language import Language

log = logging.getLogger(__name__)

# ── Language-specific system prompts (EXP-017: two prompts saves ~80 tokens/call)
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

# Filler continuation hint — tells LLM to complete the sentence, not start fresh
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


def _lang_str(language: Language | None) -> str:
    """Map Pipecat Language enum → 'en' or 'ar' string."""
    if language is None:
        return "en"
    name = language.name.lower() if hasattr(language, "name") else str(language).lower()
    return "ar" if name.startswith("ar") else "en"


def _build_system_prompt(lang: str, filler_text: str | None = None) -> str:
    base = _SYSTEM_PROMPT_AR if lang == "ar" else _SYSTEM_PROMPT_EN
    if filler_text:
        suffix = _FILLER_SUFFIX_AR if lang == "ar" else _FILLER_SUFFIX_EN
        return base + suffix.format(filler=filler_text)
    return base


class LanguageRouter(FrameProcessor):
    """
    Per-utterance: language detection + system prompt swap + optional filler.
    Wire between STT and context_aggregator.user() in the pipeline.
    """

    def __init__(
        self,
        context: LLMContext,
        llm_svc,
        tts_svc,
        *,
        filler_enabled: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._context = context
        self._llm = llm_svc
        self._tts = tts_svc
        self._filler_enabled = filler_enabled

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            lang = _lang_str(frame.language)
            filler_text: str | None = None

            # ── 1. Pick + play filler immediately (before LLM starts) ────────
            if self._filler_enabled:
                try:
                    from filler_player import pick_filler, wav_to_pcm
                    wav_bytes, filler_text = pick_filler(frame.text, lang)
                    if wav_bytes:
                        pcm, sr = wav_to_pcm(wav_bytes)
                        # Push filler audio downstream BEFORE TranscriptionFrame
                        # so it plays while LLM network call is in-flight
                        await self.push_frame(
                            TTSAudioRawFrame(audio=pcm, sample_rate=sr, num_channels=1),
                            direction,
                        )
                        log.info("filler.played", extra={"text": filler_text, "lang": lang})
                except Exception:
                    log.exception("filler.failed — continuing without filler")
                    filler_text = None

            # ── 2. Swap system prompt (language-specific + optional filler hint)
            prompt = _build_system_prompt(lang, filler_text)
            if self._context.messages:
                self._context.messages[0] = {"role": "system", "content": prompt}
            else:
                self._context.messages.insert(0, {"role": "system", "content": prompt})

            # ── 3. Update service languages ──────────────────────────────────
            if hasattr(self._llm, "set_language"):
                self._llm.set_language(lang)
            self._tts.set_language(lang)

            lang_flag = "🇦🇪" if lang == "ar" else "🇬🇧"
            print(f"📝  {lang_flag} [{lang}] {frame.text}", flush=True)
            log.info(
                "language_router.routed",
                extra={
                    "lang": lang,
                    "text_preview": frame.text[:60],
                    "filler": filler_text or "none",
                },
            )

        await self.push_frame(frame, direction)
