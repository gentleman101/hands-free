"""
SILMA TTS as a Pipecat TTSService.

Plugs SILMA (in-process F5-TTS-based EN+AR voice cloner) into Pipecat's
TTSService contract. Pipecat handles sentence aggregation, interruption,
and audio frame plumbing — we just provide the synthesis.

The model is loaded lazily on first run_tts() because Vocos's cuDNN init
must happen inside the main thread of the running event loop (verified
in the spike). After that, inference can run in an executor without
issue.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
from collections.abc import AsyncIterator

from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType, BaseTextAggregator
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language

log = logging.getLogger(__name__)


# Reference clips live in the main repo's assets/voices directory.
# The worktree shares that directory via the main repo path.
_MAIN_REPO = Path(__file__).resolve().parents[2] / "Hands-Free"
if not (_MAIN_REPO / "assets" / "voices").exists():
    # Fallback: try resolving relative to the worktree root
    _MAIN_REPO = Path(r"C:\Users\dewac\Documents\Naman\Hands-Free")
_VOICES_DIR = _MAIN_REPO / "assets" / "voices"

_REF_FILES: dict[str, str] = {
    "en": "f5_ref_en_eric.wav",       # en-US-EricNeural (Edge TTS)
    "ar": "ar_ref_hamdan_uae.wav",    # ar-AE-HamdanNeural (Edge TTS, UAE)
}

_REF_TEXTS: dict[str, str] = {
    "en": "Welcome to DEWA. I can help with your account, billing, and service requests today.",
    "ar": "أهلاً بكم في ديوا. أنا هنا لمساعدتكم في استفساراتكم.",
}


# EXP-005 equivalent for Pipecat: split on commas + Arabic comma (،) in addition to
# the standard sentence-ending punctuation (.?!). This ships the first clause to
# SILMA ~200-400ms earlier than waiting for a full sentence boundary.
_COMMA_CHARS = frozenset({",", "،"})   # EN comma + Arabic comma
_MIN_CHARS_BEFORE_COMMA = 25           # must accumulate ≥25 chars before splitting on
                                       # a comma. 8 was too low — list commas like
                                       # "billing, outages, or…" each reset the buffer,
                                       # leaving single words ("outages,") as isolated
                                       # SILMA calls which produce artifact noise.
                                       # 25 ensures only genuine long opening clauses
                                       # fire early; list items fall through to the
                                       # sentence-end boundary instead.


class CommaAwareTextAggregator(SimpleTextAggregator):
    """SimpleTextAggregator + comma/Arabic-comma sentence boundaries (EXP-005)."""

    async def aggregate(self, text: str) -> AsyncIterator[Aggregation]:
        """Yield sentences; also split on commas when buffer is long enough."""
        # In TOKEN mode fall through to parent (shouldn't be used, but safe).
        if self._aggregation_type == AggregationType.TOKEN:
            async for agg in super().aggregate(text):
                yield agg
            return

        for char in text:
            self._text += char

            # Comma split: fire early if we have a meaningful clause.
            if char in _COMMA_CHARS and len(self._text.rstrip()) >= _MIN_CHARS_BEFORE_COMMA:
                result = self._text.strip()
                self._text = ""
                self._needs_lookahead = False
                if result:
                    yield Aggregation(text=result, type=AggregationType.SENTENCE)
                continue

            # Standard sentence-boundary detection (from SimpleTextAggregator).
            result = await self._check_sentence_with_lookahead(char)
            if result:
                yield result


def _lang_code(lang: Language | str) -> str:
    """Normalise a Pipecat Language enum (or string) to "en"/"ar"."""
    if isinstance(lang, str):
        return lang[:2].lower()
    name = lang.name.lower() if hasattr(lang, "name") else str(lang).lower()
    if name.startswith("ar"):
        return "ar"
    return "en"


class SilmaTTSService(TTSService):
    """
    In-process SILMA TTS — F5-TTS-based, supports EN + AR.

    Args:
        language: Pipecat Language enum or string. Default Language.EN.
        speed: SILMA speech rate. 1.0 = natural, 1.3 = slightly faster.
        sample_rate: Output rate; SILMA always emits 24 kHz.
    """

    def __init__(
        self,
        *,
        language: Language | str = Language.EN,
        speed: float = 1.0,
        sample_rate: int = 24000,
        **kwargs,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._speed = speed
        self._silma = None
        self._load_lock = asyncio.Lock()
        # Replace Pipecat's default SimpleTextAggregator with one that also
        # splits on commas (EXP-005: ships first clause 200-400ms earlier).
        self._text_aggregator = CommaAwareTextAggregator()
        # Cache ref file + text so _synth_blocking never does a dict lookup.
        self._language: str = ""
        self._ref_file: str = ""
        self._ref_text: str = ""
        self.set_language(language)

    def set_language(self, language: Language | str) -> None:
        """Hot-swap synthesis language (per-utterance ok)."""
        lang = _lang_code(language)
        if lang == self._language:
            return
        self._language = lang
        self._ref_file = str(_VOICES_DIR / _REF_FILES[lang])
        self._ref_text = _REF_TEXTS[lang]

    async def _ensure_loaded(self) -> None:
        if self._silma is not None:
            return
        async with self._load_lock:
            if self._silma is None:
                from silma_tts.api import SilmaTTS

                log.info("silma.loading")
                # Block the executor — cuDNN init still happens on main
                # thread because run_in_executor with None uses the loop's
                # default executor which is *separate* from main, but
                # importantly the calling thread (main) is what owns the
                # CUDA context. The verified pattern from the spike is to
                # call SilmaTTS() inline on main; do that here.
                self._silma = SilmaTTS()
                log.info("silma.loaded")

    def _synth_blocking(self, text: str) -> tuple[np.ndarray, int]:
        """Run one SILMA inference. Returns (float32 PCM, sample_rate)."""
        _noop = lambda *a, **k: None  # noqa: E731

        wav, sr, _ = self._silma.infer(
            ref_file=self._ref_file,
            ref_text=self._ref_text,
            gen_text=text,
            file_wave=None,              # skip disk write — wav returned directly
            speed=self._speed + (0.1 if self._language == "ar" else 0.0),  # AR +0.1 for natural pacing; EN at base speed
            force_tashkeel=(self._language == "ar"),  # force diacritics on Arabic text for cleaner pronunciation
            normalize_numbers=(self._language == "ar"),  # Arabic needs NeMo to expand digits → words; EN LLM already writes spoken form
            nfe_step=16,              # 16 ODE steps — full quality (8 confirmed robotic even with clean ref audio)
            show_info=_noop,          # suppress SILMA's print() calls cleanly
            progress=_noop,           # suppress tqdm progress bar
        )

        # Normalise to mono float32 numpy (avoid copy if already float32)
        if hasattr(wav, "cpu"):
            wav = wav.cpu().numpy()
        if not isinstance(wav, np.ndarray) or wav.dtype != np.float32:
            wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=0) if wav.shape[0] < wav.shape[-1] else wav.mean(axis=-1)
        return wav, int(sr)

    async def run_tts(
        self, text: str, context_id: str
    ) -> AsyncGenerator[Frame, None]:
        """Synthesise one text chunk; yield audio frames."""
        text = (text or "").strip()
        if not text:
            return

        await self._ensure_loaded()

        yield TTSStartedFrame(context_id=context_id)
        try:
            wav, sr = await asyncio.get_event_loop().run_in_executor(
                None, self._synth_blocking, text
            )
        except Exception as e:
            log.exception("silma.synth_failed")
            yield ErrorFrame(error=f"SILMA TTS failed: {e}")
            yield TTSStoppedFrame(context_id=context_id)
            return

        # Append 200ms silence so the transport fully drains before TTSStoppedFrame arrives
        tail = np.zeros(sr // 5, dtype=np.float32)  # 200ms @ sample_rate
        wav = np.concatenate([wav, tail])

        # Convert float32 [-1,1] → int16 PCM
        pcm_i16 = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)

        log.debug(
            "silma.synth_done lang=%s chars=%d audio_ms=%d",
            self._language, len(text), int(len(wav) / sr * 1000),
        )

        # Yield in 20ms chunks so the output transport starts playing
        # the first packet immediately rather than waiting for the whole
        # audio blob to be processed as a single frame.
        chunk_samples = sr // 50  # 20ms @ 24kHz = 480 samples
        for i in range(0, len(pcm_i16), chunk_samples):
            chunk = pcm_i16[i : i + chunk_samples].tobytes()
            yield TTSAudioRawFrame(audio=chunk, sample_rate=sr, num_channels=1)

        yield TTSStoppedFrame(context_id=context_id)
