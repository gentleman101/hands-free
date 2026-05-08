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


def _load() -> None:
    for lang in ("en", "ar"):
        _cache[lang] = {}
        for intent in _INTENTS:
            clips: list[bytes] = []
            for i in range(1, _VARIANTS + 1):
                path = _FILLERS_DIR / lang / f"{intent}_{i}.wav"
                if path.exists():
                    clips.append(path.read_bytes())
                else:
                    log.debug("filler missing", extra={"path": str(path)})
            _cache[lang][intent] = clips
    loaded = sum(len(v) for d in _cache.values() for v in d.values())
    log.info("fillers loaded", extra={"total": loaded})


def _ensure_loaded() -> None:
    if not _cache:
        _load()


def get_filler_bytes(intent: str, language: str) -> bytes | None:
    """Return raw WAV bytes for a random filler matching intent + language."""
    _ensure_loaded()
    lang = language if language in _cache else "en"
    bucket = _cache[lang].get(intent) or _cache[lang].get("general") or []
    if not bucket:
        return None
    return random.choice(bucket)


async def stream_filler(intent: str, language: str, chunk_size: int = 4096) -> AsyncIterator[bytes]:
    """
    Yield filler WAV in chunks — same interface as tts.synthesize_stream.
    Pipeline can yield these first, then switch to real TTS chunks.
    """
    data = get_filler_bytes(intent, language)
    if not data:
        return
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]
        await asyncio.sleep(0)  # allow other coroutines to run between chunks
