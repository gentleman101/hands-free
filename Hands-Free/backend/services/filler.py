"""
Filler audio player.

Plays pre-recorded "thinking" audio while the LLM is generating the real
response, so the user hears no awkward silence. Cancelled the moment the
first real TTS sentence is ready.

5+ files per language are kept in assets/fillers/ so random rotation
prevents repetition fatigue. Generate them via scripts/generate_fillers.py.
"""
from __future__ import annotations

import asyncio
import logging
import random
from pathlib import Path
from typing import Callable, Awaitable

log = logging.getLogger(__name__)


_FILLERS_DIR = Path("assets/fillers")

_FILES = {
    "en": [
        "en_let_me_check.wav",
        "en_one_moment.wav",
        "en_looking_into_it.wav",
        "en_just_a_second.wav",
        "en_processing.wav",
    ],
    "ar": [
        "ar_lahza_min_fadlek.wav",
        "ar_da3ni_atahaqaq.wav",
        "ar_thaniya_wahida.wav",
        "ar_jari_almoaalja.wav",
        "ar_intazher_qaleelan.wav",
    ],
}


# Type alias: a callable that plays raw WAV bytes (e.g. via sounddevice).
# Pipeline injects this so we don't hardcode an audio backend here.
PlayFn = Callable[[bytes], Awaitable[None]]


class FillerPlayer:
    """Loads filler WAVs once, plays a random one on demand, cancellable."""

    def __init__(self, play_fn: PlayFn, fillers_dir: Path = _FILLERS_DIR) -> None:
        self._play_fn = play_fn
        self._fillers_dir = fillers_dir
        self._cache: dict[str, list[bytes]] = {"en": [], "ar": []}
        self._task: asyncio.Task[None] | None = None
        self._load()

    def _load(self) -> None:
        for lang, files in _FILES.items():
            for fname in files:
                path = self._fillers_dir / fname
                if path.exists():
                    self._cache[lang].append(path.read_bytes())
                else:
                    log.warning("missing filler audio", extra={"path": str(path)})
        log.info(
            "filler audio loaded",
            extra={
                "en_count": len(self._cache["en"]),
                "ar_count": len(self._cache["ar"]),
            },
        )

    def has_fillers(self, language: str) -> bool:
        return bool(self._cache.get(language))

    async def start(self, language: str) -> None:
        """Pick a random filler and start playing it (non-blocking)."""
        bucket = self._cache.get(language) or self._cache.get("en") or []
        if not bucket:
            log.debug("no filler available", extra={"language": language})
            return

        audio = random.choice(bucket)
        # Cancel any in-flight filler before starting a new one
        await self.stop()

        self._task = asyncio.create_task(self._play_fn(audio))

    async def stop(self) -> None:
        """Cancel filler playback (e.g. when real TTS audio is ready)."""
        if self._task is None or self._task.done():
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("filler playback failed")
        self._task = None
