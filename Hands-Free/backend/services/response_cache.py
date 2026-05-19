"""
Response cache — short-circuits LLM + TTS for repeated/known queries.

Cache key:   md5(normalized_text + language)
Cache value: full TTS audio (WAV bytes) + response text + language

Hit  → yield audio immediately, skip LLM + TTS entirely (~50ms total)
Miss → caller runs normal pipeline, then calls store() to populate cache

Storage:
- In-memory dict for fast lookup
- WAV files on disk (data/response_cache/) for persistence across restarts
- Index JSON for metadata (text, language, created_at, hit_count)
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path

log = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "response_cache"
_INDEX_PATH = _CACHE_DIR / "index.json"


@dataclass
class CachedResponse:
    key: str
    text: str            # original user query (for debugging)
    response: str        # LLM response text
    language: str        # "en" | "ar"
    audio_path: str      # relative path to WAV file
    created_at: float
    hit_count: int = 0


# In-memory index — keyed by cache key
_index: dict[str, CachedResponse] = {}
_loaded = False


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[.,!?¿¡،؟ـ\"'`]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _make_key(text: str, language: str) -> str:
    return hashlib.md5(f"{language}:{_normalize(text)}".encode("utf-8")).hexdigest()


def _ensure_loaded() -> None:
    """Lazy-load the on-disk index."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not _INDEX_PATH.exists():
        return
    try:
        raw = json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
        for k, v in raw.items():
            _index[k] = CachedResponse(**v)
        log.info("response cache loaded", extra={"entries": len(_index)})
    except Exception as e:
        log.warning("response cache load failed", extra={"err": str(e)})


def _save_index() -> None:
    """Persist the index to disk."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    serializable = {k: asdict(v) for k, v in _index.items()}
    _INDEX_PATH.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")


def lookup(text: str, language: str) -> tuple[bytes, str] | None:
    """
    Return (audio_bytes, response_text) if cached, None otherwise.
    Increments hit_count on hit.
    """
    _ensure_loaded()
    key = _make_key(text, language)
    entry = _index.get(key)
    if entry is None:
        return None
    audio_file = _CACHE_DIR / entry.audio_path
    if not audio_file.exists():
        log.warning("cache entry missing audio file", extra={"key": key, "path": entry.audio_path})
        del _index[key]
        return None
    entry.hit_count += 1
    log.info(
        "response cache HIT",
        extra={"text": text[:60], "lang": language, "hits": entry.hit_count},
    )
    _save_index()  # cheap write; tracks hit counts
    return audio_file.read_bytes(), entry.response


def store(text: str, response: str, language: str, audio_bytes: bytes) -> None:
    """Save a new response to cache. Called after a successful pipeline run."""
    _ensure_loaded()
    if not audio_bytes or not response.strip():
        return
    key = _make_key(text, language)
    audio_path = f"{key}.wav"
    (_CACHE_DIR / audio_path).write_bytes(audio_bytes)
    _index[key] = CachedResponse(
        key=key,
        text=text,
        response=response,
        language=language,
        audio_path=audio_path,
        created_at=time.time(),
        hit_count=0,
    )
    _save_index()
    log.info("response cache STORE", extra={"text": text[:60], "lang": language, "key": key})


def stats() -> dict:
    """Return summary stats for debugging."""
    _ensure_loaded()
    return {
        "entries": len(_index),
        "total_hits": sum(e.hit_count for e in _index.values()),
        "by_language": {
            "en": sum(1 for e in _index.values() if e.language == "en"),
            "ar": sum(1 for e in _index.values() if e.language == "ar"),
        },
    }
