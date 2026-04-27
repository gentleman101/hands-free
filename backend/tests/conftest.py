"""Shared pytest fixtures for Hands-Free test suite."""
import asyncio
import io
import wave
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Single event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture()
def fixture_audio_dir() -> Path:
    return Path(__file__).parents[2] / "fixtures" / "audio"


def make_wav_bytes(
    duration_s: float = 0.5,
    sample_rate: int = 16000,
    frequency: float = 0.0,
) -> bytes:
    """Create WAV bytes. frequency=0 → silence; >0 → sine wave."""
    n = int(duration_s * sample_rate)
    if frequency > 0:
        t = np.linspace(0, duration_s, n, endpoint=False)
        pcm = (np.sin(2 * np.pi * frequency * t) * 0.5 * 32767).astype(np.int16)
    else:
        pcm = np.zeros(n, dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


@pytest.fixture()
def silence_wav() -> bytes:
    return make_wav_bytes(0.5)


@pytest.fixture()
def speech_wav() -> bytes:
    return make_wav_bytes(1.0, frequency=440.0)


# Auto-patch database writes in all tests to avoid real SQLite I/O
@pytest.fixture(autouse=True)
def no_db_writes(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    with patch("backend.database.insert_event"):
        yield
