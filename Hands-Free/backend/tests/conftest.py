"""Shared test fixtures."""
from __future__ import annotations

import io
import wave

import numpy as np
import pytest


@pytest.fixture
def silent_wav_bytes() -> bytes:
    """0.5s of silence at 16kHz mono — for STT smoke tests."""
    sr = 16000
    silence = np.zeros(sr // 2, dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(silence.tobytes())
    return buf.getvalue()


@pytest.fixture(autouse=True)
def patch_metrics(monkeypatch):
    """Stub the metrics writer so tests don't touch the DB."""
    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr("backend.database.insert_event", _noop)
    monkeypatch.setattr("backend.database.fire_and_forget", lambda coro: None)
