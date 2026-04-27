"""Tests for backend/api/voice.py"""
import io
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _make_wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.zeros(8000, dtype=np.int16).tobytes())
    return buf.getvalue()


@pytest.fixture()
def wav_bytes() -> bytes:
    return _make_wav_bytes()


class TestTranscribeEndpoint:
    def test_returns_200_with_valid_audio(self, client: TestClient, wav_bytes: bytes) -> None:
        mock_result = MagicMock()
        mock_result.text = "Hello world"
        mock_result.language = "en"
        mock_result.confidence = 0.95
        mock_result.duration_ms = 150

        with (
            patch("backend.api.voice.stt_svc.transcribe", return_value=mock_result),
            patch("backend.database.init_db"),
        ):
            resp = client.post(
                "/api/voice/transcribe",
                files={"audio": ("test.wav", wav_bytes, "audio/wav")},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "Hello world"
        assert data["language"] == "en"

    def test_returns_422_on_empty_audio(self, client: TestClient) -> None:
        with patch("backend.database.init_db"):
            resp = client.post(
                "/api/voice/transcribe",
                files={"audio": ("test.wav", b"", "audio/wav")},
            )
        assert resp.status_code == 422


class TestPipelineEndpoint:
    def test_returns_200_with_latency(self, client: TestClient, wav_bytes: bytes) -> None:
        from backend.services.pipeline import PipelineLatency, PipelineResult

        mock_result = PipelineResult(
            transcription="test",
            language="en",
            response_text="Here is the answer.",
            audio_bytes=b"\x00" * 100,
            latency=PipelineLatency(
                stt_ms=150, llm_ttft_ms=210, llm_total_ms=450,
                tts_ttfa_ms=280, tts_total_ms=600,
                pipeline_ttfa_ms=640, total_ms=900,
            ),
        )

        with (
            patch("backend.api.voice.pipeline_svc.run", return_value=mock_result),
            patch("backend.database.init_db"),
        ):
            resp = client.post(
                "/api/voice/pipeline",
                files={"audio": ("test.wav", wav_bytes, "audio/wav")},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "pipeline_ttfa_ms" in data["latency"]
        assert data["latency"]["pipeline_ttfa_ms"] == 640
