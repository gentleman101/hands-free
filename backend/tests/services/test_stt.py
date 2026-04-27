"""Tests for backend/services/stt.py"""
import io
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.services.stt import TranscribeResult, _transcribe_sync


def _make_wav_bytes(duration_s: float = 0.5, sample_rate: int = 16000) -> bytes:
    """Create minimal valid WAV bytes with silence."""
    n_samples = int(duration_s * sample_rate)
    pcm = (np.zeros(n_samples, dtype=np.float32) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


@pytest.fixture()
def silence_wav() -> bytes:
    return _make_wav_bytes(0.5)


@pytest.fixture()
def mock_segment() -> MagicMock:
    seg = MagicMock()
    seg.text = " Hello world "
    return seg


@pytest.fixture()
def mock_info_en() -> MagicMock:
    info = MagicMock()
    info.language = "en"
    info.language_probability = 0.95
    return info


@pytest.fixture()
def mock_info_ar() -> MagicMock:
    info = MagicMock()
    info.language = "ar"
    info.language_probability = 0.88
    return info


class TestTranscribeSync:
    def test_english_transcription(
        self,
        silence_wav: bytes,
        mock_segment: MagicMock,
        mock_info_en: MagicMock,
    ) -> None:
        with patch("backend.services.stt._load_model") as mock_load:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = ([mock_segment], mock_info_en)
            mock_load.return_value = mock_model

            result = _transcribe_sync(silence_wav, "small")

        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.confidence == 0.95
        assert result.duration_ms >= 0

    def test_arabic_transcription(
        self,
        silence_wav: bytes,
        mock_info_ar: MagicMock,
    ) -> None:
        ar_segment = MagicMock()
        ar_segment.text = " مرحبا "
        with patch("backend.services.stt._load_model") as mock_load:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = ([ar_segment], mock_info_ar)
            mock_load.return_value = mock_model

            result = _transcribe_sync(silence_wav, "small")

        assert result.text == "مرحبا"
        assert result.language == "ar"

    def test_empty_audio_returns_empty_text(self, silence_wav: bytes) -> None:
        empty_info = MagicMock()
        empty_info.language = "en"
        empty_info.language_probability = 0.5
        with patch("backend.services.stt._load_model") as mock_load:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = ([], empty_info)
            mock_load.return_value = mock_model

            result = _transcribe_sync(silence_wav, "small")

        assert result.text == ""

    def test_multiple_segments_joined(self, silence_wav: bytes, mock_info_en: MagicMock) -> None:
        seg1, seg2 = MagicMock(), MagicMock()
        seg1.text = " First "
        seg2.text = " second "
        with patch("backend.services.stt._load_model") as mock_load:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = ([seg1, seg2], mock_info_en)
            mock_load.return_value = mock_model

            result = _transcribe_sync(silence_wav, "small")

        assert result.text == "First second"


@pytest.mark.asyncio
class TestTranscribeAsync:
    async def test_returns_transcribe_result(self, silence_wav: bytes, mock_segment: MagicMock, mock_info_en: MagicMock) -> None:
        with (
            patch("backend.services.stt._load_model") as mock_load,
            patch("backend.database.insert_event"),
        ):
            mock_model = MagicMock()
            mock_model.transcribe.return_value = ([mock_segment], mock_info_en)
            mock_load.return_value = mock_model

            from backend.services.stt import transcribe
            result = await transcribe(silence_wav, request_id="test-req-1")

        assert isinstance(result, TranscribeResult)
        assert result.language == "en"
