"""Tests for backend/services/pipeline.py — SentenceBuffer + run()"""
from unittest.mock import MagicMock, patch

import pytest

from backend.services.pipeline import (
    PipelineResult,
    SentenceBuffer,
    run,
)


class TestSentenceBuffer:
    def test_emits_on_period(self) -> None:
        buf = SentenceBuffer()
        # Feed enough chars first
        for ch in "Hello how are you":
            result = buf.feed(ch)
        result = buf.feed(".")
        assert result == "Hello how are you."

    def test_emits_on_question_mark(self) -> None:
        buf = SentenceBuffer()
        for ch in "How can I help you":
            buf.feed(ch)
        result = buf.feed("?")
        assert result == "How can I help you?"

    def test_emits_arabic_question_mark(self) -> None:
        buf = SentenceBuffer()
        for ch in "كيف يمكنني مساعدتك":
            buf.feed(ch)
        result = buf.feed("؟")
        assert result == "كيف يمكنني مساعدتك؟"

    def test_no_emit_below_min_chars(self) -> None:
        buf = SentenceBuffer()
        result = buf.feed("Hi.")
        assert result is None  # too short

    def test_flush_returns_remaining(self) -> None:
        buf = SentenceBuffer()
        for ch in "No boundary here":
            buf.feed(ch)
        remaining = buf.flush()
        assert remaining == "No boundary here"

    def test_flush_empty_returns_none(self) -> None:
        buf = SentenceBuffer()
        assert buf.flush() is None

    def test_second_sentence_after_first(self) -> None:
        buf = SentenceBuffer()
        text1 = "Hello, how can I help you?"
        for ch in text1:
            result = buf.feed(ch)
        assert result == text1
        # Now feed a second sentence
        text2 = "I am here to assist."
        for ch in text2:
            result = buf.feed(ch)
        assert result == text2


@pytest.mark.asyncio
class TestPipelineRun:
    async def test_full_round_trip(self) -> None:
        import io
        import wave
        import numpy as np

        # Create minimal WAV fixture
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(np.zeros(8000, dtype=np.int16).tobytes())
        wav_bytes = buf.getvalue()

        mock_transcript = MagicMock()
        mock_transcript.text = "What is the leave policy?"
        mock_transcript.language = "en"
        mock_transcript.confidence = 0.95
        mock_transcript.duration_ms = 150

        mock_llm = MagicMock()
        mock_llm.text = "You get 30 days of annual leave."
        mock_llm.ttft_ms = 210
        mock_llm.duration_ms = 450

        mock_tts = MagicMock()
        mock_tts.audio_bytes = b"\x00" * 1000
        mock_tts.sample_rate = 22050
        mock_tts.ttfa_ms = 280
        mock_tts.duration_ms = 600

        with (
            patch("backend.services.pipeline.stt_svc.transcribe", return_value=mock_transcript),
            patch("backend.services.pipeline.llm_svc.generate", return_value=mock_llm),
            patch("backend.services.pipeline.tts_svc.synthesize", return_value=mock_tts),
            patch("backend.database.insert_event"),
        ):
            result = await run(wav_bytes)

        assert isinstance(result, PipelineResult)
        assert result.transcription == "What is the leave policy?"
        assert result.language == "en"
        assert result.audio_bytes == b"\x00" * 1000
        assert result.latency.stt_ms >= 0
        assert result.latency.pipeline_ttfa_ms == (
            result.latency.stt_ms + mock_llm.ttft_ms + mock_tts.ttfa_ms
        )

    async def test_arabic_round_trip(self) -> None:
        import io
        import wave
        import numpy as np

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(np.zeros(8000, dtype=np.int16).tobytes())
        wav_bytes = buf.getvalue()

        mock_transcript = MagicMock()
        mock_transcript.text = "ما هي سياسة الإجازات؟"
        mock_transcript.language = "ar"
        mock_transcript.confidence = 0.90
        mock_transcript.duration_ms = 180

        mock_llm = MagicMock()
        mock_llm.text = "لديك ثلاثون يوم إجازة سنوية."
        mock_llm.ttft_ms = 230
        mock_llm.duration_ms = 500

        mock_tts = MagicMock()
        mock_tts.audio_bytes = b"\x01" * 1000
        mock_tts.sample_rate = 22050
        mock_tts.ttfa_ms = 300
        mock_tts.duration_ms = 650

        with (
            patch("backend.services.pipeline.stt_svc.transcribe", return_value=mock_transcript),
            patch("backend.services.pipeline.llm_svc.generate", return_value=mock_llm),
            patch("backend.services.pipeline.tts_svc.synthesize", return_value=mock_tts),
            patch("backend.database.insert_event"),
        ):
            result = await run(wav_bytes)

        assert result.language == "ar"
        assert "ثلاثون" in result.response_text
