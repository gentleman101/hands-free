"""Tests for backend/services/tts.py"""
from unittest.mock import patch

import pytest

from backend.services.tts import SynthesizeResult, strip_prosody_tags, synthesize


async def _fake_audio_stream(chunks: list[bytes]):  # type: ignore[return]
    for chunk in chunks:
        yield chunk


class TestStripProsodyTags:
    def test_removes_known_tags(self) -> None:
        text = "Hello [laugh] how are you [emphasis] today?"
        assert strip_prosody_tags(text) == "Hello  how are you  today?"

    def test_no_tags_unchanged(self) -> None:
        text = "Hello, how are you?"
        assert strip_prosody_tags(text) == text

    def test_arabic_with_tags(self) -> None:
        text = "مرحبا [sigh] كيف حالك؟"
        assert strip_prosody_tags(text) == "مرحبا  كيف حالك؟"

    def test_multiple_tags(self) -> None:
        text = "[laugh] [whisper] Hello [emphasis] world"
        cleaned = strip_prosody_tags(text)
        assert "[" not in cleaned
        assert "Hello" in cleaned


@pytest.mark.asyncio
class TestSynthesize:
    async def test_returns_audio_bytes(self) -> None:
        fake_chunks = [b"\x00\x01" * 100, b"\x02\x03" * 100]
        with (
            patch("backend.services.tts.synthesize_stream") as mock_stream,
            patch("backend.database.insert_event"),
        ):
            mock_stream.return_value = _fake_audio_stream(fake_chunks)
            result = await synthesize("Hello world", "en")

        assert isinstance(result, SynthesizeResult)
        assert result.audio_bytes == b"".join(fake_chunks)
        assert result.sample_rate == 22050

    async def test_language_switch_arabic(self) -> None:
        fake_chunks = [b"\x00" * 50]
        with (
            patch("backend.services.tts.synthesize_stream") as mock_stream,
            patch("backend.database.insert_event"),
        ):
            mock_stream.return_value = _fake_audio_stream(fake_chunks)
            result = await synthesize("مرحبا", "ar")

        assert len(result.audio_bytes) > 0
        # Verify arabic voice was selected
        call_args = mock_stream.call_args
        assert call_args[0][1] == "ar"  # second positional arg is language

    async def test_empty_audio_handled(self) -> None:
        with (
            patch("backend.services.tts.synthesize_stream") as mock_stream,
            patch("backend.database.insert_event"),
        ):
            mock_stream.return_value = _fake_audio_stream([])
            result = await synthesize("", "en")

        assert result.audio_bytes == b""
        assert result.ttfa_ms == 0
