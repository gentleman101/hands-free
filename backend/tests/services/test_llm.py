"""Tests for backend/services/llm.py"""
from unittest.mock import patch

import pytest

from backend.services.llm import GenerateResult, generate


@pytest.fixture()
def mock_stream_chunks() -> list[str]:
    return ["Hello", ",", " how", " can", " I", " help", "?"]


async def _fake_stream(chunks: list[str]):  # type: ignore[return]
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
class TestGenerate:
    async def test_english_response(self, mock_stream_chunks: list[str]) -> None:
        with (
            patch("backend.services.llm.generate_stream") as mock_gs,
            patch("backend.database.insert_event"),
        ):
            mock_gs.return_value = _fake_stream(mock_stream_chunks)
            result = await generate(
                [{"role": "user", "content": "hi"}], language="en"
            )

        assert isinstance(result, GenerateResult)
        assert result.text == "Hello, how can I help?"
        assert result.tokens_generated == len(mock_stream_chunks)

    async def test_arabic_response(self) -> None:
        ar_chunks = ["مرحبا", "،", " كيف", " يمكنني", " مساعدتك", "؟"]
        with (
            patch("backend.services.llm.generate_stream") as mock_gs,
            patch("backend.database.insert_event"),
        ):
            mock_gs.return_value = _fake_stream(ar_chunks)
            result = await generate(
                [{"role": "user", "content": "مرحبا"}], language="ar"
            )

        assert "مرحبا" in result.text
        assert result.tokens_generated == len(ar_chunks)

    async def test_max_tokens_respected(self, mock_stream_chunks: list[str]) -> None:
        with (
            patch("backend.services.llm.generate_stream") as mock_gs,
            patch("backend.database.insert_event"),
        ):
            mock_gs.return_value = _fake_stream(mock_stream_chunks)
            await generate(
                [{"role": "user", "content": "test"}], max_tokens=5
            )

        # Verify max_tokens was passed through
        call_kwargs = mock_gs.call_args.kwargs
        assert call_kwargs.get("max_tokens") == 5

    async def test_empty_response(self) -> None:
        with (
            patch("backend.services.llm.generate_stream") as mock_gs,
            patch("backend.database.insert_event"),
        ):
            mock_gs.return_value = _fake_stream([])
            result = await generate([{"role": "user", "content": "test"}])

        assert result.text == ""
        assert result.tokens_generated == 0
