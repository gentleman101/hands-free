"""TTS helper tests — strip_prosody_tags."""
from __future__ import annotations

from backend.services.tts import strip_prosody_tags


def test_strips_simple_tag():
    assert strip_prosody_tags("Hello [emphasis]world[/emphasis]") == "Hello world"


def test_strips_multiple_tags():
    text = "[laugh] welcome [emphasis] back"
    assert strip_prosody_tags(text) == "welcome  back"


def test_no_tags_returns_unchanged():
    assert strip_prosody_tags("plain text") == "plain text"


def test_arabic_text_with_tag_preserved():
    assert strip_prosody_tags("[emphasis]مرحبا[/emphasis]") == "مرحبا"


def test_uppercase_tags_also_stripped():
    """Tags should be case-insensitive."""
    assert strip_prosody_tags("[LAUGH] hi") == "hi"
