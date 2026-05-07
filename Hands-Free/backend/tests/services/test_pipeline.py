"""SentenceBuffer behaviour (min_chars, comma + period boundaries, AR boundaries)."""
from __future__ import annotations

from backend.services.pipeline import SentenceBuffer


def test_short_buffer_returns_none():
    buf = SentenceBuffer()
    # Below _MIN_SENTENCE_CHARS (8) — should not emit even with a period
    assert buf.feed("Hi.") is None


def test_emits_on_period_after_min_chars():
    buf = SentenceBuffer()
    out = buf.feed("Hello world.")
    assert out == "Hello world."
    # Buffer should now be empty
    assert buf.flush() is None


def test_emits_on_question_mark():
    buf = SentenceBuffer()
    assert buf.feed("Where to go?") == "Where to go?"


def test_emits_on_arabic_period():
    buf = SentenceBuffer()
    # Arabic comma + question mark are boundaries
    assert buf.feed("ما هي الخدمات؟") == "ما هي الخدمات؟"


def test_emits_on_comma_for_first_clause():
    """EXP-005: comma boundary ships first clause to TTS earlier."""
    buf = SentenceBuffer()
    # Feeding tokens incrementally
    out = buf.feed("Your bill is")
    assert out is None
    out = buf.feed(" 450 AED,")
    # "Your bill is 450 AED," — first comma boundary
    assert out == "Your bill is 450 AED,"


def test_remaining_text_via_flush():
    buf = SentenceBuffer()
    buf.feed("Pending text without boundary")
    # No emit yet
    assert buf.flush() == "Pending text without boundary"
    # Buffer cleared
    assert buf.flush() is None


def test_multiple_sentences_in_one_token():
    """If two boundaries arrive in one token, only the first sentence is emitted; rest stays buffered."""
    buf = SentenceBuffer()
    out = buf.feed("First done. Second clause too.")
    # Only the first sentence is emitted
    assert out == "First done."
    # The remainder is still buffered; flush surfaces it (or next feed splits more)
    rest = buf.flush()
    assert rest == "Second clause too."
