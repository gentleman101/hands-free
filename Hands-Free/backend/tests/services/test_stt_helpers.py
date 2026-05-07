"""Pure-function tests for stt.py helpers (no model loading)."""
from __future__ import annotations

from backend.services.stt import _script_matches_language


def test_arabic_text_matches_ar():
    assert _script_matches_language("مرحبا كيف الحال", "ar") is True


def test_english_text_matches_en():
    assert _script_matches_language("Hello how are you", "en") is True


def test_arabic_misclassified_as_en_is_rejected():
    """User said Arabic but Whisper guessed 'en' — reject (no Latin chars)."""
    assert _script_matches_language("مرحبا كيف الحال", "en") is False


def test_english_misclassified_as_ar_is_rejected():
    """User said English but Whisper guessed 'ar' — reject (no Arabic chars)."""
    assert _script_matches_language("Hello how are you", "ar") is False


def test_third_language_with_latin_script_passes_en_gate():
    """French/Spanish use Latin script; passes en script gate.
    Language-prob gate (separate) is what catches this.
    """
    assert _script_matches_language("Bonjour comment allez vous", "en") is True


def test_empty_text_is_rejected():
    assert _script_matches_language("", "en") is False
    assert _script_matches_language("", "ar") is False


def test_text_with_only_digits_is_rejected():
    """No alphabetic characters at all → reject."""
    assert _script_matches_language("12345 67", "en") is False


def test_mixed_arabic_with_some_english_passes_ar_gate():
    """Common in UAE: Arabic with some English brand names — should pass AR gate."""
    text = "أنا في DEWA الآن"  # Arabic > 30% of alpha chars
    assert _script_matches_language(text, "ar") is True
