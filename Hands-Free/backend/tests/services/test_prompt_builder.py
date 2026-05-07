"""prompt_builder: language selection, message order, history inclusion."""
from __future__ import annotations

from backend.services import prompt_builder


def test_english_uses_english_prompt():
    msgs = prompt_builder.build_messages("How do I pay my bill?", language="en")
    assert msgs[0]["role"] == "system"
    assert "Rammas" in msgs[0]["content"]
    assert "Dubai" in msgs[0]["content"]
    # Must NOT contain the Arabic prompt
    assert "أنت رمّاس" not in msgs[0]["content"]


def test_arabic_uses_arabic_prompt():
    msgs = prompt_builder.build_messages("كم فاتورتي؟", language="ar")
    assert msgs[0]["role"] == "system"
    assert "أنت رمّاس" in msgs[0]["content"]
    assert "ديوا" in msgs[0]["content"]
    # Must NOT contain English brevity hedging
    assert "Rammas, DEWA" not in msgs[0]["content"]


def test_user_message_is_last():
    msgs = prompt_builder.build_messages("hello", language="en")
    assert msgs[-1] == {"role": "user", "content": "hello"}


def test_history_included_in_order():
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
    ]
    msgs = prompt_builder.build_messages(
        "second", language="en", conversation_history=history
    )
    assert msgs[0]["role"] == "system"
    assert msgs[1] == history[0]
    assert msgs[2] == history[1]
    assert msgs[3] == {"role": "user", "content": "second"}


def test_unknown_language_falls_back_to_english():
    msgs = prompt_builder.build_messages("test", language="zz")
    # Falls back to English (the conditional `if language == "ar"` only matches ar)
    assert "Rammas, DEWA" in msgs[0]["content"]
