"""Tests for backend/services/prompt_builder.py"""
from backend.services.prompt_builder import build_messages, get_system_prompt


class TestBuildMessages:
    def test_system_message_first(self) -> None:
        msgs = build_messages("hello", language="en")
        assert msgs[0]["role"] == "system"
        assert len(msgs[0]["content"]) > 0

    def test_user_message_last(self) -> None:
        msgs = build_messages("What is the leave policy?", language="en")
        assert msgs[-1]["role"] == "user"
        assert msgs[-1]["content"] == "What is the leave policy?"

    def test_arabic_input_included(self) -> None:
        msgs = build_messages("ما هي سياسة الإجازات؟", language="ar")
        assert msgs[-1]["content"] == "ما هي سياسة الإجازات؟"

    def test_conversation_history_included(self) -> None:
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        msgs = build_messages("follow up", language="en", conversation_history=history)
        # system + history(2) + new user = 4
        assert len(msgs) == 4
        assert msgs[1]["content"] == "Hi"
        assert msgs[2]["content"] == "Hello!"

    def test_prosody_instructions_in_system_prompt(self) -> None:
        prompt = get_system_prompt()
        assert "[laugh]" in prompt
        assert "[emphasis]" in prompt
        assert "contractions" in prompt.lower() or "it's" in prompt

    def test_no_history_is_three_messages(self) -> None:
        msgs = build_messages("test", language="en")
        assert len(msgs) == 2  # system + user
