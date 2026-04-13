"""Tests for context window management."""

import pytest

from rune.agent.context import ContextWindow, estimate_tokens, estimate_messages_tokens
from rune.llm.client import ChatMessage


class TestTokenEstimation:
    def test_english_estimation(self):
        text = "Hello world, this is a test."
        tokens = estimate_tokens(text)
        assert 5 <= tokens <= 15

    def test_cjk_estimation(self):
        text = "你好世界"
        tokens = estimate_tokens(text)
        assert tokens > 0

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_mixed_text(self):
        text = "Hello 你好 World 世界"
        tokens = estimate_tokens(text)
        assert tokens > 0


class TestContextWindow:
    def test_basic_usage(self):
        ctx = ContextWindow(max_tokens=1000, reserve_for_response=200)
        ctx.set_system_prompt("You are a helpful assistant.")
        ctx.add_message(ChatMessage(role="user", content="Hello"))

        messages = ctx.messages
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"

    def test_usage_ratio(self):
        ctx = ContextWindow(max_tokens=1000, reserve_for_response=200)
        assert ctx.usage_ratio == 0.0

        ctx.add_message(ChatMessage(role="user", content="Hello"))
        assert ctx.usage_ratio > 0.0

    def test_auto_compact_triggered(self):
        ctx = ContextWindow(max_tokens=200, reserve_for_response=50, compact_threshold=0.5)

        # Fill up context with many messages
        for i in range(20):
            ctx.add_message(ChatMessage(role="user", content=f"Message {i} with some padding text to use tokens"))
            ctx.add_message(ChatMessage(role="assistant", content=f"Response {i} also with padding"))

        # Should have compacted
        assert ctx._compaction_count > 0
        # Messages should be reduced
        assert len(ctx._messages) < 40

    def test_clear(self):
        ctx = ContextWindow(max_tokens=1000)
        ctx.set_system_prompt("System")
        ctx.add_message(ChatMessage(role="user", content="Hello"))
        ctx.clear()

        messages = ctx.messages
        assert len(messages) == 1  # Only system prompt remains
        assert messages[0].role == "system"


class TestContextCompaction:
    def test_preserves_recent_messages(self):
        ctx = ContextWindow(max_tokens=300, reserve_for_response=50, compact_threshold=0.3)

        # Add enough messages to trigger compaction
        for i in range(15):
            ctx.add_message(ChatMessage(role="user", content=f"User message number {i} with extra words to take space"))
            ctx.add_message(ChatMessage(role="assistant", content=f"Assistant response number {i} with extra words"))

        # Last messages should still be present
        last_messages = ctx._messages[-6:]
        assert any("14" in (m.content or "") for m in last_messages)

    def test_compaction_summary(self):
        ctx = ContextWindow(max_tokens=300, reserve_for_response=50, compact_threshold=0.3)

        for i in range(10):
            ctx.add_message(ChatMessage(role="user", content=f"Question {i}"))
            ctx.add_message(ChatMessage(role="assistant", content=f"Answer {i}"))

        # First message should be a compaction summary
        first = ctx._messages[0]
        assert "compacted" in (first.content or "").lower()
