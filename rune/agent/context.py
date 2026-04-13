"""Context window management for small local models.

This is the key innovation — intelligently managing an 8K context window
by implementing:
1. Conversation compaction (summarizing old turns)
2. Smart truncation (preserving recent + important context)
3. Token estimation without expensive tokenizer calls
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rune.llm.client import ChatMessage

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Fast token estimation without a tokenizer.

    Uses a heuristic: ~4 characters per token for English,
    ~2 characters per token for Chinese/CJK.
    This avoids the overhead of loading tiktoken for every call.
    """
    if not text:
        return 0

    cjk_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other_chars = len(text) - cjk_chars

    return int(cjk_chars / 1.5 + other_chars / 4)


def estimate_messages_tokens(messages: list[ChatMessage]) -> int:
    """Estimate total tokens for a list of messages."""
    total = 0
    for msg in messages:
        # Role + formatting overhead
        total += 4
        if msg.content:
            total += estimate_tokens(msg.content)
        if msg.tool_calls:
            # Estimate tool call tokens
            import json

            total += estimate_tokens(json.dumps(msg.tool_calls))
    return total


@dataclass
class ContextWindow:
    """Manages conversation history within a limited context window.

    Key strategies for small context windows (8K):
    1. System prompt is always preserved
    2. Last N turns are always preserved
    3. Middle turns are compacted (summarized) when space runs low
    4. Tool results are truncated aggressively
    """

    max_tokens: int = 8192
    reserve_for_response: int = 2048
    compact_threshold: float = 0.75  # Trigger compaction at 75% usage

    _messages: list[ChatMessage] = field(default_factory=list)
    _system_message: ChatMessage | None = None
    _compaction_count: int = 0

    @property
    def available_tokens(self) -> int:
        return self.max_tokens - self.reserve_for_response

    @property
    def current_usage(self) -> int:
        return estimate_messages_tokens(self._messages)

    @property
    def usage_ratio(self) -> float:
        available = self.available_tokens
        if available <= 0:
            return 1.0
        return self.current_usage / available

    @property
    def messages(self) -> list[ChatMessage]:
        """Get the current message list (system + conversation)."""
        result = []
        if self._system_message:
            result.append(self._system_message)
        result.extend(self._messages)
        return result

    def set_system_prompt(self, content: str) -> None:
        """Set the system prompt (always preserved)."""
        self._system_message = ChatMessage(role="system", content=content)

    def add_message(self, message: ChatMessage) -> None:
        """Add a message and auto-compact if needed."""
        self._messages.append(message)

        if self.usage_ratio > self.compact_threshold:
            self._auto_compact()

    def add_messages(self, messages: list[ChatMessage]) -> None:
        """Add multiple messages."""
        for msg in messages:
            self._messages.append(msg)

        if self.usage_ratio > self.compact_threshold:
            self._auto_compact()

    def _auto_compact(self) -> None:
        """Automatically compact conversation history to fit the context window.

        Strategy:
        1. Always keep the last 6 messages (3 turns)
        2. Summarize everything else into a single compaction message
        3. Truncate long tool results
        """
        if len(self._messages) <= 6:
            # Not enough to compact, try truncating tool results instead
            self._truncate_tool_results()
            return

        self._compaction_count += 1
        logger.info(
            "Context compaction #%d (usage: %.0f%%)",
            self._compaction_count,
            self.usage_ratio * 100,
        )

        # Keep the last 6 messages
        keep_recent = 6
        old_messages = self._messages[:-keep_recent]
        recent_messages = self._messages[-keep_recent:]

        # Build a summary of old messages
        summary_parts = []
        for msg in old_messages:
            if msg.role == "user":
                content_preview = (msg.content or "")[:150]
                summary_parts.append(f"- User asked: {content_preview}")
            elif msg.role == "assistant" and msg.content:
                content_preview = msg.content[:150]
                summary_parts.append(f"- Assistant: {content_preview}")
            elif msg.role == "tool":
                summary_parts.append(f"- Tool result ({msg.name}): [executed]")

        summary = (
            f"[Context compacted — {len(old_messages)} earlier messages summarized]\n"
            + "\n".join(summary_parts[-10:])  # Keep last 10 summary items
        )

        compaction_msg = ChatMessage(role="user", content=summary)
        self._messages = [compaction_msg] + recent_messages

        # If still over threshold, truncate tool results
        if self.usage_ratio > self.compact_threshold:
            self._truncate_tool_results()

    def _truncate_tool_results(self) -> None:
        """Truncate long tool result messages to save context space."""
        max_tool_result_length = 2000

        for msg in self._messages:
            if msg.role == "tool" and msg.content and len(msg.content) > max_tool_result_length:
                half = max_tool_result_length // 2
                msg.content = (
                    msg.content[:half]
                    + "\n[...truncated...]\n"
                    + msg.content[-half:]
                )

    def clear(self) -> None:
        """Clear conversation history (preserves system prompt)."""
        self._messages.clear()
        self._compaction_count = 0
