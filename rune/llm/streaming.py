"""Streaming SSE parser for OpenAI-compatible endpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class StreamDelta:
    """A single delta from a streaming response."""

    content: str | None = None
    tool_calls: list[dict] | None = None
    finish_reason: str | None = None
    # For tracking accumulated tool call data
    role: str | None = None


@dataclass
class StreamAccumulator:
    """Accumulates streaming deltas into a complete response.

    Handles the complexity of tool_call streaming where each chunk contains
    incremental function name/argument fragments that must be assembled.
    """

    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str | None = None
    _tool_call_buffers: dict[int, dict] = field(default_factory=dict)

    def feed(self, delta: StreamDelta) -> None:
        """Feed a delta into the accumulator."""
        if delta.content:
            self.content += delta.content

        if delta.finish_reason:
            self.finish_reason = delta.finish_reason

        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.get("index", 0)
                if idx not in self._tool_call_buffers:
                    self._tool_call_buffers[idx] = {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": "",
                            "arguments": "",
                        },
                    }
                buf = self._tool_call_buffers[idx]
                if tc.get("id"):
                    buf["id"] = tc["id"]
                func = tc.get("function", {})
                if func.get("name"):
                    buf["function"]["name"] += func["name"]
                if func.get("arguments"):
                    buf["function"]["arguments"] += func["arguments"]

    def finalize(self) -> None:
        """Finalize the accumulation, converting buffers into tool_calls."""
        self.tool_calls = [
            self._tool_call_buffers[idx]
            for idx in sorted(self._tool_call_buffers.keys())
        ]


async def parse_sse_stream(
    byte_stream: AsyncIterator[bytes],
) -> AsyncIterator[StreamDelta]:
    """Parse an SSE byte stream into StreamDelta objects.

    Handles the Server-Sent Events protocol:
    - Lines starting with 'data: ' contain JSON payloads
    - 'data: [DONE]' signals stream end
    - Empty lines separate events
    """
    buffer = ""

    async for chunk in byte_stream:
        buffer += chunk.decode("utf-8", errors="replace")

        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()

            if not line:
                continue

            if line.startswith("data: "):
                data_str = line[6:]

                if data_str.strip() == "[DONE]":
                    return

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Parse the choice delta
                choices = data.get("choices", [])
                if not choices:
                    continue

                choice = choices[0]
                delta = choice.get("delta", {})
                finish = choice.get("finish_reason")

                yield StreamDelta(
                    content=delta.get("content"),
                    tool_calls=delta.get("tool_calls"),
                    finish_reason=finish,
                    role=delta.get("role"),
                )
