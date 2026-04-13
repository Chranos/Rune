"""Tests for the LLM streaming module."""

import asyncio
import json

import pytest

from rune.llm.streaming import StreamAccumulator, StreamDelta, parse_sse_stream


class TestStreamDelta:
    def test_basic_delta(self):
        delta = StreamDelta(content="hello")
        assert delta.content == "hello"
        assert delta.tool_calls is None
        assert delta.finish_reason is None


class TestStreamAccumulator:
    def test_accumulate_content(self):
        acc = StreamAccumulator()
        acc.feed(StreamDelta(content="Hello"))
        acc.feed(StreamDelta(content=" world"))
        acc.feed(StreamDelta(finish_reason="stop"))
        acc.finalize()

        assert acc.content == "Hello world"
        assert acc.finish_reason == "stop"
        assert acc.tool_calls == []

    def test_accumulate_tool_calls(self):
        acc = StreamAccumulator()
        # Simulate incremental tool call streaming
        acc.feed(StreamDelta(tool_calls=[{
            "index": 0,
            "id": "call_1",
            "function": {"name": "bash", "arguments": ""},
        }]))
        acc.feed(StreamDelta(tool_calls=[{
            "index": 0,
            "function": {"arguments": '{"com'},
        }]))
        acc.feed(StreamDelta(tool_calls=[{
            "index": 0,
            "function": {"arguments": 'mand": "ls"}'},
        }]))
        acc.feed(StreamDelta(finish_reason="tool_calls"))
        acc.finalize()

        assert len(acc.tool_calls) == 1
        tc = acc.tool_calls[0]
        assert tc["id"] == "call_1"
        assert tc["function"]["name"] == "bash"

        args = json.loads(tc["function"]["arguments"])
        assert args["command"] == "ls"

    def test_multiple_tool_calls(self):
        acc = StreamAccumulator()
        acc.feed(StreamDelta(tool_calls=[{
            "index": 0, "id": "call_1",
            "function": {"name": "glob", "arguments": '{"pattern": "*.py"}'},
        }]))
        acc.feed(StreamDelta(tool_calls=[{
            "index": 1, "id": "call_2",
            "function": {"name": "grep", "arguments": '{"pattern": "def"}'},
        }]))
        acc.finalize()

        assert len(acc.tool_calls) == 2
        assert acc.tool_calls[0]["function"]["name"] == "glob"
        assert acc.tool_calls[1]["function"]["name"] == "grep"


class TestParseSSEStream:
    @pytest.mark.asyncio
    async def test_parse_content_stream(self):
        # Simulate SSE chunks
        async def mock_stream():
            chunks = [
                b'data: {"choices":[{"delta":{"role":"assistant"},"index":0}]}\n\n',
                b'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n',
                b'data: {"choices":[{"delta":{"content":" world"},"index":0}]}\n\n',
                b'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}\n\n',
                b'data: [DONE]\n\n',
            ]
            for chunk in chunks:
                yield chunk

        deltas = []
        async for delta in parse_sse_stream(mock_stream()):
            deltas.append(delta)

        assert len(deltas) == 4
        assert deltas[1].content == "Hello"
        assert deltas[2].content == " world"
        assert deltas[3].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_parse_handles_malformed_json(self):
        async def mock_stream():
            yield b'data: {invalid json}\n\ndata: {"choices":[{"delta":{"content":"ok"},"index":0}]}\n\n'
            yield b'data: [DONE]\n\n'

        deltas = []
        async for delta in parse_sse_stream(mock_stream()):
            deltas.append(delta)

        # Should skip malformed and parse valid
        assert len(deltas) == 1
        assert deltas[0].content == "ok"
