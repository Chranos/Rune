"""Async LLM client for OpenAI-compatible endpoints (llama.cpp, ollama, vLLM, etc.)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

from rune.config.settings import ModelConfig
from rune.llm.models import ModelInfo
from rune.llm.streaming import StreamAccumulator, StreamDelta, parse_sse_stream

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """A single message in a conversation."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_api_dict(self) -> dict[str, Any]:
        """Serialize to the OpenAI API message format."""
        msg: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            msg["content"] = self.content
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg


@dataclass
class LLMResponse:
    """Complete response from the LLM."""

    content: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class LLMClient:
    """Async client for OpenAI-compatible chat completions.

    Designed to work with llama.cpp server, ollama, vLLM, or any endpoint
    implementing the /v1/chat/completions API.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
        )
        self._model_info: ModelInfo | None = None

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def get_model_info(self) -> ModelInfo:
        """Fetch model information from the server and auto-detect model name."""
        if self._model_info:
            return self._model_info

        try:
            resp = await self._client.get("/v1/models")
            resp.raise_for_status()
            data = resp.json()
            models = data.get("data", [])
            if models:
                # Find a loaded model, preferring non-default ones
                chosen = models[0]
                for m in models:
                    status = m.get("status", {})
                    if isinstance(status, dict) and status.get("value") == "loaded":
                        chosen = m
                        break

                model_id = chosen.get("id", self.config.model)
                # Auto-update the model name so API calls use the correct name
                if self.config.model == "local-model":
                    self.config.model = model_id

                self._model_info = ModelInfo(
                    model_id=model_id,
                    context_window=self.config.context_window,
                    supports_tool_use=self.config.supports_tool_use,
                )
            else:
                self._model_info = ModelInfo(
                    model_id=self.config.model,
                    context_window=self.config.context_window,
                    supports_tool_use=self.config.supports_tool_use,
                )
        except httpx.HTTPError:
            # Fallback if /v1/models is unavailable
            self._model_info = ModelInfo(
                model_id=self.config.model,
                context_window=self.config.context_window,
                supports_tool_use=self.config.supports_tool_use,
            )

        return self._model_info

    def _build_request_body(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
        stream: bool = True,
    ) -> dict[str, Any]:
        """Build the request payload for /v1/chat/completions."""
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [m.to_api_dict() for m in messages],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": stream,
        }

        if tools:
            body["tools"] = tools
            # Let the model decide when to use tools
            body["tool_choice"] = "auto"

        return body

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Send a non-streaming chat completion request.

        Returns the complete response at once.
        """
        body = self._build_request_body(messages, tools, stream=False)

        try:
            resp = await self._client.post("/v1/chat/completions", json=body)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            logger.error("LLM API error: %s %s", e.response.status_code, e.response.text)
            raise
        except httpx.HTTPError as e:
            logger.error("LLM connection error: %s", e)
            raise

        choice = data["choices"][0]
        message = choice["message"]

        return LLMResponse(
            content=message.get("content"),
            tool_calls=message.get("tool_calls", []),
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage", {}),
        )

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[StreamDelta]:
        """Send a streaming chat completion request.

        Yields StreamDelta objects as they arrive from the server.
        """
        body = self._build_request_body(messages, tools, stream=True)

        try:
            async with self._client.stream(
                "POST", "/v1/chat/completions", json=body
            ) as resp:
                if resp.status_code != 200:
                    # Must read the body before accessing it in streaming mode
                    error_body = await resp.aread()
                    error_text = error_body.decode("utf-8", errors="replace")
                    logger.error("LLM API error: %s %s", resp.status_code, error_text)
                    raise httpx.HTTPStatusError(
                        f"LLM returned {resp.status_code}: {error_text}",
                        request=resp.request,
                        response=resp,
                    )
                async for delta in parse_sse_stream(resp.aiter_bytes()):
                    yield delta
        except httpx.HTTPStatusError:
            raise
        except httpx.HTTPError as e:
            logger.error("LLM connection error: %s", e)
            raise

    async def chat_stream_complete(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
        on_content: Any | None = None,
    ) -> LLMResponse:
        """Stream a chat completion and return the accumulated result.

        Args:
            messages: Conversation history.
            tools: Available tool definitions.
            on_content: Optional async callback called with each content chunk for live display.

        Returns:
            The complete LLMResponse after streaming finishes.
        """
        accumulator = StreamAccumulator()

        async for delta in self.chat_stream(messages, tools):
            accumulator.feed(delta)

            if delta.content and on_content:
                await on_content(delta.content)

        accumulator.finalize()

        return LLMResponse(
            content=accumulator.content or None,
            tool_calls=accumulator.tool_calls,
            finish_reason=accumulator.finish_reason,
        )

    async def check_health(self) -> bool:
        """Check if the LLM server is reachable."""
        try:
            resp = await self._client.get("/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
