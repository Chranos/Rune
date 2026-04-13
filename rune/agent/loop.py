"""Core Agent loop — the ReAct (Reason + Act) engine.

This is the heart of Rune. It implements:
1. User message → LLM with tools → Tool execution → LLM with results → repeat
2. Streaming output for real-time terminal display
3. Error recovery and retry logic
4. Parallel tool execution when possible
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from rune.agent.context import ContextWindow
from rune.config.settings import RuneConfig
from rune.llm.client import ChatMessage, LLMClient, LLMResponse
from rune.safety.permissions import PermissionLevel, PermissionManager
from rune.tools.base import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

# Maximum iterations to prevent infinite loops
MAX_AGENT_ITERATIONS = 25

# System prompt that defines Rune's behavior
SYSTEM_PROMPT = """\
You are Rune, a local AI coding assistant running in the user's terminal.
You help users with software engineering tasks by reading, writing, and editing code,
running shell commands, and navigating codebases.

Current working directory: {cwd}

## Guidelines
- Be concise and direct. Lead with the answer or action.
- Read files before editing them. Understand existing code before modifying it.
- Use the appropriate tool for each task. Prefer dedicated tools over bash when possible.
- Be careful with destructive operations — always confirm before deleting files or running dangerous commands.
- Keep solutions simple. Only make changes that are directly requested.
- When referencing code, include file paths and line numbers.
- If you encounter errors, diagnose the root cause rather than blindly retrying.

## Available Tools
You have access to the following tools. Use them by calling functions:
- bash: Execute shell commands
- read_file: Read file contents with line numbers
- write_file: Create or overwrite files
- edit_file: Make precise string replacements in files
- glob: Find files by pattern
- grep: Search file contents with regex
"""


@dataclass
class AgentEvent:
    """Events emitted by the agent loop for UI rendering."""

    type: str  # "content" | "tool_call" | "tool_result" | "error" | "status" | "done"
    data: Any = None


class AgentLoop:
    """The core ReAct agent loop.

    Orchestrates the conversation between the user, the LLM, and the tools.
    Handles streaming, tool execution, permission checks, and error recovery.
    """

    def __init__(
        self,
        config: RuneConfig,
        llm: LLMClient,
        tools: ToolRegistry,
        permissions: PermissionManager,
        confirm_callback: Callable | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.tools = tools
        self.permissions = permissions
        self.confirm_callback = confirm_callback  # async (PermissionRequest) -> bool

        self.context = ContextWindow(
            max_tokens=config.model.context_window,
            compact_threshold=config.auto_compact_threshold,
        )
        self.context.set_system_prompt(
            SYSTEM_PROMPT.format(cwd=os.getcwd())
        )

        self._iteration_count = 0
        self._is_running = False

    async def process_user_message(
        self, user_input: str
    ) -> AsyncIterator[AgentEvent]:
        """Process a user message through the full agent loop.

        This is an async generator that yields AgentEvents for the UI to render.
        The loop continues until the LLM responds without tool calls or
        the maximum iteration count is reached.
        """
        self._is_running = True
        self._iteration_count = 0

        # Add user message to context
        self.context.add_message(ChatMessage(role="user", content=user_input))

        try:
            while self._is_running and self._iteration_count < MAX_AGENT_ITERATIONS:
                self._iteration_count += 1

                # Get LLM response with streaming
                yield AgentEvent(type="status", data="Thinking...")

                response = await self._call_llm()

                # Stream text content
                if response.content:
                    yield AgentEvent(type="content", data=response.content)

                # If no tool calls, we're done
                if not response.has_tool_calls:
                    # Add assistant message to context
                    self.context.add_message(
                        ChatMessage(
                            role="assistant",
                            content=response.content,
                        )
                    )
                    yield AgentEvent(type="done")
                    return

                # Add assistant message with tool calls
                self.context.add_message(
                    ChatMessage(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )

                # Execute tool calls one by one, yielding events between each
                for tool_call in response.tool_calls:
                    func = tool_call.get("function", {})
                    tc_name = func.get("name", "unknown")
                    tc_args = func.get("arguments", "{}")

                    # Yield tool_call event FIRST so UI can stop streaming
                    yield AgentEvent(
                        type="tool_call",
                        data={
                            "name": tc_name,
                            "arguments": tc_args,
                        },
                    )

                    # Now execute (permission check may prompt user)
                    result = await self._execute_single_tool(tool_call)

                    yield AgentEvent(
                        type="tool_result",
                        data={
                            "name": tc_name,
                            "output": result.output,
                            "is_error": result.is_error,
                        },
                    )

                    # Add tool result to context
                    self.context.add_message(
                        ChatMessage(
                            role="tool",
                            content=result.output,
                            tool_call_id=tool_call.get("id", ""),
                            name=tc_name,
                        )
                    )

            # Max iterations reached
            if self._iteration_count >= MAX_AGENT_ITERATIONS:
                yield AgentEvent(
                    type="error",
                    data=f"Agent stopped: exceeded {MAX_AGENT_ITERATIONS} iterations",
                )
                yield AgentEvent(type="done")

        except Exception as e:
            logger.exception("Agent loop error")
            yield AgentEvent(type="error", data=str(e))
            yield AgentEvent(type="done")

        finally:
            self._is_running = False

    async def _call_llm(self) -> LLMResponse:
        """Call the LLM with the current context and tools."""
        messages = self.context.messages
        tool_schemas = self.tools.get_schemas()

        # Use streaming to get real-time output
        content_chunks: list[str] = []

        async def on_content(chunk: str) -> None:
            content_chunks.append(chunk)

        response = await self.llm.chat_stream_complete(
            messages=messages,
            tools=tool_schemas if tool_schemas else None,
            on_content=on_content,
        )

        return response

    async def _execute_tools(
        self, tool_calls: list[dict]
    ) -> list[tuple[dict, ToolResult]]:
        """Execute tool calls, potentially in parallel.

        Handles permission checks and error recovery.
        """
        results: list[tuple[dict, ToolResult]] = []

        # Check if all tools can run in parallel (all read-only)
        all_read_only = all(
            self._is_read_only_call(tc) for tc in tool_calls
        )

        if all_read_only and len(tool_calls) > 1:
            # Execute in parallel
            tasks = [self._execute_single_tool(tc) for tc in tool_calls]
            tool_results = await asyncio.gather(*tasks)
            results = list(zip(tool_calls, tool_results))
        else:
            # Execute sequentially
            for tc in tool_calls:
                result = await self._execute_single_tool(tc)
                results.append((tc, result))

        return results

    async def _execute_single_tool(self, tool_call: dict) -> ToolResult:
        """Execute a single tool call with permission checking."""
        func = tool_call.get("function", {})
        name = func.get("name", "unknown")
        raw_args = func.get("arguments", "{}")

        # Parse arguments
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            args = {}

        # Check permissions
        perm_request = self.permissions.check_permission(name, args)

        if perm_request.risk_level == PermissionLevel.DENY:
            return ToolResult(f"Permission denied for {name}", is_error=True)

        if perm_request.risk_level == PermissionLevel.CONFIRM:
            if self.confirm_callback:
                approved = await self.confirm_callback(perm_request)
                if not approved:
                    return ToolResult(f"User denied execution of {name}", is_error=True)
            # If no confirm callback, default to allowing

        # Execute the tool
        tool = self.tools.get(name)
        if not tool:
            return ToolResult(f"Unknown tool: {name}", is_error=True)

        result = await tool.safe_execute(args)

        # Audit log
        self.permissions.record_action(name, args, result.output[:200])

        return result

    def _is_read_only_call(self, tool_call: dict) -> bool:
        """Check if a tool call is read-only."""
        name = tool_call.get("function", {}).get("name", "")
        return name in self.tools.get_read_only_tools()

    def stop(self) -> None:
        """Stop the agent loop."""
        self._is_running = False
