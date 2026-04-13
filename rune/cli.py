"""Rune CLI — the main entry point and REPL.

Provides an interactive terminal interface with:
- prompt_toolkit for input handling (history, multi-line, keybindings)
- Rich for output rendering
- Async event loop for streaming
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PTStyle

from rune.agent.loop import AgentLoop, AgentEvent
from rune.config.settings import RUNE_HOME, RuneConfig
from rune.llm.client import LLMClient
from rune.safety.permissions import PermissionLevel, PermissionManager, PermissionRequest
from rune.tools.base import create_default_registry
from rune.ui.renderer import Renderer

logger = logging.getLogger(__name__)

# prompt_toolkit styling
PT_STYLE = PTStyle.from_dict({
    "prompt": "cyan bold",
    "": "",
})


class RuneSession:
    """A single interactive Rune session."""

    def __init__(self, config: RuneConfig) -> None:
        self.config = config
        self.renderer = Renderer()
        self.llm = LLMClient(config.model)
        self.tools = create_default_registry()
        self.permissions = PermissionManager(config.safety)
        self.agent = AgentLoop(
            config=config,
            llm=self.llm,
            tools=self.tools,
            permissions=self.permissions,
            confirm_callback=self._handle_permission,
        )

        # Ensure history directory exists
        RUNE_HOME.mkdir(parents=True, exist_ok=True)
        history_file = RUNE_HOME / "history"
        self.prompt_session: PromptSession = PromptSession(
            history=FileHistory(str(history_file)),
            style=PT_STYLE,
            multiline=False,
        )

    async def _handle_permission(self, request: PermissionRequest) -> bool:
        """Handle permission confirmation from the agent loop."""
        allowed, grant_always = self.renderer.prompt_permission(
            request.tool_name, request.arguments, request.risk_reason
        )
        if grant_always:
            self.permissions.grant_session_permission(request.tool_name)
        return allowed

    async def _process_events(self, user_input: str) -> None:
        """Process agent events and render them."""
        streaming = False

        async for event in self.agent.process_user_message(user_input):
            if event.type == "status":
                if not streaming:
                    self.renderer.start_streaming()
                    streaming = True

            elif event.type == "content":
                if not streaming:
                    self.renderer.start_streaming()
                    streaming = True
                self.renderer.stream_content(event.data)

            elif event.type == "tool_call":
                if streaming:
                    self.renderer.stop_streaming()
                    streaming = False
                self.renderer.render_tool_call(
                    event.data["name"],
                    event.data["arguments"],
                )

            elif event.type == "tool_result":
                self.renderer.render_tool_result(
                    event.data["name"],
                    event.data["output"],
                    event.data["is_error"],
                )

            elif event.type == "error":
                if streaming:
                    self.renderer.stop_streaming()
                    streaming = False
                self.renderer.print_error(event.data)

            elif event.type == "done":
                if streaming:
                    self.renderer.stop_streaming()
                    streaming = False

                # Show context usage
                self.renderer.render_context_info(
                    self.agent.context.usage_ratio,
                    self.agent.context._compaction_count,
                )

    async def run(self) -> None:
        """Run the interactive REPL."""
        self.renderer.print_banner()

        # Check model health
        healthy = await self.llm.check_health()
        if not healthy:
            self.renderer.print_error(
                f"Cannot connect to LLM server at {self.config.model.base_url}\n"
                "  Make sure your llama.cpp server is running:\n"
                "  ./start-qwen35.sh"
            )
            return

        # Show model info
        model_info = await self.llm.get_model_info()
        self.renderer.print_model_info(model_info.model_id, model_info.context_window)
        self.renderer.console.print(
            "  Type your message to begin. Use [cyan]/help[/] for commands, "
            "[cyan]Ctrl+C[/] to cancel, [cyan]Ctrl+D[/] to exit.\n",
            style="dim",
        )

        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.prompt_session.prompt(
                        [("class:prompt", "❯ rune ")],
                    ),
                )

                user_input = user_input.strip()
                if not user_input:
                    continue

                # Handle built-in commands
                if user_input.startswith("/"):
                    if await self._handle_command(user_input):
                        continue

                # Process through the agent
                await self._process_events(user_input)
                self.renderer.console.print()  # blank line between turns

            except KeyboardInterrupt:
                self.agent.stop()
                self.renderer.console.print("\n[dim]  Interrupted[/]")
                continue
            except EOFError:
                self.renderer.console.print("\n[cyan]  Goodbye![/]")
                break
            except Exception as e:
                logger.exception("Session error")
                self.renderer.print_error(str(e))

        await self.llm.close()

    async def _handle_command(self, command: str) -> bool:
        """Handle slash commands. Returns True if handled."""
        cmd = command.lower().strip()

        if cmd in ("/exit", "/quit", "/q"):
            raise EOFError

        elif cmd == "/help":
            self.renderer.console.print(
                Panel_help(), style="dim"
            )
            return True

        elif cmd == "/clear":
            self.agent.context.clear()
            self.renderer.console.print("  [green]✓[/] Conversation cleared")
            return True

        elif cmd == "/context":
            self.renderer.render_context_info(
                self.agent.context.usage_ratio,
                self.agent.context._compaction_count,
            )
            return True

        elif cmd.startswith("/model"):
            parts = cmd.split(maxsplit=1)
            if len(parts) > 1:
                self.config.model.base_url = parts[1]
                self.llm = LLMClient(self.config.model)
                self.agent.llm = self.llm
                self.renderer.print_success(f"Model endpoint updated to: {parts[1]}")
            else:
                self.renderer.console.print(f"  Model: {self.config.model.base_url}")
            return True

        return False


def Panel_help() -> str:
    return """  [cyan bold]Rune Commands[/]
  /help     — Show this help message
  /clear    — Clear conversation history
  /context  — Show context window usage
  /model    — Show or change model endpoint
  /exit     — Exit Rune

  [cyan bold]Keyboard Shortcuts[/]
  Ctrl+C    — Cancel current generation
  Ctrl+D    — Exit Rune
"""


@click.command()
@click.option(
    "--base-url",
    default=None,
    help="LLM server base URL (default: http://127.0.0.1:8080)",
)
@click.option(
    "--model",
    "model_name",
    default=None,
    help="Model name to use",
)
@click.option(
    "--context-window",
    default=None,
    type=int,
    help="Context window size in tokens",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(),
    help="Path to config file",
)
@click.version_option(package_name="rune-agent")
def main(
    base_url: str | None,
    model_name: str | None,
    context_window: int | None,
    config_path: str | None,
) -> None:
    """Rune — Local-First AI Coding Agent for Your Terminal."""
    # Load config
    config = RuneConfig.load(Path(config_path) if config_path else None)

    # CLI overrides
    if base_url:
        config.model.base_url = base_url
    if model_name:
        config.model.model = model_name
    if context_window:
        config.model.context_window = context_window

    # Set up logging
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    # Run the session
    session = RuneSession(config)
    try:
        asyncio.run(session.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
