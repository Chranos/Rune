"""Terminal UI rendering using Rich.

Provides beautiful output for:
- Streaming LLM responses with Markdown
- Tool call display with syntax highlighting
- Spinners and progress indicators
- Permission confirmation prompts
"""

from __future__ import annotations

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.theme import Theme

# Custom theme matching a coding-focused aesthetic
RUNE_THEME = Theme({
    "rune.accent": "bold cyan",
    "rune.success": "bold green",
    "rune.error": "bold red",
    "rune.warning": "bold yellow",
    "rune.tool": "bold magenta",
    "rune.dim": "dim white",
})


class Renderer:
    """Main terminal renderer for Rune."""

    def __init__(self) -> None:
        self.console = Console(theme=RUNE_THEME)
        self._live: Live | None = None
        self._streaming_buffer: str = ""

    def print_banner(self) -> None:
        """Print the Rune startup banner."""
        banner = Text()
        banner.append("  ✦ ", style="cyan")
        banner.append("Rune", style="bold cyan")
        banner.append(" — Local-First AI Coding Agent\n", style="dim")
        banner.append("    Powered by your local LLM. Private. Fast. Free.\n", style="dim")
        self.console.print(Panel(banner, border_style="cyan", padding=(0, 1)))

    def print_model_info(self, model_id: str, context_window: int) -> None:
        """Print connected model information."""
        self.console.print(
            f"  Model: [cyan]{model_id}[/]  Context: [cyan]{context_window}[/] tokens\n",
        )

    def print_error(self, message: str) -> None:
        """Print an error message."""
        self.console.print(f"[rune.error]  Error:[/] {message}")

    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        self.console.print(f"[rune.warning]  Warning:[/] {message}")

    def print_success(self, message: str) -> None:
        """Print a success message."""
        self.console.print(f"[rune.success]  ✓[/] {message}")

    # ── Streaming response ──────────────────────────────────────────────

    def start_streaming(self) -> None:
        """Begin streaming mode for live LLM output."""
        self._streaming_buffer = ""
        self._live = Live(
            Text(""),
            console=self.console,
            refresh_per_second=12,
            vertical_overflow="visible",
        )
        self._live.start()

    def stream_content(self, chunk: str) -> None:
        """Feed a chunk of content to the streaming display."""
        self._streaming_buffer += chunk
        if self._live:
            try:
                md = Markdown(self._streaming_buffer)
                self._live.update(md)
            except Exception:
                self._live.update(Text(self._streaming_buffer))

    def stop_streaming(self) -> str:
        """End streaming mode and return the complete output."""
        if self._live:
            self._live.stop()
            self._live = None
        result = self._streaming_buffer
        self._streaming_buffer = ""
        return result

    # ── Content rendering ───────────────────────────────────────────────

    def render_assistant_message(self, content: str) -> None:
        """Render a complete assistant message as Markdown."""
        try:
            self.console.print(Markdown(content))
        except Exception:
            self.console.print(content)

    def render_tool_call(self, name: str, arguments: str) -> None:
        """Render a tool call with its arguments."""
        # Parse arguments for pretty display
        try:
            import json
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
            args_display = json.dumps(args, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            args_display = str(arguments)

        header = Text()
        header.append("  ◆ ", style="magenta")
        header.append(name, style="bold magenta")

        self.console.print(header)

        # Show arguments based on tool type
        if name == "bash":
            command = args.get("command", args_display) if isinstance(args, dict) else args_display
            self.console.print(
                Syntax(command, "bash", padding=(0, 2), theme="monokai")
            )
        elif name in ("write_file", "edit_file"):
            self.console.print(
                Syntax(args_display, "json", padding=(0, 2), theme="monokai")
            )
        else:
            self.console.print(f"    {args_display}", style="dim")

    def render_tool_result(self, name: str, output: str, is_error: bool) -> None:
        """Render a tool result."""
        style = "red" if is_error else "green"
        icon = "✗" if is_error else "✓"

        # Truncate long output for display
        display = output
        if len(display) > 2000:
            display = display[:1000] + "\n[...truncated...]\n" + display[-500:]

        self.console.print(f"  [{style}]{icon}[/] {name}", style="dim")

        if display.strip():
            if name in ("read_file", "grep", "glob"):
                # Show file content with appropriate syntax highlighting
                self.console.print(
                    Syntax(display, "text", padding=(0, 2), theme="monokai", line_numbers=False)
                )
            else:
                self.console.print(f"    {display}", style="dim")

    def render_context_info(self, usage_ratio: float, compaction_count: int) -> None:
        """Show context window usage."""
        bar_width = 20
        filled = int(usage_ratio * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        color = "green" if usage_ratio < 0.6 else "yellow" if usage_ratio < 0.85 else "red"

        info = f"  Context: [{color}]{bar}[/] {usage_ratio:.0%}"
        if compaction_count > 0:
            info += f"  (compacted ×{compaction_count})"
        self.console.print(info, style="dim")

    # ── Permission prompts ──────────────────────────────────────────────

    def prompt_permission(self, tool_name: str, args: dict, risk_reason: str | None) -> bool:
        """Ask the user to approve a tool execution."""
        self.console.print()

        panel_content = Text()
        panel_content.append(f"Tool: {tool_name}\n", style="bold")

        if tool_name == "bash":
            cmd = args.get("command", "")
            panel_content.append(f"Command: {cmd}\n", style="cyan")
        elif tool_name in ("write_file", "edit_file"):
            path = args.get("file_path", "")
            panel_content.append(f"File: {path}\n", style="cyan")

        if risk_reason:
            panel_content.append(f"⚠ {risk_reason}\n", style="yellow")

        self.console.print(
            Panel(panel_content, title="Permission Required", border_style="yellow")
        )

        response = self.console.input("[yellow]Allow? (y)es / (n)o / (a)lways: [/]").strip().lower()

        return response in ("y", "yes", "a", "always"), response in ("a", "always")
