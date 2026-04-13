"""Bash tool — execute shell commands with safety controls."""

from __future__ import annotations

import asyncio
import os
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from rune.tools.base import Tool, ToolResult


class BashTool(Tool):
    name: ClassVar[str] = "bash"
    description: ClassVar[str] = (
        "Execute a bash command and return its output. "
        "Use for running tests, installing packages, git operations, "
        "and other shell tasks. Commands run in the current working directory."
    )
    is_read_only: ClassVar[bool] = False

    class InputSchema(BaseModel):
        command: str = Field(description="The bash command to execute")
        timeout: int = Field(
            default=120,
            description="Timeout in seconds (max 600)",
            ge=1,
            le=600,
        )

    async def execute(self, command: str, timeout: int = 120, **kwargs: Any) -> ToolResult:
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd(),
                env={**os.environ, "TERM": "dumb"},
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
                return ToolResult(
                    f"Command timed out after {timeout} seconds",
                    is_error=True,
                )

            output_parts = []
            if stdout:
                output_parts.append(stdout.decode("utf-8", errors="replace"))
            if stderr:
                output_parts.append(f"STDERR:\n{stderr.decode('utf-8', errors='replace')}")

            output = "\n".join(output_parts) if output_parts else "(no output)"
            is_error = process.returncode != 0

            if is_error:
                output = f"Exit code: {process.returncode}\n{output}"

            return ToolResult(output, is_error=is_error).truncate()

        except Exception as e:
            return ToolResult(f"Failed to execute command: {e}", is_error=True)
