"""ReadFile tool — read file contents with line numbers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from rune.tools.base import Tool, ToolResult


class ReadFileTool(Tool):
    name: ClassVar[str] = "read_file"
    description: ClassVar[str] = (
        "Read the contents of a file. Returns the file content with line numbers. "
        "You can optionally specify offset and limit to read a portion of the file."
    )
    is_read_only: ClassVar[bool] = True

    class InputSchema(BaseModel):
        file_path: str = Field(description="Absolute or relative path to the file to read")
        offset: int | None = Field(
            default=None,
            description="Line number to start reading from (1-based)",
            ge=1,
        )
        limit: int | None = Field(
            default=None,
            description="Maximum number of lines to read",
            ge=1,
        )

    async def execute(
        self, file_path: str, offset: int | None = None, limit: int | None = None, **kwargs: Any
    ) -> ToolResult:
        path = Path(file_path).expanduser()

        if not path.exists():
            return ToolResult(f"File not found: {file_path}", is_error=True)

        if path.is_dir():
            return ToolResult(f"Path is a directory, not a file: {file_path}", is_error=True)

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except PermissionError:
            return ToolResult(f"Permission denied: {file_path}", is_error=True)
        except Exception as e:
            return ToolResult(f"Error reading file: {e}", is_error=True)

        lines = content.splitlines()
        total_lines = len(lines)

        # Apply offset and limit
        start = (offset - 1) if offset else 0
        end = (start + limit) if limit else total_lines

        selected = lines[start:end]

        # Format with line numbers
        numbered = []
        for i, line in enumerate(selected, start=start + 1):
            numbered.append(f"{i:>6}\t{line}")

        result = "\n".join(numbered)

        if not result:
            result = "(empty file)"

        metadata = {"total_lines": total_lines, "showing": f"{start + 1}-{min(end, total_lines)}"}
        return ToolResult(result, metadata=metadata)
