"""WriteFile tool — create or overwrite files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from rune.tools.base import Tool, ToolResult


class WriteFileTool(Tool):
    name: ClassVar[str] = "write_file"
    description: ClassVar[str] = (
        "Write content to a file. Creates the file if it doesn't exist, "
        "or overwrites it if it does. Parent directories are created automatically."
    )
    is_read_only: ClassVar[bool] = False

    class InputSchema(BaseModel):
        file_path: str = Field(description="Absolute or relative path to the file")
        content: str = Field(description="The content to write to the file")

    async def execute(self, file_path: str, content: str, **kwargs: Any) -> ToolResult:
        path = Path(file_path).expanduser()

        try:
            # Ensure parent directories exist
            path.parent.mkdir(parents=True, exist_ok=True)

            existed = path.exists()
            path.write_text(content, encoding="utf-8")

            lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            action = "Updated" if existed else "Created"
            return ToolResult(f"{action} {file_path} ({lines} lines)")

        except PermissionError:
            return ToolResult(f"Permission denied: {file_path}", is_error=True)
        except Exception as e:
            return ToolResult(f"Error writing file: {e}", is_error=True)
