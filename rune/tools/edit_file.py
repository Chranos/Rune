"""EditFile tool — precise string replacement in files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from rune.tools.base import Tool, ToolResult


class EditFileTool(Tool):
    name: ClassVar[str] = "edit_file"
    description: ClassVar[str] = (
        "Edit a file by replacing an exact string with a new string. "
        "The old_string must be unique in the file to avoid ambiguous edits. "
        "Use replace_all=true to replace all occurrences."
    )
    is_read_only: ClassVar[bool] = False

    class InputSchema(BaseModel):
        file_path: str = Field(description="Path to the file to edit")
        old_string: str = Field(description="The exact text to find and replace")
        new_string: str = Field(description="The replacement text")
        replace_all: bool = Field(
            default=False,
            description="If true, replace all occurrences instead of requiring uniqueness",
        )

    async def execute(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        path = Path(file_path).expanduser()

        if not path.exists():
            return ToolResult(f"File not found: {file_path}", is_error=True)

        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            return ToolResult(f"Error reading file: {e}", is_error=True)

        count = content.count(old_string)

        if count == 0:
            return ToolResult(
                f"old_string not found in {file_path}. Make sure it matches exactly.",
                is_error=True,
            )

        if count > 1 and not replace_all:
            return ToolResult(
                f"old_string found {count} times in {file_path}. "
                "Use replace_all=true to replace all, or provide a more specific string.",
                is_error=True,
            )

        new_content = content.replace(old_string, new_string) if replace_all else content.replace(
            old_string, new_string, 1
        )

        try:
            path.write_text(new_content, encoding="utf-8")
        except Exception as e:
            return ToolResult(f"Error writing file: {e}", is_error=True)

        replacements = count if replace_all else 1
        return ToolResult(f"Edited {file_path} ({replacements} replacement{'s' if replacements > 1 else ''})")
