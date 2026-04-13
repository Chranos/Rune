"""Glob tool — fast file pattern matching."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from rune.tools.base import Tool, ToolResult


class GlobTool(Tool):
    name: ClassVar[str] = "glob"
    description: ClassVar[str] = (
        "Find files matching a glob pattern. "
        "Supports patterns like '**/*.py', 'src/**/*.ts', '*.json'. "
        "Returns matching file paths sorted by modification time."
    )
    is_read_only: ClassVar[bool] = True

    class InputSchema(BaseModel):
        pattern: str = Field(description="Glob pattern to match files (e.g. '**/*.py')")
        path: str | None = Field(
            default=None,
            description="Directory to search in. Defaults to current directory.",
        )

    async def execute(self, pattern: str, path: str | None = None, **kwargs: Any) -> ToolResult:
        search_dir = Path(path).expanduser() if path else Path.cwd()

        if not search_dir.exists():
            return ToolResult(f"Directory not found: {search_dir}", is_error=True)

        try:
            matches = sorted(search_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

            # Filter out common unwanted directories
            skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox"}
            filtered = [
                m for m in matches
                if not any(part in skip_dirs for part in m.parts) and m.is_file()
            ]

            if not filtered:
                return ToolResult(f"No files match pattern '{pattern}' in {search_dir}")

            # Limit output
            max_results = 200
            lines = [str(f) for f in filtered[:max_results]]
            result = "\n".join(lines)

            if len(filtered) > max_results:
                result += f"\n... and {len(filtered) - max_results} more files"

            return ToolResult(result, metadata={"count": len(filtered)})

        except Exception as e:
            return ToolResult(f"Glob error: {e}", is_error=True)
