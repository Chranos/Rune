"""Grep tool — content search using regex."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from rune.tools.base import Tool, ToolResult


class GrepTool(Tool):
    name: ClassVar[str] = "grep"
    description: ClassVar[str] = (
        "Search file contents using regex patterns. "
        "Supports full regex syntax. Returns matching lines with file paths and line numbers."
    )
    is_read_only: ClassVar[bool] = True

    class InputSchema(BaseModel):
        pattern: str = Field(description="Regex pattern to search for")
        path: str | None = Field(
            default=None,
            description="File or directory to search in. Defaults to current directory.",
        )
        glob: str | None = Field(
            default=None,
            description="Glob pattern to filter files (e.g. '*.py')",
        )
        case_insensitive: bool = Field(
            default=False,
            description="Case insensitive search",
        )
        context: int = Field(
            default=0,
            description="Number of context lines before and after each match",
            ge=0,
            le=10,
        )

    async def execute(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        case_insensitive: bool = False,
        context: int = 0,
        **kwargs: Any,
    ) -> ToolResult:
        search_path = Path(path).expanduser() if path else Path.cwd()

        flags = re.IGNORECASE if case_insensitive else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return ToolResult(f"Invalid regex pattern: {e}", is_error=True)

        # Collect files to search
        if search_path.is_file():
            files = [search_path]
        else:
            file_pattern = glob or "**/*"
            skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv"}
            files = [
                f for f in search_path.glob(file_pattern)
                if f.is_file()
                and not any(part in skip_dirs for part in f.parts)
                and not _is_binary(f)
            ]

        results: list[str] = []
        max_results = 200
        match_count = 0

        for file_path in sorted(files):
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()

                for i, line in enumerate(lines):
                    if regex.search(line):
                        match_count += 1
                        if match_count > max_results:
                            break

                        # Add context lines
                        start = max(0, i - context)
                        end = min(len(lines), i + context + 1)

                        if context > 0:
                            results.append(f"--- {file_path}:{i + 1} ---")
                            for j in range(start, end):
                                marker = ">" if j == i else " "
                                results.append(f"{marker} {j + 1:>6}\t{lines[j]}")
                        else:
                            results.append(f"{file_path}:{i + 1}:\t{line}")

                if match_count > max_results:
                    break

            except (PermissionError, UnicodeDecodeError):
                continue

        if not results:
            return ToolResult(f"No matches found for pattern '{pattern}'")

        output = "\n".join(results)
        if match_count > max_results:
            output += f"\n\n... {match_count - max_results} more matches (truncated)"

        return ToolResult(output, metadata={"match_count": match_count})


def _is_binary(path: Path) -> bool:
    """Quick heuristic check if a file is binary."""
    try:
        chunk = path.read_bytes()[:1024]
        return b"\x00" in chunk
    except Exception:
        return True
