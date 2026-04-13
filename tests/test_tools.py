"""Tests for the Tool system."""

import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest

from rune.tools.base import ToolRegistry, create_default_registry
from rune.tools.bash import BashTool
from rune.tools.edit_file import EditFileTool
from rune.tools.glob_tool import GlobTool
from rune.tools.grep_tool import GrepTool
from rune.tools.read_file import ReadFileTool
from rune.tools.write_file import WriteFileTool


@pytest.fixture
def registry():
    return create_default_registry()


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


class TestToolRegistry:
    def test_all_tools_registered(self, registry: ToolRegistry):
        tools = registry.list_tools()
        names = {t.name for t in tools}
        assert names == {"bash", "read_file", "write_file", "edit_file", "glob", "grep"}

    def test_get_tool(self, registry: ToolRegistry):
        tool = registry.get("bash")
        assert tool is not None
        assert tool.name == "bash"

    def test_get_unknown_tool(self, registry: ToolRegistry):
        assert registry.get("nonexistent") is None

    def test_schemas_generated(self, registry: ToolRegistry):
        schemas = registry.get_schemas()
        assert len(schemas) == 6
        for s in schemas:
            assert s["type"] == "function"
            assert "function" in s
            assert "name" in s["function"]
            assert "parameters" in s["function"]

    def test_read_only_tools(self, registry: ToolRegistry):
        read_only = registry.get_read_only_tools()
        assert "read_file" in read_only
        assert "glob" in read_only
        assert "grep" in read_only
        assert "bash" not in read_only


class TestBashTool:
    @pytest.mark.asyncio
    async def test_simple_command(self):
        tool = BashTool()
        result = await tool.safe_execute({"command": "echo hello"})
        assert "hello" in result.output
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_command_failure(self):
        tool = BashTool()
        result = await tool.safe_execute({"command": "false"})
        assert result.is_error

    @pytest.mark.asyncio
    async def test_command_timeout(self):
        tool = BashTool()
        result = await tool.safe_execute({"command": "sleep 10", "timeout": 1})
        assert result.is_error
        assert "timed out" in result.output

    @pytest.mark.asyncio
    async def test_command_with_stderr(self):
        tool = BashTool()
        result = await tool.safe_execute({"command": "echo error >&2"})
        assert "error" in result.output


class TestReadFileTool:
    @pytest.mark.asyncio
    async def test_read_file(self, tmp_dir: Path):
        test_file = tmp_dir / "test.txt"
        test_file.write_text("line1\nline2\nline3\n")

        tool = ReadFileTool()
        result = await tool.safe_execute({"file_path": str(test_file)})
        assert not result.is_error
        assert "line1" in result.output
        assert "line2" in result.output

    @pytest.mark.asyncio
    async def test_read_with_offset_limit(self, tmp_dir: Path):
        test_file = tmp_dir / "test.txt"
        test_file.write_text("line1\nline2\nline3\nline4\nline5\n")

        tool = ReadFileTool()
        result = await tool.safe_execute({"file_path": str(test_file), "offset": 2, "limit": 2})
        assert not result.is_error
        assert "line2" in result.output
        assert "line3" in result.output
        assert "line1" not in result.output

    @pytest.mark.asyncio
    async def test_read_nonexistent(self):
        tool = ReadFileTool()
        result = await tool.safe_execute({"file_path": "/nonexistent/file.txt"})
        assert result.is_error


class TestWriteFileTool:
    @pytest.mark.asyncio
    async def test_write_new_file(self, tmp_dir: Path):
        test_file = tmp_dir / "new.txt"
        tool = WriteFileTool()
        result = await tool.safe_execute({
            "file_path": str(test_file),
            "content": "hello world\n",
        })
        assert not result.is_error
        assert "Created" in result.output
        assert test_file.read_text() == "hello world\n"

    @pytest.mark.asyncio
    async def test_write_creates_directories(self, tmp_dir: Path):
        test_file = tmp_dir / "sub" / "dir" / "file.txt"
        tool = WriteFileTool()
        result = await tool.safe_execute({
            "file_path": str(test_file),
            "content": "nested\n",
        })
        assert not result.is_error
        assert test_file.exists()


class TestEditFileTool:
    @pytest.mark.asyncio
    async def test_edit_file(self, tmp_dir: Path):
        test_file = tmp_dir / "edit.txt"
        test_file.write_text("hello world\n")

        tool = EditFileTool()
        result = await tool.safe_execute({
            "file_path": str(test_file),
            "old_string": "world",
            "new_string": "rune",
        })
        assert not result.is_error
        assert test_file.read_text() == "hello rune\n"

    @pytest.mark.asyncio
    async def test_edit_ambiguous(self, tmp_dir: Path):
        test_file = tmp_dir / "edit.txt"
        test_file.write_text("aaa\naaa\n")

        tool = EditFileTool()
        result = await tool.safe_execute({
            "file_path": str(test_file),
            "old_string": "aaa",
            "new_string": "bbb",
        })
        assert result.is_error
        assert "2 times" in result.output

    @pytest.mark.asyncio
    async def test_edit_replace_all(self, tmp_dir: Path):
        test_file = tmp_dir / "edit.txt"
        test_file.write_text("aaa\naaa\n")

        tool = EditFileTool()
        result = await tool.safe_execute({
            "file_path": str(test_file),
            "old_string": "aaa",
            "new_string": "bbb",
            "replace_all": True,
        })
        assert not result.is_error
        assert test_file.read_text() == "bbb\nbbb\n"


class TestGlobTool:
    @pytest.mark.asyncio
    async def test_glob_pattern(self, tmp_dir: Path):
        (tmp_dir / "a.py").write_text("# a")
        (tmp_dir / "b.py").write_text("# b")
        (tmp_dir / "c.txt").write_text("c")

        tool = GlobTool()
        result = await tool.safe_execute({"pattern": "*.py", "path": str(tmp_dir)})
        assert not result.is_error
        assert "a.py" in result.output
        assert "b.py" in result.output
        assert "c.txt" not in result.output


class TestGrepTool:
    @pytest.mark.asyncio
    async def test_grep_pattern(self, tmp_dir: Path):
        (tmp_dir / "a.py").write_text("def hello():\n    pass\n")
        (tmp_dir / "b.py").write_text("def world():\n    pass\n")

        tool = GrepTool()
        result = await tool.safe_execute({
            "pattern": "def hello",
            "path": str(tmp_dir),
        })
        assert not result.is_error
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_grep_no_match(self, tmp_dir: Path):
        (tmp_dir / "a.py").write_text("def hello():\n    pass\n")

        tool = GrepTool()
        result = await tool.safe_execute({
            "pattern": "nonexistent",
            "path": str(tmp_dir),
        })
        assert "No matches" in result.output
