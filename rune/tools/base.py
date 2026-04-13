"""Tool framework — base class, registry, and schema generation.

Each tool is defined as a subclass of `Tool` with:
- A Pydantic model for input validation (InputSchema)
- An async `execute()` method
- Metadata (name, description, is_read_only)

Tools are automatically registered and their JSON schemas are generated
for the LLM's function-calling interface.
"""

from __future__ import annotations

import abc
import json
import logging
from typing import Any, ClassVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ToolResult:
    """Encapsulates the result of a tool execution."""

    def __init__(
        self,
        output: str,
        is_error: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.output = output
        self.is_error = is_error
        self.metadata = metadata or {}

    def __str__(self) -> str:
        return self.output

    def truncate(self, max_length: int = 8000) -> ToolResult:
        """Return a truncated copy if output exceeds max_length."""
        if len(self.output) <= max_length:
            return self
        half = max_length // 2
        truncated = (
            self.output[:half]
            + f"\n\n... [truncated {len(self.output) - max_length} characters] ...\n\n"
            + self.output[-half:]
        )
        return ToolResult(truncated, self.is_error, self.metadata)


class Tool(abc.ABC):
    """Abstract base class for all Rune tools."""

    # Subclasses must define these
    name: ClassVar[str]
    description: ClassVar[str]
    is_read_only: ClassVar[bool] = False

    class InputSchema(BaseModel):
        """Override in subclass to define input parameters."""
        pass

    @abc.abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with validated input parameters."""
        ...

    @classmethod
    def get_function_schema(cls) -> dict[str, Any]:
        """Generate OpenAI function-calling compatible schema."""
        schema = cls.InputSchema.model_json_schema()

        # Clean up the schema for the LLM
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Remove Pydantic internal fields
        for key in list(properties.keys()):
            prop = properties[key]
            prop.pop("title", None)

        cleaned: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            cleaned["required"] = required

        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
                "parameters": cleaned,
            },
        }

    async def safe_execute(self, raw_args: str | dict) -> ToolResult:
        """Parse, validate, and execute with error handling."""
        try:
            # Parse raw arguments
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError as e:
                    return ToolResult(
                        f"Invalid JSON arguments: {e}",
                        is_error=True,
                    )
            else:
                args = raw_args

            # Validate with Pydantic
            validated = self.InputSchema.model_validate(args)
            kwargs = validated.model_dump()

            # Execute
            return await self.execute(**kwargs)

        except Exception as e:
            logger.exception("Tool %s execution failed", self.name)
            return ToolResult(
                f"Tool execution error: {type(e).__name__}: {e}",
                is_error=True,
            )


class ToolRegistry:
    """Registry for discovering and managing tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Get all tool schemas for the LLM."""
        return [tool.get_function_schema() for tool in self._tools.values()]

    def get_read_only_tools(self) -> list[str]:
        """Get names of read-only tools."""
        return [name for name, tool in self._tools.items() if tool.is_read_only]


def create_default_registry() -> ToolRegistry:
    """Create a registry with all built-in tools."""
    from rune.tools.bash import BashTool
    from rune.tools.edit_file import EditFileTool
    from rune.tools.glob_tool import GlobTool
    from rune.tools.grep_tool import GrepTool
    from rune.tools.read_file import ReadFileTool
    from rune.tools.write_file import WriteFileTool

    registry = ToolRegistry()
    registry.register(BashTool())
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(EditFileTool())
    registry.register(GlobTool())
    registry.register(GrepTool())
    return registry
