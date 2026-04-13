"""Permission and safety system for Rune.

Handles:
- Dangerous command detection
- User confirmation prompts for write operations
- Operation auditing
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field

from rune.config.settings import SafetyConfig

logger = logging.getLogger(__name__)


class PermissionLevel(enum.Enum):
    """Permission levels for tool execution."""

    AUTO = "auto"  # Always allowed (read-only tools)
    CONFIRM = "confirm"  # Requires user confirmation
    DENY = "deny"  # Always blocked


@dataclass
class PermissionRequest:
    """A request to execute a tool, pending approval."""

    tool_name: str
    arguments: dict
    risk_level: PermissionLevel
    risk_reason: str | None = None


class PermissionManager:
    """Manages tool execution permissions and safety checks."""

    def __init__(self, config: SafetyConfig) -> None:
        self.config = config
        self._session_grants: set[str] = set()  # "tool:pattern" grants for this session
        self._audit_log: list[dict] = []

    def check_permission(self, tool_name: str, arguments: dict) -> PermissionRequest:
        """Evaluate whether a tool call should be auto-approved, confirmed, or denied."""
        # Read-only tools are always allowed
        if self.config.auto_approve_reads and tool_name in _READ_ONLY_TOOLS:
            return PermissionRequest(tool_name, arguments, PermissionLevel.AUTO)

        # Check for dangerous patterns in bash commands
        if tool_name == "bash":
            command = arguments.get("command", "")
            danger = self._check_dangerous_command(command)
            if danger:
                return PermissionRequest(
                    tool_name, arguments, PermissionLevel.CONFIRM,
                    risk_reason=f"Potentially dangerous: {danger}",
                )

        # Check session-level grants
        grant_key = f"{tool_name}:*"
        if grant_key in self._session_grants:
            return PermissionRequest(tool_name, arguments, PermissionLevel.AUTO)

        # Write operations need confirmation by default
        if tool_name not in _READ_ONLY_TOOLS:
            return PermissionRequest(tool_name, arguments, PermissionLevel.CONFIRM)

        return PermissionRequest(tool_name, arguments, PermissionLevel.AUTO)

    def _check_dangerous_command(self, command: str) -> str | None:
        """Check if a bash command matches any dangerous patterns."""
        command_lower = command.lower().strip()
        for pattern in self.config.dangerous_patterns:
            if pattern.lower() in command_lower:
                return pattern
        return None

    def grant_session_permission(self, tool_name: str) -> None:
        """Grant blanket permission for a tool for the rest of the session."""
        self._session_grants.add(f"{tool_name}:*")
        logger.info("Session permission granted for tool: %s", tool_name)

    def record_action(self, tool_name: str, arguments: dict, result_summary: str) -> None:
        """Record an action in the audit log."""
        self._audit_log.append({
            "tool": tool_name,
            "args": arguments,
            "result": result_summary[:200],
        })

    def get_audit_log(self) -> list[dict]:
        return list(self._audit_log)


# Tools that only read and never modify state
_READ_ONLY_TOOLS = {"read_file", "glob", "grep"}
