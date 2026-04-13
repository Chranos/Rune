"""Model configuration and management."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Runtime information about a loaded model."""

    model_id: str
    context_window: int
    supports_tool_use: bool
    supports_streaming: bool = True

    @property
    def effective_context(self) -> int:
        """Usable context after reserving space for response."""
        # Reserve 25% of context for model's response
        return int(self.context_window * 0.75)
