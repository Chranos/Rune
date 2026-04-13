"""Configuration management for Rune."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


# Default configuration directory
RUNE_HOME = Path(os.environ.get("RUNE_HOME", Path.home() / ".rune"))
DEFAULT_CONFIG_PATH = RUNE_HOME / "config.yaml"
HISTORY_PATH = RUNE_HOME / "history"


@dataclass
class ModelConfig:
    """Configuration for a single model endpoint."""

    name: str = "default"
    base_url: str = "http://127.0.0.1:8080"
    api_key: str = "not-needed"
    model: str = "local-model"
    max_tokens: int = 4096
    temperature: float = 0.7
    context_window: int = 8192
    supports_tool_use: bool = True


@dataclass
class SafetyConfig:
    """Configuration for safety and permissions."""

    # Commands that always require user confirmation
    dangerous_patterns: list[str] = field(default_factory=lambda: [
        "rm -rf", "rm -r", "rmdir",
        "git push --force", "git reset --hard",
        "DROP TABLE", "DELETE FROM",
        "sudo", "chmod 777",
        "> /dev/", "mkfs",
    ])
    # Whether to auto-approve read-only tool calls
    auto_approve_reads: bool = True
    # Whether to sandbox bash commands
    sandbox_bash: bool = False


@dataclass
class UIConfig:
    """Configuration for UI rendering."""

    theme: str = "monokai"
    show_thinking: bool = False
    show_token_count: bool = True
    markdown_enabled: bool = True


@dataclass
class RuneConfig:
    """Top-level Rune configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    # Session
    max_conversation_turns: int = 100
    auto_compact_threshold: float = 0.75  # Compact when context usage exceeds this ratio

    @classmethod
    def load(cls, path: Path | None = None) -> RuneConfig:
        """Load configuration from a YAML file, falling back to defaults."""
        config_path = path or DEFAULT_CONFIG_PATH
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
            return cls._from_dict(data)
        return cls()

    @classmethod
    def _from_dict(cls, data: dict) -> RuneConfig:
        """Build config from a dictionary, applying only recognized fields."""
        model_data = data.get("model", {})
        safety_data = data.get("safety", {})
        ui_data = data.get("ui", {})

        model = ModelConfig(**{
            k: v for k, v in model_data.items()
            if k in ModelConfig.__dataclass_fields__
        })
        safety = SafetyConfig(**{
            k: v for k, v in safety_data.items()
            if k in SafetyConfig.__dataclass_fields__
        })
        ui = UIConfig(**{
            k: v for k, v in ui_data.items()
            if k in UIConfig.__dataclass_fields__
        })

        return cls(
            model=model,
            safety=safety,
            ui=ui,
            max_conversation_turns=data.get("max_conversation_turns", 100),
            auto_compact_threshold=data.get("auto_compact_threshold", 0.75),
        )

    def save(self, path: Path | None = None) -> None:
        """Persist configuration to a YAML file."""
        config_path = path or DEFAULT_CONFIG_PATH
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": {
                "name": self.model.name,
                "base_url": self.model.base_url,
                "api_key": self.model.api_key,
                "model": self.model.model,
                "max_tokens": self.model.max_tokens,
                "temperature": self.model.temperature,
                "context_window": self.model.context_window,
                "supports_tool_use": self.model.supports_tool_use,
            },
            "safety": {
                "dangerous_patterns": self.safety.dangerous_patterns,
                "auto_approve_reads": self.safety.auto_approve_reads,
                "sandbox_bash": self.safety.sandbox_bash,
            },
            "ui": {
                "theme": self.ui.theme,
                "show_thinking": self.ui.show_thinking,
                "show_token_count": self.ui.show_token_count,
                "markdown_enabled": self.ui.markdown_enabled,
            },
            "max_conversation_turns": self.max_conversation_turns,
            "auto_compact_threshold": self.auto_compact_threshold,
        }

        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
