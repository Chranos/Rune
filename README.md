# Rune ✦

**Local-First AI Coding Agent for Your Terminal**

Rune is an open-source AI coding assistant that runs entirely on your local machine. Powered by open-source LLMs via [llama.cpp](https://github.com/ggerganov/llama.cpp), it provides a Claude Code-like experience with zero API costs and complete privacy.

## Features

- **Fully Local** — Your code never leaves your machine. No API keys, no cloud dependencies.
- **Tool-Augmented Agent** — ReAct loop with bash, file read/write/edit, glob, and grep tools.
- **Smart Context Management** — Intelligent compaction strategies for small context windows (8K).
- **Streaming Output** — Real-time response streaming with rich Markdown rendering.
- **Permission System** — Safety checks for destructive operations with user confirmation.
- **Model Agnostic** — Works with any OpenAI-compatible API (llama.cpp, ollama, vLLM, LM Studio).

## Quick Start

### Prerequisites

- Python 3.11+
- A local LLM server running an OpenAI-compatible API (e.g., llama.cpp with `--jinja` flag)

### Install

```bash
pip install -e .
```

### Run

```bash
# Start your local model server first
./start-qwen35.sh

# In another terminal, launch Rune
rune
```

### Configuration

```bash
# Custom endpoint
rune --base-url http://localhost:11434/v1

# Custom context window
rune --context-window 16384
```

Configuration file: `~/.rune/config.yaml`

## Architecture

```
rune/
├── cli.py              # Entry point & REPL (prompt_toolkit)
├── agent/
│   ├── loop.py         # Core ReAct agent loop
│   └── context.py      # Context window management & auto-compaction
├── llm/
│   ├── client.py       # Async OpenAI-compatible client (httpx)
│   └── streaming.py    # SSE stream parser & accumulator
├── tools/
│   ├── base.py         # Tool framework with Pydantic schema generation
│   ├── bash.py         # Shell command execution
│   ├── read_file.py    # File reading with line numbers
│   ├── write_file.py   # File creation/overwriting
│   ├── edit_file.py    # Precise string replacement editing
│   ├── glob_tool.py    # File pattern matching
│   └── grep_tool.py    # Content search with regex
├── safety/
│   └── permissions.py  # Permission checking & audit logging
├── ui/
│   └── renderer.py     # Rich terminal UI rendering
└── config/
    └── settings.py     # YAML-based configuration management
```

## Key Technical Highlights

### ReAct Agent Loop
Implements the Observe → Think → Act cycle with automatic error recovery and parallel tool execution for read-only operations.

### Smart Context Compaction
Novel approach to managing small context windows (8K tokens):
- Automatic summarization of old conversation turns
- Aggressive truncation of tool outputs
- Token estimation heuristics (CJK-aware, no tokenizer dependency)

### Tool System
Extensible tool framework with:
- Pydantic-based input validation and JSON Schema generation
- Automatic OpenAI function-calling schema export
- Safety-aware execution with permission levels

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| HTTP Client | httpx (async) |
| Validation | Pydantic v2 |
| Terminal UI | Rich + prompt_toolkit |
| CLI Framework | Click |
| Config | PyYAML |
| Testing | pytest + pytest-asyncio |

## License

MIT
